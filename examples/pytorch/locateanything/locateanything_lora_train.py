#!/usr/bin/env python
"""Minimal native Transformers LoRA fine-tuning smoke test for LocateAnything.

This example exercises standard autoregressive next-token prediction (NTP).
It does not reproduce NVIDIA's custom PBD/MTP continual-SFT training pipeline.
"""

import argparse
from io import BytesIO
from pathlib import Path

import requests
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from transformers import AutoProcessor, LocateAnythingForConditionalGeneration, set_seed


MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
IMAGE_URL = "https://live.staticflickr.com/6061/6097691785_925e4687b4_o.jpg"
QUESTION = (
    "Locate all the instances that matches the following description: "
    "forklift</c>pallet</c>stacked supply boxes</c>warehouse aisle."
)
ANSWER = (
    "<ref>forklift</ref><box><586><126><979><956></box>"
    "<ref>pallet</ref><box><388><113><597><951></box><box><588><191><645><831></box>"
    "<ref>stacked supply boxes</ref><box><322><264><396><826></box>"
    "<box><388><113><597><951></box><box><588><191><645><831></box>"
    "<ref>warehouse aisle</ref><box><0><0><1000><1000></box>"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--image-url", default=IMAGE_URL)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--answer", default=ANSWER)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--in-token-limit",
        type=int,
        default=8192,
        help="Maximum number of vision patches before projector merging.",
    )
    parser.add_argument("--output-dir", default="locateanything-warehouse-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate image processing and assistant-only labels without loading model weights.",
    )
    return parser.parse_args()


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def prepare_training_batch(
    processor,
    image: Image.Image,
    question: str,
    answer: str,
    in_token_limit: int,
):
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }
    prompt = processor.apply_chat_template(
        [user_message],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_conversation = processor.apply_chat_template(
        [
            user_message,
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        add_generation_prompt=False,
        tokenize=False,
    )

    processor_kwargs = {
        "images": image,
        "return_tensors": "pt",
        "in_token_limit": in_token_limit,
    }
    batch = processor(text=[full_conversation], **processor_kwargs)
    prompt_batch = processor(text=[prompt], **processor_kwargs)
    prompt_length = prompt_batch["input_ids"].shape[1]

    if not torch.equal(batch["input_ids"][:, :prompt_length], prompt_batch["input_ids"]):
        raise ValueError("The prompt is not an exact prefix of the supervised conversation.")

    labels = batch["input_ids"].clone()
    labels[:, :prompt_length] = -100
    labels[batch["attention_mask"] == 0] = -100
    if not torch.any(labels != -100):
        raise ValueError("No assistant tokens remain after label masking.")
    batch["labels"] = labels
    return batch


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.in_token_limit < 4:
        raise ValueError("--in-token-limit must be at least 4.")

    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    image = download_image(args.image_url)
    batch = prepare_training_batch(
        processor,
        image,
        args.question,
        args.answer,
        args.in_token_limit,
    )

    supervised_ids = batch["labels"][batch["labels"] != -100]
    decoded_target = processor.decode(supervised_ids, skip_special_tokens=False)
    print(f"image_size={image.size}")
    print(f"sequence_tokens={batch['input_ids'].shape[1]}")
    print(f"supervised_tokens={supervised_ids.numel()}")
    print(f"decoded_target={decoded_target!r}")
    if args.answer not in decoded_target:
        raise ValueError("The decoded supervised tokens do not contain the expected answer.")
    if args.dry_run:
        print("Dry run passed; model weights were not loaded.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Use an A10G, L4, A100, H100, or newer GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This smoke test requires a bfloat16-capable GPU.")

    model = LocateAnythingForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()
    model.train()

    batch = {name: tensor.to("cuda") for name, tensor in batch.items()}
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=1.0,
        )
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f} grad_norm={float(grad_norm):.6f}")

    output_dir = Path(args.output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and processor to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
