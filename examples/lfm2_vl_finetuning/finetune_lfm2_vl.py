# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LoRA fine-tuning for Liquid AI's LFM2-VL-450M vision-language model on a *tiny*
image-text dataset (fewer than 50 images).

LFM2-VL-450M (https://huggingface.co/LiquidAI/LFM2-VL-450M) pairs a 350M LFM2
language backbone with an 86M SigLIP2 NaFlex vision encoder. It is small enough
to run in a web browser, which makes it a great target for quick, low-resource
fine-tuning experiments.

This script follows the path recommended by Liquid AI (LoRA adapters + TRL's
`SFTTrainer`) and is deliberately minimal so the whole loop fits on a single
small GPU (or even CPU for a smoke test).

Two data sources are supported:

* ``--demo`` (default): a fully synthetic, self-contained dataset of colored
  geometric shapes generated with Pillow. No download required, exactly
  ``--max_samples`` images, perfectly reproducible. Great for verifying the
  pipeline end-to-end and for teaching the model a brand-new behaviour you can
  measure (it learns to answer "What colored shape is this?").
* ``--dataset_name <hub id>``: a slice of a real vision-language dataset from the
  Hub (e.g. ``HuggingFaceH4/llava-instruct-mix-vsft``), capped to
  ``--max_samples`` (< 50) examples.

Example
-------
Synthetic smoke test (no dataset download, ~16 images)::

    python finetune_lfm2_vl.py --demo --max_samples 16 --num_train_epochs 8

Tiny slice of a real dataset (40 images)::

    python finetune_lfm2_vl.py \
        --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
        --max_samples 40
"""

import argparse
import logging
import random

import torch
from PIL import Image, ImageDraw

from transformers import AutoModelForImageTextToText, AutoProcessor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for LFM2-VL on a tiny image-text dataset.")

    # Model
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-VL-450M", help="Hub id of the LFM2-VL model.")
    parser.add_argument("--output_dir", type=str, default="./lfm2-vl-450m-lora", help="Where to save the adapter.")

    # Data
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the self-contained synthetic shape dataset instead of a Hub dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hub dataset id (image-text). Ignored when --demo is set.",
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to slice from.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=16,
        help="Number of (image, text) pairs to train on. Kept below 50 on purpose.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

    # Training
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length (text + image tokens).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip the before/after qualitative comparison at the end.",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Synthetic dataset: colored geometric shapes (< 50 images, fully offline)
# --------------------------------------------------------------------------------------
SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (220, 50, 50),
    "green": (50, 180, 80),
    "blue": (60, 90, 220),
    "yellow": (240, 200, 40),
}


def _draw_shape(color_name, shape_name, size=256):
    """Render a single solid shape on a white background."""
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    fill = COLORS[color_name]
    pad = size // 5
    box = [pad, pad, size - pad, size - pad]
    if shape_name == "circle":
        draw.ellipse(box, fill=fill)
    elif shape_name == "square":
        draw.rectangle(box, fill=fill)
    else:  # triangle
        draw.polygon([(size // 2, pad), (pad, size - pad), (size - pad, size - pad)], fill=fill)
    return image


def build_demo_dataset(max_samples, seed):
    """Build up to ``max_samples`` (image, question, answer) records of colored shapes."""
    rng = random.Random(seed)
    combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(combos)
    if max_samples < len(combos):
        combos = combos[:max_samples]
    else:
        # repeat combos so every (color, shape) pair is well represented
        combos = [combos[i % len(combos)] for i in range(max_samples)]

    records = []
    for color_name, shape_name in combos:
        image = _draw_shape(color_name, shape_name)
        records.append(
            {
                "image": image,
                "question": "What colored shape is in this image? Answer with '<color> <shape>'.",
                "answer": f"{color_name} {shape_name}",
            }
        )
    logger.info("Built synthetic demo dataset with %d images.", len(records))
    return records


# --------------------------------------------------------------------------------------
# Hub dataset loading + normalization to a common conversation format
# --------------------------------------------------------------------------------------
def _to_conversation(record):
    """Turn a normalized {image, question, answer} record into an LFM2-VL chat conversation."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record["image"]},
                    {"type": "text", "text": record["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record["answer"]}],
            },
        ]
    }


def _normalize_hub_record(example):
    """Best-effort mapping of a Hub example to {image, question, answer}.

    Handles the two most common vision-SFT layouts:
      * TRL/LLaVA style: ``images`` (list) + ``messages`` (chat with image/text turns)
      * Flat VQA/caption style: ``image`` + (``question``/``query``) + (``answer``/``caption``/``label``)
    """
    # TRL / LLaVA "messages + images" layout ------------------------------------------
    if "messages" in example and ("images" in example or "image" in example):
        images = example.get("images") or [example["image"]]
        question, answer = "", ""
        for msg in example["messages"]:
            content = msg["content"]
            # content may be a plain string or a list of typed parts. Image parts
            # often carry a ``text`` key explicitly set to ``None``, so guard with ``or ""``.
            if isinstance(content, list):
                text = " ".join((part.get("text") or "") for part in content if isinstance(part, dict))
            else:
                text = content
            text = text.replace("<image>", "").strip()
            if msg["role"] == "user" and not question:
                question = text
            elif msg["role"] == "assistant" and not answer:
                answer = text
        return {"image": images[0], "question": question or "Describe this image.", "answer": answer}

    # Flat VQA / captioning layout ----------------------------------------------------
    image = example.get("image") or (example.get("images") or [None])[0]
    question = example.get("question") or example.get("query") or "Describe this image."
    answer = example.get("answer") or example.get("caption") or example.get("label") or example.get("text") or ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return {"image": image, "question": str(question), "answer": str(answer)}


def build_hub_dataset(dataset_name, split, max_samples):
    from datasets import load_dataset

    logger.info("Loading %d examples from '%s' (split=%s)...", max_samples, dataset_name, split)
    # Slice in the split string so we never download more than we need.
    ds = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
    records = []
    for example in ds:
        record = _normalize_hub_record(example)
        if record["image"] is None or not record["answer"]:
            continue
        if record["image"].mode != "RGB":
            record["image"] = record["image"].convert("RGB")
        records.append(record)
    logger.info("Prepared %d usable image-text pairs.", len(records))
    return records


# --------------------------------------------------------------------------------------
# Inference helper (used for the before/after comparison)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate_answer(model, processor, image, question, max_new_tokens=32):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Only decode the newly generated tokens.
    generated = output[0, inputs["input_ids"].shape[1] :]
    return processor.decode(generated, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.demo and args.dataset_name is None:
        logger.info("No --dataset_name provided; falling back to the synthetic --demo dataset.")
        args.demo = True
    if args.max_samples >= 50:
        raise ValueError("This example is meant for tiny datasets: please keep --max_samples below 50.")

    # ---- Processor & model ----------------------------------------------------------
    logger.info("Loading processor and model '%s'...", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=dtype)

    # ---- Data -----------------------------------------------------------------------
    if args.demo:
        records = build_demo_dataset(args.max_samples, args.seed)
    else:
        records = build_hub_dataset(args.dataset_name, args.dataset_split, args.max_samples)
    if not records:
        raise RuntimeError("No usable training records were produced. Check the dataset/columns.")

    train_dataset = [_to_conversation(r) for r in records]
    # Keep one held-out example for the qualitative before/after check.
    eval_record = records[0]

    # ---- Optional: model behaviour BEFORE fine-tuning -------------------------------
    if not args.no_eval:
        model.eval()
        before = generate_answer(model, processor, eval_record["image"], eval_record["question"])
        logger.info("[before] Q: %s", eval_record["question"])
        logger.info("[before] A: %s", before)

    # ---- LoRA -----------------------------------------------------------------------
    from peft import LoraConfig

    # Adapt both the language backbone projections and the multimodal connector.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # ---- Collator: build a padded batch and mask pad + image tokens in the labels ---
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_id = processor.image_token_id

    def collate_fn(examples):
        batch = processor.apply_chat_template(
            [ex["messages"] for ex in examples],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        # Do not compute loss on padding or on the image placeholder tokens.
        labels[labels == pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    # ---- Trainer --------------------------------------------------------------------
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=1,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        report_to="none",
        seed=args.seed,
        # We feed a list of pre-formatted conversations and do tokenization in the
        # collator, so disable TRL's own dataset preparation / column pruning.
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )

    logger.info("Starting LoRA fine-tuning on %d images...", len(train_dataset))
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Saved LoRA adapter and processor to '%s'.", args.output_dir)

    # ---- Optional: model behaviour AFTER fine-tuning --------------------------------
    if not args.no_eval:
        trainer.model.eval()
        after = generate_answer(trainer.model, processor, eval_record["image"], eval_record["question"])
        logger.info("[after]  Q: %s", eval_record["question"])
        logger.info("[after]  A: %s", after)
        logger.info("[target] A: %s", eval_record["answer"])


if __name__ == "__main__":
    main()
