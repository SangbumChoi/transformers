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
Where should the LoRA budget go? A component / target-module ablation for LFM2-VL.

Runs several LoRA configurations on the *same* train / held-out-test split (from
``finetune_lfm2_vl.build_demo_split``) and reports, per config: number of adapted
layers, trainable parameters, train-set accuracy (memorization), held-out accuracy
(generalization) and the median relative weight update ||ΔW|| / ||W||.

Configurations (``--max_samples`` train images, see ``finetune_lfm2_vl`` for the data):

* ``attn_both``     – attention q/k/v/out_proj only, vision + language (the leanest)
* ``all_both``      – every transformer linear (attn + conv + MLP), vision + language
* ``vision_all``    – every transformer linear, vision tower only
* ``language_all``  – every transformer linear, language model only

It also re-runs ``vision_all`` and ``language_all`` at ranks chosen so their trainable
parameter counts roughly match, giving a *parameter-budget-controlled* vision-vs-language
comparison.

Run from the repository root::

    python examples/lfm2_vl_finetuning/ablation_components.py

See ABLATION.md for a pre-computed results table and the takeaways.
"""

import argparse
import json

import finetune_lfm2_vl as L
import torch
import torch.nn as nn

from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "LiquidAI/LFM2-VL-450M"
SEED = 42

ATTN = {"q_proj", "k_proj", "v_proj", "out_proj"}
# All transformer linears (excludes the lm_head and the vision patch-embedding):
# attention (q/k/v/out_proj), LFM2 conv mixer (in_proj/out_proj), LFM2 MLP (w1/w2/w3),
# vision MLP (fc1/fc2) and the multimodal connector (linear_1/linear_2).
CURATED = ATTN | {"in_proj", "w1", "w2", "w3", "fc1", "fc2", "linear_1", "linear_2"}
SKIP = {"lm_head", "patch_embedding"}

# (config name, LoRA rank). The last two are the parameter-matched counterparts.
CONFIGS = [
    ("attn_both", 16),
    ("all_both", 16),
    ("vision_all", 16),
    ("language_all", 16),
    ("vision_all", 36),  # ~match language_all @ r=16 (~6.0M params)
    ("language_all", 7),  # ~match vision_all @ r=16 (~2.65M params)
]


def component(name):
    if "vision_tower" in name:
        return "vision"
    if "language_model" in name:
        return "language"
    return "connector"


def target_names(model, which):
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = name.split(".")[-1]
        if suffix in SKIP:
            continue
        comp = component(name)
        if which == "attn_both" and suffix in ATTN:
            names.append(name)
        elif which == "all_both" and suffix in CURATED:
            names.append(name)
        elif which == "vision_all" and suffix in CURATED and comp == "vision":
            names.append(name)
        elif which == "language_all" and suffix in CURATED and comp == "language":
            names.append(name)
    return names


def median_relative_update(model):
    rels = []
    for _, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None or "default" not in lora_a:
            continue
        A = module.lora_A["default"].weight.detach().float()
        B = module.lora_B["default"].weight.detach().float()
        dW = module.scaling["default"] * (B @ A)
        rels.append((dW.norm() / module.base_layer.weight.detach().float().norm()).item())
    rels.sort()
    return rels[len(rels) // 2] if rels else 0.0


def run_config(which, rank, processor, train_records, test_records, train_dataset, epochs, lr):
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)
    targets = target_names(model, which)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.05,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pad_id, img_id = processor.tokenizer.pad_token_id, processor.image_token_id

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
        labels[labels == pad_id] = -100
        labels[labels == img_id] = -100
        batch["labels"] = labels
        return batch

    sft = SFTConfig(
        output_dir=f"/tmp/abl_{which}_{rank}",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        max_length=1024,
        logging_steps=10,
        bf16=False,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )
    trainer.train()
    trainer.model.eval()

    _, train_acc = L.evaluate(trainer.model, processor, train_records)
    _, test_acc = L.evaluate(trainer.model, processor, test_records)
    result = {
        "config": which,
        "rank": rank,
        "n_layers": len(targets),
        "trainable_params": trainable,
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "median_rel_update": round(median_relative_update(trainer.model), 4),
    }
    print("RESULT", json.dumps(result))
    del model, trainer
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--n_test", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--out", type=str, default="ablation_results.json")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_records, test_records = L.build_demo_split(args.max_samples, SEED, n_test=args.n_test)
    train_dataset = [L._to_conversation(r) for r in train_records]

    results = []
    for which, rank in CONFIGS:
        print(f"\n===== {which} (r={rank}) =====")
        results.append(
            run_config(
                which,
                rank,
                processor,
                train_records,
                test_records,
                train_dataset,
                args.num_train_epochs,
                args.learning_rate,
            )
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n==== SUMMARY ====")
    print(f"{'config':14s} {'rank':>4} {'params':>10} {'train':>7} {'test':>7} {'med_dW':>7}")
    for r in results:
        print(
            f"{r['config']:14s} {r['rank']:4d} {r['trainable_params']:10d} "
            f"{r['train_acc']:7.0%} {r['test_acc']:7.0%} {r['median_rel_update']:7.3f}"
        )


if __name__ == "__main__":
    main()
