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
Training-set-size sweep: how far does more data push the generalization ceiling?

Earlier experiments found a ~69% held-out ceiling with only 30 training images. Here we
fix the model config (all transformer linears, LoRA r=16) and a fixed **held-out test set
of 100 unseen combinations**, then vary the training-set size (e.g. 50 / 500 / 5000).

To stay tractable on CPU (no GPU here), each run uses the **same number of optimizer steps**
(``--max_steps``) rather than a fixed number of epochs — so compute is constant across sizes
and only the *diversity of data seen* changes. Results save incrementally (resume-safe).

    python examples/lfm2_vl_finetuning/data_size_sweep.py
"""

import argparse
import json
import os
import random

import ablation_components as A
import finetune_lfm2_vl as L
import hard_dataset as H
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor


def _sample_combo(rng, forbidden):
    """Sample a (kind, ...) combo not in ``forbidden`` (duplicates across train are allowed)."""
    while True:
        if rng.random() < 0.4:
            combo = ("single", rng.choice(list(H.COLORS)), rng.choice(H.SHAPES))
            if (combo[1], combo[2]) not in forbidden:
                return combo
        else:
            rel = rng.choice(H.RELATIONS)
            pool = H.SHAPES_2D if rel == "inside" else H.SHAPES
            c1, c2 = rng.sample(list(H.COLORS), 2)
            s1, s2 = rng.sample(pool, 2)
            combo = ("spatial", c1, s1, rel, c2, s2)
            if combo not in forbidden:
                return combo


def _record(rng, combo):
    if combo[0] == "single":
        return H._single_record(rng, combo[1], combo[2])
    return H._spatial_record(rng, combo[1], combo[2], combo[3], combo[4], combo[5])


def build_fixed_test(n_test, seed):
    """A fixed held-out set of distinct combinations, reused for every training size."""
    rng = random.Random(seed)
    test, combos = [], set()
    while len(test) < n_test:
        combo = _sample_combo(rng, set())
        key = (combo[1], combo[2]) if combo[0] == "single" else combo
        if key in combos:
            continue
        combos.add(key)
        test.append(_record(rng, combo))
    # forbidden = every test combination (so no training image can ever reproduce one)
    forbidden = set()
    for r in test:
        m = r["meta"]
        forbidden.add((m[1], m[2]) if m[0] == "single" else m)
    return test, forbidden


def build_train(n_train, forbidden, seed):
    rng = random.Random(seed)
    records = []
    while len(records) < n_train:
        combo = _sample_combo(rng, forbidden)
        records.append(_record(rng, combo))
    return records


def run_size(n_train, processor, test_records, max_steps, lr, rank):
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForImageTextToText.from_pretrained(A.MODEL_ID, dtype=torch.float32)
    targets = A.target_names(model, "all_both")
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none", target_modules=targets, task_type="CAUSAL_LM"
        ),
    )

    train_records = build_train(n_train, FORBIDDEN, A.SEED + n_train)
    train_dataset = [L._to_conversation(r) for r in train_records]
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
        output_dir=f"/tmp/size_{n_train}",
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        max_length=1024,
        logging_steps=50,
        bf16=False,
        report_to="none",
        seed=A.SEED,
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

    # memorization: subsample of the (possibly huge) train set; generalization: the fixed test
    sub = train_records if len(train_records) <= 100 else random.Random(0).sample(train_records, 100)
    _, train_acc = L.evaluate(trainer.model, processor, sub)
    _, test_acc = L.evaluate(trainer.model, processor, test_records)
    result = {
        "n_train": n_train,
        "max_steps": max_steps,
        "rank": rank,
        "train_acc_sub": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
    }
    print("RESULT", json.dumps(result))
    del model, trainer
    return result


FORBIDDEN = set()


def main():
    global FORBIDDEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 500, 5000])
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=900)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--out", type=str, default="/tmp/size_results.json")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(A.MODEL_ID)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    test_records, FORBIDDEN = build_fixed_test(args.n_test, A.SEED)
    print(f"fixed test set: {len(test_records)} unseen combinations; max_steps={args.max_steps}")

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {r["n_train"] for r in results}

    for n in args.sizes:
        if n in done:
            print(f"skip n_train={n} (already done)")
            continue
        print(f"\n===== n_train={n} =====")
        try:
            res = run_size(n, processor, test_records, args.max_steps, args.learning_rate, args.rank)
        except Exception as exc:
            print(f"FAILED n_train={n}: {exc}")
            res = {"n_train": n, "error": str(exc)}
        results.append(res)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n==== SIZE SWEEP SUMMARY ====")
    print(f"{'n_train':>8} {'train(sub)':>11} {'test':>7}")
    for r in sorted([x for x in results if "error" not in x], key=lambda x: x["n_train"]):
        print(f"{r['n_train']:8d} {r['train_acc_sub']:11.0%} {r['test_acc']:7.0%}")


if __name__ == "__main__":
    main()
