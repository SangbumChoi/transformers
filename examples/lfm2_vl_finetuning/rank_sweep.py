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
Dense LoRA-rank sweep on the *harder* dataset (``hard_dataset``).

For each component family (vision-only, language-only, all-modules) we sweep the LoRA
rank across many values and record train (memorization) and held-out (generalization)
accuracy plus trainable parameters. The goal is to see whether accuracy vs. parameter
budget traces a smooth, continuous curve, and how the three families compare.

Results are appended to the output JSON after *every* run, so a long sweep survives
an interruption. Reproduce (CPU-friendly but slow — several hours for the full grid):

    python examples/lfm2_vl_finetuning/rank_sweep.py
"""

import argparse
import json
import os

import ablation_components as A
import finetune_lfm2_vl as L
import hard_dataset as H

from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["vision_all", "language_all", "all_both"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--n_test", type=int, default=36)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--out", type=str, default="/tmp/sweep_results.json")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(A.MODEL_ID)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_records, test_records, seen = H.build_split(args.max_samples, args.n_test, A.SEED)
    train_dataset = [L._to_conversation(r) for r in train_records]
    print(f"HARD split: train={len(train_records)} test={len(test_records)} seen={seen}")

    # resume support: skip (family, rank) pairs already present in the output file
    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {(r["config"], r["rank"]) for r in results}

    for family in args.families:
        for rank in args.ranks:
            if (family, rank) in done:
                print(f"skip {family} r={rank} (already done)")
                continue
            print(f"\n===== {family} (r={rank}) =====")
            try:
                res = A.run_config(
                    family,
                    rank,
                    processor,
                    train_records,
                    test_records,
                    train_dataset,
                    args.num_train_epochs,
                    args.learning_rate,
                )
            except Exception as exc:  # keep the sweep alive; record the failure
                print(f"FAILED {family} r={rank}: {exc}")
                res = {"config": family, "rank": rank, "error": str(exc)}
            results.append(res)
            with open(args.out, "w") as f:  # incremental save
                json.dump(results, f, indent=2)

    print("\n==== SWEEP SUMMARY ====")
    print(f"{'config':14s} {'rank':>4} {'params':>10} {'train':>7} {'test':>7}")
    for r in sorted([x for x in results if "error" not in x], key=lambda x: (x["config"], x["rank"])):
        print(
            f"{r['config']:14s} {r['rank']:4d} {r['trainable_params']:10d} {r['train_acc']:7.0%} {r['test_acc']:7.0%}"
        )


if __name__ == "__main__":
    main()
