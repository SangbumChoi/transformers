<!--
Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Fine-tuning zero-shot object detectors — benchmark findings

Empirical notes from fine-tuning the four zero-shot detectors that now support a training loss
(Grounding DINO, OWL-ViT, OWLv2, OmDet-Turbo) with
[`benchmark_zero_shot_finetuning.py`](./benchmark_zero_shot_finetuning.py).

**Setup:** each model fine-tuned on 5 datasets spanning a domain/difficulty gradient
(250–300 train / 80–100 val), **frozen backbones** (only detection heads / encoder-decoder trained),
h-flip augmentation, **per-model learning rate**, early stopping (patience 3–4). Evaluation uses each
model's own `post_process_*` with `target_sizes`; the **vanilla (zero-shot) mAP is measured at epoch −1**,
before any gradient step. Artifacts (per-dataset metrics + sample images + plots) live in the
[`danelcsb/zsod-finetune-benchmark`](https://huggingface.co/datasets/danelcsb/zsod-finetune-benchmark) dataset.

![sweep summary](https://huggingface.co/datasets/danelcsb/zsod-finetune-benchmark/resolve/main/sweep/summary.png)

## Results (vanilla zero-shot → best fine-tuned mAP)

| Dataset (characteristics) | grounding_dino | owlvit | owlv2 | omdet_turbo | best |
|---|---|---|---|---|---|
| `animals` (everyday, large, sparse) | 0.21→0.32 | 0.46→0.57 | 0.65→0.65 | 0.36→**0.68** | omdet |
| `aquarium` (underwater, small, dense) | 0.14→0.46 | 0.11→0.33 | 0.37→0.37 | 0.13→**0.48** | omdet |
| `chess` (fine-grained, 13 pieces) | 0.10→0.75 | 0.02→0.69 | 0.04→0.76 | 0.10→**0.83** | omdet |
| `excavators` (industrial, large, sparse) | 0.26→0.71 | 0.23→0.54 | 0.55→0.64 | 0.36→**0.77** | omdet |
| `thermal` (known classes, alien modality) | 0.70→**0.86** | 0.43→0.72 | 0.48→0.71 | 0.61→0.85 | gdino |

## Know-how

1. **OmDet-Turbo is the most consistent model to fine-tune** — best fine-tuned mAP on 4/5 datasets and the
   largest average gain (**+0.41**). If you only fine-tune one zero-shot detector, start here.
2. **Average fine-tuning gain per model:** OmDet-Turbo +0.41 › Grounding DINO +0.34 › OWL-ViT +0.32 › OWLv2 +0.21.
3. **Fine-tuning pays off most where zero-shot is weakest.** Avg vanilla mAP by dataset:
   thermal 0.56 > animals 0.42 > excavators 0.35 > aquarium 0.19 > chess 0.06. The biggest jumps happen on the
   least-familiar data (chess 0.04–0.10 → 0.69–0.83); on already-strong priors there is little headroom
   (OWLv2 on animals 0.65→0.65).
4. **OWLv2 has the strongest zero-shot priors but the least fine-tuning headroom** with frozen heads.
5. **Object size matters more than domain** for difficulty: small/dense sets (aquarium, chess) start low across
   the board; large/sparse ones (animals, excavators, thermal) start higher. Thermal scoring the *highest*
   vanilla AP was the surprise — known semantics (dog/person) survive the modality shift.
6. **Learning rate matters per model:** Grounding DINO needs a smaller lr (5e-5) — at 1e-4 the first epoch
   *destroyed* its pretrained weights (its vanilla 0.15 dropped to ~0.02 before recovering). OWL/OWLv2 are fine at 1e-4.

## Case study: Grounding DINO on `chess` (an evaluation pitfall)

Grounding DINO initially scored **0.00 mAP on chess at every epoch** while the others reached 0.69–0.83.
Splitting the metric showed it was **not** a model/training failure:

| | class-agnostic mAP (boxes only) | with-class mAP |
|---|---|---|
| text-phrase decoding | 0.385 | **0.000** |
| token-span class scoring | 0.420 | **0.094** (vanilla) → **0.745** (fine-tuned) |

**Cause:** Grounding DINO classifies a box by token-similarity to the prompt. With 13 near-identical hyphenated
classes (`bishop` vs `black-bishop` vs `white-bishop`, …) the decoded text phrase is empty/ambiguous, so every
box was mislabeled onto a single class.

**Fix:** do not decode text — score each query against every class directly by mean-pooling the token logits over
each class's token span (`build_label_maps`) and taking `argmax` over classes, the same index-based scheme OWL-ViT
and OmDet-Turbo use. This recovered chess (0.00→0.75) and improved Grounding DINO across the board (e.g. aquarium
0.21→0.46), lifting its average gain from +0.17 to +0.34. **Takeaway: evaluate Grounding DINO with per-class token
pooling, not text-phrase decoding, on fine-grained or token-overlapping label sets.**

> These are short runs on sub-sampled datasets, meant to show that the new losses optimize and to surface
> practical fine-tuning know-how — not to be SOTA. This work was AI-assisted.
