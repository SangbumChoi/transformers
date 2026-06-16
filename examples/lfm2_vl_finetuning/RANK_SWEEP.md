<!---
Copyright 2026 The HuggingFace Team. All rights reserved.

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

# Does accuracy trace a continuous line as you add LoRA capacity?

The [ABLATION.md](./ABLATION.md) experiment compared families at one or two ranks. Here we
**densely sweep the LoRA rank** (`r ∈ {1, 2, 4, 8, 16, 32, 64}`) for each family, on a
deliberately **harder** dataset (`hard_dataset.py`: ~60 colors, ~17 shapes incl. pseudo-3D,
with per-instance color/size/position/rotation jitter so the classes *overlap* and the task
no longer saturates near 100%). The question: is accuracy-vs-capacity a smooth, continuous
curve — and where does each component plateau?

Reproduce (CPU-friendly but slow; results save incrementally so it is resume-safe):

```bash
python examples/lfm2_vl_finetuning/rank_sweep.py
```

## Results

Same 30-image training set and 36-image held-out set of **unseen combinations**, 6 epochs,
lr 2e-4, `lora_alpha = 2·r`. "train" = memorization, "test" = generalization.

| rank | vision-only test | language-only test | all-modules test |
| ---: | ---: | ---: | ---: |
| 1  | 14% | 64% | 72% |
| 2  | 11% | 67% | 69% |
| 4  | 28% | 69% | 69% |
| 8  | 42% | 69% | 69% |
| 16 | 50% | 69% | 69% |
| 32 | 64% | 69% | 69% |
| 64 | **67%** | **69%** | **69%** |

(trainable params range from 0.17M at vision r=1 to 35M at all-modules r=64.)

![rank sweep](./rank_sweep_results.png)

## Takeaways

**1. Yes — for the capacity-limited family it is a smooth, continuous curve.** `vision-only`
climbs monotonically 14% → 28% → 42% → 50% → 64% → 67% as the rank doubles. Each extra LoRA
dimension buys a bit more generalization; the points connect into a clean rising line. Its
*training* accuracy rises the same way (13% → 87%) and never reaches 100% even at r=64 — with
the language head frozen, the vision encoder is genuinely capacity-limited on this task.

**2. Language-only and all-modules are flat — they are *not* capacity-limited.** Both hit the
task ceiling (~69%) at **r = 1–4** and then stay flat out to r = 64, i.e. across a **~50–60×**
increase in trainable parameters (language: 0.37M → 24M; all: 0.55M → 35M). Spending more LoRA
budget there does nothing. So "continuous line" only shows up when capacity is the binding
constraint; once it is not, the curve is a horizontal plateau.

**3. There is a data-imposed ceiling (~69%) that no amount of capacity breaks.** With only 30
training images, generalization to unseen color·shape·relation combinations tops out around
69% — even all-modules at 35M params can't beat it. The bottleneck there is *data*, not model
capacity. (Earlier, on the easier dataset, this ceiling was ~88%.)

**4. Putting it together with the earlier analyses.** The leanest useful adapter is a *low-rank
language-only* one: `language-only r=4` reaches the ceiling with **1.5M** params, whereas
`vision-only` needs **10.6M** (r=64) just to get near it. This is the same lesson as the
weight-change analysis (adaptation concentrates in the language model) and the component
ablation (language ≫ vision per parameter), now traced out continuously across the whole
capacity axis.

> Caveat: small CPU-scale runs on a synthetic task at a fixed 6 epochs; absolute ceilings move
> with data/epochs/scale. The shapes of the curves — vision a rising line, language/all flat
> plateaus, a shared data ceiling — are the transferable lesson.

## Does a big accuracy gain come with a big weight change?

Each run also recorded the median relative weight update `||ΔW|| / ||W||` (with `ΔW = (α/r)·B·A`).
Correlating it with held-out accuracy across the rank sweep:

| family | corr(test acc, ‖ΔW‖/‖W‖) |
| --- | ---: |
| vision-only | **+0.95** |
| language-only | +0.62 |
| all-modules | −0.47 |

![weight change vs accuracy](./weight_change_vs_accuracy.png)

For **vision-only** the relationship is almost perfectly monotonic: as the weight update grows
(0.006 → 0.029) the held-out accuracy climbs (14% → 67%), and vision has the **largest ΔW at every
rank**. So where vision improves a lot, its weights move a lot — the hypothesis holds for the
capacity-limited component. The caveat: `||ΔW||` grows with rank *mechanically* for every family
(bigger `B·A`), yet language/all-modules gain nothing from it because they are already at the data
ceiling. So a large weight change only converts into accuracy where there is headroom to improve;
magnitude alone does not predict the gain across families.


## Rank sweep again, but data-rich (train = 5000)

The plateaus above were at **30** training images, where the task is data-limited. Re-running the
sweep at **train = 5000** (fixed 100-image held-out test, 2000 steps, run on an A10G via HF Jobs with
[`hf_job_rank_sweep_5000.py`](./hf_job_rank_sweep_5000.py)) shows the regime shift:

| rank | vision-only | language-only | all modules |
| ---: | ---: | ---: | ---: |
| 1  | 76% | 85% | 89% |
| 4  | 89% | 86% | 91% |
| 16 | 93% | 86% | 91% |
| 64 | **94%** | 86% | **94%** |

![rank sweep @5000](./rank_sweep_5000_results.png)

**What changed with 100× more data:**

- **Every curve shifts up** — the whole task got easier per the data-scaling result (the ~69% ceiling
  became ~86–94%).
- **Vision-only goes from capped to top.** At 30 images it saturated at 67%; at 5000 it climbs
  76% → 94% with rank and now *reaches* the best accuracy. With enough data, adapting the vision
  encoder pays off — but it needs the capacity (rank) to do it.
- **Language-only is *still* flat with rank** (85% → 86% from r=1 to r=64) — it remains
  parameter-efficient/saturating — but its plateau is now the **lowest** of the three. The roles flip:
  at low data language-only was the efficient winner; at high data on a visually hard task the
  **vision encoder becomes the bottleneck**, so language-only alone leaves accuracy on the table.
- **Best at scale:** `all_both r=64` and `vision_all r=64`, both 94%. When data is plentiful, spend
  capacity broadly (or on vision); when data is scarce, a low-rank language adapter is the efficient
  choice. The right place for the LoRA budget depends on which resource — data or capacity — is the
  binding constraint.
