<!---
Copyright 2026 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# LinkedIn post + summary figures

Three high-resolution figures (drop them into the post as a carousel, in order):

1. `linkedin_1_setup.png` — what data and what model were used (LFM2-VL-450M pipeline + the synthetic visual-understanding dataset).
2. `linkedin_2_results_tables.png` — every number: data scaling, the rank sweep (30 vs 5000 images), and weight change.
3. `linkedin_3_vision_vs_language.png` — the core insight: vision vs. language attribution + the weight-change evidence.

---

## Post

**I fine-tuned a 450M vision-language model that runs in a browser — on fewer than 50 images. Then I designed the dataset to answer one question: when a VLM gets better, is it the VISION encoder learning, or the LANGUAGE model?** 🧪

**Start with the data — and why I built it this way.**
I *generated* the data with Pillow: colored shapes on white. The point was a **perception-heavy task** — the model has to read fine COLOR, SHAPE and SPATIAL detail (e.g. "a red circle inside a blue star", "a green cube to the left of a gold sphere"). That detail is exactly what the **vision encoder** is responsible for. So this dataset is, by construction, a probe of the vision side.

Synthetic also buys honesty: I know every label and can hold out **unseen combinations** — every color/shape/relation appears in training, but never in that arrangement — which cleanly separates *generalization* from *memorization*. I scaled it 12 → 5,000 images, with 62 colors, 17 shapes (2D + pseudo-3D), 5 relations, plus color/size/position/rotation jitter to make it genuinely hard.

Model: **LiquidAI/LFM2-VL-450M** (350M LFM2 LM + 86M SigLIP2 encoder). LoRA + TRL.

**What the numbers said:**

🎯 **Tiny data already works.** ~30 images, minutes on CPU: generic captions → exact structured answers on unseen compositions (88% held-out).

🔁 **Vision vs. language flips with data.**
• Data-starved (30 imgs): a low-rank, language-only adapter was the efficient winner; adding vision capacity barely moved.
• Data-rich (5,000 imgs): the bottleneck moved to the **vision encoder** — vision-LoRA climbed **67% → 94%** as I added rank, while **language-only flat-lined at ~86% no matter the rank.** With the encoder frozen, the LM simply can't recover visual distinctions it was never given.

📊 **More data lifts the ceiling.** Held-out: 50→73%, 500→80%, 1,500→86%, 5,000→**93%**. Past a point the model stops memorizing and starts learning the *rule* (test accuracy overtook train).

🔬 **Weight forensics back it up.** The size of the LoRA weight update (‖ΔW‖/‖W‖) tracked accuracy gains almost perfectly for the vision encoder (**r = +0.95**) and not at all for the already-saturated language side — and **vision's weight updates were the largest of all.**

**Takeaway:** for a visual-understanding task, the gains — and the weight changes — concentrate on the **vision side**. So that's where the budget should go: **as you scale, give the vision encoder more capacity (rank) and more data**, not the language model. More broadly: the right place for fine-tuning budget depends on which resource is the *binding constraint* — diagnose that first.

⚙️ Small stuff on a CPU box; the big sweeps on **Hugging Face Jobs (A10G GPU)** — total GPU spend ~$5. Small models + synthetic data = ML science you can actually afford to run end-to-end.

\#MachineLearning #LLM #VisionLanguageModels #LoRA #FineTuning #HuggingFace #LiquidAI #AI
