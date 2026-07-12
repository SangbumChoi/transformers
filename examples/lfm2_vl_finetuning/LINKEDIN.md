<!--- LinkedIn post (English), question-first, ordered to match the 7-image carousel. -->

# LinkedIn post — English (attach images 1–7 at each 🖼️)

Carousel order: 1) `linkedin_1_setup.png` 2) `linkedin_3_vision_vs_language.png`
3) `data_size_results.png` 4) `rank_sweep_5000_results.png` 5) `error_analysis_results.png`
6) `color_confusion.png` 7) `linkedin_2_results_tables.png`

---

It started with one question.

I built a tiny dataset that is *deliberately* about vision — reading fine COLOR, SHAPE and SPATIAL
detail — and asked: **when a vision-language model gets better at it, where does the learning actually
go — into the VISION encoder, or the LANGUAGE model?**

To find out, I fine-tuned LiquidAI's **LFM2-VL-450M** (small enough to run in a browser via WebGPU) on
**fewer than 50 images** with LoRA + TRL… and then I couldn't stop pulling the thread. 🧵

🖼️ [1 — Setup: model + data]
The model is a 86M SigLIP2 vision encoder → connector → 350M LFM2 language model. The data is
synthetic (rendered with Pillow): 62 colors × 17 shapes (2D + pseudo-3D) × 5 spatial relations. The
key trick: the test set contains only **unseen combinations** — every color/shape/relation appears in
training, but never in that arrangement — so it measures *generalization*, not memorization. Because
the task is pure perception, it's a clean probe of the vision side.

🖼️ [2 — Vision vs. language + weight change]
I put LoRA on the vision side only / the language side only / everything, and measured the weight
update ‖ΔW‖/‖W‖. The vision encoder changes the most, and its weight change correlates with accuracy at
**r = +0.95**; the language side barely moves the needle. First signal that the *vision* side is doing
the learning.

🖼️ [3 — More data lifts the ceiling]
Same model, same fixed 100-image held-out test, only the training-set size changes: 50 → 73%, 500 →
80%, 1,500 → 86%, 5,000 → **93%**. Past a point the model stops memorizing and starts learning the
rule (held-out accuracy overtakes the training subset). The early ceiling was data starvation, not a
model limit.

🖼️ [4 — The roles flip with data (rank sweep: 30 vs 5,000 images)]
• Data-starved (30 imgs): a low-rank, language-only adapter is the efficient winner; vision is stuck.
• Data-rich (5,000 imgs): **vision climbs 76% → 94% as you add rank and takes the lead, while
language-only flat-lines at ~86% no matter the rank** — now the weakest. Where you should spend the
LoRA budget depends entirely on whether *data* or *capacity* is the binding constraint.

🖼️ [5 — Why does language-only plateau? A perception bottleneck]
I split every held-out error by type. All three setups solve the spatial-relation questions **100%** —
reasoning isn't the problem. Almost every mistake is a **fine-grained color confusion**. On the
pure-perception questions, language-only scores 60% vs vision 71% vs all 80%. With the encoder frozen,
the language model can't recover detail the encoder threw away — only adapting vision fixes it.

🖼️ [6 — Why *those specific* colors?]
Every confused color is a **nearest neighbor in RGB space** (sapphire→cobalt, brown→chocolate,
cyan↔turquoise; Δ 22–63, palette average ≈ 33). Four causes stack up: a dense color palette;
per-instance jitter (~24 RGB) that makes neighbor distributions overlap (partly irreducible); a frozen
SigLIP encoder + pixel-unshuffle that compresses fine color away; and a language head that defaults to
the common/prototype name (forestgreen → "green"). The model didn't "fail at colors" — it learned them
up to the resolution its frozen eyes allow.

🖼️ [7 — All the numbers, for reference]

**The answer to the original question:** on a visual-understanding task, the binding constraint is the
**vision encoder's perceptual resolution** — not the language model. Weight change, rank, and data
scaling all point the same way. So as you scale, spend the budget on the **vision side (capacity +
data)**. More generally: diagnose *which* resource is the bottleneck before you crank anything.

⚙️ Small runs on CPU, big sweeps on **Hugging Face Jobs (A10G GPU)** — total GPU cost ~$5. Small models
+ synthetic data = ML science you can actually run end-to-end.

#MachineLearning #LLM #VisionLanguageModels #LoRA #FineTuning #HuggingFace #LiquidAI #AI
