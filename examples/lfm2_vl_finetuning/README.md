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

# Fine-tuning LFM2-VL-450M on a tiny image-text dataset

[LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) is the smallest of
Liquid AI's first vision-language foundation models. It combines a **350M LFM2**
language backbone with an **86M SigLIP2 NaFlex** vision encoder and a 2-layer MLP
connector with pixel-unshuffle. At ~450M parameters it is light enough to run in
a web browser (WebGPU), and small enough that you can LoRA fine-tune it on a
**single small GPU — using fewer than 50 images.**

This example reproduces the path Liquid AI recommends for adapting LFM2-VL to a
new use case: **LoRA adapters + TRL's `SFTTrainer`**.

## What's here

| File | Purpose |
| --- | --- |
| `finetune_lfm2_vl.py` | End-to-end LoRA fine-tuning script (data → train → save → before/after check). |
| `requirements.txt` | Python dependencies. |

## Install

```bash
pip install -r requirements.txt
```

## Quickstart — synthetic demo (no download)

The fastest way to see the whole loop work is the built-in **synthetic dataset**,
rendered on the fly with Pillow. It is fully offline, perfectly reproducible, and —
by construction — fewer than 50 images. To avoid trivial memorization it mixes
three task types over **8 colors** and **6 shapes** (circle, square, triangle,
star, pentagon, diamond):

- **single shape** → `"<color> <shape>"` (e.g. `red star`)
- **inside** → `"a <color> <shape> inside a <color> <shape>"` (e.g. *a red circle inside a blue star*)
- **left of** → `"a <color> <shape> to the left of a <color> <shape>"`

```bash
python finetune_lfm2_vl.py --demo --max_samples 30 --num_train_epochs 10
```

The script prints the model's answer **before** and **after** fine-tuning, so you
can watch it learn both the naming format and the compositional spatial relations.

## Fine-tune on a real (tiny) dataset

Point the script at any image-text dataset on the Hub and cap it below 50 images.
It auto-detects the two most common layouts:

* **TRL / LLaVA style** — an `images` column plus a `messages` chat column
  (e.g. [`HuggingFaceH4/llava-instruct-mix-vsft`](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)).
* **Flat VQA / caption style** — an `image` column plus `question`/`query` and
  `answer`/`caption`/`label` columns.

```bash
python finetune_lfm2_vl.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --max_samples 40 \
    --num_train_epochs 3
```

Only the requested slice is downloaded (`split="train[:40]"`), so this stays fast
and cheap.

## How it works

1. **Load** `AutoProcessor` + `AutoModelForImageTextToText` for `LiquidAI/LFM2-VL-450M`.
2. **Format** each `(image, question, answer)` triple into an LFM2-VL chat
   conversation (`user` turn with an inline image + question, `assistant` turn
   with the answer).
3. **Collate** a batch with `processor.apply_chat_template(..., tokenize=True)`,
   then build `labels` by masking **pad tokens** and the **`<image>` placeholder
   tokens** (id `396`) to `-100` so loss is only computed on real text.
4. **LoRA** adapters are attached to the attention projections (`q/k/v/out_proj`)
   and the MLP (`gate/up/down_proj`) via `peft.LoraConfig`.
5. **Train** with `trl.SFTTrainer`. Because we pre-format conversations and
   tokenize inside the collator, TRL's own dataset preparation is disabled
   (`skip_prepare_dataset=True`, `remove_unused_columns=False`).
6. **Save** the adapter + processor, and run a qualitative before/after check.

## Useful flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_id` | `LiquidAI/LFM2-VL-450M` | Try `LiquidAI/LFM2-VL-1.6B` for the larger sibling. |
| `--max_samples` | `16` | Number of images. Must be **< 50**. |
| `--lora_r` / `--lora_alpha` | `8` / `16` | LoRA capacity. |
| `--num_train_epochs` | `5` | Bump this up for tiny datasets. |
| `--learning_rate` | `1e-4` | |
| `--no_eval` | off | Skip the before/after generation. |

## Loading the fine-tuned adapter

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM2-VL-450M", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "./lfm2-vl-450m-lora")
processor = AutoProcessor.from_pretrained("./lfm2-vl-450m-lora")
# (optional) model = model.merge_and_unload()  # fold LoRA into the base weights
```

## Notes & references

- LFM2-VL processes images at native resolution up to 512×512 and tiles larger
  images; the number of image tokens is tunable via the processor
  (`min_image_tokens` / `max_image_tokens`).
- For best results on real tasks, Liquid AI recommends fine-tuning on your own
  data — even small, high-quality sets help a lot at this model size.
- Liquid AI blog: <https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models>
- LFM2-VL TRL guide: <https://docs.liquid.ai/customization/finetuning-frameworks/trl>
- Model card: <https://huggingface.co/LiquidAI/LFM2-VL-450M>
