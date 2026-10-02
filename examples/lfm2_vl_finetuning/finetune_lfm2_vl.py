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

* ``--demo`` (default): a fully synthetic, self-contained dataset generated with
  Pillow. It spans **40 colors** and **12 shapes** (2D plus shaded pseudo-3D cube /
  sphere / cylinder / cone / pyramid), and mixes single-shape naming with five
  *spatial relations* — inside / left / right / above / below (e.g. "the red circle
  is inside the blue star", "the green cube is above the gold sphere"). No download
  required, exactly ``--max_samples`` images, perfectly reproducible. Each image is
  essentially unique, so the tiny dataset cannot be solved by memorizing a few labels.
* ``--dataset_name <hub id>``: a slice of a real vision-language dataset from the
  Hub (e.g. ``HuggingFaceH4/llava-instruct-mix-vsft``), capped to
  ``--max_samples`` (< 50) examples.

Example
-------
Synthetic smoke test (no dataset download, ~40 images)::

    python finetune_lfm2_vl.py --demo --max_samples 40 --num_train_epochs 12

Tiny slice of a real dataset (40 images)::

    python finetune_lfm2_vl.py \
        --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
        --max_samples 40
"""

import argparse
import logging
import math
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
        default=40,
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
# Synthetic dataset: colored 2D/3D shapes + spatial relations (< 50 images, fully offline)
# --------------------------------------------------------------------------------------
# A diverse task: 40 colors, 12 shapes (2D + shaded pseudo-3D), and five spatial
# relations (inside / left / right / above / below). Each image is essentially unique,
# so the tiny dataset cannot be memorized by a few flat labels.
#
# Every record is {image, question, answer}. The model's *input* is the image plus the
# `question`; the `answer` is the ground-truth label it is trained to produce.
COLORS = {
    "red": (220, 50, 50),
    "green": (40, 170, 80),
    "blue": (60, 90, 220),
    "yellow": (240, 200, 40),
    "orange": (240, 140, 30),
    "purple": (150, 60, 200),
    "cyan": (40, 190, 200),
    "pink": (240, 120, 180),
    "magenta": (200, 40, 160),
    "lime": (140, 210, 40),
    "teal": (20, 140, 140),
    "navy": (30, 40, 120),
    "maroon": (130, 30, 40),
    "olive": (120, 120, 30),
    "brown": (140, 80, 40),
    "gold": (212, 175, 55),
    "salmon": (240, 130, 110),
    "coral": (240, 110, 80),
    "crimson": (200, 20, 60),
    "indigo": (75, 0, 130),
    "violet": (180, 90, 220),
    "turquoise": (50, 200, 170),
    "tan": (190, 150, 100),
    "khaki": (180, 170, 90),
    "plum": (160, 90, 150),
    "orchid": (190, 90, 180),
    "chocolate": (160, 90, 40),
    "tomato": (240, 90, 60),
    "sienna": (140, 80, 50),
    "slateblue": (95, 95, 205),
    "steelblue": (70, 120, 170),
    "skyblue": (95, 170, 220),
    "seagreen": (40, 140, 95),
    "forestgreen": (30, 115, 50),
    "hotpink": (240, 100, 170),
    "deeppink": (220, 40, 120),
    "mustard": (210, 170, 40),
    "mint": (90, 205, 150),
    "lavender": (170, 150, 225),
    "rose": (220, 90, 130),
}
SHAPES_2D = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon", "star"]
SHAPES_3D = ["cube", "sphere", "cylinder", "cone", "pyramid"]
SHAPES = SHAPES_2D + SHAPES_3D

QUESTION_SINGLE = "What colored shape is in this image? Answer with '<color> <shape>'."
QUESTION_SPATIAL = "Where is the {a} located relative to the {b}? Describe the spatial relationship."


def _shade(rgb, factor):
    """Lighten (factor > 1) or darken (factor < 1) a color for pseudo-3D faces."""
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def _ngon(n, cx, cy, r, rot=-math.pi / 2):
    return [
        (cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)
    ]


def _star(cx, cy, r):
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        ang = math.pi * i / 5 - math.pi / 2
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


def _draw_one(draw, shape_name, color_name, cx, cy, r):
    """Draw a single solid 2D or shaded pseudo-3D shape centered at ``(cx, cy)``."""
    rgb = COLORS[color_name]
    if shape_name == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape_name == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape_name == "triangle":
        draw.polygon([(cx, cy - r), (cx - r * 0.92, cy + r * 0.8), (cx + r * 0.92, cy + r * 0.8)], fill=rgb)
    elif shape_name == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb)
    elif shape_name == "pentagon":
        draw.polygon(_ngon(5, cx, cy, r), fill=rgb)
    elif shape_name == "hexagon":
        draw.polygon(_ngon(6, cx, cy, r), fill=rgb)
    elif shape_name == "star":
        draw.polygon(_star(cx, cy, r), fill=rgb)
    elif shape_name == "cube":
        draw.polygon([(cx - r, cy - r / 2), (cx, cy), (cx, cy + r), (cx - r, cy + r / 2)], fill=_shade(rgb, 0.6))
        draw.polygon([(cx, cy), (cx + r, cy - r / 2), (cx + r, cy + r / 2), (cx, cy + r)], fill=_shade(rgb, 0.85))
        draw.polygon([(cx, cy - r), (cx + r, cy - r / 2), (cx, cy), (cx - r, cy - r / 2)], fill=_shade(rgb, 1.2))
    elif shape_name == "sphere":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_shade(rgb, 0.8))
        for i in range(7):
            t = i / 7
            rr = r * (1 - t) * 0.85
            ox, oy = cx - r * 0.28, cy - r * 0.28
            draw.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], fill=_shade(rgb, 0.8 + 0.55 * t))
    elif shape_name == "cylinder":
        eh = r * 0.32
        draw.ellipse([cx - r, cy + r - eh, cx + r, cy + r + eh], fill=_shade(rgb, 0.65))
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_shade(rgb, 0.85))
        draw.ellipse([cx - r, cy - r - eh, cx + r, cy - r + eh], fill=_shade(rgb, 1.15))
    elif shape_name == "cone":
        eh = r * 0.32
        draw.ellipse([cx - r, cy + r - eh, cx + r, cy + r + eh], fill=_shade(rgb, 0.65))
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=_shade(rgb, 0.95))
    elif shape_name == "pyramid":
        apex, front = (cx, cy - r), (cx, cy + r)
        draw.polygon([apex, (cx - r, cy + r * 0.35), front], fill=_shade(rgb, 0.7))
        draw.polygon([apex, front, (cx + r, cy + r * 0.35)], fill=_shade(rgb, 1.0))
    else:
        raise ValueError(f"Unknown shape: {shape_name}")


def _new_canvas(size=256):
    return Image.new("RGB", (size, size), (255, 255, 255))


def _render_single(color_name, shape_name, size=256):
    image = _new_canvas(size)
    _draw_one(ImageDraw.Draw(image), shape_name, color_name, size // 2, size // 2, size // 3)
    return image


def _render_inside(c_out, s_out, c_in, s_in, size=256):
    image = _new_canvas(size)
    draw = ImageDraw.Draw(image)
    _draw_one(draw, s_out, c_out, size // 2, size // 2, size // 2 - 14)
    _draw_one(draw, s_in, c_in, size // 2, size // 2, size // 7)
    return image


def _render_pair(c1, s1, c2, s2, axis, size=256):
    """Render two shapes side by side ('h') or stacked ('v'). Returns the image."""
    image = _new_canvas(size)
    draw = ImageDraw.Draw(image)
    r = size // 7
    if axis == "h":
        _draw_one(draw, s1, c1, size // 4, size // 2, r)
        _draw_one(draw, s2, c2, 3 * size // 4, size // 2, r)
    else:
        _draw_one(draw, s1, c1, size // 2, size // 4, r)
        _draw_one(draw, s2, c2, size // 2, 3 * size // 4, r)
    return image


def _render_relation(c1, s1, relation, c2, s2, size=256):
    """Render the scene that matches the sentence 'the c1 s1 is <relation> the c2 s2'."""
    if relation == "inside":
        return _render_inside(c2, s2, c1, s1, size)  # c1/s1 is the inner shape
    if relation == "to the left of":
        return _render_pair(c1, s1, c2, s2, "h", size)  # c1/s1 on the left
    if relation == "to the right of":
        return _render_pair(c2, s2, c1, s1, "h", size)  # c1/s1 on the right
    if relation == "above":
        return _render_pair(c1, s1, c2, s2, "v", size)  # c1/s1 on top
    if relation == "below":
        return _render_pair(c2, s2, c1, s1, "v", size)  # c1/s1 on the bottom
    raise ValueError(f"Unknown relation: {relation}")


RELATIONS = ["inside", "to the left of", "to the right of", "above", "below"]


def _single_record(color_name, shape_name):
    return {
        "image": _render_single(color_name, shape_name),
        "question": QUESTION_SINGLE,
        "answer": f"{color_name} {shape_name}",
        "meta": ("single", color_name, shape_name),
    }


def _spatial_record(c1, s1, relation, c2, s2):
    # The question names both shapes, so the relation (left vs right, above vs below) is unambiguous.
    return {
        "image": _render_relation(c1, s1, relation, c2, s2),
        "question": QUESTION_SPATIAL.format(a=f"{c1} {s1}", b=f"{c2} {s2}"),
        "answer": f"the {c1} {s1} is {relation} the {c2} {s2}",
        "meta": ("spatial", c1, s1, relation, c2, s2),
    }


def _meta_colors_shapes(meta):
    """Return (colors, shapes) appearing in a record's meta tuple."""
    if meta[0] == "single":
        return {meta[1]}, {meta[2]}
    return {meta[1], meta[4]}, {meta[2], meta[5]}


def build_demo_dataset(max_samples, seed):
    """Build ``max_samples`` records spanning single-shape and five spatial-relation tasks."""
    rng = random.Random(seed)
    n_single = max(1, round(max_samples * 0.4))
    n_spatial = max_samples - n_single

    records = []

    # Single shapes: distinct (color, shape) combinations.
    single_combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(single_combos)
    for color_name, shape_name in single_combos[:n_single]:
        records.append(_single_record(color_name, shape_name))

    # Spatial relations, distributed across the five relation types.
    for i in range(n_spatial):
        relation = RELATIONS[i % len(RELATIONS)]
        shape_pool = SHAPES_2D if relation == "inside" else SHAPES
        c1, c2 = rng.sample(list(COLORS), 2)
        s1, s2 = rng.sample(shape_pool, 2)
        records.append(_spatial_record(c1, s1, relation, c2, s2))

    rng.shuffle(records)
    logger.info(
        "Built synthetic demo dataset with %d images (%d single, %d spatial).", len(records), n_single, n_spatial
    )
    return records


def build_demo_split(max_samples, seed, n_test=16):
    """Build a (train, test) split where every color/shape/relation is seen in training,
    but the test combinations are novel — a compositional generalization test."""
    train = build_demo_dataset(max_samples, seed)

    seen_singles, seen_spatial = set(), set()
    seen_colors, seen_shapes, seen_relations = set(), set(), set()
    for record in train:
        meta = record["meta"]
        colors, shapes = _meta_colors_shapes(meta)
        seen_colors |= colors
        seen_shapes |= shapes
        if meta[0] == "single":
            seen_singles.add((meta[1], meta[2]))
        else:
            seen_spatial.add(meta)
            seen_relations.add(meta[3])

    seen_colors, seen_shapes, seen_relations = sorted(seen_colors), sorted(seen_shapes), sorted(seen_relations)
    rng = random.Random(seed + 1)

    n_test_single = max(1, round(n_test * 0.4))
    test, used = [], set()

    # Novel single (color, shape) combinations built only from seen colors/shapes.
    attempts = 0
    while sum(1 for r in test if r["meta"][0] == "single") < n_test_single and attempts < 2000:
        attempts += 1
        combo = (rng.choice(seen_colors), rng.choice(seen_shapes))
        if combo in seen_singles or combo in used:
            continue
        used.add(combo)
        test.append(_single_record(*combo))

    # Novel spatial combinations (seen tokens + seen relations, unseen tuple).
    attempts = 0
    while len(test) < n_test and attempts < 4000:
        attempts += 1
        relation = rng.choice(seen_relations)
        pool = [s for s in seen_shapes if s in SHAPES_2D] if relation == "inside" else seen_shapes
        if len(pool) < 2 or len(seen_colors) < 2:
            continue
        c1, c2 = rng.sample(seen_colors, 2)
        s1, s2 = rng.sample(pool, 2)
        tup = ("spatial", c1, s1, relation, c2, s2)
        if tup in seen_spatial or tup in used:
            continue
        used.add(tup)
        test.append(_spatial_record(c1, s1, relation, c2, s2))

    rng.shuffle(test)
    logger.info(
        "Held-out test set: %d novel combinations (seen %d colors, %d shapes, %d relations during training).",
        len(test),
        len(seen_colors),
        len(seen_shapes),
        len(seen_relations),
    )
    return train, test


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


_STOPWORDS = {"a", "an", "the", "to", "of", "is", "there"}


def _normalize(text):
    """Lowercase and drop articles/punctuation so predictions can be matched to answers."""
    return [w for w in text.lower().replace(",", " ").replace(".", " ").split() if w not in _STOPWORDS]


def is_correct(prediction, answer):
    target = _normalize(answer)
    return _normalize(prediction)[: len(target)] == target


def evaluate(model, processor, records):
    """Return (predictions, accuracy) over a list of {image, question, answer} records."""
    predictions = [generate_answer(model, processor, r["image"], r["question"]) for r in records]
    accuracy = sum(is_correct(p, r["answer"]) for p, r in zip(predictions, records)) / max(1, len(records))
    return predictions, accuracy


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
    # In demo mode we also build a held-out test set of *novel* combinations (every color,
    # shape and relation is seen in training, but never in these arrangements) to measure
    # generalization rather than memorization.
    test_records = []
    if args.demo:
        records, test_records = build_demo_split(args.max_samples, args.seed)
    else:
        records = build_hub_dataset(args.dataset_name, args.dataset_split, args.max_samples)
    if not records:
        raise RuntimeError("No usable training records were produced. Check the dataset/columns.")

    train_dataset = [_to_conversation(r) for r in records]
    # Qualitative before/after check: the held-out test set in demo mode, else one train example.
    eval_records = test_records or records[:1]

    # ---- Optional: model behaviour BEFORE fine-tuning -------------------------------
    before_preds = None
    if not args.no_eval:
        model.eval()
        before_preds, before_acc = evaluate(model, processor, eval_records)
        logger.info("[before] accuracy on %d held-out examples: %.0f%%", len(eval_records), 100 * before_acc)

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
        after_preds, after_acc = evaluate(trainer.model, processor, eval_records)
        split_name = "held-out (unseen combinations)" if test_records else "train"
        logger.info("[after]  accuracy on %d %s examples: %.0f%%", len(eval_records), split_name, 100 * after_acc)
        if before_preds is not None:
            logger.info("[before -> after] %.0f%% -> %.0f%%", 100 * before_acc, 100 * after_acc)
        if test_records:
            _, train_acc = evaluate(trainer.model, processor, records)
            logger.info("[after]  train-set accuracy (memorization): %.0f%%", 100 * train_acc)
        # Print a few example predictions for inspection.
        for record, after in zip(eval_records[:8], after_preds[:8]):
            mark = "OK" if is_correct(after, record["answer"]) else "  "
            logger.info("  [%s] gt=%r  after=%r", mark, record["answer"], after)


if __name__ == "__main__":
    main()
