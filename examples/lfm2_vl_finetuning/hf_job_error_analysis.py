# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.57",
#   "trl>=0.12",
#   "peft>=0.13",
#   "accelerate",
#   "datasets",
#   "pillow",
#   "torchvision",
#   "numpy<2",
# ]
# ///
"""Self-contained training-set-size sweep for LFM2-VL-450M, for HF Jobs (GPU).

Fixed held-out test of 100 unseen combinations; vary train size; fixed optimizer-step
budget so compute is constant. Prints `RESULT {json}` lines and a final summary.
"""

import json
import math
import random

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "LiquidAI/LFM2-VL-450M"
SEED = 42
N_TRAIN = 5000
CONFIGS = [("vision_all", 16), ("language_all", 16), ("all_both", 16)]
N_TEST = 100
MAX_STEPS = 2000
BATCH = 8
LR = 2e-4
RANK = 16

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
    "scarlet": (230, 40, 30),
    "amber": (235, 170, 20),
    "emerald": (35, 175, 100),
    "ruby": (190, 30, 70),
    "sapphire": (40, 80, 190),
    "jade": (60, 170, 130),
    "rust": (180, 80, 40),
    "lemon": (230, 220, 70),
    "azure": (60, 150, 230),
    "fuchsia": (220, 60, 200),
    "periwinkle": (140, 150, 235),
    "apricot": (240, 170, 110),
    "burgundy": (110, 30, 50),
    "charcoal": (70, 70, 80),
    "aqua": (60, 210, 200),
    "moss": (110, 140, 60),
    "cobalt": (40, 70, 170),
    "peach": (245, 180, 140),
    "wine": (130, 40, 70),
    "cerulean": (30, 130, 200),
    "chartreuse": (160, 210, 30),
    "ochre": (190, 140, 50),
}
POLY = ["square", "triangle", "diamond", "pentagon", "hexagon", "heptagon", "octagon", "star", "cross", "trapezoid"]
SMOOTH = ["circle", "ellipse"]
SHAPES_3D = ["cube", "sphere", "cylinder", "cone", "pyramid"]
SHAPES_2D = POLY + SMOOTH
SHAPES = SHAPES_2D + SHAPES_3D
RELATIONS = ["inside", "to the left of", "to the right of", "above", "below"]
Q_SINGLE = "What colored shape is in this image? Answer with '<color> <shape>'."
Q_SPATIAL = "Where is the {a} located relative to the {b}? Describe the spatial relationship."


def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def _rot(points, cx, cy, a):
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in points]


def _poly(shape, cx, cy, r):
    if shape == "triangle":
        return [(cx, cy - r), (cx - r * 0.92, cy + r * 0.8), (cx + r * 0.92, cy + r * 0.8)]
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if shape == "trapezoid":
        return [
            (cx - r, cy + r * 0.7),
            (cx + r, cy + r * 0.7),
            (cx + r * 0.55, cy - r * 0.7),
            (cx - r * 0.55, cy - r * 0.7),
        ]
    if shape == "cross":
        a = r * 0.38
        return [
            (cx - a, cy - r),
            (cx + a, cy - r),
            (cx + a, cy - a),
            (cx + r, cy - a),
            (cx + r, cy + a),
            (cx + a, cy + a),
            (cx + a, cy + r),
            (cx - a, cy + r),
            (cx - a, cy + a),
            (cx - r, cy + a),
            (cx - r, cy - a),
            (cx - a, cy - a),
        ]
    if shape == "star":
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            ang = math.pi * i / 5 - math.pi / 2
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        return pts
    n = {"pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[shape]
    return [
        (cx + r * math.cos(-math.pi / 2 + 2 * math.pi * i / n), cy + r * math.sin(-math.pi / 2 + 2 * math.pi * i / n))
        for i in range(n)
    ]


def draw_shape(d, shape, rgb, cx, cy, r, angle=0.0):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "ellipse":
        d.ellipse([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6], fill=rgb)
    elif shape in POLY:
        pts = _poly(shape, cx, cy, r)
        d.polygon(_rot(pts, cx, cy, angle) if angle else pts, fill=rgb)
    elif shape == "cube":
        d.polygon([(cx - r, cy - r / 2), (cx, cy), (cx, cy + r), (cx - r, cy + r / 2)], fill=_shade(rgb, 0.6))
        d.polygon([(cx, cy), (cx + r, cy - r / 2), (cx + r, cy + r / 2), (cx, cy + r)], fill=_shade(rgb, 0.85))
        d.polygon([(cx, cy - r), (cx + r, cy - r / 2), (cx, cy), (cx - r, cy - r / 2)], fill=_shade(rgb, 1.2))
    elif shape == "sphere":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_shade(rgb, 0.8))
        for i in range(7):
            t = i / 7
            rr = r * (1 - t) * 0.85
            ox, oy = cx - r * 0.28, cy - r * 0.28
            d.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], fill=_shade(rgb, 0.8 + 0.55 * t))
    elif shape == "cylinder":
        eh = r * 0.32
        d.ellipse([cx - r, cy + r - eh, cx + r, cy + r + eh], fill=_shade(rgb, 0.65))
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_shade(rgb, 0.85))
        d.ellipse([cx - r, cy - r - eh, cx + r, cy - r + eh], fill=_shade(rgb, 1.15))
    elif shape == "cone":
        eh = r * 0.32
        d.ellipse([cx - r, cy + r - eh, cx + r, cy + r + eh], fill=_shade(rgb, 0.65))
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=_shade(rgb, 0.95))
    elif shape == "pyramid":
        apex, front = (cx, cy - r), (cx, cy + r)
        d.polygon([apex, (cx - r, cy + r * 0.35), front], fill=_shade(rgb, 0.7))
        d.polygon([apex, front, (cx + r, cy + r * 0.35)], fill=_shade(rgb, 1.0))


def _canvas():
    return Image.new("RGB", (256, 256), "white")


def _jc(rng, c):
    return tuple(max(0, min(255, v + int(rng.gauss(0, 14)))) for v in COLORS[c])


def _jr(rng, b):
    return int(b * rng.uniform(0.82, 1.18))


def _ja(rng, s):
    return rng.uniform(-0.35, 0.35) if s in POLY else 0.0


def _pl(rng, xy, sp):
    return xy[0] + rng.randint(-sp, sp), xy[1] + rng.randint(-sp, sp)


def render_single(rng, c, s):
    img = _canvas()
    draw_shape(ImageDraw.Draw(img), s, _jc(rng, c), *_pl(rng, (128, 128), 22), _jr(rng, 80), _ja(rng, s))
    return img


def render_relation(rng, c1, s1, rel, c2, s2):
    img = _canvas()
    d = ImageDraw.Draw(img)
    if rel == "inside":
        draw_shape(d, s2, _jc(rng, c2), 128, 128, _jr(rng, 110), _ja(rng, s2))
        draw_shape(d, s1, _jc(rng, c1), *_pl(rng, (128, 128), 8), _jr(rng, 34), _ja(rng, s1))
        return img
    horizontal = rel in ("to the left of", "to the right of")
    subject_first = rel in ("to the left of", "above")
    slot_a, slot_b = ((64, 128), (192, 128)) if horizontal else ((128, 64), (128, 192))
    if subject_first:
        draw_shape(d, s1, _jc(rng, c1), *_pl(rng, slot_a, 16), _jr(rng, 38), _ja(rng, s1))
        draw_shape(d, s2, _jc(rng, c2), *_pl(rng, slot_b, 16), _jr(rng, 38), _ja(rng, s2))
    else:
        draw_shape(d, s2, _jc(rng, c2), *_pl(rng, slot_a, 16), _jr(rng, 38), _ja(rng, s2))
        draw_shape(d, s1, _jc(rng, c1), *_pl(rng, slot_b, 16), _jr(rng, 38), _ja(rng, s1))
    return img


def single_rec(rng, c, s):
    return {"image": render_single(rng, c, s), "question": Q_SINGLE, "answer": f"{c} {s}", "meta": ("single", c, s)}


def spatial_rec(rng, c1, s1, rel, c2, s2):
    return {
        "image": render_relation(rng, c1, s1, rel, c2, s2),
        "question": Q_SPATIAL.format(a=f"{c1} {s1}", b=f"{c2} {s2}"),
        "answer": f"the {c1} {s1} is {rel} the {c2} {s2}",
        "meta": ("spatial", c1, s1, rel, c2, s2),
    }


def sample_combo(rng, forbidden):
    while True:
        if rng.random() < 0.4:
            c = ("single", rng.choice(list(COLORS)), rng.choice(SHAPES))
            if (c[1], c[2]) not in forbidden:
                return c
        else:
            rel = rng.choice(RELATIONS)
            pool = SHAPES_2D if rel == "inside" else SHAPES
            c1, c2 = rng.sample(list(COLORS), 2)
            s1, s2 = rng.sample(pool, 2)
            c = ("spatial", c1, s1, rel, c2, s2)
            if c not in forbidden:
                return c


def rec(rng, c):
    return single_rec(rng, c[1], c[2]) if c[0] == "single" else spatial_rec(rng, c[1], c[2], c[3], c[4], c[5])


def to_conv(r):
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": r["image"]}, {"type": "text", "text": r["question"]}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
        ]
    }


STOP = {"a", "an", "the", "to", "of", "is", "there"}


def norm(t):
    return [w for w in t.lower().replace(",", " ").replace(".", " ").split() if w not in STOP]


def is_correct(p, t):
    tn = norm(t)
    return norm(p)[: len(tn)] == tn


ATTN = {"q_proj", "k_proj", "v_proj", "out_proj"}
CURATED = ATTN | {"in_proj", "w1", "w2", "w3", "fc1", "fc2", "linear_1", "linear_2"}
SKIP = {"lm_head", "patch_embedding"}


def target_names(model, which):
    names = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        suffix = name.split(".")[-1]
        if suffix in SKIP:
            continue
        comp = "vision" if "vision_tower" in name else ("language" if "language_model" in name else "connector")
        if which == "all_both" and suffix in CURATED:
            names.append(name)
        elif which == "vision_all" and suffix in CURATED and comp == "vision":
            names.append(name)
        elif which == "language_all" and suffix in CURATED and comp == "language":
            names.append(name)
    return names


def median_rel_update(model):
    rels = []
    for _, m in model.named_modules():
        la = getattr(m, "lora_A", None)
        if la is None or "default" not in la:
            continue
        A = m.lora_A["default"].weight.detach().float()
        B = m.lora_B["default"].weight.detach().float()
        dW = m.scaling["default"] * (B @ A)
        rels.append((dW.norm() / m.base_layer.weight.detach().float().norm()).item())
    rels.sort()
    return rels[len(rels) // 2] if rels else 0.0


@torch.no_grad()
def predict(model, processor, record):
    conv = [
        {
            "role": "user",
            "content": [{"type": "image", "image": record["image"]}, {"type": "text", "text": record["question"]}],
        }
    ]
    inp = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inp, max_new_tokens=28, do_sample=False)
    return processor.decode(out[0, inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def accuracy(model, processor, recs):
    return sum(is_correct(predict(model, processor, r), r["answer"]) for r in recs) / len(recs)


COLOR_SET = set(COLORS)
SHAPE_SET = set(SHAPES)
REL_WORDS = {"inside", "left", "right", "above", "below"}


def parse(text):
    """Extract ordered colors, shapes and the relation word from an answer/prediction."""
    toks = norm(text)  # drops a/an/the/to/of/is/there -> 'to the left of' becomes 'left'
    colors = [t for t in toks if t in COLOR_SET]
    shapes = [t for t in toks if t in SHAPE_SET]
    rel = next((t for t in toks if t in REL_WORDS), None)
    return colors, shapes, rel


def categorize(gt, pred):
    """Return the set of error kinds for one (gt, pred) pair (empty set == correct)."""
    gc, gs, gr = parse(gt)
    pc, ps, pr = parse(pred)
    errs = set()
    if pc != gc:
        errs.add("color")
    if ps != gs:
        errs.add("shape")
    if gr is not None and pr != gr:
        errs.add("relation")
    # unparseable / wrong structure (wrong number of colors or shapes recovered)
    if len(pc) != len(gc) or len(ps) != len(gs):
        errs.add("format")
    return errs


def main():
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"device={device} dtype={dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    pad_id, img_id = processor.tokenizer.pad_token_id, processor.image_token_id

    rng = random.Random(SEED)
    test, combos = [], set()
    while len(test) < N_TEST:
        c = sample_combo(rng, set())
        key = (c[1], c[2]) if c[0] == "single" else c
        if key in combos:
            continue
        combos.add(key)
        test.append(rec(rng, c))
    forbidden = {((r["meta"][1], r["meta"][2]) if r["meta"][0] == "single" else r["meta"]) for r in test}

    trng = random.Random(SEED + N_TRAIN)
    train = [rec(trng, sample_combo(trng, forbidden)) for _ in range(N_TRAIN)]
    train_ds = [to_conv(r) for r in train]
    print(
        f"train={len(train)} fixed test={len(test)}; error analysis @ r=16; max_steps={MAX_STEPS} batch={BATCH}",
        flush=True,
    )

    def collate(examples):
        b = processor.apply_chat_template(
            [e["messages"] for e in examples],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        lab = b["input_ids"].clone()
        lab[lab == pad_id] = -100
        lab[lab == img_id] = -100
        b["labels"] = lab
        return b

    results = []
    for family, rank in CONFIGS:
        print(f"\n===== {family} r={rank} =====", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(device)
        model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_names(model, family),
                task_type="CAUSAL_LM",
            ),
        )
        sft = SFTConfig(
            output_dir=f"/tmp/ea_{family}",
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=1,
            learning_rate=LR,
            max_length=1024,
            logging_steps=500,
            bf16=torch.cuda.is_available(),
            report_to="none",
            seed=SEED,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        trainer = SFTTrainer(
            model=model, args=sft, train_dataset=train_ds, data_collator=collate, processing_class=processor.tokenizer
        )
        trainer.train()
        trainer.model.eval()

        # per-example predictions on the held-out test, split by single vs spatial
        tally = {"color": 0, "shape": 0, "relation": 0, "format": 0}
        n_single = n_spatial = correct = 0
        single_correct = spatial_correct = 0
        examples = []
        for r in test:
            pred = predict(trainer.model, processor, r)
            errs = categorize(r["answer"], pred)
            is_single = r["meta"][0] == "single"
            n_single += is_single
            n_spatial += not is_single
            if not errs:
                correct += 1
                single_correct += is_single
                spatial_correct += not is_single
            else:
                for e in errs:
                    tally[e] += 1
                if len(examples) < 8:
                    examples.append({"gt": r["answer"], "pred": pred, "errs": sorted(errs)})
        res = {
            "config": family,
            "test_acc": round(correct / len(test), 4),
            "single_acc": round(single_correct / max(1, n_single), 4),
            "spatial_acc": round(spatial_correct / max(1, n_spatial), 4),
            "errors": tally,
            "n_test": len(test),
            "examples": examples,
        }
        print("RESULT", json.dumps(res), flush=True)
        results.append(res)
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n==== ERROR ANALYSIS SUMMARY (held-out test) ====")
    print(
        f"{'config':14s}{'acc':>6}{'single':>8}{'spatial':>9}{'color_err':>11}{'shape_err':>11}{'rel_err':>9}{'fmt':>6}"
    )
    for r in results:
        e = r["errors"]
        print(
            f"{r['config']:14s}{r['test_acc']:6.0%}{r['single_acc']:8.0%}{r['spatial_acc']:9.0%}"
            f"{e['color']:11d}{e['shape']:11d}{e['relation']:9d}{e['format']:6d}"
        )
    print("FINAL_JSON", json.dumps(results))


if __name__ == "__main__":
    main()
