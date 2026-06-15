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
A *harder*, higher-diversity variant of the synthetic LFM2-VL demo dataset.

Compared to ``finetune_lfm2_vl`` this:

* grows the discrete vocabulary (~60 colors, ~18 shapes incl. pseudo-3D), and
* adds continuous per-instance **jitter** to color, size, position and (for flat
  polygons) rotation, so each class's appearance distribution spreads out and
  **overlaps** its neighbours. The task therefore does not saturate, which makes a
  capacity (LoRA-rank) sweep informative.

Records share the same schema and label format as ``finetune_lfm2_vl`` so the same
``_to_conversation`` / ``evaluate`` / ``is_correct`` helpers can be reused.
"""

import math
import random

from PIL import Image, ImageDraw


# ~60 reasonably distinguishable named colors.
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

# ~18 shapes: flat polygons (rotatable) + smooth (circle/ellipse) + pseudo-3D.
POLY_SHAPES = [
    "square",
    "triangle",
    "diamond",
    "pentagon",
    "hexagon",
    "heptagon",
    "octagon",
    "star",
    "cross",
    "trapezoid",
]
SMOOTH_SHAPES = ["circle", "ellipse"]
SHAPES_3D = ["cube", "sphere", "cylinder", "cone", "pyramid"]
SHAPES_2D = POLY_SHAPES + SMOOTH_SHAPES
SHAPES = SHAPES_2D + SHAPES_3D
RELATIONS = ["inside", "to the left of", "to the right of", "above", "below"]

QUESTION_SINGLE = "What colored shape is in this image? Answer with '<color> <shape>'."
QUESTION_SPATIAL = "Where is the {a} located relative to the {b}? Describe the spatial relationship."


def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def _rotate(points, cx, cy, angle):
    ca, sa = math.cos(angle), math.sin(angle)
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


def draw_shape(draw, shape, rgb, cx, cy, r, angle=0.0):
    """Draw a single shape with a given RGB color, centered at (cx, cy)."""
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "ellipse":
        draw.ellipse([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6], fill=rgb)
    elif shape in POLY_SHAPES:
        pts = _poly(shape, cx, cy, r)
        if angle:
            pts = _rotate(pts, cx, cy, angle)
        draw.polygon(pts, fill=rgb)
    elif shape == "cube":
        draw.polygon([(cx - r, cy - r / 2), (cx, cy), (cx, cy + r), (cx - r, cy + r / 2)], fill=_shade(rgb, 0.6))
        draw.polygon([(cx, cy), (cx + r, cy - r / 2), (cx + r, cy + r / 2), (cx, cy + r)], fill=_shade(rgb, 0.85))
        draw.polygon([(cx, cy - r), (cx + r, cy - r / 2), (cx, cy), (cx - r, cy - r / 2)], fill=_shade(rgb, 1.2))
    elif shape == "sphere":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_shade(rgb, 0.8))
        for i in range(7):
            t = i / 7
            rr = r * (1 - t) * 0.85
            ox, oy = cx - r * 0.28, cy - r * 0.28
            draw.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], fill=_shade(rgb, 0.8 + 0.55 * t))
    elif shape == "cylinder":
        eh = r * 0.32
        draw.ellipse([cx - r, cy + r - eh, cx + r, cy + r + eh], fill=_shade(rgb, 0.65))
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_shade(rgb, 0.85))
        draw.ellipse([cx - r, cy - r - eh, cx + r, cy - r + eh], fill=_shade(rgb, 1.15))
    elif shape == "cone":
        eh = r * 0.32
        draw.ellipse([cx - r, cy + r - eh, cx + r, cy + r + eh], fill=_shade(rgb, 0.65))
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=_shade(rgb, 0.95))
    elif shape == "pyramid":
        apex, front = (cx, cy - r), (cx, cy + r)
        draw.polygon([apex, (cx - r, cy + r * 0.35), front], fill=_shade(rgb, 0.7))
        draw.polygon([apex, front, (cx + r, cy + r * 0.35)], fill=_shade(rgb, 1.0))
    else:
        raise ValueError(shape)


# ---- jitter ("두 배씩 더 어렵게": overlapping color / shape / position distributions) ----
def _jit_color(rng, color):
    return tuple(max(0, min(255, c + int(rng.gauss(0, 14)))) for c in COLORS[color])


def _jit_radius(rng, base):
    return int(base * rng.uniform(0.82, 1.18))


def _jit_angle(rng, shape):
    # small rotation for flat polygons only (avoid collapsing e.g. square<->diamond)
    return rng.uniform(-0.35, 0.35) if shape in POLY_SHAPES else 0.0


def _place(rng, base_xy, spread):
    return base_xy[0] + rng.randint(-spread, spread), base_xy[1] + rng.randint(-spread, spread)


def _canvas():
    return Image.new("RGB", (256, 256), "white")


def render_single(rng, color, shape):
    img = _canvas()
    cx, cy = _place(rng, (128, 128), 22)
    draw_shape(
        ImageDraw.Draw(img), shape, _jit_color(rng, color), cx, cy, _jit_radius(rng, 80), _jit_angle(rng, shape)
    )
    return img


def render_relation(rng, c1, s1, relation, c2, s2):
    """Render the scene matching 'the c1 s1 is <relation> the c2 s2', with jitter."""
    img = _canvas()
    d = ImageDraw.Draw(img)
    r = _jit_radius(rng, 38)
    if relation == "inside":
        ro = _jit_radius(rng, 110)
        draw_shape(d, s2, _jit_color(rng, c2), 128, 128, ro, _jit_angle(rng, s2))
        draw_shape(d, s1, _jit_color(rng, c1), *_place(rng, (128, 128), 8), _jit_radius(rng, 34), _jit_angle(rng, s1))
        return img
    # left/right -> horizontal; above/below -> vertical. Map subject (c1 s1) to its side.
    horizontal = relation in ("to the left of", "to the right of")
    subject_first = relation in ("to the left of", "above")  # subject occupies the first slot
    if horizontal:
        slot_a, slot_b = (64, 128), (192, 128)
    else:
        slot_a, slot_b = (128, 64), (128, 192)
    spread = 16
    if subject_first:
        draw_shape(d, s1, _jit_color(rng, c1), *_place(rng, slot_a, spread), r, _jit_angle(rng, s1))
        draw_shape(d, s2, _jit_color(rng, c2), *_place(rng, slot_b, spread), _jit_radius(rng, 38), _jit_angle(rng, s2))
    else:
        draw_shape(d, s2, _jit_color(rng, c2), *_place(rng, slot_a, spread), r, _jit_angle(rng, s2))
        draw_shape(d, s1, _jit_color(rng, c1), *_place(rng, slot_b, spread), _jit_radius(rng, 38), _jit_angle(rng, s1))
    return img


def _single_record(rng, color, shape):
    return {
        "image": render_single(rng, color, shape),
        "question": QUESTION_SINGLE,
        "answer": f"{color} {shape}",
        "meta": ("single", color, shape),
    }


def _spatial_record(rng, c1, s1, relation, c2, s2):
    return {
        "image": render_relation(rng, c1, s1, relation, c2, s2),
        "question": QUESTION_SPATIAL.format(a=f"{c1} {s1}", b=f"{c2} {s2}"),
        "answer": f"the {c1} {s1} is {relation} the {c2} {s2}",
        "meta": ("spatial", c1, s1, relation, c2, s2),
    }


def _colors_shapes(meta):
    return ({meta[1]}, {meta[2]}) if meta[0] == "single" else ({meta[1], meta[4]}, {meta[2], meta[5]})


def build_split(n_train, n_test, seed):
    """Build (train, test) where test combinations are novel but use only seen tokens."""
    rng = random.Random(seed)
    n_single = max(1, round(n_train * 0.4))
    records = []

    combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(combos)
    for c, s in combos[:n_single]:
        records.append(_single_record(rng, c, s))
    for i in range(n_train - n_single):
        rel = RELATIONS[i % len(RELATIONS)]
        pool = SHAPES_2D if rel == "inside" else SHAPES
        c1, c2 = rng.sample(list(COLORS), 2)
        s1, s2 = rng.sample(pool, 2)
        records.append(_spatial_record(rng, c1, s1, rel, c2, s2))
    rng.shuffle(records)

    seen_singles = {(m[1], m[2]) for r in records if (m := r["meta"])[0] == "single"}
    seen_spatial = {m for r in records if (m := r["meta"])[0] == "spatial"}
    seen_colors, seen_shapes, seen_rels = set(), set(), set()
    for r in records:
        cs, ss = _colors_shapes(r["meta"])
        seen_colors |= cs
        seen_shapes |= ss
        if r["meta"][0] == "spatial":
            seen_rels.add(r["meta"][3])
    seen_colors, seen_shapes, seen_rels = sorted(seen_colors), sorted(seen_shapes), sorted(seen_rels)

    trng = random.Random(seed + 1)
    test, used = [], set()
    while sum(r["meta"][0] == "single" for r in test) < round(n_test * 0.4):
        combo = (trng.choice(seen_colors), trng.choice(seen_shapes))
        if combo in seen_singles or combo in used:
            continue
        used.add(combo)
        test.append(_single_record(trng, *combo))
    while len(test) < n_test:
        rel = trng.choice(seen_rels)
        pool = [s for s in seen_shapes if s in SHAPES_2D] if rel == "inside" else seen_shapes
        if len(pool) < 2:
            continue
        c1, c2 = trng.sample(seen_colors, 2)
        s1, s2 = trng.sample(pool, 2)
        tup = ("spatial", c1, s1, rel, c2, s2)
        if tup in seen_spatial or tup in used:
            continue
        used.add(tup)
        test.append(_spatial_record(trng, c1, s1, rel, c2, s2))
    trng.shuffle(test)
    return records, test, {"colors": len(seen_colors), "shapes": len(seen_shapes), "relations": len(seen_rels)}
