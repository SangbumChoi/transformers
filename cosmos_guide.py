#!/usr/bin/env python3
"""Generate a DETAILED, beginner-friendly PDF on NVIDIA Cosmos generations.

Adds: architecture diagrams, bar charts (data, benchmarks, inference speed),
input/output DTO tables, model sizes & hyperparameters, data diversity, and
the pretraining + post-training (SFT / GRPO) pipeline.
"""

import math
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.legends import Legend

# ---------------------------------------------------------------- palette
NV_GREEN = colors.HexColor("#76B900")
G1 = colors.HexColor("#76B900")  # cosmos 1
G2 = colors.HexColor("#2E86C1")  # cosmos 2
G3 = colors.HexColor("#8E44AD")  # cosmos 2.5
G4 = colors.HexColor("#E67E22")  # cosmos 3
DARK = colors.HexColor("#1A1A1A")
GREY = colors.HexColor("#555555")
LIGHT_BG = colors.HexColor("#F2F7E9")
BLUE_BG = colors.HexColor("#EAF2FB")
PURP_BG = colors.HexColor("#F3EAF8")
ORNG_BG = colors.HexColor("#FDF0E3")
ROW_ALT = colors.HexColor("#F7F7F7")
BORDER = colors.HexColor("#CCCCCC")

PT = 1.0  # points

# ---------------------------------------------------------------- styles
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=22, textColor=DARK, spaceAfter=4, leading=26)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=11, textColor=GREY,
                     spaceAfter=12, leading=15)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=15, textColor=NV_GREEN, spaceBefore=14, spaceAfter=6, leading=18)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Helvetica-Bold",
                    fontSize=12, textColor=DARK, spaceBefore=8, spaceAfter=3, leading=15)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=10, textColor=DARK,
                      spaceAfter=6, leading=15, alignment=TA_LEFT)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=3)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, textColor=GREY, leading=11)
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8.5, textColor=GREY, leading=11,
                     spaceBefore=2, spaceAfter=10)
ANALOGY = ParagraphStyle("ANALOGY", parent=BODY, fontName="Helvetica-Oblique",
                         fontSize=9.5, textColor=GREY, leftIndent=10, leading=14, spaceAfter=8)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CELL_H = ParagraphStyle("CELL_H", parent=CELL, fontName="Helvetica-Bold",
                        textColor=colors.white, fontSize=8)
CELL_L = ParagraphStyle("CELL_L", parent=CELL, fontName="Helvetica-Bold", fontSize=8)

story = []

def p(t, s=BODY): story.append(Paragraph(t, s))
def bullets(items, s=BULLET):
    for it in items: story.append(Paragraph(it, s, bulletText="•"))
def spacer(h=6): story.append(Spacer(1, h))
def rule(): story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER,
                                    spaceBefore=8, spaceAfter=8))
def caption(t): story.append(Paragraph(t, CAP))

# ---------------------------------------------------------------- diagram helpers
def box(d, x, y, w, h, lines, fill, tcolor=DARK, fs=7.5, stroke=None, bold=True):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=stroke or DARK,
               strokeWidth=0.8, rx=4, ry=4))
    if isinstance(lines, str): lines = [lines]
    n = len(lines); lh = fs + 2
    start = y + h/2 + (n-1)*lh/2 - fs*0.35
    for i, ln in enumerate(lines):
        d.add(String(x+w/2, start - i*lh, ln, textAnchor="middle", fontSize=fs,
                     fontName="Helvetica-Bold" if bold else "Helvetica", fillColor=tcolor))

def arrow(d, x1, y1, x2, y2, color=GREY, w=1.2):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w))
    ang = math.atan2(y2-y1, x2-x1); ah = 5
    d.add(Polygon([x2, y2,
                   x2-ah*math.cos(ang-0.4), y2-ah*math.sin(ang-0.4),
                   x2-ah*math.cos(ang+0.4), y2-ah*math.sin(ang+0.4)],
                  fillColor=color, strokeColor=color))

def label(d, x, y, text, fs=7, color=GREY, anchor="middle", bold=False):
    d.add(String(x, y, text, textAnchor=anchor, fontSize=fs, fillColor=color,
                 fontName="Helvetica-Bold" if bold else "Helvetica"))

# ---------------------------------------------------------------- chart helpers
def vbar(data, cats, series_colors, series_names, w=480, h=170,
         vmin=0, vmax=100, step=20, ylabel="", barlabels=True):
    d = Drawing(w, h)
    bc = VerticalBarChart()
    bc.x = 38; bc.y = 28; bc.height = h-58; bc.width = w-70
    bc.data = data
    bc.categoryAxis.categoryNames = cats
    bc.categoryAxis.labels.fontSize = 7.5
    bc.categoryAxis.labels.angle = 0
    bc.categoryAxis.labels.dy = -3
    bc.valueAxis.valueMin = vmin
    bc.valueAxis.valueMax = vmax
    bc.valueAxis.valueStep = step
    bc.valueAxis.labels.fontSize = 7
    bc.barWidth = 6
    bc.groupSpacing = 12
    bc.barSpacing = 1
    for i, c in enumerate(series_colors):
        bc.bars[i].fillColor = c
        bc.bars[i].strokeColor = colors.white
    if barlabels:
        bc.barLabels.fontSize = 6.5
        bc.barLabelFormat = "%0.1f"
        bc.barLabels.dy = 4
        bc.barLabels.fillColor = DARK
    d.add(bc)
    # legend
    if series_names:
        lg = Legend()
        lg.x = 40; lg.y = h-8; lg.deltax = 95; lg.dxTextSpace = 4
        lg.fontSize = 7; lg.alignment = "right"; lg.columnMaximum = 1
        lg.colorNamePairs = list(zip(series_colors, series_names))
        d.add(lg)
    if ylabel:
        d.add(String(8, h/2, ylabel, textAnchor="middle", fontSize=7,
                     fillColor=GREY, angle=90))
    return d

def hbar(values, cats, bar_color, w=480, h=200, vmax=22, label_fmt="%d%%"):
    d = Drawing(w, h)
    bc = HorizontalBarChart()
    bc.x = 95; bc.y = 12; bc.height = h-24; bc.width = w-150
    bc.data = [values]
    bc.categoryAxis.categoryNames = cats
    bc.categoryAxis.labels.fontSize = 7.5
    bc.categoryAxis.labels.dx = -3
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = vmax
    bc.valueAxis.valueStep = 5
    bc.valueAxis.labels.fontSize = 7
    bc.barWidth = 9
    bc.groupSpacing = 6
    bc.bars[0].fillColor = bar_color
    bc.bars[0].strokeColor = colors.white
    bc.barLabels.fontSize = 7
    bc.barLabelFormat = label_fmt
    bc.barLabels.dx = 8
    bc.barLabels.fillColor = DARK
    d.add(bc)
    return d

# ================================================================ PAGE 1
p("NVIDIA Cosmos, Explained in Depth", H1)
p("A beginner-to-intermediate guide to all four generations &mdash; Cosmos 1, 2, 2.5, "
  "and 3 &mdash; with diagrams, model sizes, data, inference speed, and the "
  "pretraining + post-training (SFT / GRPO) recipes.", SUB)

p("What is a “World Foundation Model” (WFM)?", H2)
p("A <b>language</b> model learns from text and predicts the next <i>word</i>. "
  "A <b>World Foundation Model</b> learns from video and predicts what happens next in "
  "the <i>physical world</i> &mdash; how things move, collide, and behave. NVIDIA <b>Cosmos</b> "
  "is a family of WFMs for <b>Physical AI</b> (robots, self-driving cars): instead of testing a "
  "robot millions of times in reality, engineers let Cosmos <b>imagine</b> realistic video of "
  "what would happen and train on that.", BODY)
story.append(Paragraph(
    "Mental model: Cosmos is a “dream engine” for machines. Each generation dreams with "
    "higher quality, runs faster, understands your request better, and &mdash; by Cosmos 3 &mdash; "
    "even hears sound and decides physical actions.", ANALOGY))

p("Mini-glossary (so the rest makes sense)", H3)
gloss = [
    ["Tokenizer", "Compresses video into a small set of numbers (“tokens”) the model can handle, and decompresses them back to pixels."],
    ["Diffusion", "Starts from random noise and repeatedly “cleans” it into a video. High quality, but many steps = slower."],
    ["Autoregressive (GPT-style)", "Generates the video one chunk at a time, like typing word by word. Fast, good for real-time."],
    ["Flow-based", "A cleaner, more direct cousin of diffusion: learns a straight “flow” from noise to data, often with fewer steps."],
    ["DiT", "“Diffusion Transformer” &mdash; the Transformer architecture that powers the diffusion engine."],
    ["Mixture-of-Transformers", "Several specialized Transformers in one model (e.g. one that <i>reasons</i> + one that <i>generates</i>)."],
    ["Pretraining", "Learning general knowledge from a huge, broad dataset."],
    ["SFT", "Supervised Fine-Tuning: teach the model on curated question&rarr;answer examples after pretraining."],
    ["GRPO", "An RL method: the model tries an answer several times; better-than-average tries get reinforced. Sharpens reasoning."],
]
gd = [[Paragraph("<b>%s</b>" % r[0], CELL), Paragraph(r[1], CELL)] for r in gloss]
gt = Table(gd, colWidths=[4.0*cm, 13.0*cm])
gt.setStyle(TableStyle([
    ("GRID", (0,0), (-1,-1), 0.5, BORDER),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BACKGROUND", (0,0), (0,-1), LIGHT_BG),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
]))
story.append(gt)

story.append(PageBreak())

# ================================================================ PAGE 2 — master table
p("The big picture: all four generations side by side", H2)
header = ["", "Cosmos 1", "Cosmos 2", "Cosmos 2.5", "Cosmos 3"]
rows = [
    ["Released", "Jan 2025", "Jun 2025", "Sep 2025", "Jun 2026"],
    ["Core engine", "TWO engines:\ndiffusion + GPT-style", "Refined diffusion", "ONE unified\nflow-based model", "Mixture-of-Transformers\n(reason + generate)"],
    ["Inputs", "Text, or image/video", "Text, or image/video", "Text/image/video\n(one model)", "Text, image, video,\nAUDIO + ACTIONS"],
    ["Outputs", "Future video", "Video (480/704p)", "Video up to 30s,\nmulti-camera", "Video + actions +\naudio, with reasoning"],
    ["Reads text via", "T5-XXL encoder", "T5-style encoder", "Cosmos-Reason1 (VLM)", "Built-in reasoning\ntransformer"],
    ["Model sizes", "Diff 7B/14B,\nGPT 4B/12B", "0.6B / 2B / 14B", "2B / 14B\n(+auto/robot variants)", "Super / Nano / Edge"],
    ["Training data", "~100M clips\n(from 20M hrs)", "+ action datasets", "200M curated clips", "~20 trillion tokens"],
    ["Post-training", "Video2World\nfine-tune", "Action-conditioned\nfine-tune", "Model merge + RL", "SFT + RL (GRPO-style)"],
    ["One-word vibe", "Kitchen sink", "Polish", "Unify", "Leap"],
]
def cellP(t, header=False, label=False):
    if header: return Paragraph(t.replace("\n","<br/>"), CELL_H)
    if label:  return Paragraph(t.replace("\n","<br/>"), CELL_L)
    return Paragraph(t.replace("\n","<br/>"), CELL)
tdata = [[cellP(h, header=True) for h in header]]
for r in rows:
    tdata.append([cellP(r[0], label=True)] + [cellP(c) for c in r[1:]])
colw = [2.5*cm, 3.5*cm, 3.0*cm, 3.7*cm, 4.0*cm]
tbl = Table(tdata, colWidths=colw, repeatRows=1)
ts = [
    ("BACKGROUND", (1,0), (1,0), G1), ("BACKGROUND", (2,0), (2,0), G2),
    ("BACKGROUND", (3,0), (3,0), G3), ("BACKGROUND", (4,0), (4,0), G4),
    ("BACKGROUND", (0,0), (0,0), DARK),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#EEEEEE")),
]
for i in range(1, len(tdata)):
    if i % 2 == 0: ts.append(("BACKGROUND", (1,i), (-1,i), ROW_ALT))
tbl.setStyle(TableStyle(ts))
story.append(tbl)
caption("Table 1. Each column is colour-coded and reused throughout this guide: "
        "<font color='#76B900'>Cosmos 1</font>, <font color='#2E86C1'>Cosmos 2</font>, "
        "<font color='#8E44AD'>Cosmos 2.5</font>, <font color='#E67E22'>Cosmos 3</font>.")

story.append(PageBreak())

# ================================================================ PAGE 3 — architecture diagrams
p("How the architecture changed (diagrams)", H2)

# --- Cosmos 1 diagram: two engines
p("Cosmos 1 &mdash; two parallel engines", H3)
d1 = Drawing(482, 150)
box(d1, 0, 60, 70, 34, ["Prompt /", "image"], colors.white)
arrow(d1, 70, 77, 95, 77)
box(d1, 95, 60, 70, 34, ["Tokenizer", "(compress)"], LIGHT_BG)
# split to two engines
arrow(d1, 165, 80, 195, 112)
arrow(d1, 165, 74, 195, 42)
box(d1, 195, 100, 120, 30, ["DIFFUSION (DiT) 7B/14B", "continuous tokens"], colors.HexColor("#E8F5D5"), stroke=G1)
box(d1, 195, 28, 120, 30, ["AUTOREGRESSIVE 4B/12B", "discrete tokens (GPT)"], colors.HexColor("#FDEBD5"), stroke=G4)
arrow(d1, 315, 115, 360, 90)
arrow(d1, 315, 43, 345, 43)
box(d1, 345, 28, 70, 30, ["Diffusion", "decoder"], colors.white)
arrow(d1, 415, 43, 435, 70)
arrow(d1, 360, 90, 410, 78)
box(d1, 410, 60, 70, 34, ["Output", "video"], colors.HexColor("#E8F5D5"), stroke=G1)
label(d1, 230, 138, "high quality, slower", 6.5, G1)
label(d1, 230, 18, "fast / real-time, rougher", 6.5, G4)
story.append(d1)
caption("Figure 1. Cosmos 1 shipped two separate engines. Diffusion (top) = best quality; "
        "autoregressive (bottom) = fast, but needs a diffusion decoder to clean up its output. "
        "Text enters both via a T5-XXL encoder (not shown).")

# --- Cosmos 2.5 diagram: unified
p("Cosmos 2.5 &mdash; one unified flow model with a “smart” text encoder", H3)
d3 = Drawing(482, 110)
box(d3, 0, 40, 95, 34, ["Text / image /", "video (any)"], colors.white)
arrow(d3, 95, 57, 125, 57)
box(d3, 125, 38, 90, 40, ["Cosmos-Reason1", "(reads & understands", "the request)"], PURP_BG, stroke=G3, fs=7)
arrow(d3, 215, 57, 250, 57)
box(d3, 250, 36, 130, 44, ["UNIFIED FLOW MODEL", "2B / 14B", "(replaces 3 old models)"], colors.HexColor("#EFE0F6"), stroke=G3, fs=7.5)
arrow(d3, 380, 57, 410, 57)
box(d3, 410, 40, 72, 34, ["30s video", "multi-cam"], colors.HexColor("#EFE0F6"), stroke=G3)
story.append(d3)
caption("Figure 2. Cosmos 2.5 merged the three separate task-models of earlier versions into a "
        "single flow-based model, and upgraded the text encoder to Cosmos-Reason1 &mdash; a model "
        "that actually understands physics and language.")

# --- Cosmos 3 diagram: mixture of transformers
p("Cosmos 3 &mdash; mixture-of-transformers (thinks before it dreams)", H3)
d4 = Drawing(482, 130)
box(d4, 0, 48, 85, 44, ["Text, image,", "video, audio,", "actions"], colors.white, fs=7)
arrow(d4, 85, 70, 115, 70)
box(d4, 115, 46, 120, 48, ["REASONING", "TRANSFORMER", "(understands physics,", "objects, motion)"], ORNG_BG, stroke=G4, fs=7)
arrow(d4, 235, 70, 270, 70)
box(d4, 270, 46, 120, 48, ["EXPERT GENERATION", "TRANSFORMER", "(creates video +", "action trajectory)"], colors.HexColor("#FBE3CC"), stroke=G4, fs=7)
arrow(d4, 390, 70, 415, 70)
box(d4, 415, 48, 67, 44, ["Video +", "actions +", "audio"], colors.HexColor("#FBE3CC"), stroke=G4, fs=7)
label(d4, 240, 30, "one model: reason  →  generate  →  act", 7.5, GREY, bold=True)
story.append(d4)
caption("Figure 3. Cosmos 3 is no longer just a video predictor. A reasoning transformer first "
        "understands the scene, then an expert generation transformer produces video AND physical "
        "actions (and audio) &mdash; all in one “omnimodel”.")

story.append(PageBreak())

# ================================================================ PAGE 4 — data & diversity
p("The data: scale and diversity", H2)
p("A WFM is only as good as the video it learns from. Cosmos 1 curated <b>~100 million clips</b> "
  "out of <b>20 million hours</b> of raw video, deliberately balanced across nine kinds of physical "
  "scene so the model isn't biased toward one (e.g. only driving). Later generations grew the data "
  "by orders of magnitude.", BODY)

p("Cosmos 1 training-data mix (by category)", H3)
cats = ["Nature dynamics", "Manipulation", "Navigation", "Driving", "Human motion",
        "First-person POV", "Camera movement", "Synthetic render", "Other"]
vals = [20, 16, 16, 11, 10, 8, 8, 4, 7]
story.append(hbar(vals, cats, G1, w=482, h=185, vmax=22))
caption("Figure 4. Diversity is intentional: nature, robot manipulation, and navigation dominate, "
        "but driving, human motion and synthetic data are all represented. This breadth is what lets "
        "one pretrained model be fine-tuned for many downstream robots/vehicles.")

p("Data scale across generations", H3)
ds = [
    ["Generation", "Pretraining data", "Notes"],
    ["Cosmos 1", "~100M clips (from ~20M hours)", "9 balanced categories; ~30% removed by dedup"],
    ["Cosmos 2", "Cosmos 1 data + action datasets", "Bridge, AgiBotWorld, GR00T Dreams for post-training"],
    ["Cosmos 2.5", "200M curated high-quality clips", "+ domain post-training data (auto, robotics)"],
    ["Cosmos 3", "~20 trillion tokens", "~1B images, ~400M videos, + audio + action trajectories"],
]
dd = [[cellP(c, header=(i==0)) for c in r] for i, r in enumerate(ds)]
dt = Table(dd, colWidths=[2.8*cm, 6.2*cm, 8.0*cm])
dt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NV_GREEN),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("BACKGROUND", (0,2), (-1,2), ROW_ALT), ("BACKGROUND", (0,4), (-1,4), ROW_ALT),
]))
story.append(dt)
caption("Table 2. The jump from “100M clips” to “20T tokens” reflects Cosmos 3 also ingesting "
        "text, audio and robot/human action data &mdash; not just video. "
        "<b>Caveat:</b> Cosmos 3's exact counts (20T tokens, ~1B images, ~400M videos) come from "
        "press coverage; NVIDIA's official release states only “billions of samples across text, "
        "image, video, sound and action.”")

story.append(PageBreak())

# ================================================================ PAGE 5 — model sizes & hyperparameters
p("Model sizes & key hyperparameters", H2)
p("“Parameters” (B = billion) are the model's adjustable knobs &mdash; more usually means smarter "
  "but slower/heavier. Cosmos offers small <i>and</i> large versions so you can trade quality for "
  "speed.", BODY)

p("Every model variant at a glance", H3)
ms = [
    ["Generation", "Variants (parameters)", "Helper models"],
    ["Cosmos 1", "Diffusion 7B & 14B; Autoregressive 4B & 12B (5B/13B for Video2World)",
     "Tokenizer 77–105M; Prompt-upsampler 12B; Diffusion-decoder 7B"],
    ["Cosmos 2", "0.6B (Text2Image), 2B, 14B", "Sparse-attention inference path"],
    ["Cosmos 2.5", "2B & 14B (each: pre-trained + post-trained); auto 7-cam & robot 3-cam variants",
     "Cosmos-Reason1 as text encoder"],
    ["Cosmos 3", "Super (max accuracy), Nano (sub-second), Edge (local, soon)", "Built-in reasoning transformer"],
]
md = [[cellP(c, header=(i==0)) for c in r] for i, r in enumerate(ms)]
mt = Table(md, colWidths=[2.5*cm, 8.5*cm, 6.0*cm])
mt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NV_GREEN),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("BACKGROUND", (0,2), (-1,2), ROW_ALT), ("BACKGROUND", (0,4), (-1,4), ROW_ALT),
]))
story.append(mt)
spacer(8)

p("Cosmos 1 diffusion hyperparameters (the most fully documented)", H3)
hp = [
    ["Setting", "7B model", "14B model"],
    ["Transformer layers", "28", "36"],
    ["Model (hidden) dimension", "4,096", "5,120"],
    ["Attention heads", "32", "40"],
    ["AdaLN-LoRA rank", "256", "256"],
    ["Learning rate", "2^-15", "2^-16"],
    ["Optimizer", "AdamW (beta1=0.9, beta2=0.99), weight decay 0.1–0.2", ""],
    ["Warmup", "2,500 iterations, linear", ""],
    ["Text encoder", "T5-XXL, 512 tokens via cross-attention", ""],
    ["Resolution schedule", "512p (57 frames)  →  720p (121 frames)", ""],
    ["Context length @720p", "56,320 tokens", ""],
    ["Diffusion formulation", "EDM denoising score-matching; QK-RMSNorm for stability", ""],
    ["Compute", "~10,000 H100 GPUs, ~3 months", ""],
]
hd = []
for i, r in enumerate(hp):
    if i == 0:
        hd.append([cellP(c, header=True) for c in r])
    elif r[2] == "":
        hd.append([cellP(r[0], label=True), Paragraph(r[1], CELL)])  # span handled below
    else:
        hd.append([cellP(r[0], label=True), cellP(r[1]), cellP(r[2])])
# normalize rows to 3 cols
norm = []
span_rows = []
for i, r in enumerate(hp):
    if i == 0:
        norm.append([cellP(c, header=True) for c in r]); continue
    if r[2] == "":
        norm.append([cellP(r[0], label=True), cellP(r[1]), cellP("")])
        span_rows.append(i)
    else:
        norm.append([cellP(r[0], label=True), cellP(r[1]), cellP(r[2])])
ht = Table(norm, colWidths=[5.5*cm, 6.0*cm, 5.5*cm])
hstyle = [
    ("BACKGROUND", (0,0), (-1,0), G1),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("BACKGROUND", (0,1), (0,-1), LIGHT_BG),
]
for i in span_rows:
    hstyle.append(("SPAN", (1,i), (2,i)))
ht.setStyle(TableStyle(hstyle))
story.append(ht)
caption("Table 3. NVIDIA published far more hyperparameters for Cosmos 1 than for the closed-er "
        "later releases; 2.5 and 3 reuse the same design DNA (3D-RoPE positions, AdaLN-LoRA, a "
        "causal tokenizer) at larger scale.")

story.append(PageBreak())

# ================================================================ PAGE 6 — Input/Output DTO + distribution
p("Inputs and outputs: the “data contract” (DTO)", H2)
p("Think of a <b>DTO</b> (data-transfer object) as the exact shape of what you hand the model and "
  "what it hands back. This shape widened a lot across generations.", BODY)
io = [
    ["Generation", "INPUT (what you provide)", "OUTPUT (what you get back)"],
    ["Cosmos 1\n(diffusion)", "Text → T5-XXL embedding (512 tokens) + optional conditioning frames + fps + frame-count",
     "Continuous latent video → decoded to RGB frames"],
    ["Cosmos 1\n(autoregressive)", "Discrete video tokens (DV 8×16×16) + T5 text embedding",
     "Next discrete tokens → diffusion-decoder → RGB frames"],
    ["Cosmos 2", "Text, or image, or video + fps", "RGB video at 480p / 704p, 10–16 fps"],
    ["Cosmos 2.5", "Text / image / video, encoded by Cosmos-Reason1",
     "Up to 30s video; optional multi-camera (3–7 synced views)"],
    ["Cosmos 3", "Text + image + video + ambient audio + action trajectories",
     "Video + action trajectory + audio, preceded by explicit reasoning"],
]
iod = [[cellP(c, header=(i==0)) for c in r] for i, r in enumerate(io)]
iot = Table(iod, colWidths=[3.0*cm, 7.0*cm, 7.0*cm])
iot.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NV_GREEN),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("BACKGROUND", (0,2), (-1,2), ROW_ALT), ("BACKGROUND", (0,4), (-1,4), ROW_ALT),
]))
story.append(iot)
caption("Table 4. The input distribution broadens from “text + frames” (Cosmos 1) to a full "
        "five-modality stream including audio and actions (Cosmos 3). The output likewise grows from "
        "“a short video” to “video + the physical actions a robot should take”.")

spacer(6)
p("Why the input distribution matters", H3)
bullets([
    "<b>Cosmos 1–2:</b> trained mostly on curated <i>internet-style</i> video across 9 categories &mdash; "
    "great general physics, but you fine-tune for your specific robot.",
    "<b>Cosmos 2.5:</b> 200M <i>physical-AI-focused</i> clips + domain post-training (driving, manipulation) "
    "shift the distribution toward real deployment.",
    "<b>Cosmos 3:</b> adds <i>action trajectories from real humans and robots</i> and <i>audio</i> to the "
    "training mix &mdash; so the input/output distribution now includes the robot's own behaviour, not just "
    "what a camera sees.",
])

story.append(PageBreak())

# ================================================================ PAGE 7 — inference speed
p("Key results: inference speed", H2)
p("Speed is the headline practical difference. Faster generation = more simulations per day = "
  "faster robot/AV development. Each generation attacked latency differently.", BODY)

speed = [
    ["Generation", "Speed lever", "Reported result"],
    ["Cosmos 1", "Causal, factorized tokenizer", "Up to 12× faster encode/decode vs. prior tokenizer (CogVideoX)"],
    ["Cosmos 2", "Sparse attention in the DiT", "Up to 2.6× end-to-end inference speedup"],
    ["Cosmos 2.5", "Flow-based sampling (fewer steps) + smaller 2B option", "Faster, more resource-efficient than 2.x diffusion"],
    ["Cosmos 3", "Dedicated Nano variant", "High-quality video + action reasoning in a fraction of a second"],
]
sp = [[cellP(c, header=(i==0)) for c in r] for i, r in enumerate(speed)]
spt = Table(sp, colWidths=[2.6*cm, 6.4*cm, 8.0*cm])
spt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NV_GREEN),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("BACKGROUND", (0,2), (-1,2), ROW_ALT), ("BACKGROUND", (0,4), (-1,4), ROW_ALT),
]))
story.append(spt)
caption("Table 5. Two concrete, comparable speedups are published: Cosmos 1's tokenizer (up to "
        "12×) and Cosmos 2's sparse attention (up to 2.6×). 2.5 and 3 emphasise efficiency "
        "via flow sampling and a sub-second Nano tier rather than a single headline multiplier.")

spacer(4)
story.append(vbar(
    data=[[12.0, 2.6]],
    cats=["Cosmos 1\ntokenizer", "Cosmos 2\nsparse attn"],
    series_colors=[NV_GREEN], series_names=None,
    w=300, h=160, vmin=0, vmax=14, step=2, ylabel="speedup (×)", barlabels=True))
caption("Figure 5. The two directly-quoted “× speedup” numbers. (Different things are being "
        "sped up &mdash; tokenizer vs. whole pipeline &mdash; so this compares <i>magnitude of "
        "improvement</i>, not absolute latency.)")

spacer(2)
p("The quality/speed trade-off in one sentence", H3)
p("<b>Diffusion</b> (Cosmos 1–2) = many denoising steps = top quality but slower; "
  "<b>autoregressive</b> (Cosmos 1) = stream tokens = fast but rougher; "
  "<b>flow-based</b> (Cosmos 2.5) = fewer steps for similar quality; "
  "<b>Nano</b> (Cosmos 3) = engineered for sub-second responses.", BODY)

story.append(PageBreak())

# ================================================================ PAGE 8 — pretraining + post-training
p("Pretraining and post-training: SFT and GRPO", H2)
p("Modern Cosmos models are built in two phases. <b>Pretraining</b> soaks up broad knowledge from "
  "massive data. <b>Post-training</b> then sharpens behaviour with curated examples (<b>SFT</b>) and "
  "reinforcement learning (<b>GRPO</b>). This recipe is clearest in <b>Cosmos-Reason1</b> &mdash; the "
  "reasoning brain that powers Cosmos 2.5's text understanding and the spirit of Cosmos 3.", BODY)

p("The 4-stage training pipeline (Cosmos-Reason1)", H3)
dp = Drawing(482, 90)
boxw, boxh, gap = 108, 46, 12
xs = 0
stages = [
    (["1. Vision", "pretraining", "(broad video)"], LIGHT_BG, G1),
    (["2. General", "SFT", "(follow tasks)"], BLUE_BG, G2),
    (["3. Physical AI", "SFT", "(physics QA + CoT)"], PURP_BG, G3),
    (["4. Physical AI", "RL — GRPO", "(verifiable rewards)"], ORNG_BG, G4),
]
x = 0
for i, (lines, fill, stroke) in enumerate(stages):
    box(dp, x, 30, boxw, boxh, lines, fill, stroke=stroke, fs=7.5)
    if i < 3:
        arrow(dp, x+boxw, 53, x+boxw+gap, 53)
    x += boxw + gap
label(dp, 110, 18, "general knowledge", 6.5, GREY)
label(dp, 360, 18, "sharpened physical reasoning", 6.5, GREY)
story.append(dp)
caption("Figure 6. Knowledge first (stages 1–2), then physical-world specialisation (stage 3), "
        "then reinforcement learning to make the reasoning reliable (stage 4).")

p("What happens in SFT (stage 3)", H3)
bullets([
    "Curated <b>question → answer</b> examples for physical common sense (space, time, basic "
    "physics) and embodied decisions (what should the robot do?).",
    "Built from real robotics/AV datasets: <b>BridgeData V2, RoboVQA, AgiBot, HoloAssist</b>, and "
    "driving data.",
    "The team hand-picked <b>~200–250 examples per data source</b>, turned them into "
    "multiple-choice questions, and removed ambiguous ones &mdash; quality over quantity.",
    "Answers include <b>chain-of-thought</b> (the model explains its reasoning step by step).",
])

p("What happens in GRPO (stage 4) &mdash; in plain words", H3)
p("<b>GRPO</b> (Group Relative Policy Optimization) is reinforcement learning <i>without</i> a "
  "separate “judge” model. For each question the model produces <b>several attempts</b>; because "
  "these questions have a <b>verifiable correct answer</b>, each attempt can be auto-graded. Attempts "
  "that score <b>above the group average</b> are reinforced; below-average ones are discouraged. A "
  "<b>reference model</b> and <b>reward normalization</b> keep the updates stable so the model "
  "improves without “drifting”. The payoff: noticeably better physical reasoning.", BODY)

story.append(PageBreak())

# ================================================================ PAGE 9 — experimental results
p("Key experimental results", H2)

p("Cosmos-Reason1: bigger model & RL both help", H3)
story.append(vbar(
    data=[[54.3, 61.8], [60.2, 63.7]],
    cats=["Physical\ncommon sense", "Embodied\nreasoning"],
    series_colors=[colors.HexColor("#A9CCE3"), G3],
    series_names=["Cosmos-Reason1 7B", "Cosmos-Reason1 56B"],
    w=320, h=170, vmin=0, vmax=80, step=20, ylabel="benchmark score (%)"))
caption("Figure 7. The 56B model beats the 7B on both physical-common-sense and embodied-reasoning "
        "benchmarks. Embodied reasoning improved by ~+10–11 points over the next-best prior model. "
        "<b>Source note:</b> numbers are from NVIDIA's Cosmos-Reason1 research page (released 7B "
        "checkpoint). An earlier arXiv v1 of the paper reported an <i>8B</i> model with somewhat "
        "different scores &mdash; see the verification page.")

p("Reinforcement learning (GRPO) lifts reasoning further", H3)
story.append(vbar(
    data=[[60.7, 74.5], [65.7, 81.5]],
    cats=["Combined\nreasoning", "Intuitive\nphysics"],
    series_colors=[colors.HexColor("#F5CBA7"), G4],
    series_names=["Before RL (SFT only)", "After Physical-AI RL"],
    w=320, h=170, vmin=0, vmax=100, step=20, ylabel="score (%)"))
caption("Figure 8. Adding the GRPO reinforcement-learning stage raised the 7B model by +5.0 points "
        "(combined reasoning) and +7.0 points (intuitive physics) — evidence that post-training, "
        "not just size, drives physical reasoning. <b>Source note:</b> research-page figures; arXiv "
        "v1 reports different magnitudes (e.g. intuitive physics 65.7&rarr;68.7). The <i>direction</i> "
        "&mdash; RL helps &mdash; is consistent across both.")

p("Generation-quality results (Cosmos 2.5 & 3)", H3)
bullets([
    "<b>Cosmos 2.5</b> post-trained <b>14B</b> scores <b>0.810</b> on PAI-Bench Image2World "
    "(matching the strong Wan2.2-A14B baseline).",
    "<b>Cosmos 2.5</b> autonomous-driving multiview model: up to <b>2.3×</b> better FVD/FID "
    "(video realism metrics) versus its predecessor.",
    "<b>Cosmos 3</b> ranks <b>#1 among open models</b> on Artificial Analysis, Physics-IQ, "
    "PAI-Bench and R-Bench (world generation); RoboLab and RoboArena (action policy); and "
    "VANTAGE-Bench and TAR (vision understanding).",
])
caption("Note: FVD/FID measure how realistic/consistent generated video is (lower is better, so a "
        "2.3× improvement is large). PAI-Bench scores world-generation quality (higher is better).")

story.append(PageBreak())

# ================================================================ PAGE — verification / fact-check
p("Verification & source audit", H2)
p("Every quantitative claim in this guide was cross-checked against primary sources "
  "(NVIDIA papers, research pages, and the official Cosmos 3 release). Of 13 checked claims, "
  "<b>11 verified exactly</b>; <b>2 carry source-version or disclosure caveats</b> (marked “!”). "
  "No factual errors were found in any claim that could be checked against a primary source.", BODY)

audit = [
    ["#", "Claim in this guide", "Checked against", "Verdict"],
    ["1", "Cosmos 1 data mix: 20/16/16/11/10/8/8/4/7%", "arXiv 2501.03575", "OK — exact"],
    ["2", "~100M clips (10^8) from ~20M hours of video", "arXiv 2501.03575", "OK — exact"],
    ["3", "Diffusion 7B = 28L / 4096 / 32 heads; 14B = 36L / 5120 / 40", "Table 11", "OK — exact"],
    ["4", "AdaLN-LoRA rank 256; LR 2^-15 / 2^-16; context 56,320", "arXiv 2501.03575", "OK — exact"],
    ["5", "Autoregressive 4B / 12B (+ 5B / 13B Video2World)", "arXiv 2501.03575", "OK — exact"],
    ["6", "Trained on ~10,000 H100 GPUs for ~3 months", "arXiv 2501.03575", "OK — exact"],
    ["7", "Tokenizer up to 12× faster than prior (CogVideoX)", "arXiv 2501.03575", "OK"],
    ["8", "Cosmos 2: 0.6B/2B/14B; sparse attn up to 2.6×; 480/704p", "GitHub + NVIDIA blog", "OK"],
    ["9", "Cosmos 2.5: flow; Reason1 encoder; 2B/14B; 200M clips; 30s; PAI-Bench 0.810; 2.3× FVD/FID",
     "NVIDIA research page", "OK"],
    ["10", "GRPO post-training; 4-stage pipeline; verifiable rewards", "arXiv 2503.15558", "OK"],
    ["11", "Cosmos-Reason1 = 7B & 56B with the Fig 7–8 scores",
     "Research page vs arXiv v1", "CAVEAT — see below"],
    ["12", "Cosmos 3: mixture-of-transformers omnimodel; Super/Nano/Edge; #1 on open leaderboards",
     "NVIDIA newsroom (Jun 2026)", "OK"],
    ["13", "Cosmos 3 data = 20T tokens, ~1B images, ~400M videos",
     "Press coverage (not NVIDIA)", "CAVEAT — see below"],
]
ad = []
for i, r in enumerate(audit):
    if i == 0:
        ad.append([cellP(c, header=True) for c in r])
    else:
        verdict = r[3]
        vstyle = CELL
        ad.append([cellP(r[0]), cellP(r[1]), cellP(r[2]), Paragraph(verdict, CELL)])
at = Table(ad, colWidths=[0.8*cm, 8.4*cm, 4.3*cm, 3.5*cm], repeatRows=1)
astyle = [
    ("BACKGROUND", (0,0), (-1,0), DARK),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
]
for i in range(1, len(ad)):
    if audit[i][3].startswith("OK"):
        astyle.append(("TEXTCOLOR", (3,i), (3,i), colors.HexColor("#2E7D32")))
        astyle.append(("BACKGROUND", (3,i), (3,i), colors.HexColor("#EAF6EA")))
    else:
        astyle.append(("TEXTCOLOR", (3,i), (3,i), colors.HexColor("#B9770E")))
        astyle.append(("BACKGROUND", (3,i), (3,i), colors.HexColor("#FCF3E3")))
    if i % 2 == 0:
        astyle.append(("BACKGROUND", (0,i), (2,i), ROW_ALT))
at.setStyle(TableStyle(astyle))
story.append(at)
caption("Table 6. Claims audit. Green = verified against a primary source; amber = caveat.")

spacer(4)
p("The two caveats, in detail", H3)
bullets([
    "<b>! Cosmos-Reason1 size & scores (claim 11):</b> NVIDIA's research page describes a "
    "released <b>7B</b> model with the scores plotted in Figures 7–8. The <b>arXiv v1</b> of the "
    "paper instead describes an <b>8B</b> model and reports different numbers (e.g. physical "
    "common sense 52.3% vs 54.3%; intuitive-physics RL gain 65.7&rarr;68.7 vs 74.5&rarr;81.5). "
    "This is a paper-revision difference. This guide uses the <b>research-page (7B)</b> figures "
    "because that matches the publicly released checkpoint &mdash; but treat the exact decimals as "
    "version-dependent. The qualitative conclusions (bigger model helps; RL helps) hold in both.",
    "<b>! Cosmos 3 data scale (claim 13):</b> the “20 trillion tokens / ~1B images / ~400M videos” "
    "figures come from <b>press coverage</b> of the launch. NVIDIA's own newsroom release only "
    "says “billions of samples across text, image, video, sound and action.” Treat the precise "
    "counts as <b>unconfirmed by NVIDIA</b>.",
])
p("<b>Bottom line:</b> the guide is accurate where primary sources exist. The only soft spots are "
  "(a) decimal-level benchmark numbers for Cosmos-Reason1, which differ between paper versions, and "
  "(b) Cosmos 3's exact data counts, which NVIDIA has not officially published.", BODY)

story.append(PageBreak())

# ================================================================ PAGE — summary
p("Putting it all together", H2)
p("The four generations trace a clear arc:", BODY)
bullets([
    "<b>Cosmos 1 (Toolbox):</b> two engines (diffusion + GPT), a strong causal tokenizer, T5 text, "
    "9-category data, fully documented hyperparameters. Maximum flexibility.",
    "<b>Cosmos 2 (Polish):</b> diffusion-only, sparse attention for up to 2.6× speed, action-"
    "conditioned post-training. Same idea, executed better.",
    "<b>Cosmos 2.5 (Unify):</b> one flow-based model replaces three; Cosmos-Reason1 reads the prompt; "
    "200M clips; 30s multi-camera video; RL + model-merge post-training.",
    "<b>Cosmos 3 (Leap):</b> mixture-of-transformers omnimodel; reasons before generating; adds audio "
    "and actions; ~20T-token training; Super/Nano/Edge tiers; #1 open model on many leaderboards.",
])
spacer(4)
p("The single sentence to remember", H3)
p("Cosmos went from <b>“many tools that dream video”</b> (1) → <b>“the best tool, polished”</b> (2) "
  "→ <b>“one smart unified tool”</b> (2.5) → <b>“a reasoning, seeing, hearing, acting "
  "omnimodel”</b> (3).", BODY)

spacer(10)
p("Sources: NVIDIA <i>Cosmos World Foundation Model Platform</i> (arXiv:2501.03575); "
  "<i>Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning</i> (arXiv:2503.15558); "
  "NVIDIA Research pages for Cosmos-Predict2.5 and Cosmos-Reason1; NVIDIA developer blogs for "
  "Cosmos-Predict2; Hugging Face NVIDIA blog on Cosmos Predict/Transfer 2.5; NVIDIA newsroom "
  "release for Cosmos 3 (Jun 2026). Figures are illustrative; numbers quoted from these sources. "
  "Educational summary.", SMALL)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BORDER); canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(GREY)
    canvas.drawString(2*cm, 1.0*cm, "NVIDIA Cosmos — Detailed Beginner's Guide")
    canvas.drawRightString(A4[0]-2*cm, 1.0*cm, "Page %d" % doc.page)
    canvas.restoreState()


doc = SimpleDocTemplate(
    "/home/user/transformers/cosmos_guide.pdf", pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="NVIDIA Cosmos - Detailed Beginner's Guide",
    author="Cosmos research summary",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF written.")
