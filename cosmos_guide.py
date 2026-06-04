#!/usr/bin/env python3
"""Generate a beginner-friendly PDF explaining the differences across NVIDIA Cosmos generations."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

# ---------------------------------------------------------------- palette
NV_GREEN = colors.HexColor("#76B900")   # NVIDIA green
DARK = colors.HexColor("#1A1A1A")
GREY = colors.HexColor("#555555")
LIGHT_BG = colors.HexColor("#F2F7E9")
ROW_ALT = colors.HexColor("#F7F7F7")
BORDER = colors.HexColor("#CCCCCC")

# ---------------------------------------------------------------- styles
styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=22, textColor=DARK, spaceAfter=4, leading=26)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontName="Helvetica",
                     fontSize=11, textColor=GREY, spaceAfter=14, leading=15)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=15, textColor=NV_GREEN, spaceBefore=16, spaceAfter=6, leading=18)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Helvetica-Bold",
                    fontSize=12, textColor=DARK, spaceBefore=8, spaceAfter=3, leading=15)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName="Helvetica",
                      fontSize=10, textColor=DARK, spaceAfter=6, leading=15, alignment=TA_LEFT)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=3)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, textColor=GREY, leading=11)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELL_H = ParagraphStyle("CELL_H", parent=CELL, fontName="Helvetica-Bold",
                        textColor=colors.white, fontSize=8.5)
CELL_L = ParagraphStyle("CELL_L", parent=CELL, fontName="Helvetica-Bold", fontSize=8.5)
ANALOGY = ParagraphStyle("ANALOGY", parent=BODY, fontName="Helvetica-Oblique",
                         fontSize=9.5, textColor=GREY, leftIndent=10, leading=14)

story = []

def p(text, style=BODY):
    story.append(Paragraph(text, style))

def bullets(items, style=BULLET):
    for it in items:
        story.append(Paragraph(it, style, bulletText="•"))

def spacer(h=6):
    story.append(Spacer(1, h))

def rule():
    story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER,
                            spaceBefore=8, spaceAfter=8))

# ================================================================ TITLE
p("Understanding NVIDIA Cosmos", H1)
p("A beginner's guide to the four generations &mdash; Cosmos 1, 2, 2.5 and 3 &mdash; "
  "and what actually changes between them.", SUB)

# ================================================================ WHAT IS IT
p("First: what is a “World Foundation Model”?", H2)
p("You have probably heard of a <b>language</b> model (like the one writing this): it learns "
  "from text and predicts the next word. A <b>World Foundation Model (WFM)</b> does the same "
  "trick, but for the <i>physical world</i>. It watches enormous amounts of video and learns "
  "to predict <b>what happens next</b> &mdash; how a ball bounces, how a hand grips a cup, how a "
  "car moves through a street.", BODY)
p("NVIDIA <b>Cosmos</b> is a family of these models, built for <b>“Physical AI”</b> &mdash; "
  "robots and self-driving cars. Instead of testing a robot a million times in the real world "
  "(slow, expensive, dangerous), engineers let Cosmos <b>imagine</b> realistic video of what "
  "would happen, and train on that.", BODY)
story.append(Paragraph(
    "Analogy: think of Cosmos as a <b>“dream engine”</b> for machines. You describe a "
    "scene, and it dreams a physically-believable video of how that scene plays out. Each new "
    "generation dreams better, faster, and about more things (now even sound and physical actions).",
    ANALOGY))

# ================================================================ TIMELINE
spacer(4)
p("The story in one line", H2)
tl_data = [[
    Paragraph("<b>Cosmos 1</b><br/>Jan 2025", CELL_L),
    Paragraph("&rarr;", CELL),
    Paragraph("<b>Cosmos 2</b><br/>Jun 2025", CELL_L),
    Paragraph("&rarr;", CELL),
    Paragraph("<b>Cosmos 2.5</b><br/>Sep 2025", CELL_L),
    Paragraph("&rarr;", CELL),
    Paragraph("<b>Cosmos 3</b><br/>Jun 2026", CELL_L),
]]
tl = Table(tl_data, colWidths=[3.1*cm, 0.8*cm, 3.1*cm, 0.8*cm, 3.1*cm, 0.8*cm, 3.1*cm])
tl.setStyle(TableStyle([
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("BACKGROUND", (0,0), (0,0), LIGHT_BG),
    ("BACKGROUND", (2,0), (2,0), LIGHT_BG),
    ("BACKGROUND", (4,0), (4,0), LIGHT_BG),
    ("BACKGROUND", (6,0), (6,0), LIGHT_BG),
    ("BOX", (0,0), (0,0), 0.8, NV_GREEN),
    ("BOX", (2,0), (2,0), 0.8, NV_GREEN),
    ("BOX", (4,0), (4,0), 0.8, NV_GREEN),
    ("BOX", (6,0), (6,0), 0.8, NV_GREEN),
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
]))
story.append(tl)
spacer(4)
p("Plain-English summary of the journey: Cosmos started as a big toolbox with <b>two</b> "
  "different engines, then <b>simplified and unified</b> into one engine, then made that engine "
  "<b>smarter and longer</b>, and finally became an <b>all-in-one model that also reasons, hears, "
  "and acts</b>.", BODY)

# ================================================================ BIG TABLE
spacer(2)
p("Side-by-side comparison", H2)

def cellP(text, header=False, label=False):
    if header:
        return Paragraph(text, CELL_H)
    if label:
        return Paragraph(text, CELL_L)
    return Paragraph(text, CELL)

header = ["What", "Cosmos 1", "Cosmos 2", "Cosmos 2.5", "Cosmos 3"]
rows = [
    ["Released", "Jan 2025", "Jun 2025", "Sep 2025", "Jun 2026"],
    ["Main idea",
     "Toolbox with TWO engines (diffusion + GPT-style)",
     "Refined the diffusion engine; faster, cleaner",
     "ONE unified engine (flow-based)",
     "All-in-one “omnimodel” that reasons + generates + acts"],
    ["What goes in",
     "Text or a starting image/video",
     "Text or image/video",
     "Text, image, or video (one model)",
     "Text, image, video, sound, AND actions"],
    ["What comes out",
     "Future video",
     "Future video (480p/704p)",
     "Future video up to 30s, multi-camera",
     "Video, actions, audio — with reasoning"],
    ["How it reads text",
     "T5 text encoder",
     "T5-style encoder",
     "Cosmos-Reason1 (a smart vision-language model)",
     "Built-in reasoning transformer"],
    ["Sizes offered",
     "7B/14B (diffusion), 4B/12B (GPT)",
     "2B and 14B",
     "2B and 14B",
     "Super / Nano / Edge"],
    ["One-word vibe",
     "Kitchen sink",
     "Polish",
     "Unify",
     "Leap"],
]

table_data = [[cellP(h, header=True) for h in header]]
for r in rows:
    table_data.append([cellP(r[0], label=True)] + [cellP(c) for c in r[1:]])

col_w = [3.0*cm, 3.4*cm, 3.0*cm, 3.5*cm, 3.6*cm]
tbl = Table(table_data, colWidths=col_w, repeatRows=1)
ts = [
    ("BACKGROUND", (0,0), (-1,0), NV_GREEN),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("GRID", (0,0), (-1,-1), 0.5, BORDER),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("BACKGROUND", (0,1), (0,-1), LIGHT_BG),
]
for i in range(1, len(table_data)):
    if i % 2 == 0:
        ts.append(("BACKGROUND", (1,i), (-1,i), ROW_ALT))
tbl.setStyle(TableStyle(ts))
story.append(tbl)

# ================================================================ PER GENERATION
spacer(6)
p("Now, generation by generation", H2)

# --- Cosmos 1
story.append(KeepTogether([
    Paragraph("Cosmos 1 &mdash; “The Toolbox” (Jan 2025)", H3),
    Paragraph("This was the first release, and it shipped <b>two completely different ways</b> to "
              "dream up video:", BODY),
]))
bullets([
    "<b>Diffusion engine</b> (sizes 7B &amp; 14B): starts from random noise and gradually "
    "“cleans it up” into a video. Slower, but the picture quality is excellent.",
    "<b>Autoregressive / GPT engine</b> (sizes 4B &amp; 12B): predicts the video one chunk at a "
    "time, like typing one word after another. Faster, aimed at real-time, but rougher &mdash; so "
    "it needs a clean-up step afterwards.",
])
p("It also split tasks into <b>separate models</b>: one for text&rarr;video, one for image&rarr;video, "
  "one for video&rarr;video. Powerful, but a lot of moving parts.", BODY)
story.append(Paragraph(
    "Beginner takeaway: Cosmos 1 gave you many tools and asked you to pick the right one. "
    "Great flexibility, but you had to know what you were doing.", ANALOGY))

# --- Cosmos 2
story.append(KeepTogether([
    Paragraph("Cosmos 2 &mdash; “The Polish” (Jun 2025)", H3),
    Paragraph("Instead of adding more, Cosmos 2 <b>focused</b>. It doubled down on the diffusion "
              "engine and made it <b>faster, sharper, and more obedient</b> to your prompt &mdash; "
              "with fewer “hallucinations” (made-up nonsense). It came in a small <b>2B</b> "
              "(fast) and a big <b>14B</b> (high quality) version, at 480p/704p resolution.", BODY),
]))
story.append(Paragraph(
    "Beginner takeaway: same idea as Cosmos 1, just done much better and with fewer mistakes.",
    ANALOGY))

# --- Cosmos 2.5
story.append(KeepTogether([
    Paragraph("Cosmos 2.5 &mdash; “The Unification” (Sep 2025)", H3),
    Paragraph("The big simplification. The three separate task-models from before were <b>merged "
              "into one</b> model that accepts text, image, OR video. It switched to a <b>flow-based</b> "
              "method (a cleaner cousin of diffusion) and, importantly, started using "
              "<b>Cosmos-Reason1</b> &mdash; a model that actually <i>understands</i> language and "
              "the physical world &mdash; to read your prompt. Results: video up to <b>30 seconds</b> "
              "long and <b>multiple synchronized camera angles</b>.", BODY),
]))
story.append(Paragraph(
    "Beginner takeaway: one smart model instead of three, with a much better “ear” for "
    "what you asked, and longer videos.", ANALOGY))

# --- Cosmos 3
story.append(KeepTogether([
    Paragraph("Cosmos 3 &mdash; “The Leap” (Jun 2026)", H3),
    Paragraph("Not just a better video maker &mdash; a different kind of model. Cosmos 3 is an "
              "<b>“omnimodel”</b>: one system that natively handles <b>text, images, video, "
              "sound, and physical actions</b>. It uses a <b>mixture-of-transformers</b> design that "
              "pairs a <b>reasoning</b> part (thinks about how objects interact) with a "
              "<b>generation</b> part (creates the video and the action). In other words, it "
              "<b>thinks before it dreams</b>. It comes as <b>Super</b> (max accuracy), <b>Nano</b> "
              "(answers in a split second), and <b>Edge</b> (runs locally, coming soon).", BODY),
]))
story.append(Paragraph(
    "Beginner takeaway: earlier versions were “dream engines.” Cosmos 3 adds a brain and "
    "senses &mdash; it reasons, sees, hears, and can decide what a robot should physically do.",
    ANALOGY))

# ================================================================ THE ONE THING
spacer(8)
rule()
p("If you remember only one thing…", H3)
bullets([
    "<b>Cosmos 1</b> = many tools, two different engines. (<i>Flexible but complex.</i>)",
    "<b>Cosmos 2</b> = the best engine, polished. (<i>Better &amp; fewer mistakes.</i>)",
    "<b>Cosmos 2.5</b> = one unified, smarter engine, longer video. (<i>Simpler &amp; sharper.</i>)",
    "<b>Cosmos 3</b> = an all-in-one model that reasons, sees, hears, and acts. (<i>A new category.</i>)",
])

spacer(10)
p("Sources: NVIDIA Cosmos World Foundation Model Platform paper (arXiv:2501.03575); "
  "NVIDIA newsroom &amp; developer blogs on Cosmos Predict-2, Predict-2.5, and Cosmos 3; "
  "Hugging Face NVIDIA blog on Cosmos Predict/Transfer 2.5. Generated as an educational summary.",
  SMALL)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2*cm, 1.0*cm, "NVIDIA Cosmos — Beginner's Guide")
    canvas.drawRightString(A4[0]-2*cm, 1.0*cm, "Page %d" % doc.page)
    canvas.restoreState()


doc = SimpleDocTemplate(
    "/home/user/transformers/cosmos_guide.pdf", pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="Understanding NVIDIA Cosmos - A Beginner's Guide",
    author="Cosmos research summary",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF written.")
