#!/usr/bin/env python3
"""Build the narrated Molmo 2 explainer (voice-only): edge-tts voiceover ->
scene durations -> inject into HTML -> record 1280x720 -> mux mp4."""
import asyncio, os, re, wave, struct, array, glob, time, subprocess
import imageio_ffmpeg, edge_tts
os.environ["SSL_CERT_FILE"]="/root/.ccr/ca-bundle.crt"
os.environ["REQUESTS_CA_BUNDLE"]="/root/.ccr/ca-bundle.crt"
PROXY=os.environ.get("HTTPS_PROXY")
FF=imageio_ffmpeg.get_ffmpeg_exe()
VOICE="en-US-AriaNeural"; SR=32000

NARR=[
 "This is Molmo 2, from the Allen Institute for A I. It teaches a vision model to point — now, in video.",
 "Most vision-language models can describe an image. But describing is not the same as knowing where things are.",
 "Molmo, from twenty twenty-four, was the first fully-open model that points directly at pixels. A point lets it count, refer, and ground language to the image.",
 "Its real edge is honesty about data. Most open models distill a closed model — training on G P T four vee looking at the image. Molmo refused: no vision-language model ever labels its images. The visual understanding comes from people.",
 "For PixMo-Cap, humans describe each image out loud. The speech is transcribed, and then a text-only language model tidies the transcript into a caption. That language model never sees the image.",
 "For PixMo-Points, humans simply click on every instance. Points are fast to annotate, so they built a huge, real grounding dataset.",
 "PixMo is a family. Some parts are pure human labels — captions, points, counts. Others are augmented by text-only language models, which generate question-answer pairs from the human captions, or synthetic documents. So it isn't free of A I — it's free of vision-model distillation.",
 "And an honest caveat: Molmo is not trained from scratch. It starts from a pretrained vision encoder — CLIP, then SigLIP two — and a pretrained language model, OLMo or Qwen. Only the connector is new. The novelty is the data and the recipe. Even openness is a spectrum: the OLMo variant is fully open; the Qwen-based ones use open-weight, but not open-data, backbones.",
 "Training itself is simple. First, caption pre-training teaches the model to describe. Then, supervised fine-tuning on the PixMo mixture teaches pointing, counting, and more. No reinforcement learning.",
 "Molmo 2, in twenty twenty-five, brings this to real video. Point at an object — say, a car — and it tracks that point and box through every frame, grounding the event in time. Where, and when, the action happens.",
 "Under the hood: a SigLIP two vision encoder, a connector, and a Qwen three or OLMo language model, producing text, points, and tracks.",
 "The payoff: an eight-billion parameter model beats last year's seventy-two billion, and tops open models on video tracking.",
 "Now, the frontier: MolmoPoint. Normally, a model writes a point as text coordinates. It has to learn a coordinate system, and it spends many tokens for every single point.",
 "MolmoPoint instead emits three special grounding tokens — patch, sub-patch, and location — that select visual tokens directly, coarse to fine. This ties recognition to localization, stays pixel-precise at any resolution, and extends to video. On PointBench it reaches seventy point seven; it gains about five points on G U I pointing; and it improves video tracking.",
 "Everything is open — the weights, the data, and the code. Next, in twenty twenty-six: one model that reasons across every modality, plus robotics.",
 "Molmo 2, from Ai2. Point at what matters.",
]

async def tts(text, path):
    await edge_tts.Communicate(text, VOICE, proxy=PROXY, rate="-4%").save(path)

def wav_samples(path):
    with wave.open(path) as w:
        data=w.readframes(w.getnframes())
    a=array.array('h'); a.frombytes(data); return [v/32768.0 for v in a]

print("generating voiceover ...")
clips=[]
for i,tx in enumerate(NARR):
    mp3=f"/tmp/m{i}.mp3"; wv=f"/tmp/m{i}.wav"
    asyncio.run(tts(tx, mp3))
    subprocess.run([FF,"-y","-i",mp3,"-ac","1","-ar",str(SR),wv],capture_output=True)
    s=wav_samples(wv); clips.append(s)
    print(f"  scene {i+1:2d}: {len(s)/SR:5.1f}s")

DUR=[round(max(5.0, len(s)/SR + 1.4), 1) for s in clips]
TOTAL=sum(DUR); starts=[sum(DUR[:i]) for i in range(len(DUR))]
print("durations:", DUR, " total", round(TOTAL,1),"s")

# inject durations into the HTML (interactive version stays in sync)
html=open("molmo2_explainer.html",encoding="utf-8").read()
html=re.sub(r'const DUR=\(typeof __DUR__[^;]*;|const DUR=\[[^\]]*\];',
            f'const DUR={DUR};', html, count=1)
open("molmo2_explainer.html","w",encoding="utf-8").write(html)

# voice-only track (no background music) — place each clip at its scene start
nS=int(TOTAL*SR); buf=[0.0]*nS
for si,s in enumerate(clips):
    off=int((starts[si]+0.35)*SR)
    for k,v in enumerate(s):
        j=off+k
        if j<nS: buf[j]+=v
peak=max(1e-6,max(abs(v) for v in buf)); sc=0.97/peak
with wave.open("narration.wav","w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    fr=bytearray()
    for v in buf: fr+=struct.pack("<h", int(max(-1,min(1,v*sc))*32767))
    w.writeframes(bytes(fr))
print("wrote narration.wav (voice only)")

# record 1280x720
from playwright.sync_api import sync_playwright
os.makedirs("/tmp/vid4", exist_ok=True)
with sync_playwright() as pw:
    br=pw.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    ctx=br.new_context(viewport={"width":1280,"height":720}, record_video_dir="/tmp/vid4", record_video_size={"width":1280,"height":720})
    pg=ctx.new_page(); pg.goto("file:///home/user/transformers/molmo2_explainer.html?record=1")
    t0=time.time()
    while time.time()-t0 < TOTAL+2:
        if pg.evaluate("window.__finished"): time.sleep(0.3); break
        time.sleep(0.15)
    ctx.close(); br.close()
webm=sorted(glob.glob("/tmp/vid4/*.webm"), key=os.path.getmtime)[-1]

out="molmo2_explainer.mp4"
subprocess.run([FF,"-y","-i",webm,"-i","narration.wav","-c:v","libx264","-pix_fmt","yuv420p",
  "-crf","20","-c:a","aac","-b:a","160k","-shortest","-movflags","+faststart",out],capture_output=True)
import shutil; shutil.copy(webm,"molmo2_explainer.webm")
print("wrote", out, os.path.getsize(out), "bytes")
