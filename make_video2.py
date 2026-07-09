#!/usr/bin/env python3
"""Build the narrated Molmo 2 explainer: edge-tts voiceover -> scene durations
-> inject into HTML -> voice track -> record 1280x720 -> mux mp4."""
import asyncio, os, re, wave, struct, math, random, glob, time, array, subprocess
import imageio_ffmpeg
os.environ["SSL_CERT_FILE"]="/root/.ccr/ca-bundle.crt"
os.environ["REQUESTS_CA_BUNDLE"]="/root/.ccr/ca-bundle.crt"
PROXY=os.environ.get("HTTPS_PROXY")
FF=imageio_ffmpeg.get_ffmpeg_exe()
VOICE="en-US-AriaNeural"; SR=32000

NARR=[
 "This is Molmo 2, from the Allen Institute for A I. It teaches a vision model to point — now, in video.",
 "Most vision-language models can describe an image. But describing is not the same as knowing where things are.",
 "Molmo, released in twenty twenty-four, was the first fully-open model that points directly at pixels. A point lets it count, refer, and ground language to the image.",
 "Its real advantage wasn't the model — it was the data. Most open models quietly distill a closed model. Molmo refused, and built its own dataset, called PixMo.",
 "For PixMo-Cap, annotators describe each image out loud, for sixty to ninety seconds. Speech gives dense, natural captions that are hard to fake. No typing, no G P T.",
 "For PixMo-Points, humans simply click on every instance. Points are cheap to annotate, so they collected a huge, real grounding dataset.",
 "PixMo is a whole family — captions, question answering, pointing, counting, documents, even reading clocks. All of it, without distillation.",
 "Training is simple. First, caption pre-training teaches the model to describe. Then, supervised fine-tuning on the PixMo mixture teaches pointing, counting, and more. No reinforcement learning, no distillation.",
 "Molmo 2, in twenty twenty-five, brings this to video. It points and tracks objects across frames, and grounds events in time — where, and when, the action happens.",
 "Under the hood: a SigLIP 2 vision encoder, a connector, and a Qwen 3 or OLMo language model, producing text, points, and tracks.",
 "The payoff: an eight-billion parameter model beats last year's seventy-two billion, and tops open models on video tracking.",
 "The frontier is MolmoPoint. Instead of writing coordinates as text, it points by selecting visual tokens directly, coarse to fine — tightly coupling recognition and localization.",
 "Everything is open — the weights, the data, and the code. Next, in twenty twenty-six: one model that reasons across every modality, plus robotics.",
 "Molmo 2, from Ai2. Point at what matters.",
]

async def tts(text, path):
    c=edge_tts.Communicate(text, VOICE, proxy=PROXY, rate="-4%")
    await c.save(path)
import edge_tts

def wav_samples(path):
    with wave.open(path) as w:
        n=w.getnframes(); data=w.readframes(n)
    a=array.array('h'); a.frombytes(data)
    return [v/32768.0 for v in a]

print("generating voiceover ...")
clips=[]
for i,tx in enumerate(NARR):
    mp3=f"/tmp/n{i}.mp3"; wv=f"/tmp/n{i}.wav"
    asyncio.run(tts(tx, mp3))
    subprocess.run([FF,"-y","-i",mp3,"-ac","1","-ar",str(SR),wv],capture_output=True)
    s=wav_samples(wv); clips.append(s)
    print(f"  scene {i+1}: {len(s)/SR:5.1f}s")

DUR=[round(max(5.0, len(s)/SR + 1.6), 1) for s in clips]     # scene lasts narration + 1.6s tail
TOTAL=sum(DUR); starts=[sum(DUR[:i]) for i in range(len(DUR))]
print("scene durations:", DUR, " total", round(TOTAL,1),"s")

# ---- inject DUR into the HTML (deliverable keeps fixed durations) ----
html=open("molmo2_explainer.html",encoding="utf-8").read()
html=re.sub(r'const DUR=\(typeof __DUR__[^;]*;|const DUR=\[[^\]]*\];',
            f'const DUR={DUR};', html, count=1)
open("molmo2_explainer.html","w",encoding="utf-8").write(html)

# ---- build voice track (+ very soft pad, subtle transition whoosh) ----
nS=int(TOTAL*SR); buf=[0.0]*nS
roots=[45,45,48,41,43,45,41,48,45,50,43,48,41,45]
def midi(m): return 440.0*2**((m-69)/12.0)
for si,(st,du) in enumerate(zip(starts,DUR)):
    a=int(st*SR); b=int((st+du)*SR); f=midi(roots[si])
    for i in range(a,b):
        t=(i-a)/SR; env=min(1,t/1.0)*min(1,(du-t)/1.0)
        buf[i]+=0.040*env*(math.sin(2*math.pi*f*t)+0.6*math.sin(2*math.pi*f*1.5*t))/1.6   # quiet warm pad
for si,s in enumerate(clips):
    off=int((starts[si]+0.35)*SR)
    for k,v in enumerate(s):
        j=off+k
        if j<nS: buf[j]+=0.98*v
for s in starts[1:]:
    a=int(s*SR); prev=0
    for k in range(int(0.32*SR)):
        t=k/(0.32*SR); env=math.sin(math.pi*t)**2; w=random.uniform(-1,1); prev=0.85*prev+0.15*w
        j=a+k
        if 0<=j<nS: buf[j]+=0.05*env*prev
peak=max(1e-6,max(abs(v) for v in buf)); sc=0.96/peak
with wave.open("narration.wav","w") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    fr=bytearray()
    for v in buf:
        q=int(max(-1,min(1,v*sc))*32767); fr+=struct.pack("<hh",q,q)
    w.writeframes(bytes(fr))
print("wrote narration.wav")

# ---- record video 1280x720 ----
from playwright.sync_api import sync_playwright
os.makedirs("/tmp/vid3", exist_ok=True)
with sync_playwright() as pw:
    br=pw.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    ctx=br.new_context(viewport={"width":1280,"height":720}, record_video_dir="/tmp/vid3", record_video_size={"width":1280,"height":720})
    pg=ctx.new_page(); pg.goto("file:///home/user/transformers/molmo2_explainer.html?record=1")
    t0=time.time()
    while time.time()-t0 < TOTAL+2:
        if pg.evaluate("window.__finished"): time.sleep(0.3); break
        time.sleep(0.15)
    ctx.close(); br.close()
webm=sorted(glob.glob("/tmp/vid3/*.webm"), key=os.path.getmtime)[-1]

# ---- mux ----
out="molmo2_explainer.mp4"
subprocess.run([FF,"-y","-i",webm,"-i","narration.wav","-c:v","libx264","-pix_fmt","yuv420p",
  "-crf","20","-c:a","aac","-b:a","160k","-shortest","-movflags","+faststart",out],capture_output=True)
import shutil; shutil.copy(webm,"molmo2_explainer.webm")
print("wrote", out, os.path.getsize(out), "bytes")
