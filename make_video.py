import os, glob, time, subprocess, imageio_ffmpeg
from playwright.sync_api import sync_playwright
os.makedirs("/tmp/vid2", exist_ok=True)
url="file:///home/user/transformers/molmo2_explainer.html?record=1"
with sync_playwright() as pw:
    br=pw.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    ctx=br.new_context(viewport={"width":960,"height":540}, record_video_dir="/tmp/vid2", record_video_size={"width":960,"height":540})
    pg=ctx.new_page(); pg.goto(url)
    total=pg.evaluate("window.__total"); print("total(s)=",round(total,1))
    start=time.time()
    while time.time()-start < total+2:
        if pg.evaluate("window.__finished"): time.sleep(0.3); break
        time.sleep(0.15)
    ctx.close(); br.close()
webm=sorted(glob.glob("/tmp/vid2/*.webm"), key=os.path.getmtime)[-1]
print("webm:", os.path.getsize(webm), "bytes")
ff=imageio_ffmpeg.get_ffmpeg_exe()
out="molmo2_explainer.mp4"
r=subprocess.run([ff,"-y","-i",webm,"-i","soundtrack.wav",
  "-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-c:a","aac","-b:a","160k","-shortest","-movflags","+faststart",out],
  capture_output=True,text=True)
print("mp4:", os.path.getsize(out) if os.path.exists(out) else "FAIL")
if not os.path.exists(out): print(r.stderr[-600:])
import shutil; shutil.copy(webm,"molmo2_explainer.webm")
