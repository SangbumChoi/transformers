import os, glob, time, subprocess
from playwright.sync_api import sync_playwright
os.makedirs("/tmp/vid", exist_ok=True)
url="file:///home/user/transformers/molmo2_explainer.html?record=1"
with sync_playwright() as pw:
    br=pw.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    ctx=br.new_context(viewport={"width":960,"height":540},
                       record_video_dir="/tmp/vid", record_video_size={"width":960,"height":540})
    pg=ctx.new_page(); pg.goto(url)
    total=pg.evaluate("window.__total"); print("total(s)=",round(total,1))
    shots=[2.5,11,20,30,45,58,68,76]
    start=time.time(); i=0
    # play once; poll for finish
    deadline=total+2.0
    while time.time()-start < deadline:
        el=time.time()-start
        if i<len(shots) and el>=shots[i]:
            pg.screenshot(path=f"/tmp/shot_{i}.png"); i+=1
        if pg.evaluate("window.__finished"): 
            time.sleep(0.3); break
        time.sleep(0.1)
    ctx.close(); br.close()
vids=sorted(glob.glob("/tmp/vid/*.webm"), key=os.path.getmtime)
print("video:", vids[-1] if vids else "NONE", os.path.getsize(vids[-1]) if vids else 0, "bytes")
# convert to mp4
ff="/opt/pw-browsers/ffmpeg-1011/ffmpeg-linux"
if vids and os.path.exists(ff):
    out="/home/user/transformers/molmo2_explainer.mp4"
    r=subprocess.run([ff,"-y","-i",vids[-1],"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",out],
                     capture_output=True, text=True)
    print("mp4:", out, os.path.getsize(out) if os.path.exists(out) else "FAIL")
    if not os.path.exists(out): print(r.stderr[-500:])
    # also keep the webm
    import shutil; shutil.copy(vids[-1], "/home/user/transformers/molmo2_explainer.webm")
