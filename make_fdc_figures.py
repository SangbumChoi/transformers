#!/usr/bin/env python3
"""Render report-quality figures for the OLED 8-chamber FDC dashboard,
from the node-generated JSON (so they match the dashboard exactly)."""
import json, math, os
from PIL import Image, ImageDraw, ImageFont

D = json.load(open("oled_assets/fdc.json"))
N, MON, ref, K, TH = D["N"], D["MON"], D["ref"], D["K"], D["TH"]
CH = D["chambers"]
os.makedirs("oled_assets", exist_ok=True)

WHITE=(255,255,255); INK=(26,28,32); MUT=(120,128,138); GRID=(228,231,235)
TEAL=(28,150,148); AMBER=(200,140,30); RED=(210,59,48); BLUE=(60,120,190)
STC={"N":(46,158,79),"W":(201,146,31),"A":(210,59,48)}
LAB={"N":"NORMAL","W":"WARN","A":"ANOMALY"}

def f(sz,bold=False):
    p="/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"%("-Bold" if bold else "")
    try: return ImageFont.truetype(p,sz)
    except: return ImageFont.load_default(size=sz)

def line(dr,x,y,w,h,arr,col,lw=1):
    mn=min(arr); mx=max(arr); rng=(mx-mn) or 1
    pts=[(x+i/(len(arr)-1)*w, y+h-(arr[i]-mn)/rng*h) for i in range(len(arr))]
    dr.line(pts,fill=col,width=lw); return mn,mx

# ---------------- Figure 1: fleet overview ----------------
def fleet():
    cols,rows=4,2; cw,ch=372,232; pad=18; top=70
    W=cols*cw+(cols+1)*pad; H=top+rows*ch+(rows+1)*pad
    im=Image.new("RGB",(W,H),WHITE); d=ImageDraw.Draw(im,"RGBA")
    cnt={"N":0,"W":0,"A":0}
    for c in CH: cnt[c["status"]]+=1
    d.text((pad,18),"OLED 8-Chamber FDC — Anomaly Detection (fleet overview)",fill=INK,font=f(22,1))
    d.text((pad,46),f"NORMAL {cnt['N']}   WARN {cnt['W']}   ANOMALY {cnt['A']}   / 8 chambers   ·   "
           f"score = sum of per-sensor {K:g}sigma exceedances,  threshold {TH:g}",fill=MUT,font=f(13))
    for idx,c in enumerate(CH):
        cx=pad+(idx%cols)*(cw+pad); cy=top+pad+(idx//cols)*(ch+pad)
        st=c["status"]; sc=STC[st]
        d.rounded_rectangle([cx,cy,cx+cw,cy+ch],10,fill=(250,251,252),outline=GRID,width=1)
        d.rectangle([cx,cy+1,cx+5,cy+ch-1],fill=sc)
        d.text((cx+16,cy+10),c["name"],fill=INK,font=f(18,1))
        d.text((cx+cw-118,cy+12),LAB[st],fill=sc,font=f(13,1))
        # chart
        gx,gy,gw,gh=cx+14,cy+44,cw-28,ch-96
        d.rectangle([gx,gy,gx+gw,gy+gh],fill=(255,255,255),outline=GRID)
        score=c["score"]; mx=max(TH*2,max(score))*1.05
        px=lambda i:gx+i/(N-1)*gw; py=lambda v:gy+gh-v/mx*gh
        for i in range(N):
            if score[i]>TH: d.rectangle([px(i),gy,px(i)+gw/N+0.7,gy+gh],fill=(210,59,48,38))
        d.line([gx,py(TH),gx+gw,py(TH)],fill=(210,59,48,200),width=1)  # threshold
        d.text((gx+4,gy+2),"anomaly score",fill=MUT,font=f(10))
        d.line([(px(i),py(score[i])) for i in range(N)],fill=AMBER,width=1)
        for i in range(N):
            if score[i]>TH: d.ellipse([px(i)-1.6,py(score[i])-1.6,px(i)+1.6,py(score[i])+1.6],fill=RED)
        d.text((cx+16,cy+ch-38),f"cause: ",fill=MUT,font=f(13))
        d.text((cx+70,cy+ch-38),c["dom"],fill=sc,font=f(13,1))
        d.text((cx+cw-150,cy+ch-38),f"anomalous {c['nA']} / {N} runs",fill=MUT,font=f(12))
        d.text((cx+16,cy+ch-20),c["desc"],fill=MUT,font=f(11))
    im.save("oled_assets/fig_fdc_fleet.png"); print("fleet",im.size)

# ---------------- Figure 2: per-chamber root cause ----------------
def detail(name,fname):
    c=[x for x in CH if x["name"]==name][0]; S=c["S"]; sc=STC[c["status"]]
    cols,rows=4,2; cw,ch=350,150; pad=16; top=64
    W=cols*cw+(cols+1)*pad; H=top+rows*ch+(rows+1)*pad
    im=Image.new("RGB",(W,H),WHITE); d=ImageDraw.Draw(im,"RGBA")
    d.text((pad,16),f"{name} — root-cause breakdown  ·  diagnosis: ",fill=INK,font=f(20,1))
    wlab=d.textlength(f"{name} — root-cause breakdown  ·  diagnosis: ",font=f(20,1))
    d.text((pad+wlab,16),c["dom"],fill=sc,font=f(20,1))
    d.text((pad,44),f"{LAB[c['status']]}  ·  {c['desc']}  ·  red points = beyond {K:g}sigma of healthy baseline (CH1)",fill=MUT,font=f(13))
    for k,(nm,sh,dir_) in enumerate(MON):
        cx=pad+(k%cols)*(cw+pad); cy=top+pad+(k//cols)*(ch+pad)
        a=S[nm]; mu=ref[nm]["mu"]; sg=ref[nm]["sg"]; UCL=mu+K*sg; LCL=mu-K*sg
        d.rounded_rectangle([cx,cy,cx+cw,cy+ch],8,fill=(250,251,252),outline=GRID)
        z=(a[-1]-mu)/sg; flagged = (abs(z)>K) if dir_==0 else (dir_*z>K)
        d.text((cx+10,cy+7),nm,fill=INK,font=f(13,1))
        d.text((cx+cw-66,cy+8),f"z={z:+.1f}",fill=(RED if flagged else MUT),font=f(12,1))
        gx,gy,gw,gh=cx+10,cy+30,cw-20,ch-44
        lo=min(LCL,min(a)); hi=max(UCL,max(a)); pad2=(hi-lo)*0.08; lo-=pad2; hi+=pad2; rng=hi-lo
        px=lambda i:gx+i/(N-1)*gw; py=lambda v:gy+gh-(v-lo)/rng*gh
        d.rectangle([gx,gy,gx+gw,gy+gh],fill=(255,255,255),outline=GRID)
        for v in (UCL,LCL): d.line([gx,py(v),gx+gw,py(v)],fill=(210,59,48,150),width=1)
        d.line([gx,py(mu),gx+gw,py(mu)],fill=(28,150,148,160),width=1)
        d.line([(px(i),py(a[i])) for i in range(N)],fill=BLUE,width=1)
        for i in range(N):
            zz=(a[i]-mu)/sg; bad=(abs(zz)>K) if dir_==0 else (dir_*zz>K)
            if bad: d.ellipse([px(i)-1.7,py(a[i])-1.7,px(i)+1.7,py(a[i])+1.7],fill=RED)
    # legend cell
    cx=pad+3*(cw+pad); cy=top+pad+1*(ch+pad)
    d.text((cx+10,cy+14),"Legend",fill=INK,font=f(14,1))
    d.line([cx+10,cy+44,cx+44,cy+44],fill=BLUE,width=2); d.text((cx+52,cy+38),"sensor signal",fill=MUT,font=f(12))
    d.line([cx+10,cy+66,cx+44,cy+66],fill=(210,59,48,200),width=1); d.text((cx+52,cy+60),"UCL / LCL (3sigma)",fill=MUT,font=f(12))
    d.line([cx+10,cy+88,cx+44,cy+88],fill=(28,150,148,200),width=1); d.text((cx+52,cy+82),"baseline mean (CH1)",fill=MUT,font=f(12))
    d.ellipse([cx+22,cy+106,cx+30,cy+114],fill=RED); d.text((cx+52,cy+104),"> 3sigma exceedance",fill=MUT,font=f(12))
    im.save("oled_assets/"+fname); print(fname,im.size)

# ---------------- Figure 3: method schematic ----------------
def method():
    W,H=1500,250; im=Image.new("RGB",(W,H),WHITE); d=ImageDraw.Draw(im,"RGBA")
    d.text((24,18),"Anomaly-detection method",fill=INK,font=f(20,1))
    boxes=[(40,"7 equipment / process\nsensors per chamber",TEAL),
           (330,"z-score vs healthy\nbaseline (CH1 mean, sigma)",BLUE),
           (650,"sum of per-sensor\n3 sigma exceedances\n= anomaly score",AMBER),
           (980,"score > threshold (0.8)\n-> anomalous run",RED),
           (1280,"status + dominant\nsensor = root cause",(120,80,170))]
    bw=230
    for i,(x,t,col) in enumerate(boxes):
        d.rounded_rectangle([x,90,x+bw,170],10,fill=(250,251,252),outline=col,width=2)
        lines=t.split("\n")
        for j,ln in enumerate(lines):
            d.text((x+bw/2,118+ (j-(len(lines)-1)/2)*16),ln,fill=INK,font=f(12,1),anchor="mm")
        if i<len(boxes)-1:
            ax=x+bw+8; bx=boxes[i+1][0]-8
            d.line([ax,130,bx,130],fill=MUT,width=2)
            d.polygon([(bx,130),(bx-8,126),(bx-8,134)],fill=MUT)
    d.text((24,200),"Healthy baseline from the reference chamber (CH1); each sensor's 3-sigma control limits flag exceedances; "
           "the summed multivariate score detects faults and the largest contributor names the root cause.",fill=MUT,font=f(12))
    im.save("oled_assets/fig_fdc_method.png"); print("method",im.size)

fleet(); detail("CH3","fig_fdc_detail_ch3.png"); detail("CH8","fig_fdc_detail_ch8.png"); method()
