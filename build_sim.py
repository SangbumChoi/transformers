#!/usr/bin/env python3
"""Build diffusion_sim_3d.html (single-file, zero-dependency) per the PRD.

Path B: generates SYNTHETIC measured PL frames + sigma curve that are
physically consistent with the fitted model, so the simulator is fully
functional offline. Replace the FRAMES / DATA blocks with the original
assets (Path A) for pixel-identical output.
"""
import math, io, base64, json, random
from PIL import Image

# ---- physics (same constants as PRD) ----
Di, Dt, TAU, SIG0, c2u, FOV, TMEAS = 0.28480452, 0.02061028, 1.14018661, 0.57, 0.1, 6.0, 58.0
Di_u, Dt_u = Di * c2u, Dt * c2u
def msd(t):   return 2*TAU*(Di_u-Dt_u)*(1-math.exp(-t/TAU)) + 2*Dt_u*t
def sigma(t): return math.sqrt(SIG0*SIG0 + msd(t))

def jet(v):
    v = max(0.0, min(1.0, v))
    r = max(0.0, min(1.0, min(4*v-1.5, -4*v+4.5)))
    g = max(0.0, min(1.0, min(4*v-0.5, -4*v+3.5)))
    b = max(0.0, min(1.0, min(4*v+0.5, -4*v+2.5)))
    return (int(r*255), int(g*255), int(b*255))

GATES = [0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0,4.8,5.6,6.4,7.2,8.0,8.8,9.6,
         10.4,11.2,12.0,14.0,16.0,18.0,20.0,22.0,24.0,26.0,28.0,30.0,34.0,38.0,
         42.0,46.0,50.0,54.0,58.0]
assert len(GATES) == 37, len(GATES)

N = 180
p2u = FOV / N  # um per px

def make_frame(t, seedoff):
    random.seed(1000 + seedoff)
    sp = sigma(t) / p2u  # sigma in px
    px = []
    for y in range(N):
        dy = y - N/2 + 0.5
        for x in range(N):
            dx = x - N/2 + 0.5
            v = math.exp(-(dx*dx + dy*dy) / (2*sp*sp))
            v += random.gauss(0, 0.028)            # photon-map-like noise
            px.append(jet(v))
    img = Image.new("RGB", (N, N))
    img.putdata(px)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

print("generating 37 synthetic PL frames ...")
FRAMES = {f"{t:.1f}": make_frame(t, i) for i, t in enumerate(GATES)}

random.seed(7)
DATA = []
for t in GATES:
    if t == 0.0:
        DATA.append([0.0, 0.57])
    else:
        DATA.append([round(t, 1), round(sigma(t) * random.uniform(0.990, 1.012), 4)])

frames_kb = sum(len(v) for v in FRAMES.values()) / 1024
print(f"frames base64 total ~ {frames_kb:.0f} KB")

# ---------------------------------------------------------------- HTML template
HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Charge-Carrier Diffusion Simulator — Pristine perovskite</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#2d333b;--txt:#e6edf3;--muted:#8b949e;--teal:#2bb3b0;--amber:#e9b872;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:13px;line-height:1.45}
header{padding:18px 22px 6px}
h1{font-size:18px;margin:0 0 6px}
header p{color:var(--muted);max-width:880px;margin:4px 0}
.wrap{display:flex;flex-wrap:wrap;gap:14px;padding:14px 22px 28px;align-items:flex-start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
#left{flex:0 0 auto}
#right{display:flex;flex-direction:column;gap:14px;flex:1 1 360px;min-width:340px;max-width:460px}
h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600;letter-spacing:.02em}
canvas{display:block;border-radius:8px}
#cv3d{background:#05070b;cursor:grab}
#cv3d:active{cursor:grabbing}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;margin-top:10px}
.expbox{position:relative;width:180px;height:180px}
#expimg{width:180px;height:180px;border-radius:8px;image-rendering:pixelated;background:#05070b}
#expov{display:none;position:absolute;inset:0;border-radius:8px;border:1px solid var(--amber);
 background:rgba(58,42,20,.32);align-items:center;justify-content:center;color:var(--amber);
 font-size:11px;font-weight:700;letter-spacing:.08em}
.cap{color:var(--muted);font-size:11px;margin-top:4px}
.controls{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,select{background:#1f2630;color:var(--txt);border:1px solid var(--line);border-radius:7px;
 padding:6px 11px;font-size:12px;cursor:pointer}
button:hover{border-color:var(--teal)}
input[type=range]{flex:1 1 220px;min-width:200px;accent-color:var(--teal)}
.opts{margin-top:10px;display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px}
.opts label{display:flex;gap:6px;align-items:center;cursor:pointer}
.legend{color:var(--muted);font-size:11px;margin-top:10px}
.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}
.tag.meas{background:#143b3a;color:var(--teal)}
.tag.extr{background:#3a2a14;color:var(--amber)}
table.state{width:100%;border-collapse:collapse}
table.state td{padding:5px 4px;border-bottom:1px solid var(--line)}
table.state td:first-child{color:var(--muted)}
table.state td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.params td:last-child{color:var(--teal)}
.muted{color:var(--muted)}
</style>
</head>
<body>
<header>
 <h1>3D Charge-Carrier Diffusion Simulator — Pristine perovskite</h1>
 <p>3D photo-excited carrier cloud reconstructed from 2D time-resolved PL, compared with the measured
 photon maps and the isotropic-Gaussian model projection over the same field of view.</p>
 <p class="muted">Measured up to 58 ns; everything past 58 ns is <b>extrapolated</b> from the fit.
 Drag the 3D view to rotate.</p>
</header>

<div class="wrap">
 <div id="left" class="card">
  <h2>3D carrier cloud</h2>
  <canvas id="cv3d" width="440" height="440"></canvas>

  <div class="row">
   <div>
    <h2>Experimental PL</h2>
    <div class="expbox">
     <img id="expimg" alt="measured PL">
     <div id="expov">EXTRAPOLATED</div>
    </div>
    <div class="cap"><span id="expreg" class="tag meas">data</span> <span id="expcap">measured @ 0.0 ns</span></div>
   </div>
   <div>
    <h2>2D model projection</h2>
    <canvas id="cv2d" width="180" height="180"></canvas>
    <div class="cap">isotropic Gaussian, same FOV</div>
   </div>
  </div>

  <div class="controls">
   <button id="play">&#9654; Play</button>
   <button id="reset">&#10227; Reset</button>
   <input id="t" type="range" min="0" max="200" step="0.2" value="0">
   <select id="spd">
    <option value="0.25">0.25&times;</option>
    <option value="0.5">0.5&times;</option>
    <option value="1" selected>1&times;</option>
    <option value="2">2&times;</option>
   </select>
  </div>
  <div class="opts">
   <label><input type="checkbox" id="cbspin" checked> auto-rotate</label>
   <label><input type="checkbox" id="cbshell" checked> 2&sigma; sphere shell</label>
   <span>FOV 6 &micro;m cube</span>
  </div>
  <div class="legend">4000 carriers Gaussian-sampled &middot; colour = core&rarr;edge density (jet) &middot;
  right image = measured photon map.</div>
 </div>

 <div id="right">
  <div class="card">
   <h2>State <span id="regime" class="tag meas" style="float:right">measured</span></h2>
   <table class="state">
    <tr><td>Time t (ns)</td><td id="rt">0.0</td></tr>
    <tr><td>&sigma;(t) (&micro;m)</td><td id="rsig">0.570</td></tr>
    <tr><td>MSD &sigma;&sup2;(t)&minus;&sigma;&sup2;(0) (&micro;m&sup2;)</td><td id="rmsd">0.000</td></tr>
    <tr><td>Instantaneous D(t) (cm&sup2;/s)</td><td id="rD">0.285</td></tr>
    <tr><td>3D rms &radic;3&middot;&sigma; (&micro;m)</td><td id="rrms">0.99</td></tr>
    <tr><td>Spread &sigma;/&sigma;&#8320; (&times;)</td><td id="rspread">1.00</td></tr>
   </table>
  </div>
  <div class="card">
   <h2>MSD curve</h2>
   <canvas id="plot" width="340" height="220"></canvas>
   <div class="cap"><span class="tag meas">data</span> measured points &middot;
    <span class="tag extr">extrap</span> shaded region past 58 ns</div>
  </div>
  <div class="card">
   <h2>Fitted parameters</h2>
   <table class="state params">
    <tr><td>D_initial</td><td>0.285 cm&sup2;/s</td></tr>
    <tr><td>D_trapped</td><td>0.021 cm&sup2;/s</td></tr>
    <tr><td>&tau;_turnover</td><td>1.14 ns</td></tr>
    <tr><td>calibration</td><td>0.0308 &micro;m/px</td></tr>
   </table>
  </div>
 </div>
</div>

<script>
"use strict";
// ===== model constants =====
const Di=0.28480452, Dt=0.02061028, TAU=1.14018661, SIG0=0.57, c2u=0.1, FOV=6.0, TMEAS=58.0, TMAX=200;
const Di_u=Di*c2u, Dt_u=Dt*c2u;
function msd(t){return 2*TAU*(Di_u-Dt_u)*(1-Math.exp(-t/TAU))+2*Dt_u*t;}
function sigma(t){return Math.sqrt(SIG0*SIG0+msd(t));}
function Dinst(t){return ((Di_u-Dt_u)*Math.exp(-t/TAU)+Dt_u)/c2u;}
function jet(v){v=Math.max(0,Math.min(1,v));
 const r=Math.max(0,Math.min(1,Math.min(4*v-1.5,-4*v+4.5)));
 const g=Math.max(0,Math.min(1,Math.min(4*v-0.5,-4*v+3.5)));
 const b=Math.max(0,Math.min(1,Math.min(4*v+0.5,-4*v+2.5)));
 return [r*255|0,g*255|0,b*255|0];}

// ===== data (Path B: synthetic; replace with original FRAMES/DATA for Path A) =====
//__FRAMES__
//__DATA__
const GATES=Object.keys(FRAMES).map(Number).sort((a,b)=>a-b);
function nearest(t){let best=GATES[0],bd=1e9;for(const g of GATES){const d=Math.abs(g-t);if(d<bd){bd=d;best=g;}}return best;}

// ===== seeded gaussian point cloud =====
let seed=12345;
function rnd(){seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;}
function gss(){let u=rnd();if(u<1e-9)u=1e-9;const v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
const M=4000, P=[];
for(let i=0;i<M;i++){const x=gss(),y=gss(),z=gss();const r2=x*x+y*y+z*z;
 const c=jet(Math.exp(-r2/2));P.push({x,y,z,r:c[0],g:c[1],b:c[2]});}

// ===== elements =====
const $=id=>document.getElementById(id);
const c3=$("cv3d"), x3=c3.getContext("2d"), CW=c3.width, CH=c3.height;
const c2=$("cv2d"), x2=c2.getContext("2d"), img2=x2.createImageData(180,180);
const expimg=$("expimg"), expov=$("expov"), expcap=$("expcap"), expreg=$("expreg");
const plot=$("plot"), xp=plot.getContext("2d");
const slider=$("t"), spdSel=$("spd"), playBtn=$("play"), cbspin=$("cbspin"), cbshell=$("cbshell");

let yaw=0.6, pitch=0.32, t=0, playing=false, lastTS=0, raf=null, idle=null;

// ===== 3D point cloud =====
function render3d(tc){
 x3.fillStyle="#05070b"; x3.fillRect(0,0,CW,CH);
 const s=sigma(tc), ppu=(CW*0.46)/(FOV/2);
 const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
 const a=new Array(M); let minD=1e9, maxD=-1e9;
 for(let i=0;i<M;i++){const p=P[i];
  let X=p.x*s, Y=p.y*s, Z=p.z*s;
  let x1=X*cy+Z*sy, z1=-X*sy+Z*cy;            // yaw about Y
  let y2=Y*cp - z1*sp, z2=Y*sp + z1*cp;       // pitch about X
  const d=z2; if(d<minD)minD=d; if(d>maxD)maxD=d;
  a[i]={sx:CW/2 + x1*ppu, sy:CH/2 - y2*ppu, d:d, r:p.r, g:p.g, b:p.b};
 }
 a.sort((p,q)=>p.d-q.d);                        // far -> near
 const span=(maxD-minD)||1;
 for(let i=0;i<M;i++){const p=a[i];
  const u=(p.d-minD)/span;
  const sz=0.9+1.7*u, al=0.18+0.6*u;
  x3.fillStyle="rgba("+p.r+","+p.g+","+p.b+","+al.toFixed(3)+")";
  x3.beginPath(); x3.arc(p.sx,p.sy,sz,0,6.2832); x3.fill();
 }
 if(cbshell.checked){
  const R=2*s*ppu;
  x3.strokeStyle="rgba(255,255,255,0.16)"; x3.lineWidth=1;
  x3.beginPath(); x3.arc(CW/2,CH/2,R,0,6.2832); x3.stroke();         // silhouette
  x3.beginPath(); x3.ellipse(CW/2,CH/2,R,Math.max(2,R*Math.abs(Math.sin(pitch))),0,0,6.2832); x3.stroke();  // equator
  x3.beginPath(); x3.ellipse(CW/2,CH/2,Math.max(2,R*Math.abs(Math.sin(yaw))),R,0,0,6.2832); x3.stroke();    // meridian
 }
 // 1 um scale bar
 const bar=1*ppu, bx=20, by=CH-22;
 x3.strokeStyle="#c9d1d9"; x3.lineWidth=2;
 x3.beginPath(); x3.moveTo(bx,by); x3.lineTo(bx+bar,by); x3.stroke();
 x3.fillStyle="#c9d1d9"; x3.font="11px sans-serif"; x3.fillText("1 µm",bx,by-6);
 x3.strokeStyle="rgba(139,148,158,.35)"; x3.lineWidth=1; x3.strokeRect(8,8,CW-16,CH-16);
}

// ===== 2D model projection =====
function render2d(tc){
 const s=sigma(tc), p2u=FOV/180, sp=s/p2u, amp=(SIG0*SIG0)/(s*s), d=img2.data;
 for(let y=0;y<180;y++){const dy=y-90;
  for(let x=0;x<180;x++){const dx=x-90;
   const v=amp*Math.exp(-(dx*dx+dy*dy)/(2*sp*sp));
   const c=jet(v); const k=(y*180+x)*4;
   d[k]=c[0]; d[k+1]=c[1]; d[k+2]=c[2]; d[k+3]=255;
  }}
 x2.putImageData(img2,0,0);
}

// ===== measured frame =====
function showExp(tc){
 const key=nearest(Math.min(tc,TMEAS)).toFixed(1);
 expimg.src="data:image/png;base64,"+FRAMES[key];
 if(tc<=TMEAS){
  expov.style.display="none"; expimg.style.filter="none";
  expcap.textContent="measured @ "+(+key).toFixed(1)+" ns";
  expreg.textContent="data"; expreg.className="tag meas";
 }else{
  expov.style.display="flex"; expimg.style.filter="grayscale(0.5) brightness(0.6)";
  expcap.textContent="last frame: 58.0 ns";
  expreg.textContent="extrapolated"; expreg.className="tag extr";
 }
}

// ===== MSD plot =====
function drawPlot(tc){
 const W=plot.width, H=plot.height, L=46, R=12, T=12, B=28;
 const pw=W-L-R, ph=H-T-B, mMax=msd(TMAX)*1.05;
 const X=t=>L+(t/TMAX)*pw, Y=m=>T+ph-(m/mMax)*ph;
 xp.fillStyle="#161b22"; xp.fillRect(0,0,W,H);
 // extrapolated shading
 xp.fillStyle="rgba(233,184,114,.10)"; xp.fillRect(X(TMEAS),T,X(TMAX)-X(TMEAS),ph);
 // axes
 xp.strokeStyle="#2d333b"; xp.lineWidth=1;
 xp.beginPath(); xp.moveTo(L,T); xp.lineTo(L,T+ph); xp.lineTo(L+pw,T+ph); xp.stroke();
 xp.fillStyle="#8b949e"; xp.font="10px sans-serif"; xp.textAlign="center";
 for(let tt=0;tt<=200;tt+=50){xp.fillText(tt,X(tt),H-8);
  xp.strokeStyle="#20262e"; xp.beginPath(); xp.moveTo(X(tt),T); xp.lineTo(X(tt),T+ph); xp.stroke();}
 xp.textAlign="right";
 for(let k=0;k<=3;k++){const m=mMax*k/3; xp.fillText(m.toFixed(2),L-5,Y(m)+3);}
 xp.save(); xp.translate(12,T+ph/2); xp.rotate(-Math.PI/2); xp.textAlign="center";
 xp.fillText("σ² − σ²(0)  (µm²)",0,0); xp.restore();
 xp.textAlign="center"; xp.fillText("Time (ns)",L+pw/2,H-0.5);
 // model curve
 xp.strokeStyle="#2bb3b0"; xp.lineWidth=2; xp.beginPath();
 for(let tt=0;tt<=TMAX;tt+=2){const yy=Y(msd(tt)); if(tt===0)xp.moveTo(X(tt),yy); else xp.lineTo(X(tt),yy);}
 xp.stroke();
 // measured scatter
 xp.fillStyle="#9fe7e5";
 for(const d of DATA){const m=d[1]*d[1]-SIG0*SIG0; xp.beginPath(); xp.arc(X(d[0]),Y(m),2.4,0,6.2832); xp.fill();}
 // current marker
 const mc=msd(tc);
 xp.strokeStyle="rgba(230,237,243,.5)"; xp.setLineDash([3,3]); xp.beginPath();
 xp.moveTo(X(tc),T); xp.lineTo(X(tc),T+ph); xp.stroke(); xp.setLineDash([]);
 xp.fillStyle="#fff"; xp.beginPath(); xp.arc(X(tc),Y(mc),3.4,0,6.2832); xp.fill();
}

// ===== readouts + full update =====
function update(tc){
 t=tc;
 const s=sigma(tc);
 $("rt").textContent=tc.toFixed(1);
 $("rsig").textContent=s.toFixed(3);
 $("rmsd").textContent=msd(tc).toFixed(3);
 $("rD").textContent=Dinst(tc).toFixed(3);
 $("rrms").textContent=(Math.sqrt(3)*s).toFixed(2);
 $("rspread").textContent=(s/SIG0).toFixed(2);
 const reg=$("regime");
 if(tc<=TMEAS){reg.textContent="measured"; reg.className="tag meas";}
 else{reg.textContent="extrapolated"; reg.className="tag extr";}
 if(Math.abs(parseFloat(slider.value)-tc)>1e-6) slider.value=tc;
 render3d(tc); render2d(tc); showExp(tc); drawPlot(tc);
}

// ===== animation =====
function loop(ts){
 if(!playing){return;}
 if(!lastTS)lastTS=ts;
 const dt=Math.min(0.05,(ts-lastTS)/1000); lastTS=ts;
 t+=dt*parseFloat(spdSel.value)*12;
 if(cbspin.checked)yaw+=dt*0.5;
 if(t>=TMAX){t=TMAX; playing=false; playBtn.innerHTML="&#9654; Play";}
 update(t);
 if(playing)raf=requestAnimationFrame(loop); else startIdle();
}
function startIdle(){
 stopIdle();
 let last=0;
 function step(ts){
  if(playing||!cbspin.checked){idle=null;return;}
  if(!last)last=ts;
  const dt=Math.min(0.05,(ts-last)/1000); last=ts;
  yaw+=dt*0.5; render3d(t);
  idle=requestAnimationFrame(step);
 }
 idle=requestAnimationFrame(step);
}
function stopIdle(){if(idle){cancelAnimationFrame(idle); idle=null;}}

playBtn.onclick=()=>{
 if(playing){playing=false; playBtn.innerHTML="&#9654; Play"; startIdle();}
 else{playing=true; playBtn.innerHTML="&#10073;&#10073; Pause"; lastTS=0; stopIdle(); raf=requestAnimationFrame(loop);}
};
$("reset").onclick=()=>{playing=false; playBtn.innerHTML="&#9654; Play"; t=0; update(0); startIdle();};
slider.oninput=()=>{playing=false; playBtn.innerHTML="&#9654; Play"; update(parseFloat(slider.value)); startIdle();};
cbshell.onchange=()=>render3d(t);
cbspin.onchange=()=>{ if(cbspin.checked&&!playing)startIdle(); else stopIdle(); };

// ===== drag rotate =====
let drag=false, px=0, py=0;
c3.addEventListener("mousedown",e=>{drag=true; px=e.clientX; py=e.clientY;});
window.addEventListener("mousemove",e=>{
 if(!drag)return;
 yaw+=(e.clientX-px)*0.01; pitch+=(e.clientY-py)*0.01;
 pitch=Math.max(-1.4,Math.min(1.4,pitch));
 px=e.clientX; py=e.clientY;
 render3d(t);
});
window.addEventListener("mouseup",()=>drag=false);

// ===== init =====
update(0);
startIdle();
</script>
</body>
</html>
'''

html = (HTML
        .replace("//__FRAMES__", "const FRAMES = " + json.dumps(FRAMES) + ";")
        .replace("//__DATA__",   "const DATA = "   + json.dumps(DATA)   + ";"))

out = "/home/user/transformers/diffusion_sim_3d.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
import os
print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")
