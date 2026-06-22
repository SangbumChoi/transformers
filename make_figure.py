#!/usr/bin/env python3
"""Render a single-image time-lapse of thin-film carrier diffusion (for attachment)."""
import math, random
from PIL import Image, ImageDraw, ImageFont

# physics (same constants)
Di,Dt,TAU,SIG0,c2u,FOV,TMEAS = 0.28480452,0.02061028,1.14018661,0.57,0.1,6.0,58.0
Di_u,Dt_u = Di*c2u, Dt*c2u
msd=lambda t:2*TAU*(Di_u-Dt_u)*(1-math.exp(-t/TAU))+2*Dt_u*t
sigma=lambda t:math.sqrt(SIG0*SIG0+msd(t))
SIGZ0, LFILM = 0.15, 0.5
def jet(v):
    v=max(0,min(1,v))
    r=max(0,min(1,min(4*v-1.5,-4*v+4.5)));g=max(0,min(1,min(4*v-0.5,-4*v+3.5)));b=max(0,min(1,min(4*v+0.5,-4*v+2.5)))
    return (int(r*255),int(g*255),int(b*255))
def reflect(z,h):
    L=2*h;P=2*L;u=((z+h)%P+P)%P
    if u>L:u=P-u
    return u-h

# colours
BG=(14,17,22); PBG=(5,7,11); TXT=(230,237,243); MUT=(139,148,158)
TEAL=(43,179,176); AMBER=(233,184,114)

def font(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except Exception:
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()
def bfont(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception: return font(sz)

random.seed(1)
PTS=[(random.gauss(0,1),random.gauss(0,1),random.gauss(0,1)) for _ in range(4000)]

YAW, PITCH = 0.7, 0.55
SIZE = 320
PPU = (SIZE*0.40)/(FOV/2)
def proj(X,Y,Z):
    cy,sy,cp,sp=math.cos(YAW),math.sin(YAW),math.cos(PITCH),math.sin(PITCH)
    x1=X*cy+Z*sy;z1=-X*sy+Z*cy;y2=Y*cp-z1*sp
    return SIZE/2+x1*PPU, SIZE/2-y2*PPU

def panel(t):
    sxy=sigma(t); szf=math.sqrt(SIGZ0*SIGZ0+msd(t)); h=LFILM/2
    cy,sy,cp,sp=math.cos(YAW),math.sin(YAW),math.cos(PITCH),math.sin(PITCH)
    im=Image.new("RGB",(SIZE,SIZE),PBG); dr=ImageDraw.Draw(im,"RGBA")
    # film slab
    e=FOV/2
    cor=[(-e,-e,-h),(e,-e,-h),(e,e,-h),(-e,e,-h),(-e,-e,h),(e,-e,h),(e,e,h),(-e,e,h)]
    pj=[proj(*c) for c in cor]
    dr.polygon([pj[4],pj[5],pj[6],pj[7]],fill=(43,179,176,16))
    for i,j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        dr.line([pj[i],pj[j]],fill=(43,179,176,120),width=1)
    # cloud
    arr=[]
    for x,y,z in PTS:
        X=x*sxy;Y=y*sxy;Z=reflect(z*szf,h)
        x1=X*cy+Z*sy;z1=-X*sy+Z*cy;y2=Y*cp-z1*sp;z2=Y*sp+z1*cp
        arr.append((SIZE/2+x1*PPU,SIZE/2-y2*PPU,z2,jet(math.exp(-(x*x+y*y+z*z)/2))))
    arr.sort(key=lambda p:p[2])
    dmin=min(p[2]for p in arr);dmax=max(p[2]for p in arr);span=(dmax-dmin)or 1
    for px,py,d,c in arr:
        u=(d-dmin)/span;sz=0.9+1.7*u;al=int(255*(0.18+0.6*u))
        dr.ellipse([px-sz,py-sz,px+sz,py+sz],fill=(c[0],c[1],c[2],al))
    # scale bar
    dr.line([16,SIZE-18,16+PPU,SIZE-18],fill=(201,209,217),width=2)
    dr.text((16,SIZE-34),"1 µm",fill=(201,209,217),font=font(12))
    dr.text((16,12),f"σ={sxy:.2f} µm",fill=MUT,font=font(13))
    return im

# ---- compose ----
TIMES=[0,5,20,58,120,200]
M=26; HEAD=86; LAB=38; GAP=16; TL=160; FOOT=30
W=M*2+len(TIMES)*SIZE+(len(TIMES)-1)*GAP
H=M+HEAD+LAB+SIZE+22+TL+FOOT
img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img,"RGBA")

d.text((M,M-4),"3D Charge-Carrier Diffusion in a Thin Film — time evolution",fill=TXT,font=bfont(26))
d.text((M,M+30),"Pristine perovskite · lateral spread from the fitted MSD model; through-thickness confined by the "
       "film surfaces (L = 0.50 µm). Carriers flatten into a pancake and spread sideways.",fill=MUT,font=font(15))

def chip(x,y,txt,col,bgc):
    f=bfont(12); tb=d.textbbox((0,0),txt,font=f); w=tb[2]-tb[0]+14
    d.rounded_rectangle([x,y,x+w,y+20],8,fill=bgc)
    d.text((x+7,y+3),txt,fill=col,font=f); return w

x0=M; ytop=M+HEAD
for t in TIMES:
    p=panel(t); img.paste(p,(x0,ytop+LAB))
    fb=bfont(18); lab=f"t = {t:g} ns"
    d.text((x0+4,ytop+2),lab,fill=TXT,font=fb)
    if t<=TMEAS: chip(x0+4,ytop+26,"measured",TEAL,(20,59,58))
    else: chip(x0+4,ytop+26,"extrapolated",AMBER,(58,42,20))
    x0+=SIZE+GAP

# ---- timeline strip (MSD curve) ----
tx0=M; ty0=ytop+LAB+SIZE+22; tw=W-2*M; th=TL-44
d.rounded_rectangle([tx0,ty0,tx0+tw,ty0+th+30],10,fill=(22,27,34))
plx=tx0+54; ply=ty0+14; plw=tw-72; plh=th-10
mMax=msd(200)*1.05
X=lambda t:plx+(t/200)*plw
Y=lambda m:ply+plh-(m/mMax)*plh
# extrap shade
d.rectangle([X(58),ply,X(200),ply+plh],fill=(233,184,114,28))
# axes
d.line([plx,ply,plx,ply+plh],fill=(45,51,59),width=1); d.line([plx,ply+plh,plx+plw,ply+plh],fill=(45,51,59),width=1)
for tt in [0,50,100,150,200]:
    d.text((X(tt)-8,ply+plh+6),str(tt),fill=MUT,font=font(12))
    d.line([X(tt),ply,X(tt),ply+plh],fill=(32,38,46),width=1)
d.text((tx0+8,ply+plh//2-8),"MSD",fill=MUT,font=font(12))
d.text((X(100)-26,ty0+th+12),"Time (ns)",fill=MUT,font=font(13))
d.text((X(30),ply+4),"measured",fill=TEAL,font=font(12))
d.text((X(120),ply+4),"extrapolated",fill=AMBER,font=font(12))
# model curve
pts=[(X(tt),Y(msd(tt))) for tt in range(0,201,2)]
d.line(pts,fill=TEAL,width=2)
# sample markers
for t in TIMES:
    d.line([X(t),ply,X(t),ply+plh],fill=(230,237,243,90),width=1)
    d.ellipse([X(t)-4,Y(msd(t))-4,X(t)+4,Y(msd(t))+4],fill=(255,255,255))

# jet colorbar (top-right)
cbw,cbh=170,12; cbx=W-M-cbw; cby=M+4
for i in range(cbw):
    c=jet(i/(cbw-1)); d.line([cbx+i,cby,cbx+i,cby+cbh],fill=c)
d.text((cbx,cby+cbh+2),"edge",fill=MUT,font=font(11))
d.text((cbx+cbw-26,cby+cbh+2),"core",fill=MUT,font=font(11))
d.text((cbx,cby-16),"density",fill=MUT,font=font(11))

img.save("diffusion_timelapse.png")
print("saved diffusion_timelapse.png", img.size)
