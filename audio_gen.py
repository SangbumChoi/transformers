#!/usr/bin/env python3
"""Synthesize a soundtrack (ambient pad + SFX) for the Molmo 2 explainer — pure Python, no deps."""
import wave, struct, math, random

SR=32000
DUR=[5,6,7,6.5,8.5,7,8.5,9,9,8,7.5,8,6,5]            # must match the HTML scenes
starts=[sum(DUR[:i]) for i in range(len(DUR))]
TOTAL=sum(DUR); nS=int(TOTAL*SR)
buf=[0.0]*nS
def midi(m): return 440.0*2**((m-69)/12.0)

# ---- ambient pad: root+fifth+octave per scene, slow attack/release ----
roots=[45,45,48,41,43,45,41,48,45,50,43,48,41,45]   # A2,C3,F2,G2 ... calm modal drone
for si,(st,du) in enumerate(zip(starts,DUR)):
    a=int(st*SR); b=int((st+du)*SR)
    freqs=[midi(roots[si]),midi(roots[si]+7),midi(roots[si]+12)]
    det=[0,0.15,-0.12]
    for i in range(a,b):
        t=(i-a)/SR; env=min(1,t/1.2)*min(1,(du-t)/1.2)      # 1.2s attack/release
        trem=0.9+0.1*math.sin(2*math.pi*0.15*t)
        v=0
        for f,d in zip(freqs,det): v+=math.sin(2*math.pi*(f+d)*t)
        buf[i]+=0.10*env*trem*v/3
        buf[i]+=0.06*env*math.sin(2*math.pi*midi(roots[si]-12)*t)   # soft sub-bass

def add(t0,samps):
    a=int(t0*SR)
    for k,v in enumerate(samps):
        j=a+k
        if 0<=j<nS: buf[j]+=v

def whoosh(t0,dur=0.4,amp=0.16):
    n=int(dur*SR); prev=0; out=[]
    for k in range(n):
        t=k/n; env=math.sin(math.pi*t)**2
        w=random.uniform(-1,1); prev=0.85*prev+0.15*w      # crude low-pass
        out.append(amp*env*prev)
    add(t0,out)

def bell(t0,note,dur=1.1,amp=0.20):
    f=midi(note); n=int(dur*SR); out=[]
    for k in range(n):
        t=k/SR; env=math.exp(-3.2*t)
        v=math.sin(2*math.pi*f*t)+0.5*math.sin(2*math.pi*2*f*t)+0.25*math.sin(2*math.pi*3.01*f*t)
        out.append(amp*env*v/1.75)
    add(t0,out)

def blip(t0,note,dur=0.16,amp=0.16):
    f=midi(note); n=int(dur*SR); out=[]
    for k in range(n):
        t=k/SR; env=math.exp(-16*t)
        out.append(amp*env*math.sin(2*math.pi*f*t))
    add(t0,out)

# transitions
for s in starts[1:]: whoosh(s)
# accents
bell(0.5,81); bell(82.0,84,1.3,0.22); bell(82.35,88,1.1,0.16); bell(96.0,81,1.4,0.22)
# point "blips" during the two pointing scenes
for t in [12.4,13.66,14.92]: blip(t,88)              # Molmo 1 pointing (scene 3)
for t in [41.05,42.45,43.85]: blip(t,91)             # PixMo-Points (scene 6)
# gentle high pentatonic sparkle every 2s (A-minor pentatonic)
penta=[69,72,74,76,79,81]
tt=2.0; idx=0
while tt<TOTAL-1:
    blip(tt,penta[idx%len(penta)],0.5,0.05); tt+=2.0; idx+=1

# normalize + write 16-bit stereo wav
peak=max(1e-6,max(abs(v) for v in buf)); scale=0.92/peak
with wave.open("soundtrack.wav","w") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    frames=bytearray()
    for v in buf:
        s=int(max(-1,min(1,v*scale))*32767); frames+=struct.pack("<hh",s,s)
    w.writeframes(bytes(frames))
print(f"wrote soundtrack.wav  {TOTAL:.0f}s  {nS} samples")
