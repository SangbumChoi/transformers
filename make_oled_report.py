#!/usr/bin/env python3
"""Generate KO + EN explanation report PDFs for the OLED correlation dashboard.
Renders the correlation heatmap + example scatters, then builds two PDFs."""
import math, os
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

os.makedirs("oled_assets", exist_ok=True)

# ---------------- data (same physics as the HTML) ----------------
seed=[77]
def rnd():
    seed[0]=(seed[0]*1103515245+12345)&0x7fffffff; return seed[0]/0x7fffffff
def g():
    u=rnd() or 1e-9; return math.sqrt(-2*math.log(u))*math.cos(2*math.pi*rnd())
N=300; EV0,EV1=255,280
NAMES=["H2O","O2","BaseP","HostT","DopT","HostR","DopR","Dope%","SubT","EML","DrV","Eff","EQE","CIEx","Life"]
S=[[0.0]*N for _ in NAMES]
for i in range(N):
    ev=EV0<=i<EV1
    HostT=230+1.5*math.sin(i/60)+0.5*g(); DopT=180+2.5*math.sin(i/70)+((i-220)*0.05 if i>220 else 0)+0.7*g()
    HostR=1.0*math.exp((HostT-230)/12)+0.015*g(); DopR=0.062*math.exp((DopT-180)/10)+0.003*g()
    Dope=DopR/(HostR+DopR)*100; SubT=25+2*math.sin(i/40)+0.6*g(); EML=40+2.0*math.sin(i/27+2.1)+0.6*g()
    H2O=1.4+0.3*abs(g())+(6+2*abs(g()) if ev else 0); O2=0.8+0.2*abs(g())+(3+1.5*abs(g()) if ev else 0)
    BaseP=6.5+0.003*i+(2+abs(g()) if ev else 0)+0.25*abs(g()); DrV=3.8+0.08*(EML-40)+0.04*(H2O-1.4)+0.05*g()
    morph=1-0.01*abs(SubT-25); Eff=62*math.exp(-((Dope-6.2)/3.2)**2)*morph*(1-0.02*(H2O-1.4))+0.6*g()
    EQE=0.255*Eff+0.1*g(); CIEx=0.305+0.006*(Dope-6)+0.0015*(EML-40)+0.002*g()
    Life=5200*math.exp(-((H2O-1.4)+(O2-0.8))/9)*(0.6+0.4*morph)*(Eff/62)+60*g()
    for k,v in enumerate([H2O,O2,BaseP,HostT,DopT,HostR,DopR,Dope,SubT,EML,DrV,Eff,EQE,CIEx,Life]): S[k][i]=v
def mean(a): return sum(a)/len(a)
def pe(a,b):
    ma,mb=mean(a),mean(b); n=da=db=0
    for i in range(len(a)): x=a[i]-ma; y=b[i]-mb; n+=x*y; da+=x*x; db+=y*y
    return n/math.sqrt(da*db)
R=[[pe(S[i],S[j]) for j in range(len(NAMES))] for i in range(len(NAMES))]

def F(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",sz)
    except: return ImageFont.load_default(size=sz)

# ---------------- render heatmap ----------------
def render_heat():
    def cc(r):
        a=min(1,abs(r)); return (255,int(255-a*175),int(255-a*195)) if r>=0 else (int(255-a*195),int(255-a*120),255)
    HL,HT,W=96,58,520; CELL=(W-HL-8)/len(NAMES)
    im=Image.new("RGB",(W,W),(255,255,255)); d=ImageDraw.Draw(im)
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            x=HL+j*CELL; y=HT+i*CELL
            d.rectangle([x,y,x+CELL-1,y+CELL-1],fill=cc(R[i][j]))
            d.text((x+CELL/2,y+CELL/2),f"{R[i][j]:.2f}",fill=(20,20,20) if abs(R[i][j])>0.55 else (110,110,110),font=F(8),anchor="mm")
        d.text((HL-4,HT+i*CELL+CELL/2),NAMES[i],fill=(40,40,40),font=F(9),anchor="rm")
    for j in range(len(NAMES)):
        t=Image.new("RGBA",(60,12),(0,0,0,0)); td=ImageDraw.Draw(t); td.text((0,0),NAMES[j],fill=(40,40,40),font=F(9))
        t=t.rotate(90,expand=True); im.paste(t,(int(HL+j*CELL+CELL/2-6),HT-54),t)
    im.save("oled_assets/heat.png")

def render_scatter(ai,bi,fname,title):
    A,B=S[ai],S[bi]; W,H=420,300; L,Rm,Tp,Bm=52,12,32,34
    im=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(im,"RGBA")
    ax,bx,ay,by=min(A),max(A),min(B),max(B)
    px=lambda v:L+(v-ax)/(bx-ax)*(W-L-Rm); py=lambda v:Tp+(H-Tp-Bm)-(v-ay)/(by-ay)*(H-Tp-Bm)
    d.line([L,Tp,L,H-Bm],fill=(180,180,180)); d.line([L,H-Bm,W-Rm,H-Bm],fill=(180,180,180))
    for i in range(N):
        ev=EV0<=i<EV1
        d.ellipse([px(A[i])-2.2,py(B[i])-2.2,px(A[i])+2.2,py(B[i])+2.2],fill=(229,83,75,235) if ev else (43,150,176,150))
    ma,mb=mean(A),mean(B); nu=de=0
    for i in range(N): nu+=(A[i]-ma)*(B[i]-mb); de+=(A[i]-ma)**2
    sl=nu/de; ic=mb-sl*ma
    d.line([px(ax),py(sl*ax+ic),px(bx),py(sl*bx+ic)],fill=(220,140,40),width=2)
    d.text((L,8),f"{title}   (r = {pe(A,B):.2f})",fill=(20,20,20),font=F(13))
    d.text((W/2,H-16),NAMES[ai],fill=(90,90,90),font=F(11),anchor="mm")
    im.save("oled_assets/"+fname)

render_heat()
render_scatter(NAMES.index("Dope%"),NAMES.index("CIEx"),"sc_dope_cie.png","Doping % -> CIE x")
render_scatter(NAMES.index("H2O"),NAMES.index("Life"),"sc_h2o_life.png","H2O partial -> Lifetime")
render_scatter(NAMES.index("Dope%"),NAMES.index("Eff"),"sc_dope_eff.png","Doping % -> Efficiency (peak)")

# correlations to quote
def r(a,b): return R[NAMES.index(a)][NAMES.index(b)]
PN={"HostT":"Host Temp","HostR":"Host Rate","DopT":"Dopant Temp","DopR":"Dopant Rate","Dope%":"Doping %",
    "CIEx":"CIE x","EML":"EML thick","DrV":"Driving V","H2O":"H2O","O2":"O2","Eff":"Efficiency","EQE":"EQE","Life":"Lifetime"}
CORR=[("HostT","HostR"),("DopT","DopR"),("DopR","Dope%"),("Dope%","CIEx"),("Dope%","Eff"),
      ("EML","DrV"),("EML","CIEx"),("H2O","Life"),("O2","Life"),("Eff","Life"),("Eff","EQE"),("H2O","O2")]

# ================= PDF builders =================
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
KO="HYGothic-Medium"; EN="Helvetica"; ENB="Helvetica-Bold"
NV=colors.HexColor("#2bb3b0"); DARK=colors.HexColor("#1a1a1a"); GREY=colors.HexColor("#555")
BORDER=colors.HexColor("#cccccc"); ALT=colors.HexColor("#f5f7f9"); HEAD=colors.HexColor("#143b3a")

CONTENT={
 "ko":{
  "title":"OLED 증착 공정·소자 데이터 상관관계 분석 보고서",
  "sub":"co-evaporation 공정~소자 특성까지 — 데이터 상관관계, FDC/SPC, 파트 수명",
  "s_over":"1. 개요",
  "over":"OLED 유기물 열증착(co-evaporation) 라인에서 얻을 수 있는 15개 신호 — 진공/잔류가스, 소스 온도, "
   "QCM 증착률, 도핑 농도, 막두께, 그리고 소자 특성(구동전압·효율·EQE·CIE 색좌표·수명) — 의 상관관계를 "
   "한눈에 탐색하는 대시보드와 그 해석을 정리한다. 데이터는 실제 OLED 물리(아레니우스 증발, 도핑→색·효율, "
   "막두께→전압, 잔류수분→수명)를 반영해 생성한 합성 데이터이며, 실측 EES/MES/FDC 로그로 그대로 교체할 수 있다.",
  "s_sig":"2. 수집 데이터 신호 (15)",
  "s_phys":"3. 데이터 생성에 쓰인 물리 모델",
  "phys":["• 아레니우스 증발: 증착률 ∝ exp((T−T₀)/scale). 소스 온도가 오르면 증착률이 지수적으로 증가.",
   "• 도핑 농도 = Dopant Rate / (Host Rate + Dopant Rate) × 100. Host는 PID로 안정 제어되어 도핑은 "
   "주로 dopant rate를 추종.",
   "• 효율(cd/A): 최적 도핑(~6%)에서 정점을 갖는 역U 곡선 — 과도핑 시 농도 소광(concentration quenching)으로 감소.",
   "• EML 두께 → 구동전압: 유기막이 두꺼울수록 구동전압 상승. 두께는 마이크로캐비티로 CIE 색좌표에도 영향.",
   "• 오염(잔류 H₂O/O₂) → 수명: 잔류 수분/산소가 높을수록 다크스팟·열화로 LT95 수명이 지수적으로 감소."],
  "s_heat":"4. 상관관계 히트맵 (Pearson r)",
  "heat":"색은 상관계수 r(파랑 −1 · 흰색 0 · 빨강 +1). 좌상단 진공/오염 클러스터(H₂O·O₂·BaseP)가 서로 강한 양의 "
   "상관을 보이고, 소스온도→증착률(아레니우스)이 거의 1, 도핑→색좌표가 강한 양, 도핑→효율이 음(과도핑 영역), "
   "오염→수명이 강한 음의 상관을 보인다.",
  "s_key":"5. 주요 상관관계와 해석",
  "s_scat":"6. 대표 산점도",
  "scat_cap":["도핑 농도가 오르면 CIE x가 함께 이동 — 색좌표는 도핑으로 직접 제어됨 (강한 양의 상관).",
   "잔류 수분이 높을수록 수명이 급격히 감소 — 오염이 수명의 1차 인자 (강한 음의 상관).",
   "효율은 최적 도핑에서 정점을 갖는 역U — 선형 상관(r)만으로는 안 보이고 산점도로 드러남."],
  "s_dash":"7. 대시보드 구성요소",
  "dash":["• 상관행렬 히트맵: 15×15 Pearson r. 셀 클릭 → 해당 쌍의 산점도.",
   "• 산점도: 선택 쌍의 점 분포 + 회귀선 + r/r². 오염 이벤트 점은 빨강.",
   "• 시계열: 핵심 신호 정규화 + 오염 이벤트 음영.",
   "• SPC 관리도: 도핑/CIE 등 품질 핵심값의 3σ UCL/LCL — 말기 도핑 드리프트가 관리한계 이탈로 표시.",
   "• FDC: 잔류가스 기반 오염 anomaly score + 경보(ALARM).",
   "• 파트/소스 수명: 소스 running time, QCM 크리스털, 소재 소모량, FMM 사이클 게이지.",
   "• 인과 다이어그램: 소스온도→증착률→도핑·두께→소자특성, 오염→수명."],
  "s_fdc":"8. FDC · SPC · 파트 수명",
  "fdc":"FDC는 수백 개 센서의 실시간 이상징후를 조기 발견한다(여기서는 잔류 H₂O/O₂ 점수). SPC는 레시피별 품질값"
   "(도핑·CIE·두께)을 3σ 관리한계로 표준화해 균일성을 보장한다. 파트 수명은 소스/크리스털/마스크의 누적 사용량으로 "
   "교체주기를 산정한다. 본 데이터에서는 말기 도핑 드리프트(SPC 이탈)와 오염 이벤트(FDC 경보)가 함께 발생해 "
   "효율·색좌표·수명 저하로 이어진다.",
  "s_caus":"9. 상관 vs 인과 (주의)",
  "caus":"히트맵의 강한 상관이 모두 인과는 아니다. 예컨대 Substrate Temp와 EML 두께가 우연히 비슷한 주기로 변하면 "
   "허위(spurious) 상관이 생길 수 있다(본 버전에서는 위상을 어긋나게 해 약화시킴). 물리 인과는 다이어그램으로 해석하고, "
   "상관은 가설 수립·이상탐지의 출발점으로 사용해야 한다.",
  "s_use":"10. 활용 · 한계 · 결론",
  "use":"실측 라인에서는 합성 데이터 블록을 EES/MES/FDC 로그(동일 형식)로 교체하면 그대로 동작한다. 본 보고서의 수치는 "
   "물리 모델 기반 합성값이므로 절대값이 아닌 관계의 구조를 보기 위한 것이다. 결론적으로, OLED 공정은 "
   "‘소스온도→증착률→도핑/두께→소자특성, 오염→수명’의 인과 사슬로 이해할 수 있으며, 상관 히트맵은 그 사슬을 "
   "데이터로 빠르게 점검하고 FDC/SPC와 연결하는 강력한 출발점이다.",
  "foot":"OLED 공정 데이터 상관관계 보고서 · 합성 데이터(교육·설명용)",
  "grp":["진공/잔류가스 (RGA)","소스(열증착)","증착률 (QCM)","공정 파생","소자 특성 (QC)"],
  "sig_hdr":["그룹","신호","단위","의미"],
  "rows":[
   ["진공/잔류가스","H₂O 분압","×10⁻⁸ Torr","잔류 수분 — 수명의 1차 인자"],
   ["","O₂ 분압","×10⁻⁸ Torr","잔류 산소 — 열화/다크스팟"],
   ["","Base Pressure","×10⁻⁷ Torr","챔버 진공도/누설"],
   ["소스","Host Temp","°C","호스트 셀 온도(아레니우스)"],
   ["","Dopant Temp","°C","도펀트 셀 온도"],
   ["증착률(QCM)","Host Rate","Å/s","호스트 증착률"],
   ["","Dopant Rate","Å/s","도펀트 증착률"],
   ["공정 파생","Doping %","%","도핑 농도 = Dop/(Host+Dop)"],
   ["","Substrate Temp","°C","기판 온도(모폴로지)"],
   ["","EML thickness","nm","발광층 두께(셔터/타이밍)"],
   ["소자(QC)","Driving V","V","구동 전압"],
   ["","Efficiency","cd/A","전류 효율"],
   ["","EQE","%","외부양자효율"],
   ["","CIE x","-","색좌표 x"],
   ["","Lifetime LT95","h","휘도 95% 수명"]],
  "key_hdr":["신호 쌍","r","해석"],
 },
 "en":{
  "title":"OLED Evaporation — Process & Device Data Correlation Report",
  "sub":"From co-evaporation process to device characteristics — correlations, FDC/SPC, part life",
  "s_over":"1. Overview",
  "over":"This report documents an interactive dashboard that explores the correlations among 15 signals "
   "obtainable on an OLED organic co-evaporation line — vacuum/residual-gas, source temperatures, QCM "
   "deposition rates, doping concentration, film thickness, and device metrics (driving voltage, efficiency, "
   "EQE, CIE color, lifetime). The data is synthetic but encodes real OLED physics (Arrhenius evaporation, "
   "doping->color/efficiency, thickness->voltage, residual water->lifetime); real EES/MES/FDC logs drop in directly.",
  "s_sig":"2. Collected data signals (15)",
  "s_phys":"3. Physics models used to generate the data",
  "phys":["- Arrhenius evaporation: rate ~ exp((T-T0)/scale); source temperature raises rate exponentially.",
   "- Doping % = Dopant Rate / (Host Rate + Dopant Rate) x 100. Host is PID-controlled, so doping tracks the dopant rate.",
   "- Efficiency (cd/A): inverted-U peaking at the optimal doping (~6%); over-doping causes concentration quenching.",
   "- EML thickness -> driving voltage: thicker organics raise the voltage; thickness also shifts CIE via microcavity.",
   "- Contamination (residual H2O/O2) -> lifetime: more residual water/oxygen exponentially shortens LT95 (dark spots)."],
  "s_heat":"4. Correlation heatmap (Pearson r)",
  "heat":"Color encodes r (blue -1, white 0, red +1). The vacuum/contamination cluster (H2O, O2, Base P) is "
   "mutually strongly positive; source-temperature->rate (Arrhenius) is ~1; doping->color is strongly positive; "
   "doping->efficiency is negative (over-doping side); and contamination->lifetime is strongly negative.",
  "s_key":"5. Key correlations and interpretation",
  "s_scat":"6. Representative scatter plots",
  "scat_cap":["Higher doping shifts CIE x — color is directly set by doping (strong positive correlation).",
   "Higher residual water sharply reduces lifetime — contamination is the primary lifetime driver (strong negative).",
   "Efficiency peaks at the optimal doping (inverted-U) — invisible to linear r, revealed by the scatter."],
  "s_dash":"7. Dashboard components",
  "dash":["- Correlation heatmap: 15x15 Pearson r; click a cell -> scatter of that pair.",
   "- Scatter: point cloud + regression line + r/r^2; contamination-event points in red.",
   "- Time series: normalized key signals with the contamination event shaded.",
   "- SPC chart: 3-sigma UCL/LCL on quality-critical values (doping/CIE); late doping drift breaches limits.",
   "- FDC: residual-gas contamination anomaly score with ALARM.",
   "- Part/source life: source running time, QCM crystal, material consumed, FMM cycles gauges.",
   "- Causal diagram: source temp -> rate -> doping/thickness -> device metrics; contamination -> lifetime."],
  "s_fdc":"8. FDC, SPC and part life",
  "fdc":"FDC finds anomalies early from many sensors (here a residual H2O/O2 score). SPC standardizes recipe "
   "quality values (doping/CIE/thickness) with 3-sigma limits to ensure uniformity. Part life sizes the "
   "replacement cycle from cumulative source/crystal/mask usage. In this data a late doping drift (SPC breach) "
   "and a contamination event (FDC alarm) coincide, degrading efficiency, color and lifetime.",
  "s_caus":"9. Correlation vs causation (caution)",
  "caus":"Not every strong correlation is causal. If Substrate Temp and EML thickness happen to vary on similar "
   "periods, a spurious correlation can appear (de-phased here to weaken it). Read physical causation from the "
   "diagram, and use correlations as a starting point for hypotheses and anomaly detection.",
  "s_use":"10. Usage, limits and conclusion",
  "use":"On a real line, replace the synthetic data blocks with EES/MES/FDC logs of the same shape. The numbers "
   "here are model-based synthetic values, meant to show the structure of the relationships rather than absolute "
   "values. In short, the OLED process is a causal chain 'source temp -> rate -> doping/thickness -> device "
   "metrics; contamination -> lifetime', and the correlation heatmap is a powerful entry point to check that "
   "chain from data and tie it to FDC/SPC.",
  "foot":"OLED process-data correlation report - synthetic data (educational)",
  "grp":["Vacuum/RGA","Source (evap)","Rate (QCM)","Process-derived","Device (QC)"],
  "sig_hdr":["Group","Signal","Unit","Meaning"],
  "rows":[
   ["Vacuum/RGA","H2O partial","x10^-8 Torr","residual water - primary lifetime driver"],
   ["","O2 partial","x10^-8 Torr","residual oxygen - degradation/dark spots"],
   ["","Base Pressure","x10^-7 Torr","chamber vacuum / leak"],
   ["Source","Host Temp","C","host cell temperature (Arrhenius)"],
   ["","Dopant Temp","C","dopant cell temperature"],
   ["Rate (QCM)","Host Rate","A/s","host deposition rate"],
   ["","Dopant Rate","A/s","dopant deposition rate"],
   ["Process","Doping %","%","doping conc. = Dop/(Host+Dop)"],
   ["","Substrate Temp","C","substrate temperature (morphology)"],
   ["","EML thickness","nm","emission-layer thickness (shutter timing)"],
   ["Device (QC)","Driving V","V","driving voltage"],
   ["","Efficiency","cd/A","current efficiency"],
   ["","EQE","%","external quantum efficiency"],
   ["","CIE x","-","color coordinate x"],
   ["","Lifetime LT95","h","time to 95% luminance"]],
  "key_hdr":["Signal pair","r","Interpretation"],
 }
}
KEY_INTERP={
 "ko":{("HostT","HostR"):"아레니우스 — 소스온도↑ → 증착률↑(지수)",
  ("DopT","DopR"):"도펀트 온도↑ → 도펀트 증착률↑",
  ("DopR","Dope%"):"도펀트 증착률↑ → 도핑 농도↑",
  ("Dope%","CIEx"):"도핑↑ → 색좌표 이동(직접 제어)",
  ("Dope%","Eff"):"과도핑 → 효율↓ (농도 소광)",
  ("EML","DrV"):"막두께↑ → 구동전압↑",
  ("EML","CIEx"):"두께 → 마이크로캐비티 색이동",
  ("H2O","Life"):"잔류수분↑ → 수명↓ (강한 음)",
  ("O2","Life"):"잔류산소↑ → 수명↓",
  ("Eff","Life"):"효율↑ → 수명↑(동반)",
  ("Eff","EQE"):"효율과 EQE는 사실상 동일",
  ("H2O","O2"):"오염 신호끼리 동반 상승"},
 "en":{("HostT","HostR"):"Arrhenius - temp up -> rate up (exp)",
  ("DopT","DopR"):"dopant temp up -> dopant rate up",
  ("DopR","Dope%"):"dopant rate up -> doping up",
  ("Dope%","CIEx"):"doping up -> color shift (direct)",
  ("Dope%","Eff"):"over-doping -> efficiency down (quenching)",
  ("EML","DrV"):"thickness up -> driving voltage up",
  ("EML","CIEx"):"thickness -> microcavity color shift",
  ("H2O","Life"):"residual water up -> lifetime down (strong)",
  ("O2","Life"):"residual oxygen up -> lifetime down",
  ("Eff","Life"):"efficiency up -> lifetime up (together)",
  ("Eff","EQE"):"efficiency and EQE are essentially identical",
  ("H2O","O2"):"contamination signals rise together"}
}

def build(lang):
    C=CONTENT[lang]; base=KO if lang=="ko" else EN; bold=KO if lang=="ko" else ENB
    H1=ParagraphStyle("H1",fontName=bold,fontSize=18,textColor=DARK,leading=23,spaceAfter=3)
    SUB=ParagraphStyle("SUB",fontName=base,fontSize=10.5,textColor=GREY,leading=15,spaceAfter=12)
    H2=ParagraphStyle("H2",fontName=bold,fontSize=13.5,textColor=NV,leading=18,spaceBefore=12,spaceAfter=5)
    BODY=ParagraphStyle("BODY",fontName=base,fontSize=10,textColor=DARK,leading=16,spaceAfter=6)
    BUL=ParagraphStyle("BUL",parent=BODY,leftIndent=8,spaceAfter=3)
    CAP=ParagraphStyle("CAP",fontName=base,fontSize=8.5,textColor=GREY,leading=12,spaceAfter=10,spaceBefore=2)
    CELL=ParagraphStyle("CELL",fontName=base,fontSize=8.5,leading=11,textColor=DARK)
    CH=ParagraphStyle("CH",fontName=bold,fontSize=8.5,leading=11,textColor=colors.white)
    st=[]
    st.append(Paragraph(C["title"],H1)); st.append(Paragraph(C["sub"],SUB))
    st.append(HRFlowable(width="100%",thickness=1,color=NV,spaceAfter=8))
    st.append(Paragraph(C["s_over"],H2)); st.append(Paragraph(C["over"],BODY))
    # signals table
    st.append(Paragraph(C["s_sig"],H2))
    data=[[Paragraph(h,CH) for h in C["sig_hdr"]]]
    for rrow in C["rows"]: data.append([Paragraph(x,CELL) for x in rrow])
    t=Table(data,colWidths=[2.6*cm,3.0*cm,2.4*cm,8.0*cm],repeatRows=1)
    ts=[("BACKGROUND",(0,0),(-1,0),HEAD),("GRID",(0,0),(-1,-1),0.4,BORDER),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),5)]
    for i in range(1,len(data)):
        if i%2==0: ts.append(("BACKGROUND",(0,i),(-1,i),ALT))
    t.setStyle(TableStyle(ts)); st.append(t)
    # physics
    st.append(Paragraph(C["s_phys"],H2))
    for p in C["phys"]: st.append(Paragraph(p,BUL))
    # heatmap image
    st.append(Paragraph(C["s_heat"],H2)); st.append(Paragraph(C["heat"],BODY))
    iw,ih=Image.open("oled_assets/heat.png").size; w=12.5*cm
    st.append(RLImage("oled_assets/heat.png",width=w,height=w*ih/iw))
    # key correlations table
    st.append(Paragraph(C["s_key"],H2))
    kd=[[Paragraph(h,CH) for h in C["key_hdr"]]]
    for a,b in CORR:
        rv=r(a,b); col="#c0392b" if rv>=0 else "#1f6fb2"
        kd.append([Paragraph(f"{PN[a]} ~ {PN[b]}",CELL),
                   Paragraph(f"<font color='{col}'><b>{rv:+.2f}</b></font>",CELL),
                   Paragraph(KEY_INTERP[lang][(a,b)],CELL)])
    kt=Table(kd,colWidths=[5.2*cm,1.6*cm,9.2*cm],repeatRows=1)
    kts=[("BACKGROUND",(0,0),(-1,0),HEAD),("GRID",(0,0),(-1,-1),0.4,BORDER),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
         ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),5)]
    for i in range(1,len(kd)):
        if i%2==0: kts.append(("BACKGROUND",(0,i),(-1,i),ALT))
    kt.setStyle(TableStyle(kts)); st.append(kt)
    # scatters
    st.append(Paragraph(C["s_scat"],H2))
    scs=["sc_dope_cie.png","sc_h2o_life.png","sc_dope_eff.png"]
    imgs=[]
    for f in scs:
        iw,ih=Image.open("oled_assets/"+f).size; ww=5.3*cm
        imgs.append(RLImage("oled_assets/"+f,width=ww,height=ww*ih/iw))
    st.append(Table([imgs],colWidths=[5.6*cm]*3,style=[("ALIGN",(0,0),(-1,-1),"CENTER"),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    for cpt in C["scat_cap"]: st.append(Paragraph("• "+cpt,CAP))
    # dashboard
    st.append(Paragraph(C["s_dash"],H2))
    for x in C["dash"]: st.append(Paragraph(x,BUL))
    # fdc/spc
    st.append(Paragraph(C["s_fdc"],H2)); st.append(Paragraph(C["fdc"],BODY))
    # causation
    st.append(Paragraph(C["s_caus"],H2)); st.append(Paragraph(C["caus"],BODY))
    # usage
    st.append(Paragraph(C["s_use"],H2)); st.append(Paragraph(C["use"],BODY))

    path=f"oled_report_{lang}.pdf"
    def foot(cv,doc):
        cv.saveState(); cv.setStrokeColor(BORDER); cv.setLineWidth(0.5)
        cv.line(2*cm,1.3*cm,A4[0]-2*cm,1.3*cm)
        cv.setFont(base,8); cv.setFillColor(GREY)
        cv.drawString(2*cm,0.95*cm,C["foot"]); cv.drawRightString(A4[0]-2*cm,0.95*cm,f"p.{doc.page}")
        cv.restoreState()
    doc=SimpleDocTemplate(path,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.8*cm,bottomMargin=1.7*cm,
                          title=C["title"])
    doc.build(st,onFirstPage=foot,onLaterPages=foot)
    print("wrote",path,f"({os.path.getsize(path)/1024:.0f} KB)")

build("ko"); build("en")
