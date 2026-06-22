const fs=require('fs');
const N=300;
const NAMES=["H2O part.","O2 part.","Base P","Host Temp","Dopant Temp","Host Rate","Dopant Rate",
 "Doping %","Subs. Temp","EML thick","Driving V","Efficiency","EQE","CIE x","Lifetime"];
const configs=[{n:"CH1",desc:"reference (normal)"},{n:"CH2",hostToff:-1.0,desc:"source -1.0C offset"},
 {n:"CH3",fault:"contam",desc:"leak / contamination"},{n:"CH4",fault:"dopdrift",desc:"dopant temp drift"},
 {n:"CH5",fault:"ratedrift",desc:"host temp drift"},{n:"CH6",subOff:5,desc:"substrate overheat +5C"},
 {n:"CH7",dopToff:5,desc:"over-doping +5C"},{n:"CH8",fault:"clog",desc:"source clog (rate drop)"}];
function genCH(cfg,sd){let seed=sd;const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;};
 const g=()=>{let u=rnd();if(u<1e-9)u=1e-9;return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*rnd());};
 const S={};NAMES.forEach(n=>S[n]=[]);
 for(let i=0;i<N;i++){const contam=cfg.fault==="contam"&&i>=255&&i<285;
  const HostT=230+(cfg.hostToff||0)+1.5*Math.sin(i/60)+(cfg.fault==="ratedrift"&&i>120?(i-120)*0.03:0)+0.5*g();
  const DopT=180+(cfg.dopToff||0)+2.5*Math.sin(i/70)+(cfg.fault==="dopdrift"&&i>200?(i-200)*0.08:0)+0.7*g();
  const clog=cfg.fault==="clog"&&i>175?0.70:1.0;const HostR=1.0*Math.exp((HostT-230)/12)*clog+0.015*g();
  const DopR=0.062*Math.exp((DopT-180)/10)+0.003*g();const Dope=DopR/(HostR+DopR)*100;
  const SubT=25+(cfg.subOff||0)+2*Math.sin(i/40)+0.6*g();const EML=(40+2.0*Math.sin(i/27+2.1)+0.6*g())*(cfg.fault==="clog"&&i>175?0.86:1);
  const H2O=1.4+0.3*Math.abs(g())+(contam?6+2*Math.abs(g()):0);const O2=0.8+0.2*Math.abs(g())+(contam?3+1.5*Math.abs(g()):0);
  const BaseP=6.5+0.003*i+(contam?2+Math.abs(g()):0)+0.25*Math.abs(g());const DrV=3.8+0.08*(EML-40)+0.04*(H2O-1.4)+0.05*g();
  const morph=1-0.012*Math.abs(SubT-25);const Eff=62*Math.exp(-Math.pow((Dope-6.2)/3.2,2))*morph*(1-0.02*(H2O-1.4))+0.6*g();
  const EQE=0.255*Eff+0.1*g();const CIEx=0.305+0.006*(Dope-6)+0.0015*(EML-40)+0.002*g();
  const Life=5200*Math.exp(-((H2O-1.4)+(O2-0.8))/9)*(0.6+0.4*morph)*(Eff/62)+60*g();
  [H2O,O2,BaseP,HostT,DopT,HostR,DopR,Dope,SubT,EML,DrV,Eff,EQE,CIEx,Life].forEach((v,k)=>S[NAMES[k]].push(v));}
 return S;}
const CH=configs.map((c,i)=>({cfg:c,S:genCH(c,101+i*1337)}));
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;const sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(v=>(v-m)*(v-m))));};
const MON=[["H2O part.","H2O",1],["O2 part.","O2",1],["Subs. Temp","SubT",0],["Host Rate","Rate",0],
 ["Dopant Rate","DopR",0],["Doping %","Doping",0],["Driving V","Volt",0]];
const ref={};MON.forEach(([nm])=>{const b=CH[0].S[nm];ref[nm]={mu:mean(b),sg:sd(b)||1e-6};});
const K=3.0,TH=0.8;
CH.forEach(c=>{const score=[],cause=[];for(let i=0;i<N;i++){let sc=0,best=0,bn="-";
  MON.forEach(([nm,sh,dir])=>{const z=(c.S[nm][i]-ref[nm].mu)/ref[nm].sg;const con=dir===0?Math.max(0,Math.abs(z)-K):Math.max(0,dir*z-K);sc+=con;if(con>best){best=con;bn=sh;}});score.push(sc);cause.push(bn);}
 const recent=Math.max(...score.slice(-30))>TH,any=Math.max(...score)>TH;const nA=score.filter(v=>v>TH).length;
 const counts={};for(let i=0;i<N;i++)if(score[i]>TH)counts[cause[i]]=(counts[cause[i]]||0)+1;let dom="-",dc=0;for(const k in counts)if(counts[k]>dc){dc=counts[k];dom=k;}
 c.score=score;c.status=recent?"A":(any?"W":"N");c.nA=nA;c.dom=dom;});
fs.writeFileSync("oled_assets/fdc.json",JSON.stringify({N,MON,ref,K,TH,
 chambers:CH.map(c=>({name:c.cfg.n,desc:c.cfg.desc,status:c.status,nA:c.nA,dom:c.dom,score:c.score,S:c.S}))}));
console.log("wrote oled_assets/fdc.json");
