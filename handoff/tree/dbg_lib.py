"""切り分け：1発のあとの揺れは 外側手綱核が自分で撃つ分（i0）から来るか
  V0 今の先生 ／ V1 外側手綱核 i0 ＝ 0・境 → 外側手綱核の強さを率 30Hz に（版93f 版6 の形：g_L 1.4・i0 0）"""
import sys, json, torch
sys.argv=[sys.argv[0]]
import teacher_v95 as TV
d=json.load(open("teacher_v95_fit.json"))
gp=(0,)+tuple(d["gp"]); st=(0,)+tuple(d["st"])
DT=TV.DT

def make():
    th=TV.Teacher(gp, st)
    for k,v in d["i0"].items(): th.i0[th.sl(k)]=v
    return th

def mask(th,src,tgt):
    a,b=th.off[src],th.off[src]+TV.N; c,e=th.off[tgt],th.off[tgt]+TV.N
    return (th.s>=a)&(th.s<b)&(th.t>=c)&(th.t<e)

def rate(th,k,ms=2000):
    hz,_,_=th.run(int(ms/DT)); return float(hz[th.sl(k)].mean())

def bis(th,setter,k,tgt,lo,hi,it=10):
    for _ in range(it):
        m=(lo+hi)/2; setter(m); r=rate(th,k)
        lo,hi=(m,hi) if r<tgt else (lo,m)
    setter((lo+hi)/2); return (lo+hi)/2, rate(th,k)

def sn_i0(th):
    def s(x): th.i0[th.sl("黒質")]=x
    return bis(th,s,"黒質",4.56,0.0,0.02)

def volley(th,tag):
    T0=int(300/DT); L=int(120/DT); acc=None
    for rep in range(40):
        _,_,p=th.run(T0+L,kick={T0:{"外側手綱核":"fire"}},rec=True)
        acc=p if acc is None else acc+p
    p=acc/40; base=p[:T0].mean(0)
    print(f"[{tag}] ふだん 手綱 {float(base[5])*1000/DT:.1f}・吻 {float(base[6])*1000/DT:.1f}・黒 {float(base[7])*1000/DT:.2f}Hz",flush=True)
    for a in range(0,90,10):
        seg=p[T0+int(a/DT):T0+int((a+10)/DT)].mean(0)/base
        print(f"  {a:3d}-{a+10:<3d}ms 手綱 {float(seg[5]):.2f} 吻 {float(seg[6]):.2f} 黒 {float(seg[7]):.2f}",flush=True)
    w=lambda a,b: float(p[T0+int(a/DT):T0+int(b/DT),7].mean()/base[7])
    print(f"  黒質 10〜40ms {w(10,40):.2f}（約0.4）・50〜60ms {w(50,60):.2f}（約1）",flush=True)

