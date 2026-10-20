import sys, json, torch, math
sys.argv=[sys.argv[0]]
import teacher_v95 as TV
d=json.load(open("teacher_v95_fit.json"))
gp=(0,)+tuple(d["gp"]); st=(0,)+tuple(d["st"])
th=TV.Teacher(gp, st)
for k,v in d["i0"].items(): th.i0[th.sl(k)]=v
# 黒質の i0 を二分法で 4.56Hz に
lo,hi=0.0,0.02
for _ in range(10):
    m=(lo+hi)/2; th.i0[th.sl("黒質")]=m
    hz,_,_=th.run(int(2000/TV.DT)); r=float(hz[th.sl("黒質")].mean())
    lo,hi=(m,hi) if r<4.56 else (lo,m)
print("黒質 i0",m,"率",r,flush=True)
T0=int(300/TV.DT); L=int(150/TV.DT)
acc=None
for rep in range(40):
    _,_,p=th.run(T0+L,kick={T0:{"外側手綱核":"fire"}},rec=True,warm=int(1500/TV.DT))
    acc=p if acc is None else acc+p
p=acc/40*1000/TV.DT
base=p[:T0].mean(0)
names=TV.NUCS
print("ふだん", " ".join(f"{n} {float(base[i]):.1f}" for i,n in enumerate(names)))
for a in range(0,100,10):
    seg=p[T0+int(a/TV.DT):T0+int((a+10)/TV.DT)].mean(0)/base
    print(f"{a:3d}-{a+10}ms 手綱 {float(seg[5]):.2f} 吻 {float(seg[6]):.2f} 黒 {float(seg[7]):.2f}",flush=True)
