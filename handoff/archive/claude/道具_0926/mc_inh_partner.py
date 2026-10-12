import numpy as np
rng=np.random.default_rng(0)
DT=10.67e-3; MU,SD,TH=1.65,1.59,20.9
s2=np.log(1+(SD/MU)**2); m=np.log(MU)-s2/2
def rate(N,r,T=200000):
    k=rng.binomial(N,r,T); tot=np.zeros(T)
    idx=np.repeat(np.arange(T),k)
    np.add.at(tot,idx,rng.lognormal(m,np.sqrt(s2),k.sum()))
    return (tot>TH).mean()
def solve(r_hz,t_hz):
    r=r_hz*DT; t=t_hz*DT; lo,hi=10,20000
    while hi-lo>2:
        mid=(lo+hi)//2
        if rate(mid,r)<t: lo=mid
        else: hi=mid
    return hi, r, t
for name,rh,th in [("旧 0.45%/歩→FS 2Hz(Peyrache)",0.0045/DT,2.0),
                   ("Verma 覚醒 3.30→4.70",3.30,4.70),
                   ("LVQ 覚醒 2.6→6.4",2.6,6.4)]:
    N,r,t=solve(rh,th); print(f"{name}: 相手 {N}  (錐体 {r:.2%}/歩, 抑制の的 {t:.2%}/歩, 確認 {rate(N,r):.2%})")
