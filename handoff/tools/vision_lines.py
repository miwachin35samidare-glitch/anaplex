"""フルHD のカメラ1台（片目）から、人の網膜の形で束ねたときに要る線（神経節細胞に当たる）の数を出す（コンテナで回した。網は使わない）
  python3 vision_lines.py
形：人の神経節細胞の受容野の密度【ヒト】Watson 2014 J Vis（Curcio & Allen 1990 ほかの当てはめの式）
  ⚠️ 式の係数は Claude が覚えていた値。全視野の合計 133万（midget 79万）が Curcio & Allen の 70〜150万と合うのを確かめただけ。本文で確かめる
  ⚠️ 4つの経線の平均を半径だけで使った（上下左右の違いは入れていない）
カメラ：1920 × 1080 を横の視野 H 度に一様に割る（画素/度² ＝ (1920/H)²）
  線の数 ＝ 視野の中で min(人の密度, 画素の密度) を足したもの（画素より細かい線は作れない）。ON・OFF の2倍まで許す数も出す
"""
import numpy as np
RC=14804.6; RM=41.03; F0=1/1.12
P={'temporal':(0.9851,1.058,22.14),'superior':(0.9935,1.035,16.35),'nasal':(0.9729,1.084,7.633),'inferior':(0.996,0.9932,12.13)}
def dm(r,k):
    a,r2,re=P[k]; return 2*RC*(1+r/RM)**-1*(a*(1+r/r2)**-2+(1-a)*np.exp(-r/re))
def dg(r,k): return dm(r,k)/(F0*(1+r/RM)**-1)
# total over monocular field approx: avg of 4 meridians, r 0..90 (deg^2 measure, flat approx 2*pi*r dr)
r=np.linspace(0,90,90001); dr=r[1]-r[0]
for f,name in ((dm,'midget'),(dg,'all')):
    d=np.mean([f(r,k) for k in P],axis=0)
    print(name, f"{(d*2*np.pi*r).sum()*dr:,.0f}", "peak", f"{d[0]:,.0f}/deg2")
print()
for H in (60,70,90,120):
    V=H*1080/1920
    n=800
    xs=np.linspace(-H/2,H/2,n); ys=np.linspace(-V/2,V/2,int(n*V/H))
    X,Y=np.meshgrid(xs,ys); R=np.hypot(X,Y); dA=(xs[1]-xs[0])*(ys[1]-ys[0])
    d=np.mean([dg(R,k) for k in P],axis=0)
    p=(1920/H)**2
    hum=(d*dA).sum()
    c1=(np.minimum(d,p)*dA).sum(); c2=(np.minimum(d,2*p)*dA).sum()
    cut=np.sqrt(max(0,0))
    frac=(d>p).mean()
    print(f"横 {H}°：画素 {p:,.0f}/deg² ／ 人の線（この視野）{hum:,.0f} ／ 画素で頭打ち {c1:,.0f}（ON・OFF 2倍まで {c2:,.0f}）／ 人の方が密な所 {frac:.1%}")
