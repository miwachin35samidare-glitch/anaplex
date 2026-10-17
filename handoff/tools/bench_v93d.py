"""版93 の試し台 その4：黒質の学びの痕跡を Harnett 2009 の9点に合わせる（細胞1体、実験と同じ入力の列）
  python3 bench_v93d.py
  （網は使わない。何も書き換えない。CPU）
【み】10-17 0:40「Aでいいよ」（痕跡2本：入力側 ＝ 続くほど溜まり終わると 約0.1秒で消える ／ 跳ね側 ＝ 跳ねてから 約0.5秒残る）
実験（【ラット】Harnett 2009、緻密部 約9割、4〜7週、室温ではなく 32〜34℃ の記載は無し ⚠️）
  入力の列：50Hz × 70発（1.4秒）／ 細胞の跳ね：20Hz × 5発 ／ 20秒ごと 10回 ／ 30〜40分後の NMDA の重さ
  跳ねの時刻（列の始まりから）と変化：0ms −3% ／ +200ms +3% ／ +500ms +20% ／ +1秒 +43% ／ 列の終わり +60ms +13% ／ +120ms +6%
    跳ねが先 −250ms −22% ／ −500ms −10% ／ 入力だけ −8%
式（周回 1.067ms ごと。1本の線の重さ w の変わり方）
  a ＝ 入力の速い痕跡（τa）：撃つたびに 1 足して減る ⇒「今、入力が続いているか」（終わると τa で消える）
  b ＝ 入力の遅い痕跡（τb）：撃つたびに足して減る ⇒「どれだけ長く続いているか」
  c ＝ 跳ねの痕跡（τc）：細胞が撃つたびに 1 足して減る
  細胞が撃ったとき：Δw ＝ ＋A₊ × a × b（入力が続いている中で、長く続いているほど強まる）
  入力が撃ったとき：Δw ＝ −A₋ × c − A₀（跳ねのあとに入力が来ると弱まる ／ 入力だけでも少し弱まる）
  ⇒ τa・τb・τc・A₊・A₋・A₀ の6つを9点に合わせる（9点に6つ ⇒ 合わない点が出たら形の不足）
強まる大きさ：生理的な Mg（1.2mM）で +23%（+1秒の点）⇒ 形は 43% の9点で合わせ、A₊ だけ 23/43 倍（⚠️ 弱まる側は Mg の値が無いのでそのまま）
"""
import numpy as np
from scipy.optimize import least_squares
import os
TC_MAX = float(os.environ.get("TC_MAX", "3000"))   # 版2：跳ねの痕跡の上限（Harnett の Ca の像：500ms 前の跳ねは列とほとんど重ならない）
ORDER = os.environ.get("ORDER") == "1"            # 版2：入力が来ていないときの跳ねだけが c を作る（Harnett「跳ねが入力に先立つと弱まる」）

DT = 1.067
DATA = [("跳ね 0ms", 0, -3), ("+200ms", 200, 3), ("+500ms", 500, 20), ("+1秒", 1000, 43),
        ("終わり+60ms", 1400 + 60, 13), ("終わり+120ms", 1400 + 120, 6), ("跳ねが先 −250ms", -250, -22),
        ("跳ねが先 −500ms", -500, -10), ("入力だけ", None, -8)]


def trial(p, t_burst):
    """1回の組（入力の列と跳ね）での重さの変わり（相対）"""
    ta, tb, tc, Ap, Am, A0 = p
    T0 = 600.0                                   # 前の余白
    total = T0 + 1400 + 1000
    n = int(total / DT)
    pre = np.zeros(n); post = np.zeros(n)
    for k in range(70):                          # 50Hz × 70
        pre[int((T0 + k * 20.0) / DT)] = 1
    if t_burst is not None:
        for k in range(5):                       # 20Hz × 5
            i = int((T0 + t_burst + k * 50.0) / DT)
            if 0 <= i < n:
                post[i] = 1
    ea, eb, ec = np.exp(-DT / ta), np.exp(-DT / tb), np.exp(-DT / tc)
    a = b = c = 0.0; dw = 0.0
    for i in range(n):
        a *= ea; b *= eb; c *= ec
        if post[i]:
            dw += Ap * a * b
            if not ORDER or a < 0.05:
                c += 1.0
        if pre[i]:
            dw -= Am * c + A0
            a += 1.0; b += 1.0
    return dw


def model(p):
    return np.array([10 * 100 * trial(p, t) for _, t, _ in DATA])   # 10回の組、%


def main():
    y = np.array([v for *_, v in DATA], float)
    best = None
    for ta0 in (30, 60, 120):
        for tb0 in (300, 800, 2000):
            for tc0 in [x for x in (100, 200, 500) if x < TC_MAX]:
                x0 = [ta0, tb0, tc0, 1e-5, 5e-4, 1e-4]
                r = least_squares(lambda p: model(p) - y, x0, bounds=([5, 50, 30, 0, 0, 0], [500, 10000, TC_MAX, 1, 1, 1]),
                                  x_scale=[50, 500, 300, 1e-5, 5e-4, 1e-4])
                if best is None or r.cost < best.cost:
                    best = r
    p = best.x
    print("合わせた値（Harnett の 43% の形）", flush=True)
    print(f"  τa（入力の速い痕跡）{p[0]:.0f}ms ／ τb（入力の遅い痕跡）{p[1]:.0f}ms ／ τc（跳ねの痕跡）{p[2]:.0f}ms", flush=True)
    print(f"  A₊ {p[3]:.3g} ／ A₋ {p[4]:.3g} ／ A₀ {p[5]:.3g}", flush=True)
    m = model(p)
    for (nm, t, v), mv in zip(DATA, m):
        print(f"  {nm:<16} 的 {v:+4d}%  模型 {mv:+6.1f}%", flush=True)
    print(f"  ずれ（二乗の平均の根）{np.sqrt(np.mean((m - y) ** 2)):.1f}%", flush=True)
    q = p.copy(); q[3] *= 23 / 43
    m2 = model(q)
    print(f"\n生理的な Mg：A₊ × 23/43 ⇒ +1秒 {m2[3]:+.1f}%（的 +23%）・入力だけ {m2[8]:+.1f}%・跳ねが先 −250ms {m2[6]:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
