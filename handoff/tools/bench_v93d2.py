"""版93 の試し台 その4 の2：黒質の学びを Harnett 2009 の 11 条件に合わせ直す（細胞1体、実験と同じ入力の列）
  python3 bench_v93d2.py
  （網は使わない。何も書き換えない。CPU）
【み】10-18 17:13「A」：その4 で入れ落とした条件（跳ねだけ 0% ／ 入力の列 ＋ 1発 ≒ 0%・1発が弱まりを止める）を足す
実験の形（その4 と同じ）：入力 50Hz × 70発（1.4秒）・跳ね 20Hz × 5発・20秒ごと 10回
的（NMDA の重さの変化、Harnett 2009）
  跳ねの時刻 0ms −3% ／ +200ms +3% ／ +500ms +20% ／ +1秒 +43% ／ 列の終わり +60ms +13% ／ +120ms +6%
  跳ねが先 −250ms −22% ／ −500ms −10% ／ 入力だけ −8%
  ＋ 跳ねだけ（入力なし）0%（Fig 1E）／ 入力の列 ＋ 1発（+1秒）0%（−1% ± 10%・+1% ± 8% の2つの測り、Fig 1E・8）
式（周 1.067ms ごと。1本の線）
  a・b：入力の速い・遅い痕跡（その4 と同じ）
  z：細胞のカルシウムの代わり。撃つたびに +1、τz で減る ⇒ z が θ を越えた発火 ＝「跳ね」（1発では越えない）【Harnett：強まりは跳ねだけ、1発では起きない】
  c：跳ねの痕跡（跳ねの発火で、入力の速い痕跡 a が小さいときだけ +1。決め B）
  u：入力だけで弱まる分の溜め。入力が来るたびに +A₀、τu で減り、減った分が重さから引かれる。細胞が撃つと 0 に戻す【Harnett：1発で弱まりが止まる】
  跳ねの発火：Δw ＝ +A₊·a·b ／ 入力の発火：Δw ＝ −A₋·c、u ＋＝ A₀
"""
import numpy as np
from scipy.optimize import least_squares

DT = 1.067
DATA = [("跳ね 0ms", 0, "burst", -3), ("+200ms", 200, "burst", 3), ("+500ms", 500, "burst", 20), ("+1秒", 1000, "burst", 43),
        ("終わり+60ms", 1460, "burst", 13), ("終わり+120ms", 1520, "burst", 6), ("跳ねが先 −250ms", -250, "burst", -22),
        ("跳ねが先 −500ms", -500, "burst", -10), ("入力だけ", None, None, -8),
        ("跳ねだけ", 1000, "burst_nopre", 0), ("入力 ＋ 1発（+1秒）", 1000, "single", 0)]


def trial(p, t_post, kind):
    ta, tb, tc, Ap, Am, A0, tz, th, tu = p
    T0 = 600.0; n = int((T0 + 1400 + 1000) / DT)
    pre = np.zeros(n); post = np.zeros(n)
    if kind != "burst_nopre":
        for k in range(70):
            pre[int((T0 + k * 20.0) / DT)] = 1
    if t_post is not None:
        for k in range(1 if kind == "single" else 5):
            i = int((T0 + t_post + k * 50.0) / DT)
            if 0 <= i < n:
                post[i] = 1
    ea, eb, ec, ez, eu = (np.exp(-DT / x) for x in (ta, tb, tc, tz, tu))
    a = b = c = z = u = 0.0; dw = 0.0
    for i in range(n):
        a *= ea; b *= eb; c *= ec; z *= ez
        du = u * (1 - eu); u -= du; dw -= du
        if post[i]:
            z += 1.0
            if z > th:                                  # 跳ねの発火
                dw += Ap * a * b
                if a < 0.05:
                    c += 1.0
            u = 0.0                                     # 撃つと入力だけの弱まりは止まる
        if pre[i]:
            dw -= Am * c
            u += A0
            a += 1.0; b += 1.0
    # 列のあとに残った溜めは、次の列（20秒後）までに減り切る分を引く
    dw -= u
    return dw


def model(p):
    return np.array([10 * 100 * trial(p, t, k) for _, t, k, _ in DATA])


def main():
    y = np.array([v for *_, v in DATA], float)
    best = None
    for tz in (50.0, 100.0, 200.0):
        for th in (1.2, 1.5, 1.8):
            for tu in (200.0, 500.0, 1000.0):
                f = lambda q: model(list(q) + [tz, th, tu]) - y
                for ta0, tb0, tc0 in ((60, 800, 200), (120, 2000, 300), (130, 8000, 280)):
                    x0 = [ta0, tb0, tc0, 1e-5, 5e-4, 1e-4]
                    try:
                        r = least_squares(f, x0, bounds=([5, 50, 30, 0, 0, 0], [500, 10000, 3000, 1, 1, 1]),
                                          x_scale=[50, 500, 300, 1e-5, 5e-4, 1e-4], max_nfev=200)
                    except Exception:
                        continue
                    if best is None or r.cost < best[0]:
                        best = (r.cost, list(r.x) + [tz, th, tu])
                print(f"  τz {tz:.0f}・θ {th}・τu {tu:.0f} まで：いちばん良いずれ {np.sqrt(2 * best[0] / len(y)):.1f}%", flush=True)
    p = best[1]
    print("\n合わせた値（Harnett の 43% の形）", flush=True)
    print(f"  τa {p[0]:.0f}ms ／ τb {p[1]:.0f}ms ／ τc {p[2]:.0f}ms ／ A₊ {p[3]:.3g} ／ A₋ {p[4]:.3g} ／ A₀ {p[5]:.3g} ／ τz {p[6]:.0f}ms・θ {p[7]} ／ τu {p[8]:.0f}ms", flush=True)
    m = model(p)
    for (nm, _, _, v), mv in zip(DATA, m):
        print(f"  {nm:<18} 的 {v:+4d}%  模型 {mv:+6.1f}%", flush=True)
    print(f"  ずれ（二乗の平均の根）{np.sqrt(np.mean((m - y) ** 2)):.1f}%", flush=True)
    q = list(p); q[3] *= 23 / 43
    print(f"\n生理的な Mg：A₊ × 23/43 ⇒ +1秒 {model(q)[3]:+.1f}%（的 +23%）", flush=True)


if __name__ == "__main__":
    main()
