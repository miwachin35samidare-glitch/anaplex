"""PPN・CL・視床・皮質の揺れの帯を、取ってある 歩 × 領野 の表で見る（arkhe の ~/anaplex で。CPU だけ・網は組まない・GPU を使わない）
  python3 analyze_band112.py [表 ...]   既定：wave111c_cl256.npz wave111c_v112.npz（無い表は飛ばす）
  始めの 6000歩は捨てる（学び始めの区間を混ぜない。50 §2）
  比 ＝ その帯の強さ ÷ 同じ値の並びを歩ごとにでたらめに混ぜた列の強さ（1 ならリズムなし。撃った体の割合の列で、人の場の電位とは別の量）
  人と比べる帯：19〜45Hz（人の中心視床の起きているときの揺れ）・8〜13Hz（PPN の α）
"""
import os, sys, numpy as np
CXN = ["聴覚", "視覚", "連合", "高次", "前頭前野", "補足", "前補足", "眼窩前頭", "前帯状", "44", "45", "STS1", "STS2", "中側頭後", "TPJ",
       "島前", "島後", "縁上回", "海馬傍", "脳梁膨大後"]
BANDS = [("1〜4", 1, 4), ("4〜8", 4, 8), ("8〜13", 8, 13), ("13〜19", 13, 19), ("19〜45", 19, 45)]
NSEG = 256
def psd(x):
    x = x - x.mean(); w = np.hanning(NSEG); st = NSEG // 2
    S = [np.abs(np.fft.rfft(x[i:i + NSEG] * w)) ** 2 for i in range(0, len(x) - NSEG + 1, st)]
    return np.mean(S, 0)
fs = sys.argv[1:] or ["wave111c_cl256.npz", "wave111c_v112.npz"]
rng = np.random.default_rng(112)
for f in fs:
    if not os.path.exists(f):
        print(f"（{f} が無い ⇒ 飛ばす）\n"); continue
    z = np.load(f, allow_pickle=False)
    X = z["rate"].astype(np.float64); names = [str(x) for x in z["names"]]; dt = float(z["dt"])
    X = X[6000:]
    fr = np.fft.rfftfreq(NSEG, dt)
    print(f"━━ {f} ／ 版 {int(z['version'])} ／ 6000歩より後 {len(X)} 歩 ＝ {len(X) * dt:.0f} 秒 ━━")
    print(f"  {'':10s}{'Hz':>7s} ｜ " + "".join(f"{b[0]:>9s}" for b in BANDS) + " ｜ 比が一番高い周波数（5Hz〜）")
    S = [("PPN", X[:, names.index("PPN")]), ("CL", X[:, names.index("CL")]), ("CM", X[:, names.index("CM")]), ("Pf", X[:, names.index("Pf")]),
         ("視床", X[:, names.index("視床")]), ("視床網様核", X[:, names.index("視床網様核")]), ("皮質の平均", X[:, [names.index(k) for k in CXN]].mean(1))]
    for nm, x in S:
        P = psd(x); Q = np.mean([psd(rng.permutation(x)) for _ in range(8)], 0)
        rb = [P[(fr >= a) & (fr < b)].sum() / max(Q[(fr >= a) & (fr < b)].sum(), 1e-30) for _, a, b in BANDS]
        k = np.convolve(P, np.ones(5) / 5, "same") / np.maximum(np.convolve(Q, np.ones(5) / 5, "same"), 1e-30)
        m = (fr >= 5) & (fr <= 45); i = np.where(m)[0][int(np.argmax(k[m]))]
        print(f"  {nm:<10s}{x.mean() / dt:7.2f} ｜ " + "".join(f"{r:9.2f}" for r in rb) + f" ｜ {fr[i]:5.1f}Hz（比 {k[i]:.1f}）")
    print()
