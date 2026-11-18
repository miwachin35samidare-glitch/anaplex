"""視床網様核の率が版112 で上がった所を、取ってある 歩 × 領野 の表で見る（arkhe の ~/anaplex で。CPU だけ・網は組まない・GPU を使わない）
  python3 analyze_trn112.py [表 ...]   既定：wave111c_cl256.npz wave111c_v112.npz（無い表は飛ばす）
  出すもの（始めの 3000歩は捨てる）
    ① 視床網様核の率：3000歩ごと（下がっていくか・留まるか）
    ② 皮質が一度に撃つ歩（皮質の 20% 以上）の 次の歩・その次の歩 と、それ以外の歩に分けた率と、発火の入る割合
    ③ 視床（11 の核の平均）・皮質の平均の率（比べる用）
"""
import os, sys, numpy as np
CXN = ["聴覚", "視覚", "連合", "高次", "前頭前野", "補足", "前補足", "眼窩前頭", "前帯状", "44", "45", "STS1", "STS2", "中側頭後", "TPJ",
       "島前", "島後", "縁上回", "海馬傍", "脳梁膨大後"]
THN = ["視床", "背内側視床", "CL", "CM", "Pf", "VA", "VLa", "VLp", "視床枕", "中線核", "視床前核"]
fs = sys.argv[1:] or ["wave111c_cl256.npz", "wave111c_v112.npz"]
for f in fs:
    if not os.path.exists(f):
        print(f"（{f} が無い ⇒ 飛ばす）\n"); continue
    z = np.load(f, allow_pickle=False)
    X = z["rate"].astype(np.float64); names = [str(x) for x in z["names"]]; dt = float(z["dt"])
    X = X[min(3000, len(X) // 5):]; n = len(X)
    hz = lambda v: float(np.mean(v)) / dt
    t = X[:, names.index("視床網様核")]; cx = X[:, [names.index(k) for k in CXN]].mean(1); th = X[:, [names.index(k) for k in THN]].mean(1)
    print(f"━━ {f} ／ 版 {int(z['version'])} ／ 残り {n} 歩 ━━")
    print(f"  視床網様核 {hz(t):.2f}Hz（目標 8.2）・視床 11 核の平均 {hz(th):.2f}Hz・皮質 {hz(cx):.2f}Hz")
    print("① 3000歩ごとの視床網様核 Hz：" + "・".join(f"{hz(t[i:i + 3000]):.1f}" for i in range(0, n - 2999, 3000)))
    b = cx >= 0.2
    b1 = np.zeros(n, bool); b1[1:] = b[:-1]
    b2 = np.zeros(n, bool); b2[2:] = b[:-2]
    near = b | b1 | b2; rest = ~near
    print(f"② 皮質が一度に撃つ歩 {b.mean():.1%}")
    for lab, m in (("一度に撃つ歩そのもの", b), ("その次の歩", b1 & ~b), ("その次の次の歩", b2 & ~b & ~b1), ("それ以外の歩", rest)):
        print(f"   {lab:<12s} 歩の {m.mean():5.1%} ｜ 視床網様核 {hz(t[m]) if m.any() else 0.0:6.2f}Hz ｜ 発火の入る割合 {t[m].sum() / max(t.sum(), 1e-12):5.1%}")
    print()
