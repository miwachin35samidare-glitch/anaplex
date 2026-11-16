"""wave111c.npz（measure_wave111c が書いた 歩 × 領野 の表）を読んで、CL の撃ちと皮質のそろいの関係を出す（arkhe の ~/anaplex で。CPU だけ・網は組まない・数秒）
  python3 analyze_wave111c.py [表=wave111c_v111_1116.npz]
  何のため：wave111c で CL が皮質の1歩前に撃ち（合い方 0.79）、皮質 20領野が同じ歩にそろっていた。CL は 5体で、皮質の表現の体はその 5体から 40本ずつ受ける
    （30 §6 の見張り：送り手が重なり、覚醒の入力がそろう）⇒ 皮質のそろいが CL の少なさから来ているかを、取ってある表で見る
  出すもの（始めの 3000歩は捨てる）
    ① CL がその歩に撃った体の数（0〜5）ごとに：歩の割合・次の歩に皮質（運動の外）が撃った体の割合の平均・皮質が1体も撃たなかった歩の割合
    ② 皮質が撃った歩のうち、前の歩に CL が1体以上撃っていた割合 ／ 皮質が黙った歩のうち、前の歩に CL が 0体だった割合
    ③ 同じことを 視床・CM・Pf・背内側視床・視床枕 についても（比べる用：前の歩のその核の撃ちと、次の歩の皮質）
    ④ 皮質が撃った歩で、撃った体の割合の分布（そろい方：1歩に皮質の何割が撃つか）
"""
import sys, numpy as np
f = sys.argv[1] if len(sys.argv) > 1 else "wave111c_v111_1116.npz"
z = np.load(f, allow_pickle=False)
X = z["rate"].astype(np.float64)                      # 歩 × 領野：撃った体の割合
names = [str(x) for x in z["names"]]
CXN = ["聴覚", "視覚", "連合", "高次", "前頭前野", "補足", "前補足", "眼窩前頭", "前帯状", "44", "45", "STS1", "STS2", "中側頭後", "TPJ",
       "島前", "島後", "縁上回", "海馬傍", "脳梁膨大後"]
SKIP = min(3000, len(X) // 5)
X = X[SKIP:]
n = len(X)
cx = X[:, [names.index(k) for k in CXN]].mean(1)       # 皮質（運動の外）：領野の平均
print(f"{f} ／ 版 {int(z['version'])} ／ 残り {n} 歩 ／ 皮質の撃った体の割合 平均 {cx.mean():.4f}・1体も撃たない歩 {float((cx == 0).mean()):.0%}")
cl = X[:, names.index("CL")]
ncl = int(round(1.0 / cl[cl > 0].min())) if (cl > 0).any() else 0
k = np.rint(cl * ncl).astype(int)
print(f"\n① CL（{ncl} 体）がその歩に撃った体の数ごと ⇒ 次の歩の皮質")
print(f"  {'CL の数':>6s}{'歩の割合':>8s}{'次の歩の皮質の割合':>14s}{'皮質が黙った':>10s}")
for v in range(ncl + 1):
    m = k[:-1] == v
    if m.any():
        nx = cx[1:][m]
        print(f"  {v:6d}{m.mean():8.1%}{nx.mean():14.4f}{float((nx == 0).mean()):10.0%}")
on = cx[1:] > 0
print(f"\n② 皮質が撃った歩 {on.mean():.0%} のうち、前の歩に CL が1体以上 {float((k[:-1][on] > 0).mean()):.0%} ／ 皮質が黙った歩のうち、前の歩に CL が 0体 {float((k[:-1][~on] == 0).mean()):.0%}")
print(f"   前の歩の CL の数 と 次の歩の皮質の割合 の合い方 {np.corrcoef(k[:-1], cx[1:])[0, 1]:.2f} ／ 同じ歩 {np.corrcoef(k, cx)[0, 1]:.2f}")
print("\n③ 比べる用：前の歩にその核が撃った（1体以上）か ⇒ 次の歩の皮質")
print(f"  {'核':<8s}{'撃った歩':>7s}{'撃った次の皮質':>12s}{'撃たない次の皮質':>14s}{'合い方':>7s}")
for nm in ("CL", "視床", "CM", "Pf", "背内側視床", "視床枕", "VA", "VLa", "中線核", "視床前核", "PPN"):
    if nm in names:
        v = X[:, names.index(nm)]
        a = v[:-1] > 0
        c = np.corrcoef(v[:-1], cx[1:])[0, 1] if v.std() > 0 else float("nan")
        print(f"  {nm:<8s}{a.mean():7.0%}{cx[1:][a].mean() if a.any() else float('nan'):12.4f}{cx[1:][~a].mean() if (~a).any() else float('nan'):14.4f}{c:7.2f}")
print("\n④ 皮質が撃った歩で、皮質の何割が撃ったか（歩の割合）")
v = cx[cx > 0]
ed = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0001]
h = np.histogram(v, ed)[0] / max(1, len(v))
print("  " + " ・ ".join(f"{ed[i]:.0%}〜{min(ed[i + 1], 1):.0%}：{h[i]:.0%}" for i in range(len(h))))
print(f"  中央 {np.median(v):.3f}・平均 {v.mean():.3f}・一番多い歩 {v.max():.3f}")
