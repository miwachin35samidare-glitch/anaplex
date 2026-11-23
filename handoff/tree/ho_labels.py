"""HarvardOxford の番号ごとの場所（arkhe で。何も書き換えない）
  ~/siibra-venv/bin/python ho_labels.py
11-01 1:44：siibra の HarvardOxford は名前 96（左右別）なのに画像の値は 49種（0 ＋ 48）で、名前と番号の中身が合わない
  （左の下前頭回三角部の中央が x +48・中側頭回の側頭後頭部が y +43 など）
⇒ 名前は使わず、番号 1〜48 ごとに 左（x<0）・右（x>0）の体積と中央の x・y・z を出す。どの番号がどの回かは場所で照らす
"""
import sys
import numpy as np

try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
p = siibra.parcellations.get("HarvardOxford cort maxprob thr25")
m = siibra.get_map(parcellation=p, space="mni152", maptype="labelled")
img = m.fetch()
d = np.asarray(img.dataobj).astype(int)
vox = abs(np.linalg.det(img.affine[:3, :3]))
print(f"画像 {d.shape}・格子 {vox:.2f}mm³", flush=True)
for lab in sorted(set(np.unique(d)) - {0}):
    ijk = np.argwhere(d == lab)
    xyz = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T
    out = []
    for side, sel in (("左", xyz[:, 0] < 0), ("右", xyz[:, 0] > 0)):
        q = xyz[sel]
        if len(q):
            c = np.median(q, axis=0)
            out.append(f"{side} {len(q)*vox:,.0f}mm³ 中央 x {c[0]:.0f} y {c[1]:.0f} z {c[2]:.0f}")
    print(f"  番号 {lab}：" + "　／　".join(out), flush=True)
print("終わり", flush=True)
