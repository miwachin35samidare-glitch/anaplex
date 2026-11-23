"""Julich-Brain の地図（MNI152）で、脳梁膝より前（y ＞ y0）の皮質の体積と、その中の眼窩前頭・前帯状の割合を出す
（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python julich_pfc_share.py
なぜ：今の前頭前野 946 は Gabi 2016 の「脳梁膝より前の皮質」の数（約13億）。そこから眼窩前頭（・前帯状）を切り出すとき、
  体積 × 密度で別に数えると、数え方の違い（立体解析と等方分画）で桁が合わない ⇒ 同じ地図の中の体積の割合で分ける【み選】10-25 ア
出すもの
  1 地図の全部の領域の体積（左右込み）と、y ＞ 25・30・35mm の分
  2 そのうち眼窩前頭（Fo1〜7）・前帯状（s24・p24ab・p24c・s32・p32・33・25）の分と割合
  3 地図に印が無い領域（下の葉で取れないものは親の領域で）
"""
import sys
import numpy as np

Y0 = (25.0, 30.0, 35.0)
OFC = ("fo1", "fo2", "fo3", "fo4", "fo5", "fo6", "fo7")
ACC = ("s24", "p24ab", "p24c", "s32", "p32", "area 33", "area 25")
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
parc = None
for name in ("julich 3.1", "julich 3.0.3", "julich 3", "julich"):
    try:
        parc = siibra.parcellations.get(name)
        break
    except Exception as e:
        last = e
if parc is None:
    sys.exit(f"⚠️ 地図が取れない：{last}")
print(f"地図：{parc.name}", flush=True)
space = "mni152"


def vol(region):
    img = region.get_regional_mask(space, maptype="labelled").fetch()
    d = np.asarray(img.get_fdata()) > 0
    vox = abs(np.linalg.det(img.affine[:3, :3]))
    ijk = np.argwhere(d)
    y = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[1] if len(ijk) else np.zeros(0)
    return float(d.sum() * vox), [float((y > y0).sum() * vox) for y0 in Y0]


done, miss, rows = set(), [], []
todo = [r for r in parc if not r.children]
while todo:
    r = todo.pop(0)
    if r.name in done:
        continue
    done.add(r.name)
    try:
        v, va = vol(r)
        rows.append((r.name, v, va))
    except Exception:
        p = r.parent
        if p is not None and p.name not in done and all(c.name in done or c in todo for c in p.children):
            todo.append(p)                      # 下の葉に印が無ければ親で
        else:
            miss.append(r.name)
names = [n for n, _, _ in rows]
print(f"数えた領域 {len(rows)}・印なし {len(miss)}", flush=True)
tot = sum(v for _, v, _ in rows)
for i, y0 in enumerate(Y0):
    a = sum(va[i] for _, _, va in rows)
    o = sum(va[i] for n, _, va in rows if any(k in n.lower() for k in OFC))
    c = sum(va[i] for n, _, va in rows if any(k in n.lower() for k in ACC))
    print(f"y＞{int(y0)}：地図の全部 {a:,.0f}mm³（全体 {tot:,.0f} の {a / tot:.1%}）／ 眼窩前頭 {o:,.0f}（{o / a:.1%}）／ 前帯状 {c:,.0f}（{c / a:.1%}）", flush=True)
o_all = sum(v for n, v, _ in rows if any(k in n.lower() for k in OFC))
c_all = sum(v for n, v, _ in rows if any(k in n.lower() for k in ACC))
print(f"眼窩前頭 全部 {o_all:,.0f}mm³・前帯状 全部 {c_all:,.0f}mm³", flush=True)
print("前帯状に数えた領域：" + "／".join(n for n in names if any(k in n.lower() for k in ACC)), flush=True)
if miss:
    print("印なし：" + "／".join(miss[:40]) + (" …" if len(miss) > 40 else ""), flush=True)
