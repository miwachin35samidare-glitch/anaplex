"""統語の人の部位が今の網の領野に入っているかを、Julich の地図で確かめる（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python julich_syntax.py
見ること
  - 44野・45野（下前頭回、ブローカ野）と、その周り（下前頭溝 ifs・前頭弁蓋 op）の体積のうち、
    今の前頭前野の切り口 y0 ＝ 45 より前にある割合（前頭前野は「y0 より前の皮質」で数えた。julich_pfc_y0・10 §16）
  - 上側頭の Te3（今の高次）・Te4・上側頭溝 STS1・STS2 の前後の広がり（後ろの上側頭・上側頭溝が高次に入っているか）
  左右ごとに出す（統語は左が主【ヒト】）
"""
import sys
import numpy as np

Y0 = 45.0                                    # 今の前頭前野の切り口（10 §16）
KEYS = ("area 44", "area 45", "area ifs", "area op", "te 3", "te 4", "sts1", "sts2", "te 5", "area 6v")
BINS = np.arange(-130.0, 101.0, 1.0)

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


def ys(region):
    img = region.get_regional_mask(space, maptype="labelled").fetch()
    d = np.asarray(img.get_fdata()) > 0
    vox = abs(np.linalg.det(img.affine[:3, :3]))
    ijk = np.argwhere(d)
    if not len(ijk):
        return np.zeros(0), vox
    xyz = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3]
    return xyz[1], vox


rows = []
for r in parc:
    if r.children:
        continue
    n = r.name.lower()
    if not any(k in n for k in KEYS):
        continue
    try:
        y, vox = ys(r)
    except Exception as e:
        print(f"  印なし：{r.name}（{type(e).__name__}）", flush=True)
        continue
    if not len(y):
        continue
    rows.append((r.name, len(y) * vox, float((y > Y0).mean()), float(np.percentile(y, 5)),
                 float(np.median(y)), float(np.percentile(y, 95))))

print(f"\n前頭前野の切り口 y0 ＝ {Y0:.0f}（y が大きいほど前）", flush=True)
print("領域 ／ 体積 mm³ ／ y0 より前の割合 ／ y の 5% ・中央・95%", flush=True)
for n, v, f, a, m, b in sorted(rows):
    print(f"  {n}：{v:,.0f} ／ {f:.0%} ／ {a:.0f}・{m:.0f}・{b:.0f}", flush=True)
print(f"\n数えた領域 {len(rows)}", flush=True)


def path(r):
    out, p = [], r.parent
    while p is not None:
        out.append(p.name.lower())
        p = p.parent
    return " / ".join(out)


# 中側頭回などの区画が地図にあるか：側頭葉の下の葉の名前を全部出す
tl = sorted({r.name for r in parc if not r.children and "temporal" in path(r)})
print(f"\n側頭葉の区画（{len(tl)}）：" + "／".join(tl), flush=True)
