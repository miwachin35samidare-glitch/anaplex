"""Julich-Brain の最大確率地図で、眼窩前頭（Fo1〜Fo7）と前帯状（s24・p24・s32・p32・33・25）の体積を数える
（arkhe で。網は使わない。何も書き換えない。地図は EBRAINS から取る。julich_volume.py と同じ取り方）
  ~/siibra-venv/bin/python julich_ofc.py   （siibra は 10-16 に作った箱 ~/siibra-venv に入っている。無ければ python3 -m venv ~/siibra-venv && ~/siibra-venv/bin/pip install siibra）
出すもの
  1 領域ごとの体積（mm³、MNI152 と Colin27、左右込み）と、確かめ用の Te1（1.0＋1.1＋1.2）との比
  2 MNI152 で脳梁膝より前（y ＞ y0）にある体積の割合（y0 ＝ 25・30・35mm）
    ⇐ 今の前頭前野（Gabi 2016）は「脳梁膝より前の皮質」⇒ 切り出す眼窩前頭のうち、前頭前野の数に入っていた分を引くため
【み選】10-25 22:41 ア（眼窩前頭は前頭前野から人の区分のとおり切り出す）
⚠️ Claude はこの道具を動かせていない（コンテナから EBRAINS に繋がらない）
"""
import sys
import numpy as np

KEYS = ["fo1", "fo2", "fo3", "fo4", "fo5", "fo6", "fo7",
        "s24", "p24", "s32", "p32", "area 33", "area 25",
        "te 1.0", "te 1.1", "te 1.2"]
Y0 = (25.0, 30.0, 35.0)
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python julich_ofc.py で回す（箱が無ければ python3 -m venv ~/siibra-venv の後 ~/siibra-venv/bin/pip install siibra）")
print(f"siibra {getattr(siibra, '__version__', '?')}", flush=True)
parc = None
for name in ("julich 3.1", "julich 3.0.3", "julich 3", "julich 2.9", "julich"):
    try:
        parc = siibra.parcellations.get(name)
        break
    except Exception as e:
        last = e
if parc is None:
    sys.exit(f"⚠️ Julich-Brain の地図が取れない：{last}")
print(f"地図：{parc.name}", flush=True)
allr = [r for r in parc if not r.children]
hits = []
for k in KEYS:
    found = [r for r in allr if k in r.name.lower()]
    if not found:
        print(f"  ⚠️ 「{k}」に当たる領域なし", flush=True)
    hits += [r for r in found if r not in hits]
print(f"数える領域 {len(hits)}：" + "／".join(r.name for r in hits), flush=True)


def image(region, space):
    errs = []
    for how in ("mask", "regional", "map"):
        try:
            if how == "mask":
                return region.get_regional_mask(space, maptype="labelled").fetch(), how
            if how == "regional":
                return region.fetch_regional_map(space, maptype="labelled"), how
            mp = siibra.get_map(parcellation=parc, space=space, maptype="labelled")
            return mp.fetch(index=mp.get_index(region)), how
        except Exception as e:
            errs.append(f"{how}: {type(e).__name__} {str(e)[:80]}")
    raise RuntimeError(" ／ ".join(errs))


for space in ("mni152", "colin27"):
    print(f"\n座標 {space}（最大確率の印、mm³）", flush=True)
    tot, ant = {}, {}
    for r in hits:
        try:
            img, how = image(r, space)
            d = np.asarray(img.get_fdata()) > 0
            vox = abs(np.linalg.det(img.affine[:3, :3]))
            v = float(d.sum() * vox)
            tot[r.name] = v
            line = f"  {r.name:<48} {v:9,.0f}"
            if space == "mni152":
                ijk = np.argwhere(d)
                y = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[1] if len(ijk) else np.zeros(0)
                fr = [float((y > y0).mean()) if len(y) else 0.0 for y0 in Y0]
                ant[r.name] = fr
                line += "  膝より前 " + "・".join(f"y>{int(y0)} {f:.0%}" for y0, f in zip(Y0, fr))
            print(line + f"（{how}）", flush=True)
        except Exception as e:
            print(f"  {r.name:<48} ✗ {e}", flush=True)
    te1 = sum(v for n, v in tot.items() if "te 1." in n.lower())
    if te1:
        print(f"  ⇒ Te1（左右込み）{te1:,.0f}mm³（前の地図 5,917）", flush=True)
        for lab, keys in (("眼窩前頭 Fo1〜7", ("fo",)), ("前帯状（s24・p24・s32・p32・33・25）", ("s24", "p24", "s32", "p32", "area 33", "area 25"))):
            sel = [n for n in tot if any(k in n.lower() for k in keys)]
            v = sum(tot[n] for n in sel)
            if v:
                print(f"  ⇒ {lab} {v:,.0f}mm³ ／ Te1 の {v / te1:.2f}倍", flush=True)
                if space == "mni152":
                    for i, y0 in enumerate(Y0):
                        va = sum(tot[n] * ant[n][i] for n in sel if n in ant)
                        print(f"      y>{int(y0)} の分 {va:,.0f}mm³（{va / v:.0%}）", flush=True)
