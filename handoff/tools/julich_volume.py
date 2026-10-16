"""Julich-Brain の最大確率地図で、領域の体積（mm³）を数える（arkhe で。網は使わない。何も書き換えない。地図は EBRAINS から取る）
  pip install siibra        （はじめに1回。入らなければ pip install --user siibra）
  python3 julich_volume.py                 補足運動野（6mc）・前補足運動野（6ma）と、確かめ用の聴覚 Te1.0・1.1・1.2
  python3 julich_volume.py 6m TE           名前に含む文字で探して数える（大文字小文字は問わない）
【み】10-16 21:31「いいよ」（補足運動野の体数を、模型の皮質と同じやり方 ＝ Julich-Brain の最大確率地図の体積 × 人の密度 で出す）
前に使った値（torch_brain.py）：Te1.0 1,822 ＋ Te1.1 2,687 ＋ Te1.2 1,408 ＝ 5,917mm³（左右込み、死後脳10体、1mm等方）
  ⇒ 同じ地図・同じ数え方なら Te1 が 5,917 前後に出る。出なければ版か座標が違う（数字はそのまま比べず、Te1 との比で使う）
⚠️ Claude はこの道具を動かせていない（コンテナから EBRAINS に繋がらない）。siibra の版で呼び方が違うので、いくつかの呼び方を順に試す
"""
import sys
import numpy as np

KEYS = [a for a in sys.argv[1:]] or ["6mc", "6ma", "TE 1.0", "TE 1.1", "TE 1.2"]
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：pip install siibra（だめなら pip install --user siibra）")
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

# 名前で領域を探す（左・右・両方の葉だけ）
def leaves():
    out = []
    for r in parc:
        if not r.children:
            out.append(r)
    return out

allr = leaves()
hits = []
for k in KEYS:
    kk = k.lower()
    found = [r for r in allr if kk in r.name.lower()]
    if not found:
        print(f"  ⚠️ 「{k}」に当たる領域なし", flush=True)
    hits += [r for r in found if r not in hits]
print(f"数える領域 {len(hits)}：" + "／".join(r.name for r in hits), flush=True)


def voxels(region, space):
    """領域の最大確率の印（labelled）の体積 mm³。呼び方を順に試す"""
    errs = []
    for how in ("mask", "regional", "map"):
        try:
            if how == "mask":
                vol = region.get_regional_mask(space, maptype="labelled")
                img = vol.fetch()
            elif how == "regional":
                img = region.fetch_regional_map(space, maptype="labelled")
            else:
                mp = siibra.get_map(parcellation=parc, space=space, maptype="labelled")
                idx = mp.get_index(region)
                img = mp.fetch(index=idx)
            d = np.asarray(img.get_fdata())
            vox = abs(np.linalg.det(img.affine[:3, :3]))
            return float((d > 0).sum() * vox), how
        except Exception as e:
            errs.append(f"{how}: {type(e).__name__} {str(e)[:80]}")
    raise RuntimeError(" ／ ".join(errs))


for space in ("mni152", "colin27"):
    print(f"\n座標 {space}（最大確率の印、mm³）", flush=True)
    tot = {}
    for r in hits:
        try:
            v, how = voxels(r, space)
            print(f"  {r.name:<45} {v:10,.0f}（{how}）", flush=True)
            tot[r.name] = v
        except Exception as e:
            print(f"  {r.name:<45} ✗ {e}", flush=True)
    te1 = sum(v for n, v in tot.items() if "te 1." in n.lower())
    if te1:
        print(f"  ⇒ Te1（1.0＋1.1＋1.2、左右込み）{te1:,.0f}mm³（前に使った値 5,917）", flush=True)
        for key, lab in (("6mc", "補足運動野 6mc"), ("6ma", "前補足運動野 6ma")):
            v = sum(x for n, x in tot.items() if key in n.lower())
            if v:
                print(f"  ⇒ {lab}（左右込み）{v:,.0f}mm³ ／ Te1 の {v / te1:.2f}倍", flush=True)
