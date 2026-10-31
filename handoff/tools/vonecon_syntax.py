"""統語の部位の体積（中側頭回の後ろ）と、von Economo の区画（＝皮質の型）への当てはめ（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python vonecon_syntax.py
11-01：HCP-MMP は siibra に無かった（hcp_mtg.py）。siibra にある人の地図で代わりに引く
見ること
  A HarvardOxford（人の MRI の地図）の中側頭回・上側頭回・下前頭回などの体積と y の中央（MNI152、mm³）
    ⇒ 中側頭回の後ろの体積。同じ地図の下前頭回の弁蓋部・三角部を Julich の 44・45 と並べて、地図の違いの大きさを見る
  B von Economo の地図（siibra の Von Economo Koskinas Brain Atlas）に、Julich v3.1 の領域と HarvardOxford の中側頭回・上側頭回が
    どの区画で何割重なるか ⇒ 型の表（García-Cabezas 2020）の区画に当てる（運動 6v が FBop か FAop か・上側頭溝・中側頭回の後ろ）
  重なりは 1mm に丸めた MNI の座標の集合どうしで数える（地図ごとに格子が違ってもよいように）
"""
import sys
import numpy as np

try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
print(f"siibra {getattr(siibra, '__version__', '?')}", flush=True)
SPACE = "mni152"
JU_KEYS = ("area 44", "area 45", "sts1", "sts2", "area tpj", "te 3", "area 6v", "area op", "te 4", "te 5")
HO_KEYS = ("middle temporal", "superior temporal", "inferior frontal", "supramarginal", "angular", "temporal pole", "precentral")


def pts(region):
    img = region.get_regional_mask(SPACE, maptype="labelled").fetch()
    d = np.asarray(img.get_fdata()) > 0
    ijk = np.argwhere(d)
    vox = abs(np.linalg.det(img.affine[:3, :3]))
    if not len(ijk):
        return set(), 0.0, 0.0
    xyz = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T
    s = set(map(tuple, np.round(xyz).astype(int)))
    return s, len(ijk) * vox, float(np.median(xyz[:, 1]))


def getp(names):
    last = None
    for n in names:
        try:
            return siibra.parcellations.get(n)
        except Exception as e:
            last = e
    print(f"⚠️ 地図が取れない：{names}（{last}）", flush=True)
    return None


def leaves(parc, keys):
    out = []
    for r in parc:
        if r.children:
            continue
        if any(k in r.name.lower() for k in keys):
            out.append(r)
    return out


# ── A HarvardOxford
ho = getp(("HarvardOxford cort maxprob thr25", "harvardoxford cort maxprob thr25", "harvardoxford"))
hopts = {}
if ho is not None:
    print(f"\nA 地図：{ho.name}", flush=True)
    for r in leaves(ho, HO_KEYS):
        try:
            s, v, ym = pts(r)
        except Exception as e:
            print(f"  {r.name}：取れない（{type(e).__name__}：{e}）", flush=True)
            continue
        hopts[r.name] = s
        print(f"  {r.name}：{v:,.0f}mm³ ／ y 中央 {ym:.0f}", flush=True)

# ── B von Economo
ve = getp(("Von Economo Koskinas Brain Atlas", "von economo", "economo"))
if ve is None:
    sys.exit("⚠️ von Economo の地図が取れない（Claude に返す）")
print(f"\nB 地図：{ve.name}", flush=True)
try:
    print("  この地図の空間：" + "／".join(str(s.name) for s in ve.spaces), flush=True)
except Exception:
    pass
vepts = {}
for r in ve:
    if r.children:
        continue
    try:
        s, v, ym = pts(r)
    except Exception as e:
        print(f"  {r.name}：取れない（{type(e).__name__}：{e}）", flush=True)
        continue
    if s:
        vepts[r.name] = s
print(f"  von Economo の区画（MNI で取れた数 {len(vepts)}）：" + "／".join(sorted(vepts)), flush=True)
if not vepts:
    sys.exit("⚠️ von Economo の区画が MNI152 で取れない（Claude に返す）")


def overlap(name, s):
    if not s:
        return
    hits = sorted(((len(s & t) / len(s), k) for k, t in vepts.items() if s & t), reverse=True)
    top = "　".join(f"{k} {f:.0%}" for f, k in hits[:4])
    print(f"  {name}（{len(s):,} 点）：{top if top else '重なりなし'}", flush=True)


ju = getp(("julich 3.1", "julich 3.0.3", "julich"))
if ju is not None:
    print(f"\nB-1 Julich（{ju.name}）の領域が von Economo のどの区画に何割入るか", flush=True)
    for r in sorted(leaves(ju, JU_KEYS), key=lambda r: r.name):
        try:
            s, v, ym = pts(r)
        except Exception as e:
            print(f"  {r.name}：取れない（{type(e).__name__}）", flush=True)
            continue
        overlap(r.name, s)
if hopts:
    print("\nB-2 HarvardOxford の領域が von Economo のどの区画に何割入るか", flush=True)
    for k in sorted(hopts):
        overlap(k, hopts[k])
print("\n終わり", flush=True)
