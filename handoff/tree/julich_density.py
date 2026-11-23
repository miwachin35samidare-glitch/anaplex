"""統語の部位の体数を出すための、人の細胞の密度と体積を siibra で引く（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python julich_density.py
見ること
  1 体積（最大確率地図、左右）：44・45・STS1・STS2・TPJ・Te3・Te1.0〜1.2（Te1 は今の網の出どころ 5,917mm³ との比べ用）
  2 細胞の密度：siibra の「LayerwiseCellDensity」「CellDensityProfile」（BigBrain・死後脳の層ごとの細胞の密度）があれば全部出す
     Te1 も出す ⇒ 今の網が使った人の密度（A1 42,069/mm³・Tpt 34,147/mm³）との比で使う（単位・数え方の違いを比で消す）
  ⚠️ Claude はこの道具を動かせていない（コンテナから EBRAINS に繋がらない）。siibra の版で呼び方が違うので、いくつか試して見つかったものを全部出す
"""
import sys
import numpy as np

KEYS = ("area 44", "area 45", "sts1", "sts2", "area tpj", "te 3", "te 1.0", "te 1.1", "te 1.2")
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
print(f"siibra {getattr(siibra, '__version__', '?')}", flush=True)
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

leaves = [r for r in parc if not r.children and any(k in r.name.lower() for k in KEYS)]
print(f"\n1 体積（mm³、最大確率地図・MNI152）", flush=True)
for r in sorted(leaves, key=lambda x: x.name):
    try:
        img = r.get_regional_mask("mni152", maptype="labelled").fetch()
        v = (np.asarray(img.get_fdata()) > 0).sum() * abs(np.linalg.det(img.affine[:3, :3]))
        print(f"  {r.name}：{v:,.0f}", flush=True)
    except Exception as e:
        print(f"  {r.name}：取れない（{type(e).__name__}）", flush=True)

print(f"\n2 細胞の密度", flush=True)
mods = ("LayerwiseCellDensity", "CellDensityProfile", "cell density", "celldensity")
parents = {}
for r in leaves:                                   # 左右の葉 ⇒ 親（左右込みの名前）でも探す
    parents[r.name] = r
    if r.parent is not None:
        parents[r.parent.name] = r.parent
for name, r in sorted(parents.items()):
    found = False
    for m in mods:
        try:
            fs = siibra.features.get(r, m)
        except Exception as e:
            continue
        for f in fs or []:
            found = True
            print(f"  ── {name} ／ {m} ／ {getattr(f, 'name', type(f).__name__)}", flush=True)
            for attr in ("unit", "modality", "subject", "anchor"):
                if hasattr(f, attr):
                    try:
                        print(f"     {attr}：{getattr(f, attr)}", flush=True)
                    except Exception:
                        pass
            try:
                d = f.data
                print("     " + str(d).replace("\n", "\n     ")[:1500], flush=True)
            except Exception as e:
                print(f"     data が取れない（{type(e).__name__}：{e}）", flush=True)
        if found:
            break
    if not found:
        print(f"  {name}：密度の特徴なし", flush=True)
