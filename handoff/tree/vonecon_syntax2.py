"""vonecon_syntax.py の出直し（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python vonecon_syntax2.py
11-01 1:38 直し：probe で、fetch(index=…) が区画だけの 0/1 の画像を返すと分かった（番号 1000 などで切ると空）⇒ 丸ごと取って番号で切る。Julich は左右が別の画像（fragment）なので半球ごとに取る
11-01 1:23 の結果に2つおかしい所があった
  ① HarvardOxford：下前頭回の y 中央が 0 より後ろ（前頭葉なのに）・左の角回が左右両方の Broca に重なる
     ⇒ 区画ごとの切り出し（get_regional_mask）が名前と中身でずれている疑い。地図を丸ごと取り、番号で切り直す
  ② Julich の点のうち von Economo のどこかに入るのが 1割前後しかない
     ⇒ von Economo の地図の格子・厚みを出して、入らない点がどこにあるかを見る。3mm 以内の一番近い区画も数える
見ること
  A HarvardOxford：区画ごとの x・y・z の中央（左は x<0、下前頭回は y>0 になるはず）と体積
  B von Economo：格子の大きさ・区画の点の合計
  C Julich の領域：von Economo に入る割合／入った点の内訳／入らない点を 3mm 以内の一番近い区画に寄せた内訳
"""
import sys
import numpy as np

try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
print(f"siibra {getattr(siibra, '__version__', '?')}", flush=True)
SPACE = "mni152"
JU_KEYS = ("area 44", "area 45", "sts1", "sts2", "area tpj", "te 3", "area 6v")
HO_KEYS = ("middle temporal", "inferior frontal", "superior temporal")


def xyz_of(img, mask):
    ijk = np.argwhere(mask)
    if not len(ijk):
        return np.zeros((0, 3))
    return (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T


def labelled_sets(parc, keys=None):
    """地図を丸ごと取り、番号で区画を切る。{名前: (点の集合, 体積mm³, 中央xyz)}"""
    m = siibra.get_map(parcellation=parc, space=SPACE, maptype="labelled")
    out, cache = {}, {}
    for r in m.regions:
        if keys and not any(k in r.lower() for k in keys):
            continue
        try:
            idx = m.get_index(r)
            frag = getattr(idx, "fragment", None)
            fk = (idx.volume, frag)
            if fk not in cache:
                # fetch(index=…) は区画だけの 0/1 の画像を返すので番号で切れない（11-01 1:38 probe）。丸ごと（半球ごと）取る
                cache[fk] = m.fetch(fragment=frag) if frag else m.fetch()
                a = cache[fk].affine
                print(f"  （{parc.name[:30]} {frag or ''}：画像 {cache[fk].shape}・原点 {a[:3, 3].round(1).tolist()}・向き {np.diag(a)[:3].round(2).tolist()}）", flush=True)
            img = cache[fk]
            d = np.asarray(img.dataobj)
            mask = d == idx.label if idx.label is not None else d > 0
            xyz = xyz_of(img, mask)
            vox = abs(np.linalg.det(img.affine[:3, :3]))
        except Exception as e:
            print(f"  {r}：取れない（{type(e).__name__}：{e}）", flush=True)
            continue
        if not len(xyz):
            continue
        s = set(map(tuple, np.round(xyz).astype(int)))
        out[r] = (s, len(xyz) * vox, np.median(xyz, axis=0), vox)
    return out


def getp(names):
    for n in names:
        try:
            return siibra.parcellations.get(n)
        except Exception:
            pass
    print(f"⚠️ 地図が取れない：{names}", flush=True)
    return None


# ── A
ho = getp(("HarvardOxford cort maxprob thr25", "harvardoxford"))
if ho is not None:
    print(f"\nA {ho.name}（丸ごと取って番号で切る）", flush=True)
    for k, (s, v, c, vox) in sorted(labelled_sets(ho, HO_KEYS).items()):
        print(f"  {k}：{v:,.0f}mm³ ／ 中央 x {c[0]:.0f} y {c[1]:.0f} z {c[2]:.0f}", flush=True)

# ── B
ve = getp(("Von Economo Koskinas Brain Atlas", "von economo"))
if ve is None:
    sys.exit("⚠️ von Economo の地図が取れない")
vesets = labelled_sets(ve)
vesets = {k: v for k, v in vesets.items() if "unknown" not in k.lower()}
allve = {}
for k, (s, *_ ) in vesets.items():
    for p in s:
        allve[p] = k
vox = next(iter(vesets.values()))[3] if vesets else 0
print(f"\nB {ve.name}：区画 {len(vesets)}・格子 1 つ {vox:.2f}mm³・点の合計 {len(allve):,}", flush=True)
for k in ("FCBm", "FDT", "FA:", "FB:", "TA:", "TE:"):
    for name, (s, v, c, _) in vesets.items():
        if name.startswith(k) and name.endswith("left"):
            print(f"  {name[:40]}：{v:,.0f}mm³ ／ 中央 x {c[0]:.0f} y {c[1]:.0f} z {c[2]:.0f}", flush=True)

OFF = [(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3, 4) if a*a + b*b + c*c <= 9]
OFF.sort(key=lambda o: o[0]**2 + o[1]**2 + o[2]**2)


def nearest(p):
    for o in OFF:
        q = (p[0] + o[0], p[1] + o[1], p[2] + o[2])
        if q in allve:
            return allve[q]
    return None


def short(k):
    return k.split(":")[0] + (" 左" if k.endswith("left") else " 右")


def show(cnt, n):
    tot = sum(cnt.values())
    top = sorted(cnt.items(), key=lambda t: -t[1])[:4]
    return "　".join(f"{short(k)} {c/tot:.0%}" for k, c in top) if tot else "なし"


# ── C
ju = getp(("julich 3.1", "julich"))
if ju is not None:
    print(f"\nC {ju.name}：入る割合／入った点の内訳／3mm 以内に寄せた内訳（寄せられた割合）", flush=True)
    for k, (s, v, c, _) in sorted(labelled_sets(ju, JU_KEYS).items()):
        inside, near, miss = {}, {}, 0
        for p in s:
            a = allve.get(p)
            if a:
                inside[a] = inside.get(a, 0) + 1
                near[a] = near.get(a, 0) + 1
                continue
            b = nearest(p)
            if b:
                near[b] = near.get(b, 0) + 1
            else:
                miss += 1
        n = len(s)
        print(f"  {k}（{n:,} 点・中央 x {c[0]:.0f} y {c[1]:.0f} z {c[2]:.0f}）", flush=True)
        print(f"    入る {sum(inside.values())/n:.0%}：{show(inside, n)}", flush=True)
        print(f"    寄せる {(n-miss)/n:.0%}：{show(near, n)}", flush=True)
print("\n終わり", flush=True)
