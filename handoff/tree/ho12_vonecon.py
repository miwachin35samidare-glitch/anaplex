"""中側頭回の後ろ（HarvardOxford 番号 12）が von Economo のどの区画に入るか（arkhe で。何も書き換えない）
  ~/siibra-venv/bin/python ho12_vonecon.py
11-01 2:10：中側頭回の後ろの型を引く。von Economo の中側頭回は TE1（中側頭回の固有野、Eu-III）。後ろは PH（側頭後頭の頭頂の底）へ移る
  ⇒ 番号 12 の点が TE と PH（ほか）にどれだけ入るかを見る。番号は場所で照らした並び（ho_labels、11-01 1:50）
"""
import sys
import numpy as np
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
SPACE = "mni152"
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



ve = siibra.parcellations.get("Von Economo Koskinas Brain Atlas")
vesets = {k: v for k, v in labelled_sets(ve).items() if "unknown" not in k.lower()}
allve = {}
for k, (st, *_ ) in vesets.items():
    for p in st:
        allve[p] = k
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



ho = siibra.parcellations.get("HarvardOxford cort maxprob thr25")
m = siibra.get_map(parcellation=ho, space=SPACE, maptype="labelled")
img = m.fetch()
d = np.asarray(img.dataobj).astype(int)
for lab in (12, 13):
    xyz = xyz_of(img, d == lab)
    for side, sel in (("左", xyz[:, 0] < 0), ("右", xyz[:, 0] > 0)):
        st = set(map(tuple, np.round(xyz[sel]).astype(int)))
        inside, near, miss = {}, {}, 0
        for p in st:
            a = allve.get(p)
            if a:
                inside[a] = inside.get(a, 0) + 1
            b = a or nearest(p)
            if b:
                near[b] = near.get(b, 0) + 1
            else:
                miss += 1
        n = len(st)
        print(f"番号 {lab} {side}（{n:,} 点）", flush=True)
        print(f"  入る {sum(inside.values())/n:.0%}：{show(inside, n)}", flush=True)
        print(f"  寄せる {(n-miss)/n:.0%}：{show(near, n)}", flush=True)
print("終わり", flush=True)
