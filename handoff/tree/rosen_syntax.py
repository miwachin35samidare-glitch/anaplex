"""人の拡散の繋がり（Rosen & Halgren 2021、HCP 1065人、log10 Fpt、360×360）で、統語の部位と上側頭溝のまわりを見る（arkhe で）
  ~/siibra-venv/bin/python rosen_syntax.py      （nibabel・scipy・h5py のどれかが要る）
11-01 4:22：averageConnectivity_Fpt.csv は名前の無い 360×360（対角 NaN、左右対称＝向きは無い）
 区画の並びは parcelOrder_and_networkAssignment.mat から取る（読めなければ lh/rh の annot、左 180→右 180 と仮に置き、⚠️ を出す）
 並びの確かめ：MST と MT は隣どうしで強く繋がるはず（-1.1 前後）
 読み方：値そのものより「同じ半球の全部の組の中で何番目に強いか（百分位）」で見る。遠い組（V1⇄45 など）を比べに並べる
 ⚠️ 拡散には向きが無い ⇒ 繋がりのあるなしだけ。向きは型の表
"""
import os, sys, urllib.request, statistics, math
D = os.path.expanduser("~/anaplex/rosen")
BASE = "https://zenodo.org/api/records/10150880/files/"
for k in ("parcelOrder_and_networkAssignment.mat", "lh.HCP-MMP1.annot", "rh.HCP-MMP1.annot"):
    p = os.path.join(D, k)
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + k + "/content", p)
M = [[float(x) for x in l.strip().split(",")] for l in open(os.path.join(D, "averageConnectivity_Fpt.csv")) if l.strip()]
n = len(M)
print(f"表 {n}×{len(M[0])}")

names = None
p = os.path.join(D, "parcelOrder_and_networkAssignment.mat")
def flat(x):
    try:
        import numpy as np
        a = np.array(x, dtype=object).ravel()
        out = []
        for e in a:
            e = np.array(e).ravel()
            out.append(str(e[0]) if e.size == 1 else "".join(chr(int(c)) for c in e) if e.dtype.kind in "iuf" else str(e))
        return out
    except Exception:
        return None
try:
    import scipy.io
    m = scipy.io.loadmat(p)
    print("mat の中身：", {k: getattr(v, "shape", None) for k, v in m.items() if not k.startswith("__")})
    for k, v in m.items():
        if k.startswith("__"):
            continue
        f = flat(v)
        if f and len(f) == n and any("V1" in s for s in f):
            names = f; print(f"並びは {k} から"); break
except Exception as e:
    print("scipy で読めない：", str(e)[:120])
    try:
        import h5py
        with h5py.File(p) as h:
            print("mat(v7.3) の中身：", list(h.keys()))
            for k in h.keys():
                ds = h[k]
                if hasattr(ds, "shape") and n in ds.shape and ds.dtype.kind == "O":
                    f = ["".join(chr(int(c)) for c in h[r][()].ravel()) for r in ds[()].ravel()]
                    if any("V1" in s for s in f):
                        names = f; print(f"並びは {k} から"); break
    except Exception as e2:
        print("h5py でも読めない：", str(e2)[:120])
if names is None:
    import nibabel.freesurfer as fs
    names = []
    for h in ("lh", "rh"):
        _, _, nm = fs.read_annot(os.path.join(D, f"{h}.HCP-MMP1.annot"))
        nm = [x.decode() if isinstance(x, bytes) else x for x in nm]
        nm = [x for x in nm if x not in ("???", "Unknown", "unknown")]
        names += nm
    print(f"⚠️ 並びは annot から仮に（左→右）：{len(names)} 個")
print("頭：", names[:5], " 181番目：", names[180] if len(names) > 180 else None)

def key(s):
    s = s.replace("_ROI", "")
    h = "L" if s.startswith(("L_", "lh")) else "R" if s.startswith(("R_", "rh")) else "?"
    return h, s.split("_", 1)[1] if "_" in s else s
K = [key(s) for s in names]
def ix(h, p):
    return [i for i, (hh, pp) in enumerate(K) if pp == p and (hh == h or hh == "?")]

for h in ("L", "R"):
    a, b = ix(h, "MST"), ix(h, "MT")
    if a and b:
        print(f"並びの確かめ {h}：MST⇄MT = {M[a[0]][b[0]]:.2f}（-1 前後なら並びは合っている）")

GROUPS = {"聴覚": ["A1"], "連合": ["MBelt", "LBelt", "PBelt"], "高次": ["A4", "A5"],
          "STS1": ["STSda", "STSdp"], "STS2": ["STSva", "STSvp"], "中側頭後": ["TE1p"],
          "TPJ": ["PSL", "STV", "TPOJ1"], "44": ["44"], "45": ["45"], "運動": ["6v"],
          "比べV1": ["V1"], "比べ10pp": ["10pp"]}
for h in ("L", "R"):
    hem = [i for i, (hh, _) in enumerate(K) if hh == h]
    allv = sorted(M[i][j] for i in hem for j in hem if i != j and not math.isnan(M[i][j]))
    def pct(x):
        lo, hi = 0, len(allv)
        while lo < hi:
            mid = (lo + hi) // 2
            if allv[mid] < x: lo = mid + 1
            else: hi = mid
        return 100 * lo / len(allv)
    print(f"\n══ {'左' if h == 'L' else '右'}：組の平均 log10 Fpt ／ 同じ半球の全組での百分位（100 が最強）。中央値 {statistics.median(allv):.2f}")
    G = list(GROUPS)
    print(" " * 9 + "".join(f"{g:>11}" for g in G))
    for ga in G:
        cells = []
        for gb in G:
            v = [M[i][j] for pa in GROUPS[ga] for i in ix(h, pa) for pb in GROUPS[gb] for j in ix(h, pb)
                 if i != j and not math.isnan(M[i][j])]
            if not v:
                cells.append(f"{'―':>11}"); continue
            m = sum(v) / len(v)
            cells.append(f"{m:5.2f}/{pct(m):3.0f}")
        print(f"{ga:>8} " + "".join(f"{c:>11}" for c in cells))
print("\n終わり")
