"""版107：人の値で新しい皮質の部位の体数と繋がりを測る（arkhe で。網には触らない。何も書き換えない）
  ~/siibra-venv/bin/python v107_measure.py
  使うもの（どちらも arkhe にある。無ければ取りに行く）
    ~/anaplex/HCP-MMP1_UniqueRegionList.csv   （hcp_vol103 が置いた。Glasser 2016 の 360区画の MNI 体積）
    ~/anaplex/rosen/averageConnectivity_Fpt.csv（rosen_syntax が使った。Rosen & Halgren 2021、HCP 1065人）
  出すもの（data_v107.txt にも書く）
    1 新しい部位の体積（左＋右）と体数 ＝ 高次（A4・A5）224 × 体積比 × 0.55（版103b の縮める側の決まり）
      ⚠️ 版103 の体数は Julich の体積から。HCP の比は確かめ用（Julich に区画が無い部位だけ HCP の比を使う）
    2 新しい部位 × 今の網の領野の繋がり（同じ半球の全部の組の中での百分位、100 が最強。左／右）
      読み方は版103 2b と同じ：93 前後より上を「人の線あり」の目安（10 §7 の rosen の閾値）
  区画の当てはめ【C】（組み表_版107 §1）
"""
import csv, io, os, sys, math, statistics, urllib.request

A = os.path.expanduser("~/anaplex")
OUT = "data_v107.txt"
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

NEW = {"島前": ("AAIC", "AVI", "MI"), "島後": ("Ig", "PoI1", "PoI2"), "縁上回": ("PF", "PFop", "PFt", "PFcm"),
       "腹側運動前": ("6r", "55b"), "海馬傍": ("PeEc", "PHA1", "PHA2", "PHA3"), "脳梁膨大後": ("RSC",)}
OLD = {"聴覚": ("A1",), "連合": ("MBelt", "LBelt", "PBelt"), "高次": ("A4", "A5"), "STS1": ("STSda", "STSdp"),
       "STS2": ("STSva", "STSvp"), "中側頭後": ("TE1p",), "TPJ": ("PSL", "STV", "TPOJ1"), "44": ("44",), "45": ("45",),
       "運動": ("6v",), "前頭前野": ("46", "9-46d", "a9-46v", "p9-46v"), "眼窩前頭": ("11l", "13l", "OFC", "pOFC"),
       "前帯状": ("a24", "p24", "a24pr", "p24pr", "p32", "a32pr", "d32"), "補足": ("6mp",), "前補足": ("6ma",),
       "視覚": ("V1",), "嗅内": ("EC",), "海馬": ("H",), "比べ10pp": ("10pp",)}

# ── 1 体積 ──
p = os.path.join(A, "HCP-MMP1_UniqueRegionList.csv")
if not os.path.exists(p):
    url = "https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv"
    try:
        open(p, "w").write(urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace"))
    except Exception as e:
        sys.exit(f"⚠️ 表が無い・取れない：{e}（hcp_vol103.py と同じ表を ~/anaplex に）")
rows = list(csv.reader(open(p, encoding="utf-8", errors="replace")))
low = [h.lower() for h in rows[0]]
vcol = next(i for i, h in enumerate(low) if "vol" in h)
ncol = next((i for i, h in enumerate(low) if h in ("region", "regionname", "name", "label")),
            next(i for i, h in enumerate(low) if "name" in h or "region" in h))
vol = {}
for r in rows[1:]:
    if len(r) <= max(vcol, ncol):
        continue
    b = r[ncol].strip()
    for pre in ("L_", "R_"):
        b = b[2:] if b.startswith(pre) else b
    for suf in ("_ROI", "_L", "_R"):
        b = b[:-len(suf)] if b.endswith(suf) else b
    try:
        vol[b] = vol.get(b, 0.0) + float(r[vcol])
    except ValueError:
        pass
vh = sum(vol.get(k, 0.0) for k in OLD["高次"])
say(f"1 体積（HCP、左＋右 mm³）と体数（高次 {vh:,.0f} を 224 × 0.55 ＝ 123体として比で）")
for k, regs in list(NEW.items()) + [("（確かめ）44", ("44",)), ("（確かめ）TPJ", OLD["TPJ"])]:
    v = sum(vol.get(x, 0.0) for x in regs)
    miss = [x for x in regs if x not in vol]
    say(f"  {k}（{'・'.join(regs)}）{v:,.0f} ⇒ {224 * 0.55 * v / vh:.0f}体" + (f" ⚠️ 無い区画 {miss}" if miss else ""))
say("  （確かめ：版103 の 44 は Julich で 99体・TPJ は 54体。HCP の比が大きくずれるなら地図の差）")

# ── 2 繋がり ──
D = os.path.join(A, "rosen")
M = [[float(x) for x in l.strip().split(",")] for l in open(os.path.join(D, "averageConnectivity_Fpt.csv")) if l.strip()]
n = len(M)
names = None
try:
    import scipy.io, numpy as np
    m = scipy.io.loadmat(os.path.join(D, "parcelOrder_and_networkAssignment.mat"))
    for k, v in m.items():
        if k.startswith("__"):
            continue
        a = np.array(v, dtype=object).ravel()
        f = []
        for e in a:
            e = np.array(e).ravel()
            f.append(str(e[0]) if e.size == 1 else "".join(chr(int(c)) for c in e) if e.dtype.kind in "iuf" else str(e))
        if len(f) == n and any("V1" in s for s in f):
            names = f; break
except Exception as e:
    say(f"⚠️ 並びの mat が読めない：{str(e)[:100]}")
if names is None:
    import nibabel.freesurfer as fs
    names = []
    for h in ("lh", "rh"):
        _, _, nm = fs.read_annot(os.path.join(D, f"{h}.HCP-MMP1.annot"))
        names += [x.decode() if isinstance(x, bytes) else x for x in nm if (x.decode() if isinstance(x, bytes) else x) not in ("???", "Unknown", "unknown")]
    say("⚠️ 並びは annot から仮に（左→右）")
def key(s):
    s = s.replace("_ROI", "")
    h = "L" if s.startswith(("L_", "lh")) else "R" if s.startswith(("R_", "rh")) else "?"
    return h, s.split("_", 1)[1] if "_" in s else s
K = [key(s) for s in names]
def ix(h, p_):
    return [i for i, (hh, pp) in enumerate(K) if pp == p_ and (hh == h or hh == "?")]
for h in ("L", "R"):
    a, b = ix(h, "MST"), ix(h, "MT")
    if a and b:
        say(f"並びの確かめ {h}：MST⇄MT = {M[a[0]][b[0]]:.2f}（-1 前後なら合っている）")
G = {**NEW, **OLD}
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
    say(f"\n2 {'左' if h == 'L' else '右'}：新しい部位（行）× 相手（列）の百分位（100 が最強）。93 以上に ＊")
    cols = list(G)
    for ga in NEW:
        cells = []
        for gb in cols:
            if gb == ga:
                cells.append(f"{gb}:―"); continue
            v = [M[i][j] for pa in G[ga] for i in ix(h, pa) for pb in G[gb] for j in ix(h, pb) if i != j and not math.isnan(M[i][j])]
            if not v:
                cells.append(f"{gb}:無"); continue
            q = pct(sum(v) / len(v))
            cells.append(f"{gb}:{q:.0f}{'＊' if q >= 93 else ''}")
        say(f"  {ga} ｜ " + " ".join(cells))
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n（{OUT} に書いた）")
