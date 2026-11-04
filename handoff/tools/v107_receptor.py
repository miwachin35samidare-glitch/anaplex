"""版107 §11：人の受容体・終末の地図を、網の皮質の領野ごとに平均する（arkhe で。網には触らない）
  ~/siibra-venv/bin/pip install neuromaps nibabel      （はじめに1回）
  ~/siibra-venv/bin/python v107_receptor.py
  地図（neuromaps が配る人の PET の地図）
    beliveau2017：5-HT1A・5-HT2A ほか（210人）     ⇒ セロトニンで膜を下げる量・上げる量（組み表_版107 §11b）
    aghourian2017：FEOBV（コリンの終末）            ⇒ アセチルコリンの量（§11c）
  区画：~/anaplex/rosen/lh.HCP-MMP1.annot・rh.HCP-MMP1.annot（rosen_syntax と同じ。fsaverage 164k の頂点に付く）
  出すもの：地図ごと × 網の領野ごとの平均（左／右）と、全皮質の平均との比（data_v107_receptor.txt）
"""
import os, sys
import numpy as np

A = os.path.expanduser("~/anaplex")
OUT = "data_v107_receptor.txt"
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

try:
    from neuromaps.datasets import fetch_annotation, available_annotations
    import nibabel as nib
    import nibabel.freesurfer as fs
except ImportError as e:
    sys.exit(f"⚠️ {e}：~/siibra-venv/bin/pip install neuromaps nibabel")

WANT = [a for a in available_annotations() if a[0] in ("beliveau2017", "aghourian2017")]
say("使える地図：" + " ／ ".join("・".join(a) for a in WANT))
if not WANT:
    sys.exit("⚠️ beliveau2017・aghourian2017 が neuromaps に無い（版を確かめる）")

AREAS = {"聴覚": ("A1",), "連合": ("MBelt", "LBelt", "PBelt"), "高次": ("A4", "A5"), "STS1": ("STSda", "STSdp"),
         "STS2": ("STSva", "STSvp"), "中側頭後": ("TE1p",), "TPJ": ("PSL", "STV", "TPOJ1"), "44": ("44",), "45": ("45",),
         "運動": ("6v",), "前頭前野": ("46", "9-46d", "a9-46v", "p9-46v"), "眼窩前頭": ("11l", "13l", "OFC", "pOFC"),
         "前帯状": ("a24", "p24", "a24pr", "p24pr", "p32", "a32pr", "d32"), "補足": ("6mp",), "前補足": ("6ma",),
         "視覚": ("V1",), "嗅内": ("EC",), "島前": ("AAIC", "AVI", "MI"), "島後": ("Ig", "PoI1", "PoI2"),
         "縁上回": ("PF", "PFop", "PFt", "PFcm"), "腹側運動前": ("6r", "55b"), "海馬傍": ("PeEc", "PHA1", "PHA2", "PHA3"),
         "脳梁膨大後": ("RSC",)}

lab = {}
for h in ("lh", "rh"):
    v, ctab, nm = fs.read_annot(os.path.join(A, "rosen", f"{h}.HCP-MMP1.annot"))
    nm = [x.decode() if isinstance(x, bytes) else x for x in nm]
    lab[h] = (v, nm)
    say(f"区画 {h}：頂点 {len(v)}（fsaverage 164k なら 163842）")

def base(n):
    n = n.replace("_ROI", "")
    return n.split("_", 1)[1] if n[:2] in ("L_", "R_") else n

for a in WANT:
    try:
        f = fetch_annotation(source=a[0], desc=a[1], space=a[2], den=a[3])
    except Exception as e:
        say(f"⚠️ {'・'.join(a)} が取れない：{str(e)[:120]}"); continue
    if a[2] != "fsaverage":
        say(f"（{'・'.join(a)} は {a[2]} の地図 ⇒ 皮質の区画では平均しない。皮質の外は次の道具で）"); continue
    files = f if isinstance(f, (list, tuple)) else [f]
    if len(files) != 2:
        say(f"⚠️ {'・'.join(a)} の左右のファイルが2つでない：{files}"); continue
    say(f"\n══ {'・'.join(a)}")
    res = {}
    for h, fn in zip(("lh", "rh"), files):
        d = np.asarray(nib.load(fn).agg_data()).ravel()
        v, nm = lab[h]
        if len(d) != len(v):
            say(f"⚠️ 頂点の数が合わない（地図 {len(d)}・区画 {len(v)}）⇒ den を確かめる"); break
        allm = float(np.nanmean(d[v > 0]))
        for k, regs in AREAS.items():
            idx = [i for i, n in enumerate(nm) if base(n) in regs]
            m = np.isin(v, idx)
            val = float(np.nanmean(d[m])) if m.any() else float("nan")
            res.setdefault(k, {})[h] = (val, val / allm if allm else float("nan"))
    for k, hv in res.items():
        say(f"  {k:<8} " + " ／ ".join(f"{'左' if h == 'lh' else '右'} {x[0]:.3g}（全皮質の {x[1]:.2f}倍）" for h, x in hv.items()))
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n（{OUT} に書いた）")
