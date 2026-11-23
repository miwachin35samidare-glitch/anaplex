"""版107 §12：人の PET の地図（MNI152）を、皮質の外の部位と皮質の領野ごとに平均する（arkhe で。網には触らない）
  ~/siibra-venv/bin/python v107_subcort.py
  地図（neuromaps、v107_receptor で取ったもの）
    aghourian2017 feobv（コリンの終末）・beliveau2017 cumi101（5-HT1A）・cimbi36（5-HT2A）・dasb（輸送体）
  区画：HarvardOxford（人の MRI の地図、nilearn が配る。最大確率・閾値 25%・1mm）
    皮質の外：尾状核・被殻・側坐核・淡蒼球・視床・海馬・扁桃体
    皮質（FEOBV で Ch4 の送り先の重みを出す用）：網の領野に近い回で当てる【C】
  出すもの：地図ごと × 部位ごとの平均と、皮質の外の部位どうし・皮質どうしの比（data_v107_subcort.txt）
"""
import sys
import numpy as np

OUT = "data_v107_subcort.txt"
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

try:
    from neuromaps.datasets import fetch_annotation
    from nilearn import datasets, image
    import nibabel as nib
except ImportError as e:
    sys.exit(f"⚠️ {e}：~/siibra-venv/bin/pip install neuromaps nilearn")

MAPS = [("aghourian2017", "feobv"), ("beliveau2017", "cumi101"), ("beliveau2017", "cimbi36"), ("beliveau2017", "dasb")]
SUB = {"尾状核": ("Left Caudate", "Right Caudate"), "被殻": ("Left Putamen", "Right Putamen"),
       "側坐核": ("Left Accumbens", "Right Accumbens"), "淡蒼球": ("Left Pallidum", "Right Pallidum"),
       "視床": ("Left Thalamus", "Right Thalamus"), "海馬": ("Left Hippocampus", "Right Hippocampus"),
       "扁桃体": ("Left Amygdala", "Right Amygdala")}
CX = {"聴覚": ("Heschl's Gyrus (includes H1 and H2)",), "連合": ("Planum Temporale",),
      "高次": ("Superior Temporal Gyrus, posterior division",), "中側頭後": ("Middle Temporal Gyrus, posterior division",),
      "44": ("Inferior Frontal Gyrus, pars opercularis",), "45": ("Inferior Frontal Gyrus, pars triangularis",),
      "運動": ("Precentral Gyrus",), "前頭前野": ("Middle Frontal Gyrus",), "眼窩前頭": ("Frontal Orbital Cortex",),
      "前帯状": ("Cingulate Gyrus, anterior division",), "補足": ("Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)",),
      "島": ("Insular Cortex",), "縁上回": ("Supramarginal Gyrus, anterior division", "Supramarginal Gyrus, posterior division"),
      "海馬傍": ("Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division"),
      "視覚": ("Intracalcarine Cortex",), "脳梁膨大後": ("Cingulate Gyrus, posterior division",)}

def atlas(name):
    a = datasets.fetch_atlas_harvard_oxford(name)
    img = a.maps if hasattr(a.maps, "get_fdata") else nib.load(a.maps)
    return img, list(a.labels)

sub_img, sub_lab = atlas("sub-maxprob-thr25-1mm")
cx_img, cx_lab = atlas("cort-maxprob-thr25-1mm")
say(f"HarvardOxford：皮質の外 {len(sub_lab)} ／ 皮質 {len(cx_lab)} の札")

def means(pet, img, lab, groups):
    r = image.resample_to_img(pet, img, interpolation="continuous")
    d = np.asarray(r.get_fdata()); a = np.asarray(img.get_fdata()).astype(int)
    out = {}
    for k, names in groups.items():
        idx = [lab.index(n) for n in names if n in lab]
        if not idx:
            out[k] = None; continue
        m = np.isin(a, idx)
        out[k] = float(np.nanmean(d[m])) if m.any() else None
    return out

for src, desc in MAPS:
    try:
        f = fetch_annotation(source=src, desc=desc, space="MNI152", res="1mm")
    except Exception as e:
        say(f"⚠️ {src}・{desc} が取れない：{str(e)[:100]}"); continue
    pet = nib.load(f if isinstance(f, str) else f[0])
    say(f"\n══ {src}・{desc}（MNI152）")
    for title, img, lab, groups in (("皮質の外", sub_img, sub_lab, SUB), ("皮質", cx_img, cx_lab, CX)):
        m = means(pet, img, lab, groups)
        vals = [v for v in m.values() if v is not None]
        avg = sum(vals) / len(vals) if vals else float("nan")
        say(f"  {title}（平均 {avg:.3g} を 1 とした比）")
        for k, v in m.items():
            say(f"    {k:<8} " + ("無い札" if v is None else f"{v:.3g}（{v / avg:.2f}）"))
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n（{OUT} に書いた）")
