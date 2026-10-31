"""Tziortzi 2014 の線条体の地図（Oxford-GSK-Imanova Striatal Connectivity Atlas、人 26人）で、
尾状核・被殻・腹側線条体ごとに、7つの皮質の帯に塗られた点の数を数える（arkhe で。網は触らない）
  ~/siibra-venv/bin/python striatum_zones.py
11-01 3:26【み】「人にできるなら人にして」：皮質 → 線条体の本数を人の比に置き直す
- 地図は FSL の data/atlases/Striatum（arkhe に FSL は無い ⇒ FreeSurfer の配布所にある FSL 5.0.7 の写しから取る）
- 置き場所：~/anaplex/striatum_atlas/（無ければ作る。取ったものは消さない）
"""
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
try:
    import nibabel as nib
except ImportError:
    sys.exit("⚠️ nibabel が無い：~/siibra-venv/bin/python で回す")

BASE = "https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/tutorial_packages_centos6/centos6/fsl_507/data/atlases/"
FILES = {
    "Striatum-Connectivity-7sub.xml": "",
    "Striatum-Structural.xml": "",
    "striatum-con-label-thr25-7sub-1mm.nii.gz": "Striatum/",
    "striatum-con-label-thr50-7sub-1mm.nii.gz": "Striatum/",
    "striatum-structural-1mm.nii.gz": "Striatum/",
}
D = os.path.expanduser("~/anaplex/striatum_atlas")
os.makedirs(D, exist_ok=True)
for f, sub in FILES.items():
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print(f"取る：{f}", flush=True)
        try:
            urllib.request.urlretrieve(BASE + sub + f, p)
        except Exception as e:
            sys.exit(f"⚠️ 取れない：{f}（{type(e).__name__}：{e}）")


def labels(xmlname):
    t = ET.parse(os.path.join(D, xmlname))
    out = {}
    for lab in t.iter("label"):
        out[int(lab.get("index"))] = lab.text.strip()
    return out


con_names = labels("Striatum-Connectivity-7sub.xml")
st_names = labels("Striatum-Structural.xml")
print("帯の名前（xml の番号 → 名前）：", con_names, flush=True)
print("核の名前（xml の番号 → 名前）：", st_names, flush=True)

st_img = nib.load(os.path.join(D, "striatum-structural-1mm.nii.gz"))
st = np.asarray(st_img.dataobj).astype(int)
x_mm = (st_img.affine @ np.c_[np.argwhere(np.ones(st.shape, bool)), np.ones(st.size)].T)[0].reshape(st.shape)
print(f"核の画像 {st.shape}・値 {sorted(set(np.unique(st)) - {0})}", flush=True)

for thr in ("thr25", "thr50"):
    ci = nib.load(os.path.join(D, f"striatum-con-label-{thr}-7sub-1mm.nii.gz"))
    c = np.asarray(ci.dataobj).astype(int)
    if c.shape != st.shape or not np.allclose(ci.affine, st_img.affine):
        print(f"⚠️ {thr}：格子が核の画像と違う（{c.shape}）", flush=True)
        continue
    vals = sorted(set(np.unique(c)) - {0})
    print(f"\n== {thr}（帯の画像の値 {vals}。xml の番号は 0 から、画像の値は 1 から の見込み ⇒ 値 − 1 を名前に当てる）", flush=True)
    for sv in sorted(set(np.unique(st)) - {0}):
        sname = st_names.get(sv - 1, st_names.get(sv, f"番号{sv}"))
        for side, sel in (("左", x_mm < 0), ("右", x_mm > 0)):
            m = (st == sv) & sel
            n = int(m.sum())
            if not n:
                continue
            row = []
            for cv in vals:
                k = int(((c == cv) & m).sum())
                row.append(f"{con_names.get(cv - 1, cv)} {k}（{k / n:.0%}）")
            nolab = int(((c == 0) & m).sum())
            print(f"  {sname} {side}（{n} 点）：" + "／".join(row) + f"／ 帯なし {nolab}", flush=True)
print("\n終わり", flush=True)
