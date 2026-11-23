"""中側頭回の後ろの体積を HCP-MMP1（Glasser 2016）の区画から出す（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python hcp_mtg.py
【み】10-31 23:38「それでいいよ」（中側頭回の後ろは Julich に区画が無いので HCP-MMP から引く）
見ること
  1 siibra にある地図の名前（HCP-MMP・Glasser があるか）
  2 HCP-MMP の中側頭回の後ろの区画（TE1p・PHT・TE1m）と上側頭溝の区画（STSdp・STSvp・STSda・STSva）・Te3 に当たる A4・A5 の体積（MNI152、mm³）
     ⇒ Julich の Te3（今の高次）との比で体数にするため、同じ地図の A4・A5 も出す
  ⚠️ Claude はこの道具を動かせていない（コンテナから EBRAINS に繋がらない）
"""
import sys
import numpy as np

AREAS = ("TE1p", "PHT", "TE1m", "TE1a", "STSdp", "STSvp", "STSda", "STSva", "A4", "A5", "TPOJ1", "PSL", "STV")
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
print(f"siibra {getattr(siibra, '__version__', '?')}", flush=True)
names = [p.name for p in siibra.parcellations]
print("地図：" + "／".join(names), flush=True)
cand = [p for p in siibra.parcellations if any(k in p.name.lower() for k in ("mmp", "glasser", "hcp"))]
if not cand:
    sys.exit("⚠️ HCP-MMP の地図が siibra に無い ⇒ 別の取り方が要る（Claude に返す）")
parc = cand[0]
print(f"使う地図：{parc.name}", flush=True)
for a in AREAS:
    hits = [r for r in parc if not r.children and (r.name.split()[0].upper() == a.upper() or r.name.upper().startswith(a.upper() + " ") or ("_" + a.upper() + "_") in r.name.upper())]
    if not hits:
        hits = [r for r in parc if not r.children and a.lower() in r.name.lower()]
    if not hits:
        print(f"  {a}：区画が見つからない", flush=True)
        continue
    for r in hits[:4]:
        try:
            img = r.get_regional_mask("mni152", maptype="labelled").fetch()
            d = np.asarray(img.get_fdata()) > 0
            v = d.sum() * abs(np.linalg.det(img.affine[:3, :3]))
            ijk = np.argwhere(d)
            y = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[1] if len(ijk) else np.zeros(1)
            print(f"  {a} ／ {r.name}：{v:,.0f}mm³ ／ y 中央 {np.median(y):.0f}", flush=True)
        except Exception as e:
            print(f"  {a} ／ {r.name}：取れない（{type(e).__name__}：{e}）", flush=True)
