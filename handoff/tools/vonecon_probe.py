"""vonecon_syntax2 が何も出さなかった訳を見る（arkhe で。何も書き換えない）
  ~/siibra-venv/bin/python vonecon_probe.py
11-01 1:33：地図を丸ごと取る道（siibra.get_map）で、HarvardOxford・von Economo とも区画が 0 個・エラーも無し
見ること：地図の型・区画の名前の数と頭の数個・番号の中身・画像の大きさと格子・画像の中の値の種類
"""
import sys
import numpy as np

try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")

for pname in ("HarvardOxford cort maxprob thr25", "Von Economo Koskinas Brain Atlas", "julich 3.1"):
    print(f"\n== {pname}", flush=True)
    try:
        p = siibra.parcellations.get(pname)
        m = siibra.get_map(parcellation=p, space="mni152", maptype="labelled")
    except Exception as e:
        print(f"  地図が取れない：{type(e).__name__}：{e}", flush=True)
        continue
    print(f"  型 {type(m).__name__} ／ 空間 {m.space.name} ／ 画像の数 {len(getattr(m, 'volumes', []))}", flush=True)
    regs = list(m.regions)
    print(f"  区画の名前 {len(regs)} 個：" + "／".join(map(str, regs[:5])), flush=True)
    if regs:
        try:
            idx = m.get_index(regs[0])
            print(f"  {regs[0]} の番号：{idx}", flush=True)
        except Exception as e:
            print(f"  番号が取れない：{type(e).__name__}：{e}", flush=True)
    try:
        img = m.fetch()
        d = np.asarray(img.dataobj)
        u = np.unique(d)
        vox = abs(np.linalg.det(img.affine[:3, :3]))
        print(f"  画像 {d.shape}・格子 1 つ {vox:.2f}mm³・値の種類 {len(u)}（頭 {u[:6].tolist()}）・0 でない点 {int((d > 0).sum()):,}", flush=True)
    except Exception as e:
        print(f"  画像が取れない：{type(e).__name__}：{e}", flush=True)
print("\n終わり", flush=True)
