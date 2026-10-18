"""版93 の確かめ その1：中脳の核だけ（網は使わない）。外の送り手は決まった率のポアソン（撃った周つき）
  python3 check_v93_mb.py [秒=20]
見るもの：ふだんの率（的 視床下核の辺縁 19.3・境 33・外側手綱核 30・吻内側被蓋核 17.8・黒質 4.56Hz）／ 1歩の 10周の中のそろい ／ 門 ／ 重さの動き ／ 1歩の時間
"""
import sys, time, torch
sys.path.insert(0, "/tmp/t90"); sys.path.insert(0, ".")
import midbrain as MB
LEARN = "--nolearn" not in sys.argv
SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator(device=dev).manual_seed(0)
SRC = {"前頭前野": (946, 3.30), "側坐核": (292, 2.1 / 2), "上丘": (64, 0.0), "PPN": (512, 23.2)}   # 側坐核：表現 2.1Hz・誤差 ≒0 の平均
mb = MB.Midbrain(dev=dev, gen=g); mb.wire({k: n for k, (n, _) in SRC.items()})
steps = int(SEC * 1000 / 10.67)
phase = torch.zeros(len(MB.NAMES_MB), MB.R_PASS, device=dev)
w0 = mb.w.clone(); t0 = time.time()
for t in range(steps):
    ext = {k: (torch.rand(MB.R_PASS, n, device=dev, generator=g) < hz * MB.DT / 1000) for k, (n, hz) in SRC.items()}
    gate = mb.step(ext, learn=LEARN)
el = (time.time() - t0) / steps * 1000
print(f"1歩 {el:.2f}ms（{dev}）／ 門 {float(gate):.2f}")
print("率（Hz）：" + "・".join(f"{k} {v:.1f}" for k, v in mb.rates_hz().items()))
dw = (mb.w - w0)
for nm, kind in (("黒質の興奮（Harnett）", 1), ("ほかの興奮（決め A）", 2)):
    m = mb.kind == kind
    print(f"重さ {nm}：平均 {float(mb.w[m].mean()):.3f}・幅 {float(mb.w[m].min()):.3f}〜{float(mb.w[m].max()):.3f}")
