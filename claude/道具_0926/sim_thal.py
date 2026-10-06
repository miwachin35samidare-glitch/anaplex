"""視床1領野（版50 の作り）で、中継 → 中継の興奮の本数を振る（CPU・学びながら・測るだけ）
  python3 sim_thal.py K 歩数
入力：耳 400本（40帯 × 10本）。音は歩の 19%（20歩ずつのかたまり）、音のある帯（半分）の繊維が 1歩 40% で撃つ
"""
import sys, time, torch, numpy as np
import torch_area as T
from torch_group2 import Group2
K = None if sys.argv[1] == "all" else int(sys.argv[1]); NS = int(sys.argv[2])
torch.manual_seed(0)
pr = T.Prior.load("prior_v1.npz", dev="cpu")
a = T.Area(400, pr, n_ext=0, eta=0.3, n_partner=9100, stp_path="synapse_model_human.json", dev="cpu", seed=0,
           motor=True, inh_frac=0.275, fire_th=0.3, x_gain=1.0, spont_hz=0.3, relay_exc=K)
g = Group2([a]); g.scaling_on = False
n = 400; nrow = a.n_recv; ninh = a.n_inh
rng = np.random.default_rng(0)
ext = torch.zeros(0)
snd = False; left = 0; bands = None
rates, xr, lat = [], [], []
hi_run = 0; hi_max = 0; above = 0
t0 = time.time()
for t in range(NS):
    if left == 0:
        snd = rng.random() < 0.19 / (0.19 + 0.81 * 1.0) if not snd else False
        left = 20 if snd else int(rng.integers(40, 120))
        bands = rng.random(40) < 0.5
    left -= 1
    p = np.repeat(bands, 10) * 0.4 if snd else np.zeros(400)
    x = torch.tensor((rng.random(400) < p).astype(np.float32))
    tb0 = g.tb[:, 0].clone()
    g.step(x, ext, learn=True)
    d0 = g.decay[:, 0]; v = ((g.tb[:, 0] - tb0 * d0) / (1 - d0)).round()
    rep = v[n:2 * n]; inh = v[2 * n:]
    r = float(rep.mean()); rates.append(r); xr.append(float(x.mean()))
    lat.append(float((rep * (1 - x)).sum() / max(1.0, float(rep.sum()))))   # 下から来ていない体の割合
    above += r > 0.4
hz = lambda v: f"{np.mean(v):5.1%}"
R = np.array(rates); L = np.array(lat)
edges_rr = int(((~g.e_inh) & (g.e_recv.long() < 2 * n)).sum())
print(f"K={sys.argv[1]}  中継→中継の辺/体 {edges_rr / (2 * n):.0f}  抑制の辺/体 {int((g.e_inh & (g.e_recv.long() < 2*n)).sum())/(2*n):.0f}  （{(time.time()-t0)/60:.1f}分）")
for i in range(0, NS, NS // 4):
    s = slice(i, i + NS // 4)
    print(f"  歩 {i:5d}〜  表 {hz(R[s])}  入力 {hz(np.array(xr)[s])}  撃った体のうち下から来ていない {np.mean(L[s]):5.1%}  40%超の歩 {np.mean(R[s] > 0.4):5.1%}  最大 {R[s].max():5.1%}")
