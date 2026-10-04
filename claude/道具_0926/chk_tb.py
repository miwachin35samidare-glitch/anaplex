"""Group2 の tb の足し込みの並びを確かめる（2領野・CPU。測るだけ）"""
import sys; sys.path.insert(0, ".")   # ~/anaplex で回す
import torch, torch_area as T
from torch_group2 import Group2
pr = T.Prior.load("./prior_v1.npz", dev="cpu")
A = T.Area(4, pr, n_partner=20, dev="cpu", seed=1, fire_th=1.0, spont_hz=0.0)
B = T.Area(6, pr, n_partner=20, dev="cpu", seed=2, fire_th=1.0, spont_hz=0.0)
g = Group2([A, B])
for k in ("scale_skip",):
    if not hasattr(g, k): print("no attr", k)
x = torch.zeros(10); x[:4] = 5.0            # 領野 A の 4体だけ下から越える。B は何も来ない
ext = torch.zeros(0)
tb0 = g.tb[:, 0].clone()
g.step(x, ext, learn=False)
d0 = g.decay[:, 0]
v = ((g.tb[:, 0] - tb0 * d0) / (1 - d0)).round()
kind = ["誤", "表", "抑"]
for i, a in enumerate(g.areas):
    lo = int(g.off_recv[i])
    rows = v[lo:lo + a.n_recv].int().tolist()
    print("ABAB"[i], "誤", rows[:a.n_ch], "表", rows[a.n_ch:2 * a.n_ch], "抑", rows[2 * a.n_ch:])
