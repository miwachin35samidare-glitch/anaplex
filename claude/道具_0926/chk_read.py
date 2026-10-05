"""版51 の Group2 で、tb から戻す読み方（領野ごとの行）が撃った体と合うかを確かめる（2領野・CPU）"""
import sys; sys.path.insert(0, ".")   # ~/anaplex で回す
import torch, torch_area as T
from torch_group2 import Group2
pr = T.Prior.load("./prior_v1.npz", dev="cpu")
A = T.Area(4, pr, n_partner=20, dev="cpu", seed=1, fire_th=1.0, spont_hz=0.0)
B = T.Area(6, pr, n_partner=20, dev="cpu", seed=2, fire_th=1.0, spont_hz=0.0)
g = Group2([A, B])
x = torch.zeros(10); x[[1, 2]] = 5.0; x[[4 + 3, 4 + 5]] = 5.0     # A の 1・2、B の 3・5 だけ越える
tb0 = g.tb[:, 0].clone(); g.step(x, torch.zeros(0), learn=False)
d0 = g.decay[:, 0]; v = ((g.tb[:, 0] - tb0 * d0) / (1 - d0)).round()
for i, a in enumerate(g.areas):
    lo = int(g.off_recv[i])
    print("AB"[i], "表現", v[lo + a.n_ch:lo + 2 * a.n_ch].int().tolist(), "抑制", v[lo + 2 * a.n_ch:lo + a.n_recv].int().tolist())
