"""網の辺と下からの線の本数を数える（コンテナ・CPU）。python3 count_edges.py 相手"""
import sys, time; sys.path.insert(0, ".")
import torch, babble as B, torch_area as T
from torch_brain import Brain
P = int(sys.argv[1]); B.DEV = "cpu"; torch.manual_seed(0); t0 = time.time()
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=P, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
nbu = sum(len(s.src) for s in b.bu.values())
print(f"版{B.VERSION} 相手 {P}：体 {b.n_cells:,} 網の辺 {len(b.g.w):,} 下からの線 {nbu:,}（{time.time()-t0:.0f}秒）", flush=True)
