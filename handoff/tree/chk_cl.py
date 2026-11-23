"""版67 の CL → 皮質 シナプスを小さい網で確かめる（CPU・相手 200・作り物の声・学びながら。値は読まない、形だけ）"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
B.DEV = "cpu"
from torch_brain import Brain, NAMES
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev="cpu")
b = Brain(prior, n_partner=200, hippo_scale=4.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
_, say, _ = B.voices(b, True)
w0 = b.cl_w.clone()
print(flush=True, end=""); print(f"CL→皮質 辺 {len(b.cl_w)}（表現の体 {len(b.cl_ch)} × 78）重さ 中央 {b.cl_w.median():.3f} 平均 {b.cl_w.mean():.3f}")
rng = np.random.default_rng(0)
cr, th = [], []
orig = b.step
def step(*a, **k):
    out = orig(*a, **k); cr.append(float(b.cl_rate)); th.append(float(b.g.third)); return out
b.step = step
for i in range(3):
    say(rng, True)
    ch = (b.cl_w / w0)
    print(f"{i+1}回目  CL 撃つ {np.mean(cr):.3f}/歩  門 平均 {np.mean(th):.2f}  重さ 平均 {b.cl_w.mean():.3f}  "
          f"変わった辺 {(ch != 1).float().mean():.1%}  比 最小 {ch.min():.3f} 最大 {ch.max():.3f}", flush=True)
