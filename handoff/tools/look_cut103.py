"""切り分け（見るだけ・直しではない）：線の重さを 0 にして暮らし 3 計画の声を数える。python3 look_cut.py 線名,線名"""
import sys; sys.path.insert(0, ".")
import numpy as np, torch, babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain
cut = [k for k in sys.argv[1].split(",") if k]
B.DEV = "cpu"; torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
for k in cut:
    b.bu[k].w = torch.zeros_like(b.bu[k].w)
b.bu_pack()
b.bs.set_state(BS.WAKE)
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
fr = []; orig = b.step
def st(*a, **k):
    r = orig(*a, **k); fr.append(float(b.motor_rep.mean())); return r
b.step = st
lf.run_plans(3, learn=True)
print(f"版{B.VERSION} 0 にした線 {cut}：{lf.t}歩・声 {lf.stats['voices']}・運動の表現 平均 {np.mean(fr):.3f}", flush=True)
