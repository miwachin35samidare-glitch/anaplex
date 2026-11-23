import sys; sys.path.insert(0, ".")
import numpy as np, torch, babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain
B.DEV = "cpu"; torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
for _ in range(80):
    lf.step()
b.bu_unpack()
torch.save({"w": b.g.w, "bu": {k: s.w for k, s in b.bu.items()}, "up": dict(b.up)}, sys.argv[1])
print("書いた", sys.argv[1], flush=True)
