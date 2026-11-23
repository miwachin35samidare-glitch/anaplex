import sys, numpy as np, torch, collections
sys.path.insert(0,'.')
import babble as B, torch_area as T, torch_brain as TB
from torch_brain import Brain, NAMES
B.DEV="cpu"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
xg = {nm: float(b.g.areas[i].x_gain) if hasattr(b.g.areas[i],'x_gain') else None for i,nm in enumerate(NAMES)}
off = [int(x) for x in b.off_ch]
def area_of(ch):
    for i in range(len(NAMES)):
        if off[i] <= ch < off[i+1]: return NAMES[i]
# per receiver area: total lines per receiving cell, by path
rows = collections.defaultdict(list)
for k, s in b.bu.items():
    tgt = s.tgt if hasattr(s,'tgt') else None
    n_tgt = getattr(s,'n_tgt', None)
    o = getattr(s,"ch0", None)
    if tgt is None: continue
    cnt = torch.bincount(tgt.long())
    nm = area_of(int(o)) if o is not None else "?"
    rows[nm].append((k, float(cnt[cnt>0].float().mean()), len(cnt)))
for nm in sorted(rows, key=lambda n: -sum(r[1] for r in rows[n])):
    tot = sum(r[1] for r in rows[nm])
    print(f"{nm:<8s} x_gain {xg.get(nm)!s:>6.6s}  受け手1体の下からの線 計 {tot:8.1f} ⇒ ×x_gain {tot*(xg.get(nm) or 1):9.1f}  | " + "・".join(f"{k} {m:.0f}" for k,m,_ in rows[nm]))
