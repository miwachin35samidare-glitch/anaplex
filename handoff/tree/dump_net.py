"""今の網（コードの値）を表に書き出す。python3 dump_net.py > net.txt（コンテナ・CPU・相手 200）"""
import sys, torch, numpy as np
sys.path.insert(0, ".")
import torch_area as T, torch_brain as TB, babble as B
from torch_brain import Brain, NAMES
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
print("版", B.VERSION, "体", b.n_cells)
print("領野|体|型|抑制の体|時間細胞/送り元|傾き|閾値|入口の重さ|時定数×|自発Hz|ext|相手")
for nm in NAMES:
    a = b.areas[nm]
    ch = a.ch
    sp = getattr(a, "spont_hz", None)
    print("|".join(str(x) for x in (nm, b.n[nm], TB.PRIOR_TYPE.get(nm, "?"), getattr(a, "n_inh", "?"),
          getattr(ch, "n_cell", "?"), getattr(ch, "n_ramp", "?"), round(float(getattr(a, "fire_th", float("nan"))), 3),
          round(float(getattr(a, "x_gain", float("nan"))), 3), round(float(getattr(a, "tau_scale", float("nan"))), 3),
          sp, b.n_ext.get(nm, "?"), getattr(a, "n_partner", "?"))))
print("\n下からのシナプス（受け手1体の本数）")
for k, s in b.bu.items():
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    print(f"  {k}: 受け手 {s.n_tgt}・1体 {int(per.min())}〜{int(per.max())}本・計 {len(s.tgt)}")
print("\n核→領野の抑制", sorted(b.bu_inh))
print("CL→皮質", TB.CL_CORTEX, "K_CL", TB.K_CL)
print("上からの戻り（眠りで落とす）", b.topdown_rows)
attrs = [k for k in dir(b) if not k.startswith("__")]
print("\nBrain の主な属性", [k for k in attrs if not callable(getattr(b, k, None))][:80])
