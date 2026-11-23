"""視床の型の領野の「動く最低数」（設計案_脳幹 §12-42 と同じ測り。コンテナ・CPU・相手 200・数値は合わせない）
  python3 measure_thal_min.py N [回=20] [間=120]
  CM の体数を N にして組む。耳・口の指令は 0（起きている）。
  入力：運動の表現（運動 → CM の送り手）の 30% を3歩だけ撃たせる（§12-42：外から入る 30% を3歩）
  見るもの（前 40歩 と 後 の差を t で。|t|>3 を届いたとする）
    1 CM が撃った体の割合（1歩前の表現の発火）
    2 CM → 被殻 が被殻へ届けた量（下からの表の出力の合計。CM の出口を通った分だけ）
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB

N = int(sys.argv[1])
REP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 120
PRE, POST, PULSE, FRAC = 40, 8, 3, 0.30

TB.N_NEW["CM"] = N
B.DEV = "cpu"
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev="cpu")
b = TB.Brain(prior, n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
ear = torch.zeros(B.N_EAR)
cmd = torch.zeros(b.n["運動"]) if hasattr(b, "n") else None
n_cmd = b.n["運動"]
cmd = torch.zeros(n_cmd)
gen = torch.Generator().manual_seed(1)
nm = b.n["運動"]


def one(pulse):
    if pulse:
        m = (torch.rand(nm, generator=gen) < FRAC).float()
        b.motor_rep = torch.maximum(b.motor_rep, m)
    b.step(ear, cmd)
    return float(b.cmpf["CM"].mean()), float(b._bo["CM→被殻"].sum())


for _ in range(200):                     # 慣らし
    one(False)
pre_c, pre_s, post_c, post_s = [], [], [], []
for r in range(REP):
    tr = [one(False) for _ in range(GAP - PULSE)]
    pre = tr[-PRE:]
    pr = [one(True) for _ in range(PULSE)]
    po = [one(False) for _ in range(POST)]
    pre_c.append(np.mean([a for a, _ in pre])); pre_s.append(np.mean([s for _, s in pre]))
    post_c.append(np.mean([a for a, _ in pr + po])); post_s.append(np.mean([s for _, s in pr + po]))


def t(a, b_):
    d = np.array(b_) - np.array(a)
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)


print(f"CM {N}体：撃った割合 前 {np.mean(pre_c):.3f} → 後 {np.mean(post_c):.3f}  t {t(pre_c, post_c):+.1f} ／ "
      f"CM→被殻 前 {np.mean(pre_s):.1f} → 後 {np.mean(post_s):.1f}  t {t(pre_s, post_s):+.1f}", flush=True)
