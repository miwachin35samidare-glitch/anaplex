"""版49：嗅内浅 層 II の誤差 → 歯状回の入口 dg が、少数の歩にまとまっているかを測る（arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v49h.py [喃語の回数=10]
【測】measure_v49g（10-04）：嗅内浅 II の誤差 2.6〜2.9%（20体 ⇒ 平均 1歩 0.56体）なのに、dg は平均 1歩 0.6体（上位 8体の枠の 7%）
  見立て【見】：嗅内浅 II の誤差が少数の歩にまとまって撃ち、残りの歩は 0 ⇒ dg も多くの歩で 0
dg の作り（torch_brain.step）：歯状回1体が 嗅内浅 II の 20体から 2本（貫通路）を足す ⇒ 値が 0 でない体のうち上位 8体（2.5%）
並べるもの（全歩）
  嗅内浅 II の誤差が撃った数の分布（1歩に 0体 ／ 1体 ／ 2〜4体 ／ 5体以上 の歩の割合）
  dg の数の分布（0 ／ 1〜3 ／ 4〜7 ／ 8）
  嗅内浅 II の 20体ごとの撃った割合（同じ体ばかり撃っているか）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
k = max(1, int(round(b.n["歯状回"] * TB.SPARSE)))
ec_n, dg_n, per_cell = [], [], torch.zeros(TB.EC_II, device=B.DEV)
orig = Brain.step


def hooked(self, ear, cmd, eye=None, learn=True):
    ec2 = (self.up["嗅内浅"][:TB.EC_II] > 0).float()          # step が使うのと同じ（1歩前の誤差）
    drive = ec2[self.perforant].sum(1)
    ec_n.append(int(ec2.sum())); dg_n.append(min(k, int((drive > 0).sum())))
    per_cell.add_(ec2)
    return orig(self, ear, cmd, eye, learn)


Brain.step = hooked
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
for i in range(N):
    say(rng, True)
    B.reply(b, rp, rng_r)
e = np.array(ec_n); d = np.array(dg_n); n = max(len(e), 1)
print(f"歩 {len(e)}  嗅内浅 II の誤差 平均 {e.mean():.2f}体/歩（{e.mean()/TB.EC_II:.1%}）  dg 平均 {d.mean():.2f}体/歩（枠 {k}）")
print(f"嗅内浅 II が撃った数   0体 {np.mean(e==0):5.1%}  1体 {np.mean(e==1):5.1%}  2〜4体 {np.mean((e>=2)&(e<=4)):5.1%}  5体以上 {np.mean(e>=5):5.1%}  最大 {e.max()}")
print(f"dg の数              0 {np.mean(d==0):5.1%}  1〜3 {np.mean((d>=1)&(d<=3)):5.1%}  4〜{k-1} {np.mean((d>=4)&(d<k)):5.1%}  {k} {np.mean(d==k):5.1%}")
print("嗅内浅 II の20体ごとの撃った割合：", " ".join(f"{v:.1%}" for v in (per_cell / n).tolist()))
