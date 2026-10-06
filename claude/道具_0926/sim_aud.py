"""聴覚1領野（版51 の作り）で、錐体どうしの興奮を人の値で始めるかを比べる（CPU・学びながら・測るだけ）
  python3 sim_aud.py {0|human} 歩数
入力：視床 400本（40帯 × 10本）。音は歩の 19%（20歩ずつ）、音のある帯（半分）の繊維が 1歩 70% で撃つ
      聴覚の入口 ＝ 自分の帯の視床の本数（0〜10）× 1.68（版29。9本で閾値 14.3 を越える）
human：表現の体 ← 同じ領野の表現の体（1歩前）の辺の重みを、対数正規 中央値 0.46mV（Planert 2025）・平均 1.12mV（Hunt 2023）で始める
       1本 ＝ 1.12mV ＝ 重さ 1（版45 の単位）。辺の重みは接触 3.3 で割る
"""
import sys, time, math, torch, numpy as np
import torch_area as T
from torch_group2 import Group2
MODE = sys.argv[1]; NS = int(sys.argv[2])
torch.manual_seed(0)
pr = T.Prior.load("prior_v1.npz", dev="cpu")
TH = 16.0 / 1.12
a = T.Area(400, pr, n_ext=0, eta=0.3, n_partner=9100, stp_path="synapse_model_human.json", dev="cpu", seed=1,
           fire_th=TH, x_gain=TH / 8.5, tgt_by_fire=True, spont_hz=0.3)
n = 400
src = a.e_src_sorted; rcv = a.e_recv.long()
rr = (src >= a.ch.n_cells) & (src < a.ch.n_cells + n * a.D) & (rcv >= n) & (rcv < 2 * n)   # 表現の体（1歩前）→ 表現の体
print(f"表現 → 表現 の辺 {int(rr.sum())}（1体 {int(rr.sum())/n:.0f}本 ＝ 同じ領野の表現の {int(rr.sum())/n/n:.1%}）  D={a.D}")
if MODE == "human":
    med, mean = 0.46 / 1.12, 1.0
    sig = math.sqrt(2 * math.log(mean / med)); mu = math.log(med)
    g0 = torch.Generator().manual_seed(5)
    a.w[rr] = torch.exp(mu + sig * torch.randn(int(rr.sum()), generator=g0)) / 3.3
g = Group2([a]); g.scaling_on = False
rng = np.random.default_rng(0)
snd = False; left = 0; bands = None
R, X, H = [], [], []
t0 = time.time()
for t in range(NS):
    if left == 0:
        snd = (not snd) and rng.random() < 0.6
        left = 20 if snd else int(rng.integers(40, 100))
        bands = rng.random(40) < 0.5
    left -= 1
    p = np.repeat(bands, 10) * 0.7 if snd else np.zeros(400)
    th = (rng.random(400) < p).astype(np.float32)
    x = torch.tensor(th.reshape(40, 10).sum(1).repeat(10))
    tb0 = g.tb[:, 0].clone()
    g.step(x, torch.zeros(0), learn=True)
    v = ((g.tb[:, 0] - tb0 * g.decay[:, 0]) / (1 - g.decay[:, 0])).round()
    rep = v[n:2 * n]; tgt = (x * TH / 8.5 > TH).float()
    R.append(float(rep.mean())); X.append(float(tgt.mean()))
    H.append(float((rep * tgt).sum() / max(1.0, float(tgt.sum()))) if tgt.sum() > 0 else float("nan"))
R, X, H = map(np.array, (R, X, H))
wrr = g.w[(~g.e_inh) & (g.e_recv.long() >= n) & (g.e_recv.long() < 2 * n)]
print(f"MODE={MODE}（{(time.time()-t0)/60:.1f}分）")
for i in range(0, NS, NS // 5):
    s = slice(i, i + NS // 5)
    print(f"  歩 {i:5d}〜  表現 {R[s].mean():6.2%}  的 {X[s].mean():6.2%}  的が立った体のうち撃てた {np.nanmean(H[s]):6.1%}"
          f"  40%超の歩 {np.mean(R[s] > 0.4):5.1%}  最大 {R[s].max():6.1%}")
print(f"  表現の体が受ける興奮の重み 平均 {float(wrr.mean()):.4f}  最大 {float(wrr.max()):.3f}")
