"""版107：網の中の興奮の辺の重さが跳ねる所を、受け手の領野 × 送り手の領野で追う（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_wgrow107.py [番=100] [何番ごと=10]
  quiet107y（11-09 19:29）：辺の重さの平均 40番 0.12 → 70番 1.32 → 100番 29.4（10番ごとに約3倍）。興奮の重さの和 CA1 33.8倍・STS1 21倍・上丘 17倍・運動 8倍。
    quiet107w でも 80番から同じ跳ね（CA1 214倍）・quiet107v は 100番で 1.50（まだ上がる途中）
  何番ごとに：受け手の領野ごとの興奮の辺の重さの和・最大・5.4（6mV）を越える辺の数 ／ 跳ねた受け手について、重さの和を送り手の領野ごとに（上位 6）
"""
import os, sys, numpy as np, torch
# 版107h：グラフで回す（⟲ eager は取り置きが増えて 12〜15GB【測】11-05）。eager にするなら ANAPLEX_GRAPH=eager
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NP = int(args[0]) if args else 100
EV = int(args[1]) if len(args) > 1 else 10
TEST = "--test" in sys.argv
WIN = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--win=")), 20))
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
A = len(NAMES)
send_area = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
for i, r in enumerate(g.pool_send):
    send_area[torch.as_tensor(r, device=dev).long()] = i
recv_area = torch.zeros(g.n_recv_all, dtype=torch.long, device=dev)
for i in range(A):
    recv_area[int(g.off_recv[i]):int(g.off_recv[i + 1])] = i


def snap():
    g.flush_all()
    M = torch.zeros(A * A, device=dev, dtype=torch.float64)      # 受け手 × 送り手 の重さの和（興奮）
    mx = torch.zeros(A, device=dev)
    big = torch.zeros(A, device=dev, dtype=torch.float64)
    E, st = len(g.w), 1 << 24
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        ex = ~g.e_inh[sl].bool()
        w_ = g.w[sl].float()[ex]
        ra = recv_area[g.e_recv[sl].long()[ex]]
        sa = send_area[g.e_src[sl].long()[ex]].clamp(min=0)
        M.index_add_(0, ra * A + sa, w_.double())
        mx.scatter_reduce_(0, ra, w_, reduce="amax")
        big.index_add_(0, ra, (w_ > 5.4).double())
    return M.view(A, A).cpu().numpy(), mx.cpu().numpy(), big.cpu().numpy()


M0, _, _ = snap()
tot0 = M0.sum(1)
done = 0
print(f"版 {B.VERSION} ／ {NP}番まで {EV}番ごと ／ 受け手の興奮の辺の重さの和（0番比）・最大・5.4 を越える辺", flush=True)
while True:
    M, mx, big = snap()
    tot = M.sum(1)
    ratio = tot / np.maximum(tot0, 1e-9)
    top = np.argsort(-ratio)[:6]
    line = "・".join(f"{NAMES[i]} {ratio[i]:.1f}倍（最大 {mx[i]:.0f}・>5.4 {int(big[i]):,}）" for i in top)
    print(f"{done:3d}番 {life.t:6d}歩 ｜ {line}", flush=True)
    for i in top[:3]:
        srt = np.argsort(-M[i])[:6]
        print("      " + NAMES[i] + " ← " + "・".join(f"{NAMES[j]} {M[i, j]:.0f}（0番 {M0[i, j]:.0f}）" for j in srt if M[i, j] > 0), flush=True)
    if done >= NP:
        break
    life.run_plans(EV, learn=True)
    done += EV
