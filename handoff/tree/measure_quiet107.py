"""版107e：50番で網が静かになる元を分ける（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_quiet107.py [番=50] [何番ごとに見る=5]
  knobs.json があればそれで（版107e）。5番ごとに 1行：歩・起きている／眠り・門（黒質・VTA）・皮質の表現の率・
    恒常性の下げ（scale、眠りで下げる）と上げ（scale_up、起きていて上げる）の皮質・核・線条体の平均・生きている辺の数・辺の重さの平均
  終わりに領野ごとの 0番 → 終わりの比（scale・scale_up・入る辺の重さの和・辺の数）
"""
import os, sys, numpy as np, torch
# 版107h：グラフで回す（⟲ eager は取り置きが増えて 12〜15GB【測】11-05）。eager にするなら ANAPLEX_GRAPH=eager
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NP = int(args[0]) if args else 50
EV = int(args[1]) if len(args) > 1 else 5
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
CX = [nm for nm in NAMES if TB.PRIOR_TYPE.get(nm) == "皮質"]
NUCS = list(TB.NUC)
STRS = ["尾状核", "被殻", "側坐核", "ストリオソーム"]


def rows(nms):
    idx = []
    for nm in nms:
        i = NAMES.index(nm)
        idx.append(torch.arange(int(g.off_recv[i]), int(g.off_recv[i + 1]), device=dev))
    return torch.cat(idx)


R = {"皮質": rows(CX), "核": rows(NUCS), "線条体": rows(STRS)}


def per_area():
    g.flush_all()
    live = g.w > 0
    wsum = torch.zeros(g.n_recv_all, device=dev).index_add_(0, g.e_recv.long(), g.w.float())
    ecnt = torch.zeros(g.n_recv_all, device=dev).index_add_(0, g.e_recv.long(), live.float())
    out = {}
    for i, nm in enumerate(NAMES):
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        out[nm] = (float(g.scale[lo:hi].mean()), float(g.scale_up[lo:hi].mean()), float(wsum[lo:hi].sum()), float(ecnt[lo:hi].sum()))
    return out


def rate_cx(steps=20):
    acc = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    for _ in range(steps):
        life.step()
        acc += g._any[0][:g.n_ch_all].double()
    rs = []
    for nm in TB.X_CX_AREAS:
        i = NAMES.index(nm)
        o, n = int(b.off_ch[i]), b.n[nm]
        rs.append(float(acc[o:o + n].mean()) / steps / 0.01067)
    return float(np.mean(rs))


p0 = per_area()
print("番 ・ 歩 ・ 状態 ・ 門（黒質・VTA）・ 皮質の率 ・ scale（皮質・核・線条体）・ scale_up（皮質・核・線条体）・ 生きている辺 ・ 辺の重さの平均", flush=True)
done = 0
while True:
    rc = rate_cx()
    g.flush_all()
    live = g.w > 0
    st = "眠り" if getattr(g, "sleep_mode", False) else "起き"
    sc = "・".join(f"{float(g.scale[r].mean()):.3f}" for r in R.values())
    su = "・".join(f"{float(g.scale_up[r].mean()):.3f}" for r in R.values())
    print(f"{done:3d}番 {life.t:6d}歩 {st} 門 {float(b.gate[0]):.2f}・{float(b.gate[1]):.2f} 皮質 {rc:5.2f}Hz scale {sc} scale_up {su} "
          f"辺 {int(live.sum()):,} 重さ {float(g.w[live].mean()):.4f}", flush=True)
    if done >= NP:
        break
    life.run_plans(EV, learn=True)
    done += EV
p1 = per_area()
print("\n領野ごと 0番 → 終わり（比）：scale ・ scale_up ・ 入る辺の重さの和 ・ 辺の数", flush=True)
for nm in NAMES:
    a, c = p0[nm], p1[nm]
    r_ = lambda x, y: (y / x) if x else float("nan")
    print(f"  {nm:<8s} {r_(a[0], c[0]):6.3f} {r_(a[1], c[1]):7.3f} {r_(a[2], c[2]):7.3f} {r_(a[3], c[3]):7.3f}", flush=True)
