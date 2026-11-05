"""版107f：50番で網が静かになる元を分ける（その2：興奮・抑制の辺を分け、領野ごとの率も）
  ⟲ その1（measure_quiet107）：恒常性（scale・scale_up）は 50番まで 1.000 のまま・辺は皮質と海馬へ増え重さの和は CA1 で 3,952倍 ⇒ 抑制の辺が強まったかを見る
  元の説明：（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_quiet107b.py [番=50] [何番ごとに見る=5]
  knobs.json があればそれで（版107e）。5番ごとに 1行：歩・起きている／眠り・門（黒質・VTA）・皮質の表現の率・
    恒常性の下げ（scale、眠りで下げる）と上げ（scale_up、起きていて上げる）の皮質・核・線条体の平均・生きている辺の数・辺の重さの平均
  終わりに領野ごとの 0番 → 終わりの比（scale・scale_up・入る辺の重さの和・辺の数）
"""
import os, sys, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
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


def bu_area():
    """下からの線（8本の道・皮質 → 線条体ほか、表で学ぶ）の重さの和を受け手の領野ごとに"""
    t_, na = b.but, g.n_ch_all
    pidx = torch.cat([(torch.arange(n_, device=dev) + (c_ if c_ >= 0 else na)) for c_, n_ in zip(t_.ch0, t_.nt)])
    rc = pidx[t_.tgt.long()]
    ok = rc < na
    s_ = torch.zeros(na, device=dev).index_add_(0, rc[ok], t_.w[ok].float())
    return {nm: float(s_[int(b.off_ch[i]):int(b.off_ch[i]) + b.n[nm]].sum()) for i, nm in enumerate(NAMES)}, float(t_.w.float().mean())


R = {"皮質": rows(CX), "核": rows(NUCS), "線条体": rows(STRS)}


def per_area(steps=20):
    acc = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    for _ in range(steps):
        life.step()
        acc += g._any[0][:g.n_ch_all].double()
    g.flush_all()
    live = (g.w > 0).float()
    inh = g.e_inh.bool()
    er = g.e_recv.long()
    z = lambda: torch.zeros(g.n_recv_all, device=dev)
    we = z().index_add_(0, er[~inh], g.w[~inh].float()); wi = z().index_add_(0, er[inh], g.w[inh].float())
    ne = z().index_add_(0, er[~inh], live[~inh]); ni = z().index_add_(0, er[inh], live[inh])
    out = {}
    for i, nm in enumerate(NAMES):
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        o, n = int(b.off_ch[i]), b.n[nm]
        out[nm] = (float(we[lo:hi].sum()), float(wi[lo:hi].sum()), float(ne[lo:hi].sum()), float(ni[lo:hi].sum()),
                   float(acc[o:o + n].mean()) / steps / 0.01067)
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
bu0, _m = bu_area()
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
          f"辺 {int(live.sum()):,} 重さ {float(g.w[live].mean()):.4f} 下からの線の重さ {bu_area()[1]:.4f}", flush=True)
    if done >= NP:
        break
    life.run_plans(EV, learn=True)
    done += EV
p1 = per_area()
bu1, _m = bu_area()
print("\n領野ごと 0番 → 終わり：表現の率 Hz（0番 → 終わり）・興奮の辺の重さの和（比）・抑制の辺の重さの和（比）・興奮の辺の数（比）・抑制の辺の数（比）・下からの線（8本の道ほか）の重さの和（比）", flush=True)
r_ = lambda x, y: (y / x) if x else float("nan")
for nm in NAMES:
    a, c = p0[nm], p1[nm]
    print(f"  {nm:<8s} {a[4]:6.1f} → {c[4]:6.1f}  興奮 {r_(a[0], c[0]):8.2f}  抑制 {r_(a[1], c[1]):8.2f}  "
          f"辺 {r_(a[2], c[2]):6.2f}・{r_(a[3], c[3]):6.2f}  下からの線 {r_(bu0[nm], bu1[nm]):7.3f}", flush=True)
