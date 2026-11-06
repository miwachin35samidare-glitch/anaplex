"""版107f：50番で網が静かになる元を分ける（その2：興奮・抑制の辺を分け、領野ごとの率も）
  ⟲ その1（measure_quiet107）：恒常性（scale・scale_up）は 50番まで 1.000 のまま・辺は皮質と海馬へ増え重さの和は CA1 で 3,952倍 ⇒ 抑制の辺が強まったかを見る
  元の説明：（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_quiet107b.py [番=50] [何番ごとに見る=5] [--win=率を見る歩 20]
  版107l-2：皮質の行に「的」（下からの入力 × 盛りだけで閾値を越えるか＝網の中の辺の学びの的。err ＝ 的 − 予測）の平均も Hz の目盛りで出す
  版107l：--win で率を見る歩を長くできる。皮質の率は 20歩ずつの塊の 一番低い〜一番高い も出す（107l で行は 0.03Hz・終わりの表は 2〜8Hz とずれた ⇒ 撃つ・黙るの波か、見る所の当たり外れか）
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
WIN = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--win=")), 20))
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
_rx = rows(list(TB.X_CX_AREAS))
RCX = _rx[~g.is_inh_r[_rx]]                     # 版107l-2：盛りを振る皮質の興奮の行（表現・誤差）


def per_area(steps=None):
    steps = WIN if steps is None else steps
    acc = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    for _ in range(steps):
        life.step()
        acc += g._any[0][:g.n_ch_all].double()
    g.flush_all()
    z = lambda: torch.zeros(g.n_recv_all, device=dev)
    we, wi, ne, ni = z(), z(), z(), z()
    E, st = len(g.w), 1 << 24                       # 版107h：辺の数の大きさの一時の置き場所を作らない（塊ごと）
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        er, w_, ih = g.e_recv[sl].long(), g.w[sl].float(), g.e_inh[sl].bool()
        lv = (w_ > 0).float()
        we.index_add_(0, er, torch.where(ih, 0.0, w_)); wi.index_add_(0, er, torch.where(ih, w_, 0.0))
        ne.index_add_(0, er, torch.where(ih, 0.0, lv)); ni.index_add_(0, er, torch.where(ih, lv, 0.0))
    out = {}
    for i, nm in enumerate(NAMES):
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        o, n = int(b.off_ch[i]), b.n[nm]
        out[nm] = (float(we[lo:hi].sum()), float(wi[lo:hi].sum()), float(ne[lo:hi].sum()), float(ni[lo:hi].sum()),
                   float(acc[o:o + n].mean()) / steps / 0.01067)
    return out


CXI = torch.cat([torch.arange(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm], device=dev) for nm in TB.X_CX_AREAS])


def rate_cx(steps=None):
    """盛りを振る皮質の表現の平均率（領野ごとの平均の平均）と、20歩ずつの塊の 一番低い・一番高い"""
    steps = WIN if steps is None else steps
    acc = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    blk, cur, k = [], torch.zeros((), device=dev, dtype=torch.float64), 0
    tg = torch.zeros((), device=dev, dtype=torch.float64)
    for _ in range(steps):
        life.step()
        a_ = g._any[0][:g.n_ch_all].double()
        acc += a_
        cur += a_[CXI].mean(); k += 1
        t_ = getattr(g, "_tgt", None)
        if t_ is not None:
            tg += t_[RCX].double().mean()
        if k == 20:
            blk.append(float(cur) / 20 / 0.01067); cur.zero_(); k = 0
    rs = []
    for nm in TB.X_CX_AREAS:
        i = NAMES.index(nm)
        o, n = int(b.off_ch[i]), b.n[nm]
        rs.append(float(acc[o:o + n].mean()) / steps / 0.01067)
    rate_cx.tgt = float(tg) / steps / 0.01067
    return float(np.mean(rs)), (min(blk) if blk else float("nan")), (max(blk) if blk else float("nan"))


p0 = per_area()
bu0, _m = bu_area()
print("番 ・ 歩 ・ 状態 ・ 門（黒質・VTA）・ 皮質の率 ・ scale（皮質・核・線条体）・ scale_up（皮質・核・線条体）・ 生きている辺 ・ 辺の重さの平均", flush=True)
done = 0
while True:
    rc, rlo, rhi = rate_cx()
    g.flush_all()
    nl_, sw_ = 0, 0.0
    for a0 in range(0, len(g.w), 1 << 24):         # 版107h：塊ごと
        w_ = g.w[a0:a0 + (1 << 24)]
        nl_ += int((w_ > 0).sum()); sw_ += float(w_.sum())
    st = "眠り" if getattr(g, "sleep_mode", False) else "起き"
    sc = "・".join(f"{float(g.scale[r].mean()):.3f}" for r in R.values())
    su = "・".join(f"{float(g.scale_up[r].mean()):.3f}" for r in R.values())
    print(f"{done:3d}番 {life.t:6d}歩 {st} 門 {float(b.gate[0]):.2f}・{float(b.gate[1]):.2f} 皮質 {rc:5.2f}Hz（20歩ずつ {rlo:.2f}〜{rhi:.2f}・的 {rate_cx.tgt:.2f}）scale {sc} scale_up {su} "
          f"辺 {nl_:,} 重さ {sw_ / max(1, nl_):.4f} 下からの線の重さ {bu_area()[1]:.4f}", flush=True)
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
