"""版111 の確かめ（容器・CPU・小さい網）：運動の表現・誤差に目標 9.6Hz・下げない印が外れ・境目が当たる。ほかは版110 のまま
  python3 check_v111.py
"""
import math, torch
import babble as B
B.DEV = "cpu"
import torch_area as T, torch_group2 as G, torch_brain as TB
from torch_brain import Brain, NAMES

assert B.VERSION == 111, B.VERSION
ok = True


def ck(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'✓' if cond else '✗'} {name}{extra}", flush=True)


torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
DT = T.DT_MS / 1000.0
i = NAMES.index("運動")
lo, hi, nc = int(g.off_recv[i]), int(g.off_recv[i + 1]), b.areas["運動"].n_ch
t = g.target_recv[lo:lo + 2 * nc] / DT
ck("1 運動の表現・誤差の目標は 9.6Hz", bool(((t - 9.6).abs() < 1e-4).all()), f"：{float(t.min()):.3f}〜{float(t.max()):.3f}Hz（{2 * nc} 行）")
ck("2 運動の表現・誤差は下げない印なし", int(g.scale_skip[lo:lo + 2 * nc].sum()) == 0)
ck("3 運動の抑制の体は下げない印のまま", bool(g.scale_skip[lo + 2 * nc:hi].all()), f"：{hi - lo - 2 * nc} 行")
ck("4 初めの率（rate_ema・rate_long）は目標から", bool((g.rate_ema[lo:lo + 2 * nc] == g.target_recv[lo:lo + 2 * nc]).all()
                                             and (g.rate_long[lo:lo + 2 * nc] == g.target_recv[lo:lo + 2 * nc]).all()))
b._bcm107("bu")
for nm, key in (("下からの線", "bu"), ("CL", "cl")):
    rows, use = b._bcm_rows[key]
    m = (rows >= lo) & (rows < lo + 2 * nc)
    if key == "bu":
        m &= ~b.but.hippo
    n_m = int(m.sum())
    ck(f"5 運動が受け手の{nm}に境目が当たる", n_m > 0 and bool(use[m].all()), f"：{n_m:,} 本")
# 式の表：運動の率 → 境目のずれ（ln・75組あたりにする前）・起きているあいだの上げ・眠りの下げ（1回ぶん）
print("  運動の率  q     境目のずれ（学びの ln から引く）  境目 d0'   起きている上げ（1回の倍 − 1）  眠りの下げ（1回の倍）")
for hz in (0.0, 2.0, 5.7, 9.6, 15.0, 29.0, 53.0):
    q = torch.tensor(hz / 9.6)
    sh = float(TB._bcm_shift(q))
    d0p = min(1.0, TB.BCM_D0 * float(q) ** TB.BCM_POW)
    up = math.exp(G.K_WAKE_UP * max(0.0, 1.0 - float(q)))
    dn = min(1.0, max(0.5, 1 - 0.01 * (float(q) - 1)))
    print(f"  {hz:5.1f}Hz {float(q):5.2f}  {sh:+.4f}                    {d0p:.3f}     {up - 1:.3e}                 {dn:.4f}")
sh0, sh1, sh3 = (float(TB._bcm_shift(torch.tensor(x))) for x in (0.0, 1.0, 3.0))
ck("6 的どおり（9.6Hz）でずれ 0・撃ちすぎで境目が上がる（引く量が正 ＝ 弱まる側）・静かで下がる（強まる側）", abs(sh1) < 1e-6 and sh3 > 0 > sh0)
print("\n全部 ✓" if ok else "\n✗ あり")
