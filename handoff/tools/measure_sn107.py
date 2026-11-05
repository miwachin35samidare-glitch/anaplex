"""版107j：門（黒質・VTA）が皮質より先に落ちる元を分ける（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_sn107.py [番=12] [何番ごとに見る=1]
  【み】11-06 07:27「いいよ」。measure_quiet107b（版107j）：5番で門 1.22 → 0.49、皮質はまだ 3.11Hz ⇒ 黒質を下げている入力を線ごとに分ける
  グラフのまま回す（eager にしない＝VRAM を増やさない）。値は「重さ × 送り手の率」で、線の重さが変わった（学び）のか、
    送り手の率が変わったのかを分ける：
    核どうしの線：送り手の領野・表現／誤差・興奮／抑制ごとに、受け手1体あたり Σ w × 送り手の率（Hz）と、重さの和の比（÷ 初め）
    下からの線：道ごとの重さの和の比 ／ 網の中の辺：興奮・抑制の重さの和の比
    受け手の側：率・閾値に掛かる倍率（th_gain ＝ 覚醒 × 5-HT・ACh）・scale・scale_up・x_gain
"""
import os, sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NP = int(args[0]) if args else 12
EV = int(args[1]) if len(args) > 1 else 1
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
n_all = g.n_ch_all
DT = 10.67 / 1000.0
TGT = ("黒質", "VTA")
ch_area = torch.cat([torch.full((b.n[nm],), i, dtype=torch.long, device=dev) for i, nm in enumerate(NAMES)])
nn_ = g.nucnet
src_all, dch_all, sg_all, _dl = nn_.flat()
s_area = ch_area[src_all % n_all]
s_kind = (src_all >= n_all).long()            # 0 表現 ／ 1 誤差
d_area = ch_area[dch_all]
LINES = {}                                     # (受け手, 送り手, 種, 符号) → 線の番号
for t in TGT:
    ti = NAMES.index(t)
    sel = torch.nonzero(d_area == ti).squeeze(1)
    if len(sel) == 0:
        continue
    key = s_area[sel] * 4 + s_kind[sel] * 2 + (sg_all[sel] > 0).long()
    for k in torch.unique(key).tolist():
        sa, r = divmod(k, 4)
        kind, sgn = divmod(r, 2)
        LINES[(t, NAMES[sa], "誤差" if kind else "表現", "興奮" if sgn else "抑制")] = sel[key == k]
BUK = {t: [k for k, s_ in b.bu.items() if int(s_.ch0) == int(b.off_ch[NAMES.index(t)])] for t in TGT}


def bu_w():
    t_ = b.but
    out = {}
    for i, k in enumerate(t_.keys):
        out[k] = float(t_.w[t_.eo[i]:t_.eo[i + 1]].float().sum())
    return out


def g_w():
    """網の中の辺：受け手の領野ごとの興奮・抑制の重さの和"""
    g.flush_all()
    we, wi = torch.zeros(g.n_recv_all, device=dev), torch.zeros(g.n_recv_all, device=dev)
    E, st = len(g.w), 1 << 24
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        er, w_, ih = g.e_recv[sl].long(), g.w[sl].float(), g.e_inh[sl].bool()
        we.index_add_(0, er, torch.where(ih, 0.0, w_)); wi.index_add_(0, er, torch.where(ih, w_, 0.0))
    out = {}
    for t in TGT:
        i = NAMES.index(t)
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        out[t] = (float(we[lo:hi].sum()), float(wi[lo:hi].sum()))
    return out


def snap(steps=20):
    ar = torch.zeros(n_all, device=dev, dtype=torch.float64)
    ae = torch.zeros(n_all, device=dev, dtype=torch.float64)
    for _ in range(steps):
        life.step()
        ar += g._any[0][:n_all].double()
        ae += g._any[1][g.err_pos_ch].double()
    rate = torch.cat([ar, ae]) / steps / DT     # [表現 | 誤差] の Hz（送り手の並び）
    w = nn_.w.double()
    row = {"番": done, "歩": life.t, "門": (float(b.gate[0]), float(b.gate[1]))}
    cx = []
    for nm in TB.X_CX_AREAS:
        i = NAMES.index(nm)
        o, n = int(b.off_ch[i]), b.n[nm]
        cx.append(float(ar[o:o + n].mean()) / steps / DT)
    row["皮質"] = float(np.mean(cx))
    thg = g.th_gain if torch.is_tensor(g.th_gain) else torch.tensor(float(g.th_gain), device=dev)
    for t in TGT:
        i = NAMES.index(t)
        o, n = int(b.off_ch[i]), b.n[t]
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        tg = float(thg[o:o + n].mean()) if thg.dim() else float(thg)
        row[t] = dict(率=float(ar[o:o + n].mean()) / steps / DT, th=tg,
                      scale=float(g.scale[lo:hi].mean()), up=float(g.scale_up[lo:hi].mean()),
                      xg=float(g.x_gain[o]))
    for k, idx in LINES.items():
        n = b.n[k[0]]
        row[k] = (float((w[idx] * rate[src_all[idx]]).sum()) / n, float(w[idx].sum()), float(rate[src_all[idx]].mean()))
    row["bu"] = bu_w()
    row["g"] = g_w()
    return row


done = 0
H = []
while True:
    H.append(snap())
    r = H[-1]
    print(f"{done:3d}番 {life.t:6d}歩 門 {r['門'][0]:.2f}・{r['門'][1]:.2f} 皮質 {r['皮質']:5.2f}Hz  "
          + "  ".join(f"{t} {r[t]['率']:5.1f}Hz th×{r[t]['th']:.3f}" for t in TGT), flush=True)
    if done >= NP:
        break
    life.run_plans(EV, learn=True)
    done += EV

nb = [h["番"] for h in H]
print("\n番:" + "".join(f"{x:>8d}" for x in nb), flush=True)
for t in TGT:
    print(f"\n■ {t}", flush=True)
    for lab, f in (("率 Hz", lambda h: h[t]["率"]), ("閾値の倍率", lambda h: h[t]["th"]), ("scale", lambda h: h[t]["scale"]),
                   ("scale_up", lambda h: h[t]["up"]), ("x_gain", lambda h: h[t]["xg"])):
        print(f"  {lab:<32s}" + "".join(f"{f(h):8.3f}" for h in H), flush=True)
    ks = sorted([k for k in LINES if k[0] == t], key=lambda k: -abs(H[0][k][0]))
    print("  核どうしの線（受け手1体あたり Σ 重さ × 送り手の率 Hz ／ 下の行：重さの和の比・送り手の平均率 Hz）", flush=True)
    for k in ks:
        lab = f"{k[3]} {k[1]}（{k[2]}）"
        print(f"  {lab:<30s}" + "".join(f"{h[k][0]:8.2f}" for h in H), flush=True)
        w0 = H[0][k][1]
        print(f"  {'  重さの比':<30s}" + "".join(f"{(h[k][1] / w0 if w0 else float('nan')):8.3f}" for h in H), flush=True)
        print(f"  {'  送り手 Hz':<30s}" + "".join(f"{h[k][2]:8.2f}" for h in H), flush=True)
    for k in BUK[t]:
        w0 = H[0]["bu"][k]
        print(f"  下から {k:<26s}" + "".join(f"{(h['bu'][k] / w0 if w0 else float('nan')):8.3f}" for h in H) + "（重さの和の比）", flush=True)
    e0, i0 = H[0]["g"][t]
    print(f"  {'網の中 興奮（比）':<30s}" + "".join(f"{(h['g'][t][0] / e0 if e0 else float('nan')):8.3f}" for h in H), flush=True)
    print(f"  {'網の中 抑制（比）':<30s}" + "".join(f"{(h['g'][t][1] / i0 if i0 else float('nan')):8.3f}" for h in H), flush=True)
