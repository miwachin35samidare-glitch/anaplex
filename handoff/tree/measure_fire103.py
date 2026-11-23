"""版103：領野ごとの発火と「撃った辺」（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_fire103.py [歩=100]
  【み】11-01 17:37「暴走で発火率が上がっているところは時間を遅くする要因になるからそれも見てね」
  1歩の網の時間の多くは、撃った送り手の辺を読む核（_p1_flush・_p1_learn 1歩 12ms）。撃った辺が多い領野＝重い領野
  出すもの（暮らしで回し、温めの 60歩のあとから数える）
    領野ごと：撃った辺（1歩あたり・全体の割合）／ 送り手（時間細胞・遅れの列）のうち撃った割合 ／
              表現の体の率（Hz）と恒常性の目標（Hz）／ 誤差の体の率（Hz）
    ⚠️ 率は「見るだけ」（数値は全部組んでから合わせる【み】10-18）。ここでは重さの出どころと暴走の場所を見る
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 100
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(60):
    life.step()
g = b.g
A = len(g.areas)
# 送り手 → 領野（時間細胞の塊 → 各領野の遅れの列）
lab = torch.empty(g.n_send_all, dtype=torch.long, device=dev)
ch_off = [0]
for a in g.areas:
    ch_off.append(ch_off[-1] + a.ch.n_cells)
buf_off = [0]
for a in g.areas:
    buf_off.append(buf_off[-1] + a.n_src * a.D)
bb = ch_off[-1]
for i in range(A):
    lab[ch_off[i]:ch_off[i + 1]] = i
    lab[bb + buf_off[i]:bb + buf_off[i + 1]] = i
deg = (g.start[1:] - g.start[:-1]).float()                 # 送り手ごとの辺の数
edges_by = torch.zeros(A, device=dev).index_add_(0, lab, deg)
fired_e, fired_s = torch.zeros(A, device=dev), torch.zeros(A, device=dev)
n_send = torch.zeros(A, device=dev).index_add_(0, lab, torch.ones_like(deg))
rep_n = torch.zeros(g.n_ch_all, device=dev)             # 表現の体がこの歩のどこかで撃った（帯の並び）
err_n = {nm: torch.zeros(b.n[nm], device=dev) for nm in NAMES}   # 誤差の体（1歩の終わりの up）
P0 = type(g)._fired_edges


def fe(self):
    out = P0(self)
    on = self._base_on.float() if getattr(self, "_base_on", None) is not None else None
    if on is not None and len(on) == len(lab) and not getattr(self, "_fe_seen", False):
        fired_e.add_(torch.zeros(A, device=dev).index_add_(0, lab, on * deg))
        fired_s.add_(torch.zeros(A, device=dev).index_add_(0, lab, on))
        self._fe_seen = True
    return out


type(g)._fired_edges = fe
for _ in range(NS):
    g._fe_seen = False
    life.step()
    rep_n += g._any[0][:g.n_ch_all].float()
    for nm in NAMES:
        err_n[nm] += b.up[nm].float()
hz = lambda c: c / NS / (10.67 / 1000.0)
tot = float(fired_e.sum())
rows = []
for i, nm in enumerate(NAMES):
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    nc = g.areas[i].n_ch
    oc = int(b.off_ch[i])
    rr = rep_n[oc:oc + nc]; ee = err_n[nm]
    tg = g.target_recv[lo + nc:lo + 2 * nc]
    rows.append((float(fired_e[i]) / NS, nm, float(fired_s[i]) / NS / max(1.0, float(n_send[i])), float(hz(rr).mean()),
                 float(tg.mean()) / (10.67 / 1000.0), float(hz(ee).mean()), float(edges_by[i])))
print(f"版 {B.VERSION} ／ 暮らし {NS}歩 ／ 撃った辺 1歩 平均 {tot / NS:,.0f}本（辺の総数 {len(g.w):,}）", flush=True)
print(f"  {'領野':<8s} {'撃った辺/歩':>12s} {'割合':>6s} {'送り手の撃つ割合':>10s} {'表現 Hz':>8s} {'目標 Hz':>8s} {'誤差 Hz':>8s} {'辺':>12s}", flush=True)
for fe_, nm, fs, rh, th, eh, ed in sorted(rows, reverse=True):
    print(f"  {nm:<8s} {fe_:12,.0f} {fe_ / max(1.0, tot / NS):6.1%} {fs:10.1%} {rh:8.1f} {th:8.1f} {eh:8.1f} {ed:12,.0f}", flush=True)
