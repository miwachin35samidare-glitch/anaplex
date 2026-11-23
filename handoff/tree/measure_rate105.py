"""版105b：皮質の盛りを振ったときの率（1行1領野。arkhe の ~/anaplex で。保存しない・網は変えない）
  ANAPLEX_XCX=値 python3 measure_rate105.py [先に回す番=0] [測る歩=40]
  【み】11-04 00:17「まぁ盛りからでいいよ」。皮質の受け手の盛り（今 X_NEW 7.14）を振って、高次の皮質の率が人の 3.3Hz に近づく値を探す
  表現・誤差・来ない体の率（1歩に1回まで数える）。⚠️ 学ぶ前（0番）の 40歩。学んで育ったあとは別に見る
"""
import os, sys, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 0
NS = int(args[1]) if len(args) > 1 else 40
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
if WARM > 0:
    life.run_plans(WARM, learn=True)
g = b.g
n_all = g.n_ch_all
any_r = torch.zeros(n_all, device=dev, dtype=torch.float64)
any_e = torch.zeros(len(g.err_i), device=dev, dtype=torch.float64)
t0 = life.t
for _ in range(NS):
    life.step()
    any_r += g._any[0][:n_all].double()
    any_e += g._any[1].double()
ns = max(1, life.t - t0)
DT = 10.67 / 1000.0
err_area, err_om = g.area_of[g.err_i], g.is_om[g.err_i]
AREAS = TB.CL_CORTEX + ("背内側視床", "VA", "視床枕", "扁桃体基底", "視床下核", "黒質", "外節", "尾状核", "被殻", "側坐核")
xcx = os.environ.get("ANAPLEX_XCX", "なし（X_NEW）")
print(f"\n=== ANAPLEX_XCX={xcx} ／ 先に {WARM}番（{t0}歩）・測る {ns}歩 ／ 率 Hz（表現・誤差・来ない）", flush=True)
cx_rep = []
for nm in AREAS:
    i = NAMES.index(nm)
    oc, n = int(b.off_ch[i]), b.n[nm]
    r = float(any_r[oc:oc + n].mean()) / ns / DT
    e_sel = (err_area == i) & ~err_om
    o_sel = (err_area == i) & err_om
    e = float(any_e[e_sel].mean()) / ns / DT if bool(e_sel.any()) else float("nan")
    o = float(any_e[o_sel].mean()) / ns / DT if bool(o_sel.any()) else float("nan")
    xg = float(g.areas[i].x_gain)
    if nm in TB.X_CX_AREAS:
        cx_rep.append(r)
    print(f"  {nm:<8s} x_gain {xg:5.2f}  表現 {r:6.1f}  誤差 {e:6.1f}  来ない {o:6.1f}", flush=True)
print(f"  ⇒ 盛りを振った 12領野の表現の平均 {np.mean(cx_rep):.1f}Hz（人の皮質 3.3Hz）", flush=True)
