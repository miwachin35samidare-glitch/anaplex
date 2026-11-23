"""版103g の確かめ（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 check_v103g.py            ⇒ 皮質 15領野の時間細胞・傾きの細胞が調整する側か／抑制の体の興奮の相手が遅れの列か
  ANAPLEX_CX_TC=drive python3 check_v103g.py   ⇒ 103f と同じ（調整する側は海馬・上丘だけ）か
"""
import os, sys, torch
import babble as B, torch_area as T
from torch_brain import Brain, NAMES, CX_TC, TC_MOD

TEST = "--test" in sys.argv
dev = "cpu" if TEST else B.DEV
drive = os.environ.get("ANAPLEX_CX_TC") == "drive"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
g = b.g
ch_off = [0]
for a in g.areas:
    ch_off.append(ch_off[-1] + a.ch.n_cells)
is_tc = torch.zeros(g.n_send_all, dtype=torch.bool, device=dev)
is_tc[:ch_off[-1]] = True
bad = 0
print(f"皮質の時間細胞：{'撃たせる側（103f と同じ）' if drive else '調整する側（103g）'}", flush=True)
print(f"  {'領野':<8s} {'時間細胞の調整の割合':>10s} {'抑制の体の興奮の相手のうち時間細胞':>14s}", flush=True)
for i, nm in enumerate(NAMES):
    a = g.areas[i]
    if a.ch.n_cells == 0:
        continue
    lo = ch_off[i]
    per = a.ch.n_cell + a.ch.n_ramp
    m = g.mod_send[lo:lo + (a.n_ch + a.n_ext) * per].float().mean().item() if per else 0.0
    # 抑制の体の行へ来る興奮の辺の送り手が時間細胞か
    r_lo = int(g.off_recv[i]) + 2 * a.n_ch
    r_hi = r_lo + a.n_inh
    e = (g.e_recv >= r_lo) & (g.e_recv < r_hi) & ~g.e_inh
    f = is_tc[g.e_src[e].long()].float().mean().item() if bool(e.any()) else float("nan")
    want_mod = nm in TC_MOD
    ok = (m > 0.999) == want_mod and ((f < 1e-9) == want_mod or not want_mod)
    if nm in CX_TC and drive:
        ok = ok and m < 1e-9
    bad += int(not ok)
    print(f"  {nm:<8s} {m:10.3f} {f:14.3f}  {'✓' if ok else '✗'}", flush=True)
print("✅ 全部 ✓" if bad == 0 else f"⚠️ ✗ が {bad} 個", flush=True)
