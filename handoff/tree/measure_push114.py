"""版114（11-24 時間細胞の辺の重さを足した）：皮質（運動の外 20領野）の表現の体が撃った歩に、押しがどこから来たか（arkhe の ~/anaplex で。網は変えない・保存しない）
  python3 measure_push114.py [歩=15000]
  ANAPLEX_RR=0 python3 measure_push114.py 15000   ← 表現 → 表現 なし（比べる元）
  何のため：版112〜114 のどれでも皮質の発火の 7割余りは「皮質の 20% 以上が同じ歩に撃つ歩」（組み表_版114_案 §8）。
    撃った歩の押しが CL・下からの線・表現 → 表現・持ち越しのどれから来ているかを、そろう歩とそれ以外の歩に分けて見る
    （measure_srccx107 は歩の平均の押しだった。50 §9 11-09「撃った歩の押しで確かめる」）
  押し（÷閾値 fire_th × th_gain。⚠️ 抑制の割り算 1＋抑制 の前。持ち越しは割り算のあとの値）
    CL       cl_drive（その歩に届いた CL の線）
    下から   x_all × x_gain（下からの線の和）
    rr       表現 → 表現 の辺：Σ 接触 × w × 送り手が前の歩に撃ったか（⚠️ 短期の減りを入れない ⇒ 上の側。誤差の行の分は入れない）
    持越     v_carry（歩に入るときの値）÷閾値
    ⚠️ 網の中の辺のうち rr 以外（外から入る遅れの列）は入れない（§12 の測りで w 和 0〜1.3）。時間細胞は既定では調整の側で撃たせない
  11-24 足した：時間細胞・傾きの細胞 → 皮質20 の表現 の辺の重さ（初めは 0、学びで育つ）を区間ごとに。
    1体あたりの和 Σ 接触 × w ÷ 閾値 ＝ その体の時間細胞が全部撃ったときの押しの上限。ANAPLEX_CX_TC=drive のとき撃たせる側
  毎歩に足す仕事：3,256体の小さい足し算と、rr の辺 約11.6万本の index_add（受け手 3,256 の箱）。読むのは区間の終わりだけ
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, torch_group2 as G, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 15000
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION not in (114, 115):                 # 11-23：版115・115b（ANAPLEX_PPNCL_FAST）でも回す
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版114・115 で回す")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
CX = [nm for nm in NAMES if TB.PRIOR_TYPE.get(nm) == "皮質" and nm != "運動"]
off_ch = [int(x) for x in g.off_ch]
Lc = torch.cat([torch.arange(off_ch[NAMES.index(n)], off_ch[NAMES.index(n) + 1], device=g.w.device) for n in CX])
nL = len(Lc)
if TB.RR_ON:
    assert torch.equal(Lc, g.rr_L_ch), "当てる領野の並びが rr と違う"
L_area = torch.cat([torch.full((g.areas[NAMES.index(n)].n_ch,), j, dtype=torch.long, device=g.w.device) for j, n in enumerate(CX)])
OH = torch.zeros(nL, len(CX), device=g.w.device)
OH[torch.arange(nL, device=g.w.device), L_area] = 1.0             # 領野ごとの和は 1 つの行列の掛け算で（箱への足し込みなし）
C_EXC = float(g.c_exc[int(g.rr_L_row[0])]) if TB.RR_ON else 3.3
COMP = ("CL", "下から", "rr", "持越")
_tcm = torch.zeros(g.n_send_all, dtype=torch.bool, device=g.w.device)
for _nm in CX:
    _a, _i = b.areas[_nm], NAMES.index(_nm); _per = _a.ch.n_cell + _a.ch.n_ramp
    if _per:
        _lo = int(g.off_send[_i]); _tcm[_lo:_lo + (_a.n_ch + _a.n_ext) * _per] = True
_rowmap = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=g.w.device)
_rowmap[g.rep_i[Lc].long()] = torch.arange(nL, device=g.w.device)
TC_E = torch.nonzero(_tcm[g.e_src.long()] & ~g.e_inh & (_rowmap[g.e_recv.long()] >= 0)).squeeze(1)
TC_R = _rowmap[g.e_recv.long()[TC_E]]
print(f"時間細胞 → 皮質20 の表現 の辺 {TC_E.numel():,} 本（1体あたり {TC_E.numel() / nL:.0f}）", flush=True)
cap = {}
_step0 = G.Group2.step


def _step_spy(self, x_all, ext_all, learn=True):
    cap["x"] = x_all
    cap["vc"] = self.v_carry[Lc].clone() if getattr(self, "v_carry", None) is not None else torch.zeros(nL, device=Lc.device)
    return _step0(self, x_all, ext_all, learn)


G.Group2.step = _step_spy
# 区間ごとの和（GPU の上）：[成分, 種類]。種類 0 そろう歩で撃った・1 それ以外の歩で撃った・2 全部の体と歩
S = torch.zeros(len(COMP), 3, len(CX), device=g.w.device, dtype=torch.float64)
N = torch.zeros(3, len(CX), device=g.w.device, dtype=torch.float64)
NSY = torch.zeros(1, device=g.w.device, dtype=torch.float64)
last = {"t": 0}


def window(t):
    L.beat()
    s_ = S.cpu().numpy(); n_ = N.cpu().numpy(); dt = t - last["t"]; nsy = float(NSY.cpu())
    print(f"\n━━ {last['t']}〜{t} 歩（そろう歩 ＝ 20領野の表現の体の 20% 以上が撃った歩：{nsy / max(dt, 1):.1%}）押しは ÷閾値 ━━", flush=True)
    tot_f = n_[0].sum() + n_[1].sum()
    print(f"  皮質20 の発火 {tot_f:,.0f}（そろう歩に {n_[0].sum() / max(tot_f, 1):.1%}）・率 {tot_f / nL / max(dt, 1) / 0.01067:.2f}Hz", flush=True)
    for kk, lab in ((0, "そろう歩で撃った体"), (1, "それ以外の歩で撃った体"), (2, "全部の体・全部の歩（平均）")):
        nn = n_[kk].sum()
        vals = [s_[c, kk].sum() / max(nn, 1) for c in range(len(COMP))]
        tot = sum(vals)
        print(f"  {lab:<16s}" + "  ".join(f"{COMP[c]} {vals[c]:.3f}（{vals[c] / tot:.0%}）" if tot > 0 else f"{COMP[c]} —" for c in range(len(COMP)))
              + f"  計 {tot:.3f}", flush=True)
    print(f"    {'領野':<8s}{'撃った':>8s}{'そろう歩':>8s} ｜ 撃った体の押し（全部の歩）：" + "".join(f"{c:>8s}" for c in COMP), flush=True)
    for j, nm in enumerate(CX):
        nf = n_[0, j] + n_[1, j]
        v = [(s_[c, 0, j] + s_[c, 1, j]) / max(nf, 1) for c in range(len(COMP))]
        print(f"    {nm:<8s}{nf:8.0f}{n_[0, j] / max(nf, 1):8.0%} ｜ " + " " * 18 + "".join(f"{x:8.3f}" for x in v), flush=True)
    th_ = (g.fire_th * g.th_gain)
    th_ = th_[Lc] if torch.is_tensor(th_) and th_.dim() > 0 and th_.numel() > 1 else torch.as_tensor(th_, device=Lc.device).expand(nL)
    wt = g.w[TC_E]
    ps = torch.zeros(nL, device=Lc.device, dtype=wt.dtype).index_add_(0, TC_R, C_EXC * wt) / th_
    print(f"  時間細胞 → 表現 の辺：w 平均 {float(wt.mean()):.4f}・0 より大きい {float((wt > 0).float().mean()):.1%}・"
          f"1体あたり Σ接触×w÷閾値 中央 {float(ps.median()):.3f}・90% {float(ps.quantile(0.9)):.3f}", flush=True)
    S.zero_(); N.zero_(); NSY.zero_(); last["t"] = t
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    L.beat()


print(f"皮質の時間細胞：{'撃たせる側（ANAPLEX_CX_TC=' + TB._CX_TC_ENV + '）' if TB._CX_TC_ENV else '調整の側（既定）'}", flush=True)
print(f"PPN → CL の速い分 f {TB.PPNCL_FAST if hasattr(TB, 'PPNCL_FAST') else 1}・g {TB.PPNCL_G if hasattr(TB, 'PPNCL_G') else 1}", flush=True)
print(f"版 {B.VERSION}（ANAPLEX_RR={'1' if TB.RR_ON else '0'}・上限 {'107z' if G.WCAP_107Z else '6mV ÷ 接触'}）／ {NS} 歩 ／ 皮質20 の表現の体 {nL:,}", flush=True)
marks = {m for m in (1000, 3000, 6000, 10000) if m < NS} | {NS}
for i in range(1, NS + 1):
    a1 = b._rr_a1.clone() if TB.RR_ON else None            # この歩に届く送り手の発火（1歩前の列）
    life.step(learn=True)
    th = (g.fire_th * g.th_gain)
    th = th[Lc] if torch.is_tensor(th) and th.dim() > 0 and th.numel() > 1 else torch.as_tensor(th, device=Lc.device).expand(nL)
    fired = g._any[0][Lc] > 0
    comp = torch.zeros(len(COMP), nL, device=Lc.device)
    comp[0] = g.cl_drive[Lc] / th
    comp[1] = (cap["x"][Lc] * g.x_gain[Lc]) / th
    if TB.RR_ON:
        idx = g._rr_idx
        comp[2] = torch.zeros(nL, device=Lc.device).index_add_(0, g._rr_rp, C_EXC * g.w[idx] * a1[g._rr_sp]) / th
    comp[3] = cap["vc"] / th          # ⟲ 11-21 01:3x の回は ÷閾値 を落としていた（持越だけ 約14倍に出た）
    fsum = fired.sum()
    sy = (fsum >= 0.2 * nL).double()
    NSY += sy
    ff = fired.float()
    m0 = ff * sy.float(); m1 = ff * (1.0 - sy.float())
    S[:, 0] += (comp * m0) @ OH
    S[:, 1] += (comp * m1) @ OH
    S[:, 2] += comp @ OH
    N[0] += m0 @ OH
    N[1] += m1 @ OH
    N[2] += OH.sum(0)
    if i % 1000 == 0:
        L.beat()
    if i in marks:
        window(i)
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
