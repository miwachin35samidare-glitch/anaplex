"""版110・111：率の揺れ（約35〜40秒）と一緒に、境目のならし（率÷目標）・下からの線の重さ・CL の重さが揺れるかを並べる（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_wave111b.py [歩=15000] [刻み=200]
  何のため：wave111・wave110b（data/arkhe_wave111_1116.txt・arkhe_wave110b_1116.txt）で、皮質が版110b でも版111 でも約3,300〜3,800歩の周期で揺れていた。
    見立て「滑る境目のならし（rate_ema 約21秒）の遅れで行き過ぎる」を分ける：
      当たりなら ⇒ 率が山のあと ならし（q）が遅れて山になり、q ＞ 1 のあいだ重さが下がり、q ＜ 1 のあいだ上がる（重さが率の約1/4周期あとを揺れる）
      外れなら ⇒ 重さは揺れない（率だけが揺れる）⇒ 元は別（門・網の中の辺・ほか）
  出すもの（刻みごとに1行）
    皮質（運動の外）：その刻みの表現の率（領野の平均の平均）・q（rate_ema ÷ 目標、表現の行の中央を領野で平均）・下からの線の重さの和/体（領野の平均）・CL の重さの和/体（領野の平均）
    運動：同じ4つ ／ CA1 の率 ／ 門（third）
  毎歩に足す仕事：表現の発火（体の数 110,406）を1本の数えへ足す 1回。線・辺には触らない・少ない箱への足し込みなし・写しなし
  刻みごと（200歩に1回）：下からの線の道ごとの切り出しの和（皮質へ入る道だけ・写しなし）・CL を (受け手, 本) に並べた和 ⇒ measure_term109 の区切りの数えと同じ（そちらは区切り 4回、こちらは 75回）
  歩は life.step（見張り・休み・止めはそのまま働く）。グラフで回す（ふだんと同じ）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 15000
EV = int(args[1]) if len(args) > 1 else 200
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION not in (110, 111):                       # 11-16：版110b でも回す（揺れが版111 からか、前からかを比べる）
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版110・111 で回す")

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
n_all = g.n_ch_all
DT = 10.67 / 1000.0
CX = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質" and n != "運動"]
HIP = [n for n in ("歯状回", "CA3", "CA1") if n in NAMES]
SL = {nm: slice(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm]) for nm in CX + ["運動"] + HIP}

ALL = CX + ["運動"]
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
is_rep_r = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev)
is_rep_r[g.rep_i.long()] = True
RR = {}
for nm in ALL:
    i = NAMES.index(nm)
    rr = torch.arange(int(g.off_recv[i]), int(g.off_recv[i + 1]), device=dev)
    RR[nm] = rr[is_rep_r[rr]]
tab = b.but
paths = {nm: [] for nm in ALL}                           # 領野 → [(切り出し, 受け手の数)]
for i, k in enumerate(tab.keys):
    nm = ch0_name.get(int(b.bu[k].ch0), "?")
    if nm in paths:
        paths[nm].append((slice(int(tab.eo[i]), int(tab.eo[i + 1])), max(1, tab.nt[i])))
cl_on = bool(getattr(b, "cl_on", False))
if cl_on:
    nr = len(b.cl_ch)
    kcl = len(b.cl_w) // nr
    assert kcl * nr == len(b.cl_w) and bool((b.cl_recv.long().view(nr, kcl) == torch.arange(nr, device=dev)[:, None]).all()), "CL の並びが思っていた形でない"
    cl_ch = b.cl_ch.long()
    CLM = {nm: (cl_ch >= SL[nm].start) & (cl_ch < SL[nm].stop) for nm in ALL}


def weights():
    q = g.rate_ema / g.target_recv.clamp(min=1e-9)
    clw = b.cl_w.view(nr, kcl).sum(1, dtype=torch.float64) if cl_on else None
    out = {}
    for nm in ALL:
        wa = sum(float(tab.w[s].sum(dtype=torch.float64)) / nt for s, nt in paths[nm])
        cw = float(clw[CLM[nm]].mean()) if cl_on and bool(CLM[nm].any()) else float("nan")
        out[nm] = (float(q[RR[nm]].median()), wa, cw)
    return out


acc = torch.zeros(n_all, device=dev)
print(f"版 {B.VERSION} ／ {NS} 歩・刻み {EV} 歩（{EV * DT:.1f} 秒）／ 皮質（運動の外）{len(CX)} 領野 ／ 体 {n_all:,}", flush=True)
print(f"  {'歩':>6s} ｜ 皮質 {'率Hz':>6s}{'q':>6s}{'下からw':>8s}{'CLw':>7s} ｜ 運動 {'率Hz':>6s}{'q':>6s}{'下からw':>8s}{'CLw':>7s} ｜ {'CA1':>5s} ｜ 門", flush=True)
for i in range(1, NS + 1):
    life.step(learn=True)
    acc.add_(g._any[0][:n_all].float())
    if i % EV == 0:
        L.beat()
        r = acc / EV / DT
        W = weights()
        L.beat()
        cx = np.array([[float(r[SL[nm]].mean()), *W[nm]] for nm in CX]).mean(0)
        mo = [float(r[SL["運動"]].mean()), *W["運動"]]
        th = g.third.reshape(-1).tolist() if torch.is_tensor(g.third) else list(g.third)
        print(f"  {life.t:6d} ｜      {cx[0]:6.2f}{cx[1]:6.2f}{cx[2]:8.3f}{cx[3]:7.3f} ｜      {mo[0]:6.2f}{mo[1]:6.2f}{mo[2]:8.3f}{mo[3]:7.3f} ｜ "
              f"{float(r[SL['CA1']].mean()):5.2f} ｜ {'・'.join(f'{x:.2f}' for x in th)}", flush=True)
        acc.zero_()
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
