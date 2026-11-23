"""版110・111：約40秒の揺れの元を分ける —— 学びの一部を止めて、揺れが消えるかを見る（arkhe の ~/anaplex で。保存しない・測るためだけ）
  python3 measure_wave111d.py 条件 [歩=15000] [刻み=200]
  条件（1回の起動で1つ。同じ種で 0 から組む）
    全部         ふだんの網（比べる元。wave111b と同じ）
    下からCLなし  下からの線と CL → 皮質 の重さを動かさない（網の中の辺・核どうし・恒常性・刈り込みはそのまま）
                 ⇒ 揺れが消えれば、元は 下からの線・CL の学び（と滑る境目）。残れば、元は別（網の中の辺・門・核どうし ほか）
    境目なし      下からの線と CL の学びはそのまま、滑る境目のずれだけ 0（版107h までの形）
                 ⇒ 揺れが消えて率が上がり続ける／張り付くなら、揺れを作っているのは境目のならしの遅れ
  ⚠️ 項を外すのは測るためだけ（人から外れた網で決めない。どれが揺れを作っているかの手がかり）。境目なしは撃ちすぎへ進むはず（守りが止めることがある）
  何のため：wave111b（data/arkhe_wave111b_1116.txt）で、重さは q ＞ 1 で下がり q ＜ 1 で上がり、黙る長さはならしの戻りと合った。
    ただ重さの揺れは CL 1〜2割・下から 2% で、率の 100倍の揺れを作っているのがこの学びかは確かめていない
  出すもの：wave111b と同じ列（刻みごとに 皮質と運動の 率・q・下からの線の重さ・CL の重さ ／ CA1 ／ 門）
  毎歩に足す仕事：表現の発火（体の数 110,406）を1本の数えへ足す 1回。学びの式は元の式を呼んで、条件のときだけ結果を差し替える（仕事は増えない）
  刻みごと（200歩に1回）：下からの線の道ごとの切り出しの和・CL の和（wave111b と同じ）
  差し替えは網を組む前（グラフに撮るときも差し替えた式が入る。measure_term109 と同じやり方）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

CONDS = ("全部", "下からCLなし", "境目なし")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args or args[0] not in CONDS:
    sys.exit(f"条件を1つ：{' ・ '.join(CONDS)}")
COND = args[0]
NS = int(args[1]) if len(args) > 1 else 15000
EV = int(args[2]) if len(args) > 2 else 200
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION not in (110, 111):                       # 11-16：版110b でも回す（揺れが版111 からか、前からかを比べる）
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版110・111 で回す")

import collections
import torch_group2 as G
calls = collections.Counter()
R0 = G._run


NOSH = COND == "境目なし"
NOW = COND == "下からCLなし"


def _bu(src, pidx, pre_now, pre_prev, tgt, hippo, tab, post_age, d, sh):
    if NOSH:
        sh = sh * 0.0
    fac, age = TB._bu_learn_core(src, pidx, pre_now, pre_prev, tgt, hippo, tab, post_age, d, sh)
    if NOW:
        fac = torch.ones_like(fac)
    return fac, age


def _cl(any0, cl_ch, cl_spk, cl_pre_prev, cl_src, cl_recv, cl_post_prev, third, cl_w, sh):
    if NOSH:
        sh = sh * 0.0
    w, post = TB._cl_learn_core(any0, cl_ch, cl_spk, cl_pre_prev, cl_src, cl_recv, cl_post_prev, third, cl_w, sh)
    if NOW:
        w = cl_w * 1.0
    return w, post


def run(fn, *a):
    if fn is TB._bu_learn_core:
        calls["下から"] += 1
        return R0(_bu, *a)
    if fn is TB._cl_learn_core:
        calls["CL"] += 1
        return R0(_cl, *a)
    return R0(fn, *a)


if COND != "全部":
    G._run = run
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
print(f"版 {B.VERSION} ／ 条件 {COND} ／ {NS} 歩・刻み {EV} 歩（{EV * DT:.1f} 秒）／ 皮質（運動の外）{len(CX)} 領野 ／ 体 {n_all:,}", flush=True)
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
print(f"\n（終わり）差し替えた学びの呼び {dict(calls)}", flush=True)
