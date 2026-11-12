"""版109・版110：皮質の撃ちすぎ（srccx109：1番 1.4〜3.8倍 ⇒ 10番 3〜12倍）を、学びの項を1つずつ外して比べる（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_term109.py 条件 [番の並び=1,3,10]
  条件（1回の起動で1つ。同じ種で 0 から組む）
    全部          ふだんの網
    学びなし      learn=False（下からの線・CL・網の中の辺・恒常性の学びが全部止まる）⇒ 学ぶ前のノブだけで撃ちすぎるかを見る
    重さ全部なし   learn=True のまま、重さを変える学びを全部 0（網の中の辺 eta・抑制への弱まり・張り替え・核どうし・下からの線・CL・平行線維）
                  ⇒ 学びなしとの差は「重さ以外で learn が変えるもの」（11-12 23:01：全部は最初の 50歩から高い・CA1 81 対 2.9Hz）
    網の中なし     網の中の辺の学び（eta を 0・抑制の体への大きい辺の弱まり INH_LTD_F を 1・張り替え 0）
    核どうしなし   核どうしの線の学び（NucNet.NUC_A_ETA を 0）
    受け手が先なし 下からの線の学びの「受け手が先」の項（窓 tab × 門の開き。滑る境目なし）を 0
    海馬の線なし   下からの線のうち海馬の線（入力が先 LN_PRE_HIP × 門・受け手が先）を学ばない
    同じ歩なし     下からの線のうち皮質の線の「同じ歩」の項（入力が先 ln_ctx・滑る境目あり）を 0
    CL受け手が先なし CL → 皮質 の学びの「受け手が先」の項（_CL_LN_POST・滑る境目なし・門なし）を 0
  ⚠️ 項を外すのは測るためだけ（人から外れた網で決めない。どの項が伸ばしているかの手がかり）
  起きているあいだの恒常性は、下げ（scaling_on は眠りだけ）も上げ（rate_long 1時間・2日で 2.5倍）もこの長さでは動かない作り【コード】⇒ 条件に入れない
  番ごとに（区切りで1回）：皮質の領野ごとに 表現Hz・目標Hz（target_recv ÷ 1歩）（区切りのあと 50歩で数える）・率÷目標（rate_ema・約21秒の遅れ）
                         ・下からの線の重さの和（受け手1体あたり）・CL の重さの和（受け手1体あたり）
                         海馬（歯状回・CA3・CA1）も同じ列で（「海馬の線」＝ 受け手が海馬の下からの線。皮質へ入る線には無い）
                         ・門（third）・有限でない重さの本数
  毎歩に足す仕事：50歩の数えだけ（皮質の体の発火の足し算 1回 ＝ 体の数。線・辺には触らない・少ない箱への足し込みなし・写しなし）
     区切り（4回）：下からの線の道ごとの和（切り出しの和・写しなし）・CL を (受け手, 本) に並べた和・重さの和が有限か（写しなし）
     学びの式は差し替えた関数で（外した項の分だけ仕事が減る）
  グラフで回す（ふだんと同じ。⟲ eager は取り置きが 12〜15GB【測】11-05 ⇒ 12GB の守りで止まる）。差し替えは網を組む前なので、グラフに撮るときも差し替えた式が入る
  大きさ（容器で数えた。相手の数に依らない）：体 110,406・下からの線 9,386,535・CL 163,840
"""
import os, sys, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

CONDS = ("全部", "学びなし", "重さ全部なし", "網の中なし", "核どうしなし", "受け手が先なし", "海馬の線なし", "同じ歩なし", "CL受け手が先なし")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args or args[0] not in CONDS:
    sys.exit(f"条件を1つ：{' ・ '.join(CONDS)}")
COND = args[0]
WIN = [int(x) for x in (args[1] if len(args) > 1 else "1,3,10").split(",")]
TEST = "--test" in sys.argv
NW = 5 if TEST else 50
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 110:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版110 の学びの式を写している（版110 で回す）")

# ── 項を外した学びの式（torch_brain の式と同じ形。外す項だけ 0）──
NOPF = COND == "受け手が先なし"
NOHIP = COND == "海馬の線なし"
NOSAME = COND == "同じ歩なし"
NOCLPF = COND == "CL受け手が先なし"
NOW = COND == "重さ全部なし"
NONET = COND in ("網の中なし", "重さ全部なし")
NONUC = COND in ("核どうしなし", "重さ全部なし")
if NONET:
    G.INH_LTD_F = 1.0                     # 抑制の体への大きい辺の弱まり（学びの式に値で渡る・グラフに撮る前に）
if NONUC:
    G.NucNet.NUC_A_ETA = 0.0
calls = collections.Counter()


def _bu_core(src, pidx, pre_now, pre_prev, tgt, hippo, tab, post_age, d, sh):
    post = src[pidx]
    pre_e = pre_now | pre_prev
    post_e = post[tgt] > 0
    g_on = (2.0 * d - 1.0).clamp(0.0, 1.0)
    ln_ctx = ((1.0 - TB._dd(d)) * TB._CL_LN_NODA + TB._dd(d) * TB._CL_LN_DA - sh) / TB.CL_PAIRS
    if NOSAME:
        ln_ctx = ln_ctx * 0.0
    ln_hip = TB.LN_PRE_HIP * g_on * (0.0 if NOHIP else 1.0)
    ln_pp = torch.where(hippo, ln_hip, ln_ctx)
    k = post_age.clamp(max=TB.WIN_MAXK + 1)[tgt]
    pf = (tab[hippo.long(), k] * g_on - TB._pf_shift(sh, k)) * (pre_now & ~post_e).float()   # 版110 の式
    if NOPF:
        pf = pf * 0.0
    if NOHIP:
        pf = torch.where(hippo, torch.zeros_like(pf), pf)
    fac = torch.exp(ln_pp * (pre_e & post_e).float() + pf)
    if NOW:
        fac = torch.ones_like(fac)
    return fac, torch.where(post > 0, torch.ones_like(post_age), post_age + 1)


def _cl_core(any0, cl_ch, cl_spk, cl_pre_prev, cl_src, cl_recv, cl_post_prev, third, cl_w, sh):
    post = any0[cl_ch]
    pre_e = (cl_spk + cl_pre_prev)[cl_src] > 0
    post_e = post[cl_recv] > 0
    d = (third / 2.0).clamp(0.0, 1.0)
    ln_pp = ((1.0 - TB._dd(d)) * TB._CL_LN_NODA + TB._dd(d) * TB._CL_LN_DA - sh) / TB.CL_PAIRS
    pf = (cl_post_prev[cl_recv] > 0) & ~post_e & (cl_spk[cl_src] > 0)
    post_term = (TB._CL_LN_POST - sh) / TB.CL_PAIRS * pf.float() * (0.0 if NOCLPF else 1.0)   # 版110 の式
    if NOW:
        return cl_w * 1.0, post
    return cl_w * torch.exp(ln_pp * (pre_e & post_e).float() + post_term), post


R0 = G._run


def run(fn, *a):
    if fn is TB._bu_learn_core:
        calls["下から"] += 1
        return R0(_bu_core, *a)
    if fn is TB._cl_learn_core:
        calls["CL"] += 1
        return R0(_cl_core, *a)
    return R0(fn, *a)


G._run = run

CX = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質"]
HIP = [n for n in ("歯状回", "CA3", "CA1") if n in NAMES]
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
if NONET:
    g.eta = g.eta * 0.0
    g.rewire_every = 0
if NOW:
    b._pf_learn = lambda *a, **k: None
n_all = g.n_ch_all
LEARN = COND != "学びなし"
DT = 10.67 / 1000.0
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
is_rep_r = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev)
is_rep_r[g.rep_i.long()] = True

# 下からの線（興奮の表）：道ごとの切り出し（受け手の領野・海馬の線か）
tab = b.but
paths = collections.defaultdict(list)          # 領野 → [(切り出し, 受け手の数, 海馬か)]
n_lines = 0
for i, k in enumerate(tab.keys):
    a0, a1 = int(tab.eo[i]), int(tab.eo[i + 1])
    n_lines += a1 - a0
    nm = ch0_name.get(int(b.bu[k].ch0), "?")
    hip = bool(tab.hippo[a0:a1].any()) if a1 > a0 else False
    paths[nm].append((slice(a0, a1), max(1, tab.nt[i]), hip))
n_inh_lines = sum(len(t.w) for t in [getattr(b, "but_inh", None)] if t is not None)
# CL：受け手（皮質の表現の体）ごとに K_CL 本が並ぶ（cl_recv ＝ arange.repeat_interleave）
cl_on = bool(getattr(b, "cl_on", False))
if cl_on:
    nr = len(b.cl_ch)
    kcl = len(b.cl_w) // nr
    assert kcl * nr == len(b.cl_w) and bool((b.cl_recv.long().view(nr, kcl) == torch.arange(nr, device=dev)[:, None]).all()), "CL の並びが思っていた形でない"
    cl_ch = b.cl_ch.long()
print(f"［条件］網の中 eta {float(g.eta) if not torch.is_tensor(g.eta) else float(g.eta.max()):.3g}・抑制への弱まり {G.INH_LTD_F:.4f}"
      f"・張り替え {g.rewire_every}・核どうし {G.NucNet.NUC_A_ETA:g}", flush=True)
print(f"版 {B.VERSION} ／ 条件 {COND} ／ 番 {WIN}・数える窓 {NW}歩 ／ 皮質 {len(CX)} 領野 ／ 体 {n_all:,}"
      f" ／ 下からの線 {n_lines:,}（抑制の表 {n_inh_lines:,}）／ CL {len(b.cl_w) if cl_on else 0:,}", flush=True)


def check(label):
    rep_n = torch.zeros(n_all, device=dev)
    t0 = life.t
    for _ in range(NW):
        life.step(learn=LEARN)
        rep_n.add_(g._any[0][:n_all].float())
    ns = max(1, life.t - t0)
    q = g.rate_ema / g.target_recv.clamp(min=1e-9)
    L.beat()
    if cl_on:
        clw = b.cl_w.view(nr, kcl).sum(1, dtype=torch.float64)               # 受け手ごとの CL の重さの和
        cl_area = {}
        for nm in CX:
            i = NAMES.index(nm)
            oc, n = int(b.off_ch[i]), b.n[nm]
            m = (cl_ch >= oc) & (cl_ch < oc + n)
            cl_area[nm] = float(clw[m].mean()) if bool(m.any()) else float("nan")
    sw = float(tab.w.sum(dtype=torch.float64)); scl = float(b.cl_w.sum(dtype=torch.float64)) if cl_on else 0.0
    bad = "あり" if not (np.isfinite(sw) and np.isfinite(scl)) else "なし"
    th = g.third.reshape(-1).tolist() if torch.is_tensor(g.third) else g.third
    L.beat()
    print(f"\n━━ {label}（{t0}〜{life.t}歩）門 third {np.round(th, 3).tolist()} ／ 有限でない重さ {bad}"
          f" ／ 差し替えた学びの呼び {dict(calls)} ━━", flush=True)
    print(f"  {'領野':<8s}{'表現Hz':>8s}{'目標Hz':>8s}{'率÷目標':>8s}{'下から w和/体':>14s}{'CL w和/体':>11s}", flush=True)
    for nm in CX + HIP:
        i = NAMES.index(nm)
        oc, n = int(b.off_ch[i]), b.n[nm]
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        rr = torch.arange(lo, hi, device=dev)
        rr = rr[is_rep_r[rr]]
        wa = sum(float(tab.w[s].sum(dtype=torch.float64)) / nt for s, nt, _ in paths.get(nm, []))
        print(f"  {nm:<8s}{float(rep_n[oc:oc + n].mean()) / ns / DT:8.2f}{float(g.target_recv[rr].median()) / DT:8.2f}{float(q[rr].median()):8.2f}"
              f"{wa:14.3f}{cl_area.get(nm, float('nan')) if cl_on else float('nan'):11.3f}", flush=True)
    L.beat()


check("0番")
done = 0
for k in WIN:
    if k > done:
        life.run_plans(k - done, learn=LEARN)
        done = k
    check(f"{k}番")
print("\n（終わり）", flush=True)
