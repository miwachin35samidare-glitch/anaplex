"""版107z：皮質の抑制の体を何が撃たせているか（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_inhcx107.py [先に回す番=10] [測る歩=100]
  問い：srccx107z（11-10）で 補足・前補足・45・眼窩前頭 は 1番から表現 0Hz・1＋抑制 49〜88（領野の中の抑制の体）。
        聴覚・運動（版64・65 の切片の抑制）・前頭前野・連合 は撃っている ⇒ 抑制の体の駆動を並べて比べる
  抑制の体が撃つか（torch_group2._out_core）：d_inh ＝ (網の中の興奮 − 調整)[抑制の行] ÷ (1＋抑制) ＋ 持ち越し ＞ INH_TH
  1 送り手の種類ごとの押し（その歩の初めに撃っている送り手 ＝ base。歩の平均・抑制の行1体あたり・÷ INH_TH）
       時間細胞（元が 表現／外から／抑制）・傾きの細胞・遅れの列（表現／外から／抑制）
       ＋ その種類の辺の重さの和・本数（抑制の行1体あたり）・送り手が撃っている割合
     ⚠️ 調整の辺（mod_send）は抑制の行の駆動から引かれるので「調整」に分けて出す。短期の増減（放出）は掛けていない
  2 1歩の中の周ごと（1〜10周目）：抑制の体の d_inh ÷ INH_TH・撃った割合・抑制の行の (1＋抑制)・表現の行の (1＋抑制)
     ⇒ 周の中で抑制が積もるか（周の中で撃った抑制の体は、その歩の残りの周にずっと足される作り【コード】）
  ⚠️ 平均で見る。1つの歩の中の流れは 2 で見る
  ⟲ 11-11 00:55 arkhe：測る区間で値を毎回 CPU へ読んでいたので 1歩 33 → 88ms ⇒ life.py の止めで止まった。値は GPU の上で足し、終わりに1回だけ読む形に直した
"""
import os, sys, collections, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 10
NS = int(args[1]) if len(args) > 1 else 100
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 107:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版107（107z）の形で読む")
AR = [n for n in ("補足", "前補足", "45", "眼窩前頭", "前頭前野", "連合", "聴覚", "運動") if n in NAMES]
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
INH_TH = T.INH_TH

# ── 送り手の種類（torch_group2 の remap と同じ並び：時間細胞 [送り元, 時間細胞＋傾き] → buf [送り元, D]）──
KIND = ("時間細胞←表現", "時間細胞←外から", "時間細胞←抑制", "傾きの細胞", "遅れ←表現", "遅れ←外から", "遅れ←抑制")
areas = g.areas
ch_off = np.cumsum([0] + [a.ch.n_cells for a in areas])
buf_base = int(ch_off[-1])
buf_off = np.cumsum([0] + [a.n_src * a.D for a in areas])
kind = torch.full((g.n_send_all,), -1, dtype=torch.int8, device=dev)
area_of_send = torch.full((g.n_send_all,), -1, dtype=torch.int16, device=dev)


def src_k(s, a):
    return torch.where(s < a.n_ch, 0, torch.where(s < a.n_ch + a.n_ext, 1, 2))


for i, a in enumerate(areas):
    nc, nr = a.ch.n_cell, a.ch.n_ramp
    w_ = nc + nr
    loc = torch.arange(a.ch.n_cells, device=dev)
    s, j = loc // max(1, w_), loc % max(1, w_)
    k = torch.where(j < nc, src_k(s, a), 3).to(torch.int8)
    kind[int(ch_off[i]):int(ch_off[i + 1])] = k
    area_of_send[int(ch_off[i]):int(ch_off[i + 1])] = i
    s = torch.arange(a.n_src, device=dev).repeat_interleave(a.D)
    kind[buf_base + int(buf_off[i]):buf_base + int(buf_off[i + 1])] = (4 + src_k(s, a)).to(torch.int8)
    area_of_send[buf_base + int(buf_off[i]):buf_base + int(buf_off[i + 1])] = i
assert int((kind < 0).sum()) == 0, "送り手の種類が埋まらない（並びが版107 と違う）"

# ── 抑制の行へ入る辺（測る領野だけ）──
g.flush_all()
inh_rows, sub = {}, {}
tgt_mask = torch.zeros(g.n_recv_all, dtype=torch.int16, device=dev) - 1
for nm in AR:
    i = NAMES.index(nm)
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    r = torch.arange(lo, hi, device=dev)
    inh_rows[nm] = r[g.is_inh_r[r]]
    tgt_mask[inh_rows[nm]] = AR.index(nm)
parts = collections.defaultdict(list)
E, st = len(g.w), 1 << 22
for a0 in range(0, E, st):
    sl = torch.arange(a0, min(E, a0 + st), device=dev)
    t = tgt_mask[g.e_recv[sl].long()]
    m = t >= 0
    for j, nm in enumerate(AR):
        mm = m & (t == j)
        if mm.any():
            parts[nm].append(sl[mm])
for nm in AR:
    sub[nm] = torch.cat(parts[nm]) if parts[nm] else torch.zeros(0, dtype=torch.long, device=dev)

# 抑制の行の位置（_out_core の d_inh・inh_spk は g.inh_i の並び）
inh_i = g.inh_i.long()
pos_in = {nm: torch.nonzero(tgt_mask[inh_i] == AR.index(nm)).squeeze(1) for nm in AR}
rep_i = g.rep_i.long()
rep_rows = {}
for nm in AR:
    i = NAMES.index(nm)
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    rep_rows[nm] = torch.nonzero((rep_i >= lo) & (rep_i < hi)).squeeze(1)

NK = len(KIND)
NB = NK + 2                      # 種類（興奮）＋ 調整 ＋ 抑制の送り手
cat_e, cnt_e, nr_ = {}, {}, {}
for nm in AR:
    e = sub[nm]
    se = g.e_src[e].long()
    ie = g.e_inh[e].bool()
    md = g.mod_send[se] & ~ie & ~g.mod_drive_inh[se]
    c = kind[se].long()
    c = torch.where(md, NK, c)
    c = torch.where(ie, NK + 1, c)
    cat_e[nm] = c
    cnt_e[nm] = torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, c, torch.ones(len(c), dtype=torch.float64, device=dev))
    nr_[nm] = float(max(1, len(inh_rows[nm])))
acc_push = {nm: torch.zeros(NB, dtype=torch.float64, device=dev) for nm in AR}
acc_on = {nm: torch.zeros(NB, dtype=torch.float64, device=dev) for nm in AR}
NP = g.recur if hasattr(g, "recur") else T.RECUR
# 周ごと：測る領野の位置に組の番号を付け、周ごとに index_add（読み出さない）
grp_inh = torch.full((len(inh_i),), -1, dtype=torch.long, device=dev)
for j, nm in enumerate(AR):
    grp_inh[pos_in[nm]] = j
grp_rep = torch.full((len(rep_i),), -1, dtype=torch.long, device=dev)
for j, nm in enumerate(AR):
    grp_rep[rep_rows[nm]] = j
m_inh = torch.nonzero(grp_inh >= 0).squeeze(1)   # 番号の並び（真偽値で切ると GPU が数を待つ）
m_rep = torch.nonzero(grp_rep >= 0).squeeze(1)
gi_, gr_ = grp_inh[m_inh], grp_rep[m_rep]
n_inh_g = torch.zeros(len(AR), dtype=torch.float64, device=dev).index_add_(0, gi_, torch.ones(len(gi_), dtype=torch.float64, device=dev)).clamp(min=1)
n_rep_g = torch.zeros(len(AR), dtype=torch.float64, device=dev).index_add_(0, gr_, torch.ones(len(gr_), dtype=torch.float64, device=dev)).clamp(min=1)
prof = torch.zeros(NP, 4, len(AR), dtype=torch.float64, device=dev)   # d_inh/TH・撃った割合・抑制の行の 1＋抑制・表現の行の 1＋抑制（和）
prof_n = [0] * NP
on = [False]
pidx = [0]
nsteps = [0]
F0 = type(g)._fired_edges


def fired_w(self):
    first = self._e_cache is None and self._within is None
    out = F0(self)
    if on[0] and first and getattr(self, "_base_on", None) is not None:
        bo = self._base_on
        for nm in AR:
            e = sub[nm]
            if len(e) == 0:
                continue
            se = g.e_src[e].long()
            ie = g.e_inh[e].bool()
            act = bo[se].double()
            amp = torch.where(ie, 4.0, g.c_exc[g.e_recv[e].long()]).double() * g.w[e].double() * g.send_gain[se].double()
            c = cat_e[nm]
            acc_push[nm].index_add_(0, c, amp * act)
            acc_on[nm].index_add_(0, c, act)
    return out


R0 = G._run
cur = {}


def run(fn, *a):
    out = R0(fn, *a)
    if not on[0]:
        return out
    if fn is G._drive_core:
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        cur["ihn"] = ihn.double()
    elif fn is G._out_core:
        d_inh, rep, err, inh_spk = out[0], out[1], out[2], out[3]
        k = min(pidx[0], NP - 1)
        ihn = cur.get("ihn")
        prof[k, 0].index_add_(0, gi_, d_inh[m_inh].double() / INH_TH)
        prof[k, 1].index_add_(0, gi_, inh_spk[m_inh].double())
        if ihn is not None:
            prof[k, 2].index_add_(0, gi_, 1.0 + ihn[inh_i[m_inh]])
            prof[k, 3].index_add_(0, gr_, 1.0 + ihn[rep_i[m_rep]])
        prof_n[k] += 1
        pidx[0] += 1
    return out


if WARM > 0:
    life.run_plans(WARM, learn=True)
g.flush_all()
# 重さ・本数（抑制の行1体あたり、種類ごと）
W = {}
for nm in AR:
    e = sub[nm]
    se = g.e_src[e].long()
    ie = g.e_inh[e].bool()
    md = g.mod_send[se] & ~ie & ~g.mod_drive_inh[se]
    kk = kind[se].long()
    nr = max(1, len(inh_rows[nm]))
    rows = []
    for j in range(NK):
        mm = (kk == j) & ~ie & ~md
        rows.append((float(g.w[e][mm].double().sum()) / nr, float(mm.sum()) / nr))
    rows.append((float(g.w[e][md].double().sum()) / nr, float(md.sum()) / nr))
    rows.append((float(g.w[e][ie].double().sum()) / nr, float(ie.sum()) / nr))
    W[nm] = rows

type(g)._fired_edges, G._run = fired_w, run
on[0] = True
t0 = life.t
for _ in range(NS):
    pidx[0] = 0
    life.step()
    nsteps[0] += 1
on[0] = False
type(g)._fired_edges, G._run = F0, R0
ns = max(1, nsteps[0])
print(f"版 {B.VERSION} ／ 先に {WARM}番・測る {ns}歩（{t0}〜{life.t}歩）／ INH_TH {INH_TH:.3f} ／ 1歩 {NP}周", flush=True)
print("\n■ 1 抑制の体の押し（歩の初めに撃っている送り手から。抑制の行1体あたり ÷ INH_TH）／［重さの和・本数（1体あたり）・撃っている割合］", flush=True)
for nm in AR:
    P = (acc_push[nm] / ns).cpu().numpy()
    O = (acc_on[nm] / ns / cnt_e[nm].clamp(min=1)).cpu().numpy()
    nr = nr_[nm]
    print(f"\n  {nm}（抑制の体 {len(inh_rows[nm])}）", flush=True)
    for j in range(NK):
        w_, n_ = W[nm][j]
        print(f"    {KIND[j]:<14s} 押し {P[j] / nr / INH_TH:8.3f}   ［重さ {w_:8.3f}・本数 {n_:7.1f}・撃っている {O[j]:6.2%}］", flush=True)
    w_, n_ = W[nm][NK]
    print(f"    {'調整（引かれる側）':<14s} 押し {P[NK] / nr / INH_TH:8.3f}   ［重さ {w_:8.3f}・本数 {n_:7.1f}・撃っている {O[NK]:6.2%}］", flush=True)
    w_, n_ = W[nm][NK + 1]
    print(f"    {'抑制の送り手':<14s} 届く {P[NK + 1] / nr:8.3f}（割り算の側）［重さ {w_:8.3f}・本数 {n_:7.1f}・撃っている {O[NK + 1]:6.2%}］", flush=True)
print("\n■ 2 1歩の中の周ごと（抑制の体の d_inh ÷ INH_TH ／ 撃った割合 ／ 抑制の行の 1＋抑制 ／ 表现の行の 1＋抑制）".replace("表现", "表現"), flush=True)
PR = prof.cpu().numpy()
ni, nrp = n_inh_g.cpu().numpy(), n_rep_g.cpu().numpy()
for j, nm in enumerate(AR):
    cells = []
    for k in range(NP):
        c = max(1, prof_n[k])
        cells.append(f"{k + 1}:{PR[k, 0, j] / c / ni[j]:.2f}/{PR[k, 1, j] / c / ni[j]:.0%}/{PR[k, 2, j] / c / ni[j]:.1f}/{PR[k, 3, j] / c / nrp[j]:.1f}")
    print(f"  {nm:<6s} " + "  ".join(cells), flush=True)
print("\n（終わり）", flush=True)
