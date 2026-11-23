"""版113 の確かめ（容器・CPU・小さい網）：皮質の 表現 → 表現 を人の本数・初めの重さ・下からの線と同じ学びに（組み表_版113_案 §13.7）
  python3 check_v113.py
  ⚠️ ANAPLEX_RR=0 で版112 と同じになることは dump113.py の2回で見る（この道具の外）
"""
import math, numpy as np, torch
import babble as B
B.DEV = "cpu"
import torch_area as T, torch_brain as TB, brainstem as BS
from torch_brain import Brain, NAMES

ok = True


def ck(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'✓' if cond else '✗'} {name}{extra}", flush=True)


ck("1 VERSION 113・SAME_NET なし（state.pt は 0 から）・ANAPLEX_RR は既定で入る", B.VERSION == 113 and B.SAME_NET == () and TB.RR_ON)

# 4（先に）：式の表（組み表 §13.3）
print("── 式（_rr_learn_core に作り物の発火を入れる）", flush=True)
tab = torch.zeros(TB.WIN_MAXK + 2)
for k_, v_ in TB.WIN_CTX.items():
    tab[k_] = v_
want = {(1.0, 0.5): (1.0024, 1.0036, 1.0036), (1.0, 1.0): (1.0000, 1.0000, 1.0000), (1.0, 2.0): (0.9974, 0.9961, 0.9961),
        (0.5, 1.0): (0.9984, 1.0000, 1.0000), (2.0, 1.0): (1.0026, 1.0046, 1.0030), (2.0, 2.0): (1.0000, 1.0007, 0.9990)}
fine = True
for (th, q), w3 in want.items():
    d = torch.tensor(min(max(th / 2.0, 0.0), 1.0))
    sh = TB._bcm_shift(torch.tensor([q, q]))
    sp, rp = torch.tensor([0]), torch.tensor([1])
    # 同じ歩：送り手は前の歩に撃った（この歩に届く）・受け手はこの歩に撃つ
    f_same, _ = TB._rr_learn_core(torch.tensor([0., 1.]), torch.tensor([1., 0.]), torch.zeros(2), sp, rp,
                                  torch.tensor([9, 9]), tab, d, sh)
    got = [float(f_same)]
    for k in (2, 6):          # 受け手が先：受け手は k 歩前に撃った・この歩は撃たない
        f_k, _ = TB._rr_learn_core(torch.zeros(2), torch.tensor([1., 0.]), torch.zeros(2), sp, rp,
                                   torch.tensor([9, k]), tab, d, sh)
        got.append(float(f_k))
    good = all(abs(a - b) < 6e-5 for a, b in zip(got, w3))
    fine &= good
    print(f"   門 {th} q {q}：{got[0]:.5f} {got[1]:.5f} {got[2]:.5f}（表 {w3}）{'' if good else ' ✗'}", flush=True)
f_q, _ = TB._rr_learn_core(torch.zeros(2), torch.tensor([1., 0.]), torch.tensor([1., 0.]), torch.tensor([0]), torch.tensor([1]),
                           torch.tensor([9, 9]), tab, torch.tensor(1.0), TB._bcm_shift(torch.tensor([0., 0.])))
f_s, _ = TB._rr_learn_core(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.tensor([0]), torch.tensor([1]),
                           torch.tensor([9, 2]), tab, torch.tensor(1.0), TB._bcm_shift(torch.tensor([0., 0.])))
ck("4 倍率が組み表 §13.3 と同じ・受け手が撃たない／送り手も来ないと 1", fine and abs(float(f_q) - 1) < 1e-7 and abs(float(f_s) - 1) < 1e-7,
   f"：送り手だけ撃つ {float(f_q):.6f}・誰も撃たない {float(f_s):.6f}")

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
RR = [nm for nm in NAMES if b.areas[nm].rr_k > 0]
EXP = ["聴覚", "視覚", "連合", "高次", "前頭前野", "補足", "前補足", "眼窩前頭", "前帯状", "44", "45", "STS1", "STS2", "中側頭後",
       "TPJ", "島前", "島後", "縁上回", "海馬傍", "脳梁膨大後"]
ck("2a 当てる領野は運動の外の皮質 20", sorted(RR) == sorted(EXP) and len(RR) == 20 and "運動" not in RR,
   "：" + "・".join(RR))

# 2：本数・重なり・自分なし（辺の表を全部読んで数える）
es, er = g.e_src.long(), g.e_recv.long()
m_rr = g.rr_send[es] & g.rr_recv[er]
row_ok, dup_ok, self_ok, line = True, True, True, []
send_ch = torch.full((g.n_send_all,), -1, dtype=torch.long)
send_ch[g.rr_L_send] = g.rr_L_ch
for nm in RR:
    i = NAMES.index(nm)
    a = b.areas[nm]
    lo = int(g.off_recv[i]) + a.n_ch
    sel = m_rr & (er >= lo) & (er < lo + a.n_ch)
    cnt = torch.bincount(er[sel] - lo, minlength=a.n_ch)
    row_ok &= bool((cnt == a.rr_k).all())
    pairs = (er[sel] - lo) * a.n_ch + (send_ch[es[sel]] - int(g.off_ch[i]))
    dup_ok &= len(torch.unique(pairs)) == len(pairs)
    self_ok &= not bool(((send_ch[es[sel]] - int(g.off_ch[i])) == (er[sel] - lo)).any())
    k_h = TB._rr_k(a.n_ch)
    row_ok &= a.rr_k == k_h
    # 表現の行の外（誤差の行）へ入る自分の表現の列の辺は版112 の引き方のまま
    line.append(f"{nm} {a.rr_k}（{a.n_ch}）")
ck("2b 受け手1体の本数 ＝ round(0.158 ×（体数 − 1））", row_ok, "：" + "・".join(line))
ck("2c 同じ送り手を2回引かない・自分は引かない", dup_ok and self_ok)
n_rr = int(m_rr.sum())
n_exp = sum(b.areas[nm].n_ch * b.areas[nm].rr_k for nm in RR)
ck("2d 辺の数 ＝ Σ 体数 × K", n_rr == n_exp, f"：{n_rr:,}（arkhe の大きさでは 115,797【計算】。容器の小さい網も皮質の体数は同じ）")
# 相手の総数は変えない
nexc = torch.bincount(er, minlength=g.n_recv_all)
tot_ok = True
for nm in RR:
    i = NAMES.index(nm); a = b.areas[nm]; lo = int(g.off_recv[i]) + a.n_ch
    tot_ok &= bool((nexc[lo:lo + a.n_ch] == a.n_partner).all())
ck("2e 表現の行の相手の総数は変わらない（領野の n_partner のまま）", tot_ok)
wr = g.w[m_rr]
med = float(wr.median())
ck("2f 初めの重さ：中央 0.148 前後（幅で切った _ln_w の中央 0.49 ÷ 接触 3.3）・幅 0.027〜1.62（0.1〜6mV ÷ 1.12 ÷ 3.3）",
   abs(med - 0.49 / 3.3) < 0.01 and float(wr.min()) >= 0.1 / 1.12 / 3.3 - 1e-6 and float(wr.max()) <= 6 / 1.12 / 3.3 + 1e-6,
   f"：中央 {med:.4f}・平均 {float(wr.mean()):.4f}・最小 {float(wr.min()):.4f}・最大 {float(wr.max()):.4f}")
# 境目を当てる行
b_use = (~g.scale_skip[g.rr_L_row]) & b._bcm_ok_rows()[g.rr_L_row]
ck("2g 滑る境目は当てる領野の表現の行の全部に", bool(b_use.all()), f"：{int(b_use.sum())}／{len(b_use)}")

# 3・5：歩く。rr の辺の跡は 0・重さは倍率どおりだけ動く
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
rec = {}
_orig = TB._rr_learn_core


def _spy(*a):
    f, age = _orig(*a)
    rec["fac"] = f.clone()
    return f, age


TB._rr_learn_core = _spy
exact, el_ok, nan_ok, moved, rep_hz = True, True, True, 0, 0.0
w_first = None
for k in range(40):
    if g._rr_idx is None:
        g.rr_rebuild()
    idx = g._rr_idx.clone()
    g.flush_all()
    w0 = g.w[idx].clone()
    if w_first is None:
        w_first = w0.clone()
    life.step()
    rep_hz = rep_hz + float(g._any[0][g.rr_L_ch].sum())
    rw = g._rr_idx is None or not torch.equal(g._rr_idx, idx)
    if not rw:
        g.flush_all()
        exp_w = w0 * rec["fac"]
        exact &= bool(torch.allclose(g.w[idx], exp_w, rtol=1e-6, atol=1e-9))
        moved += int((rec["fac"] != 1).sum())
    el_ok &= float(g.elig[g._rr_idx if g._rr_idx is not None else idx].abs().max()) == 0.0
    nan_ok &= not bool(torch.isnan(g.w).any())
TB._rr_learn_core = _orig
ck("3 rr の辺の学びの跡（elig）は 0 のまま（予測誤差の学びが入らない）", el_ok)
ck("5a 歩くあいだ rr の辺の重さは 1歩ごとの倍率だけで動く", exact, f"：40歩で倍率 ≠ 1 の組 {moved:,}")
ck("5b NaN なし", nan_ok)
wn = g.w[g._rr_idx]
q_ = g.rate_ema[g.rr_L_row] / g.target_recv[g.rr_L_row]
hz_ = float(rep_hz) / 40 / len(g.rr_L_ch) / 0.01067
print(f"   容器の小さい網（n_partner 200・学ぶ前の値）：当てる領野の表現 {hz_:.2f}Hz・q 平均 {float(q_.mean()):.2f}・門（VTA）{float(g.third.reshape(-1)[-1]):.2f}", flush=True)
print(f"   40歩のあと rr の重さ：中央 {float(wn.median()):.4f}（初め {float(w_first.median()):.4f}）・最大 {float(wn.max()):.4f}", flush=True)
# 6：刈り込みのあと番号の列を作り直しても同じ辺の組
n_before = int((g.rr_send[g.e_src.long()] & g.rr_recv[g.e_recv.long()]).sum())
g.flush_all()
g.rewire()
assert g._rr_idx is None
g.rr_rebuild()
m2 = g.rr_send[g.e_src.long()] & g.rr_recv[g.e_recv.long()]
same_set = torch.equal(torch.sort(g._rr_idx).values, torch.nonzero(m2).squeeze(1))
ck("6 刈り込みのあと：番号の列 ＝ 全部の辺を読んだ rr の辺（刈り込みは変えていない）", same_set,
   f"：前 {n_before:,} → あと {int(m2.sum()):,}（切られて時間細胞の辺に戻った分・生えた分）")
print("\n全部 ✓" if ok else "\n✗ あり")
