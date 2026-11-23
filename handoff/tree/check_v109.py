"""版109 の確かめ（コンテナ・CPU）：皮質の型の領野が全部 切片の見方の抑制になったか
  python3 check_v109.py
  見ること：
    1 皮質の型の領野が全部 inh_slice（それ以外の型は変わらない）
    2 皮質の表現・誤差の行の抑制の辺の送り手が、抑制の体の1歩前の列だけ（時間細胞・傾きの細胞から 0本）
    3 1行あたりの抑制の本数 ＝ round(抑制の体 × 0.23)・同じ体が2本にならない
"""
import os, sys, numpy as np, torch
import babble as B, torch_area as T, torch_brain as TB
from torch_brain import Brain, NAMES

B.DEV = "cpu"
if B.VERSION != 109:
    raise SystemExit(f"⚠️ 版 {B.VERSION}（109 を当ててから）")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
areas = g.areas
ch_off = np.cumsum([0] + [a.ch.n_cells for a in areas])
buf_base = int(ch_off[-1])
buf_off = np.cumsum([0] + [a.n_src * a.D for a in areas])
ok = True
for i, nm in enumerate(NAMES):
    a = areas[i]
    cx = TB.PRIOR_TYPE.get(nm) == "皮質"
    want = cx or nm in TB.INH_SLICE
    if bool(a.inh_slice) != want:
        print(f"✗ {nm}：inh_slice {a.inh_slice}（期待 {want}）"); ok = False
        continue
    if not cx:
        continue
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    rows = torch.arange(lo, lo + a.n_ch * 2)                    # 表現・誤差の行
    m = (g.e_recv.long() >= lo) & (g.e_recv.long() < lo + a.n_ch * 2) & g.e_inh.bool()
    src = g.e_src[m].long()
    col0 = buf_base + int(buf_off[i]) + (a.n_ch + a.n_ext + torch.arange(a.n_inh)) * a.D   # 抑制の体の1歩前の列
    in_col = torch.isin(src, col0)
    k = max(1, int(round(a.n_inh * TB.P_INH_CONN)))
    per = torch.bincount(g.e_recv[m].long() - lo, minlength=a.n_ch * 2)
    dup = 0
    rr = g.e_recv[m].long()
    pairs = rr * (g.n_send_all + 1) + src
    dup = len(pairs) - len(torch.unique(pairs))
    good = bool(in_col.all()) and int(per.min()) == k and int(per.max()) == k and dup == 0
    ok &= good
    print(f"{'✓' if good else '✗'} {nm}：抑制の体 {a.n_inh}・1行 {int(per.min())}〜{int(per.max())} 本（期待 {k}）・1歩前の列から {float(in_col.double().mean()):.1%}・重なり {dup}")
print("\n✅ 版109 の確かめ 全部 ✓" if ok else "\n⚠️ ✗ がある")
