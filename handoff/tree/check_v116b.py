"""版116b の確かめ（容器・CPU・小さい網）。木の中で回す
  python3 check_v116b.py run 出力.npz   … 20歩回して 重さ・発火の数・抑制の体の率を書く（ANAPLEX_V116B は外から）
  python3 check_v116b.py cmp a.npz b.npz … 2つが同じか
"""
import sys, hashlib, numpy as np, torch
if sys.argv[1] == "cmp":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in ("w", "fired"):
        same = a[k].shape == b[k].shape and bool((a[k] == b[k]).all())
        print(f"{'✓' if same else '✗'} {k} が同じ（{sys.argv[2]} と {sys.argv[3]}）")
    sys.exit(0)
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB
from torch_brain import Brain, NAMES
B.DEV = "cpu"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
cx_inh = torch.zeros(g.n_recv_all, dtype=torch.bool)
for nm in TB.CX_TC:
    cx_inh |= (g.area_of == NAMES.index(nm)) & g.is_inh_r
rows_inh = torch.nonzero(cx_inh).squeeze(1)
inh_pos = torch.searchsorted(g.inh_i, rows_inh)
assert bool((g.inh_i[inh_pos] == rows_inh).all())
fired, inh_f = [], 0.0
ok = True
for t in range(20):
    life.step(learn=True)
    fired.append(torch.cat([g._any[0], g._any[2]]).sum().item())
    inh_f += float(g._any[2][inh_pos].float().mean())
    if getattr(TB, "V116B", False) and t == 19:
        e = g.ext_exc
        bad = int(((e != 0) & ~cx_inh).sum())
        print(f"{'✓' if bad == 0 else '✗'} ext_exc が皮質20 の抑制の体の行の外で 0（外で 0 でない行 {bad}）"); ok &= bad == 0
        spk = b._tc_now
        k = 5
        v = float((spk[b._ei_idx[k].long()] * b._ei_w[k]).sum())
        r = int(b._ei_rows[k])
        print(f"{'✓' if abs(v - float(e[r])) < 1e-4 else '✗'} 1行を手で足した値 {v:.4f} ＝ ext_exc {float(e[r]):.4f}"); ok &= abs(v - float(e[r])) < 1e-4
        print(f"  表 {tuple(b._ei_idx.shape)}（行 {len(b._ei_rows)}・1行 {TB.N_TC_EI}）・重さ 平均 {float(b._ei_w.mean()):.4f}（見込み {T.INH_W_MEAN * TB.TC_EI_F:.4f}）")
        same_area = True
        for nm in TB.CX_TC:
            i = NAMES.index(nm); a = g.areas[i]; per = a.ch.n_cell + a.ch.n_ramp; lo = int(g.off_send[i])
            m = (g.area_of[b._ei_rows] == i)
            s = b._ei_idx[m].long() - lo
            same_area &= bool(((s >= 0) & (s < (a.n_ch + a.n_ext) * per) & ((s % per) < a.ch.n_cell)).all())
        print(f"{'✓' if same_area else '✗'} 送り手はどれも同じ領野の興奮の行の遅れの体"); ok &= same_area
        print(f"{'✓' if torch.isfinite(g.w).all() else '✗'} 重さが有限")
print(f"  皮質20 の抑制の体が撃った割合（20歩の平均）{inh_f / 20:.1%}・V116B {getattr(TB, "V116B", None)}")
np.savez(sys.argv[2], w=g.w.numpy(), fired=np.array(fired))
print("（終わり）" if ok else "✗ どれか外れ")
