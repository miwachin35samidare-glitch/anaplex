"""版98 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない【み】10-18 22:08
  python3 check_v98.py
確かめること
  1 組める・体数（人の中の比）・型・時間細胞なし・抑制の体（顆粒の中 ＝ ゴルジ・プルキンエ／分子層は 0・VLp 27.5%）・自発のくじなし
  2 線：皮質 → 橋核 64本の割り振り・苔状線維 4本・平行線維の確率・登上線維 1本・抑制の線・小脳核 → VLp・CM・赤核・VLp ⇄ 運動
  3 プルキンエの自分で撃つ分（核の i0 と同じ入り方）
  4 平行線維の学びの向き（登上線維の前の平行線維は弱まる・登上線維なしの平行線維は強まる）
  5 回る（作り物の声で1番）・保存と読み込みで平行線維の重さが戻る
"""
import sys, os, tempfile, numpy as np, torch
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

B.DEV = "cpu"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


K = TB.K_CB
print(f"1 組めた：体 {b.n_cells:,}・網の辺 {len(g.w):,}・領野 {len(NAMES)}", flush=True)
want = {"プルキンエ": K, "顆粒": round(K * 2991), "分子層": round(K * 50.1), "下オリーブ": max(1, round(K / 22.3)),
        "VLp": 132, "赤核": 64}
for nm, n in want.items():
    chk(b.n[nm] == n, f"{nm} {b.n[nm]}体（{n}）")
chk(b.n["小脳核"] + b.n["小脳核の抑制"] == max(2, round(K / 17.4)), f"小脳核 {b.n['小脳核']}＋抑制 {b.n['小脳核の抑制']}（人の比 1/17.4）")
chk(b.n["橋核"] == max(1, round(b.n["顆粒"] * 23.085e6 / 101e9)), f"橋核 {b.n['橋核']}体（顆粒 × 2,309万／1,010億）")
a = b.areas["顆粒"]
chk(a.n_inh == round(b.n["顆粒"] * 1.5 / 2991), f"顆粒の中の抑制の体（ゴルジ）{a.n_inh}体（プルキンエ1体に 1.5）")
for nm in ("プルキンエ", "分子層"):
    chk(b.areas[nm].n_inh == 0, f"{nm} 中の抑制の体なし（抑制は外の領野から）")
v = b.areas["VLp"]
chk(abs(v.n_inh / (v.n_ch + v.n_inh) - 0.275) < 0.02 and TB.PRIOR_TYPE["VLp"] == "視床", f"VLp 型 視床・抑制の体 {v.n_inh / (v.n_ch + v.n_inh):.1%}")
for nm in TB.CB:
    ar = b.areas[nm]
    chk(ar.ch.n_cells == 0 and b.n_ext[nm] == 0, f"{nm} 時間細胞なし・樹状突起の側なし")
o = lambda nm: int(b.off_ch[NAMES.index(nm)])
for nm in TB.CB:
    if nm != "VLp":
        chk(float(g.spont_p[o(nm):o(nm) + b.n[nm]].max()) == 0.0, f"{nm} 自発のくじなし")
chk(all(TB.PRIOR_TYPE[k] in TB.UNBAKED_OK for k in TB.CB if k != "VLp"), "小脳の型は焼くまで prior_v1（UNBAKED_OK）")

print("2 線", flush=True)
tot = 0
for k_, n_ in TB.K_CX_PONS.items():
    s = b.bu[f"{k_}→橋核"]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    tot += int(per[0])
    chk(bool((per == min(n_, b.n[k_])).all()), f"{k_}→橋核 受け手1体 {int(per.min())}本（{n_}）")
chk(tot == 64, f"皮質 → 橋核 合わせて {tot}本（64）")
s = b.bu["苔状線維CB"]
per = torch.bincount(s.tgt, minlength=s.n_tgt)
pairs = s.tgt * 10 ** 6 + s.src
chk(bool((per == TB.K_MF).all()) and len(torch.unique(pairs)) == len(pairs), f"苔状線維 顆粒1体 {TB.K_MF}本（別々の橋核から）")
s = b.bu[TB.PF_KEY]
per = torch.bincount(s.tgt, minlength=s.n_tgt).float()
mu, sd = b.n["顆粒"] * TB.P_PF, (b.n["顆粒"] * TB.P_PF * (1 - TB.P_PF)) ** 0.5
chk(bool(((per - mu).abs() < 5 * sd).all()), f"平行線維 プルキンエ1体 {int(per.min())}〜{int(per.max())}本（顆粒 × {TB.P_PF} ＝ {mu:.0f}）")
chk(TB.PF_KEY not in b.but.keys, "平行線維は表の外（登上線維の学び）")
s = b.bu["登上線維"]
chk(bool((torch.bincount(s.tgt, minlength=s.n_tgt) == 1).all()) and bool((s.w == TB.CF_W).all()), f"登上線維 プルキンエ1体に1本・重さ {TB.CF_W:.2f}")
for key, (sn, tn) in [(k_, TB.INH_TO_AREA[k_]) for k_ in ("プルキンエ→小脳核", "プルキンエ→小脳核の抑制", "分子層→プルキンエ", "小脳核の抑制→下オリーブ")]:
    s, rows = b.bu_inh[key]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    chk(bool((per == min(TB.K_NUC, b.n[sn])).all()) and bool((s.w == TB.W_INH_BU).all()),
        f"{key}（抑制）受け手1体 {int(per.min())}本（{min(TB.K_NUC, b.n[sn])}）")
for key in ("小脳核→VLp", "小脳核→赤核", "小脳核→CM", "運動→赤核", "補足→赤核", "VLp→運動", "運動→VLp", "顆粒→分子層"):
    s = b.bu[key]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    k = min(TB.K_NEW, b.n[key.split("→")[0]])
    chk(bool((per == k).all()), f"{key} 受け手1体 {int(per.min())}本（{k}）")
for key, k in (("橋核→小脳核", b.n["橋核"]), ("下オリーブ→小脳核", b.n["下オリーブ"]), ("赤核→下オリーブ", min(TB.K_NUC, 64))):
    s = b.bu[key]
    chk(bool((torch.bincount(s.tgt, minlength=s.n_tgt) == k).all()), f"{key} 受け手1体 {k}本")

print("3 プルキンエの自分で撃つ分", flush=True)
chk(bool((b.pc_i0 > 0).all()), f"i0 {float(b.pc_i0[0]):.3f}（{TB.PC_HZ:.1f}Hz の形・τm 40ms）")
b.bs.set_state(BS.WAKE)
ear = torch.zeros(B.N_EAR); cmd = torch.zeros(b.n["運動"])
b.step(ear, cmd, learn=True)
chk(torch.allclose(g.cl_drive[b.pc_ch], b.pc_i0), "網の cl_drive にプルキンエの i0 が入る")
dk = float(g.v_decay[b.pc_ch[0]])
chk(abs(dk - np.exp(-10.67 / 40.0)) < 1e-5, f"プルキンエの時間の足し合わせ τm 40ms（{dk:.4f}）")

print("4 平行線維の学びの向き", flush=True)
s = b.bu[TB.PF_KEY]
w0 = s.w.clone()
nG = b.n["顆粒"]
b.gr_age = torch.full((nG,), 10 ** 6, dtype=torch.long)
b.cf_age = torch.full((b.n["プルキンエ"],), 10 ** 6, dtype=torch.long)
g_a, g_b = 0, 1                                        # 顆粒 a：3歩前に撃つ ／ 顆粒 b：今撃つ（登上線維なし）
b.nuc["下オリーブ"] = torch.zeros(b.n["下オリーブ"])
for t in range(3):
    gr = torch.zeros(nG); b.nuc["顆粒"] = gr.clone()
    if t == 0:
        b.nuc["顆粒"][g_a] = 1.0
    b._pf_learn()
b.nuc["顆粒"] = torch.zeros(nG); b.nuc["下オリーブ"] = torch.ones(b.n["下オリーブ"])   # 登上線維が全部のプルキンエに来る
b._pf_learn()
ea = s.src == g_a
chk(bool((s.w[ea] < w0[ea]).all()), f"登上線維の 3歩前の平行線維は弱まる（{float((s.w[ea] / w0[ea]).mean()):.4f}倍）")
s.w = w0.clone()
b.cf_age = torch.full((b.n["プルキンエ"],), 10 ** 6, dtype=torch.long)
b.nuc["下オリーブ"] = torch.zeros(b.n["下オリーブ"]); b.nuc["顆粒"] = torch.zeros(nG); b.nuc["顆粒"][g_b] = 1.0
b._pf_learn()
eb, other = s.src == g_b, (s.src != g_b)
chk(bool((s.w[eb] > w0[eb]).all()) and torch.equal(s.w[other], w0[other]), f"登上線維なしに撃った平行線維は強まる（{float((s.w[eb] / w0[eb]).mean()):.4f}倍）・ほかは変わらない")
s.w = w0.clone()

print("5 回る・保存と読み込み", flush=True)
import life as L
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
lf.run_plans(1, learn=True)
nz = {k: float(b.nuc[k].mean()) for k in TB.CB}
print(f"  （見るだけ）{lf.t}歩・声 {lf.stats['voices']}・最後の歩の小脳の発火 { {k: round(v, 3) for k, v in nz.items()} }", flush=True)
chk(lf.t > 0, f"{lf.t}歩 回った")
d = tempfile.mkdtemp()
p = os.path.join(d, "s.pt")
B.save(b, 1, path=p)
w_pf = b.bu[TB.PF_KEY].w.clone()
b2 = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=p)
chk(torch.equal(b2.bu[TB.PF_KEY].w, w_pf), "平行線維の重さが保存・読み込みで戻る")
chk(B.VERSION == 98 and B.SAME_NET == (), "VERSION 98・SAME_NET なし（state.pt は 0 から）")
print("✅ 版98 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
