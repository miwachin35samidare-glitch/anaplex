"""版102 の確かめ（版101 の確かめ ＋ 縮める側 × 0.85）。元は版101（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない【み】10-18 22:08
  python3 check_v102.py
確かめること（組み表_版101）
  1 組める・来ない体は OMIT の7領野に帯ごとに1体（ほかの領野には無い）・誤差の種類の行で撃つ向きだけ逆・同じ帯の入口・相手の数は誤差の体と同じ
  2 撃ち方：来ない体は (p − 的) ＞ THETA で撃つ・誤差の体は (的 − p) ＞ THETA のまま・学びはどちらも 的 − p
  3 線：上の領野への線は送り手が同じ帯の誤差の体か来ない体（半々）・受け手1体の本数は版100 のまま ／ 前頭前野・眼窩前頭・前帯状の来ない体 → 視床下核
  4 核の線・核 → 領野の抑制は誤差の体の行だけ（来ない体を入れない）／ 恒常性の目標は来ない体も誤差の体と同じ
  5 驚きの和に来ない体の発火が入る
  6 回る・保存と読み込み・辺の数と VRAM の見込み
"""
import sys, os, tempfile, inspect, numpy as np, torch
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, torch_group2 as G2
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


print(f"1 組めた：体 {b.n_cells:,}・網の辺 {len(g.w):,}・領野 {len(NAMES)}", flush=True)
for nm, v in {"連合": 204, "高次": 224, "前頭前野": 578, "眼窩前頭": 320, "前帯状": 275}.items():
    chk(b.n[nm] == v, f"{nm} {b.n[nm]}体（版101 × 0.85 ＝ {v}）")
chk(TB.OMIT == ("聴覚", "連合", "高次", "視覚", "前頭前野", "眼窩前頭", "前帯状"), f"置く領野 {'・'.join(TB.OMIT)}【み選】10-29 22:52")
n_om_all = 0
for i, nm in enumerate(NAMES):
    a = b.areas[nm]
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    n_om = int(g.is_om[lo:hi].sum())
    want = b.n[nm] if nm in TB.OMIT else 0
    if nm in TB.OMIT or n_om:
        chk(n_om == want and a.n_om == want, f"{nm} 来ない体 {n_om}（{want}）")
    n_om_all += n_om
chk(n_om_all == sum(b.n[k] for k in TB.OMIT), f"来ない体 合わせて {n_om_all}")
for nm in ("運動", "補足", "前補足", "CA1"):
    chk(b.areas[nm].n_om == 0, f"{nm} には置かない")
om = torch.nonzero(g.is_om).squeeze(1)
chk(bool((g.kind[om] == 0).all()) and bool((g.fire_sgn[om] == -1).all()) and bool((g.fire_sgn[~g.is_om] == 1).all()),
    "来ない体は誤差の種類の行・撃つ向きだけ逆（fire_sgn −1）")
for nm in ("聴覚", "前頭前野"):
    i = NAMES.index(nm); lo = int(g.off_recv[i]); a = b.areas[nm]; nc = a.n_ch
    e_rows = torch.arange(lo, lo + nc); o_rows = torch.arange(lo + 2 * nc + a.n_inh, lo + 2 * nc + a.n_inh + nc)
    chk(bool(g.is_om[o_rows].all()) and torch.equal(g.x_index[o_rows], g.x_index[e_rows]), f"{nm} 来ない体の行は抑制の体のうしろ・同じ帯の入口（的）")
    cnt = torch.bincount(g.e_recv.long(), minlength=g.n_recv_all)
    chk(torch.equal(cnt[o_rows], cnt[e_rows]), f"{nm} 来ない体の相手の数 {int(cnt[o_rows][0])}（誤差の体と同じ）")
chk(len(g.err_ch_i) == g.n_ch_all and not bool(g.is_om[g.err_ch_i].any()) and bool(g.is_err[g.err_ch_i].all()),
    "誤差の体の行（帯の順）err_ch_i は来ない体を含まない")
chk(len(g.err_i) == g.n_ch_all + n_om_all and len(g.om_pos) == n_om_all and len(g.err_pos_ch) == g.n_ch_all,
    "誤差の発火の並び ＝ 誤差の体 ＋ 来ない体（領野ごとに 誤差 ｜ 来ない）")

print("2 撃ち方と学び", flush=True)
ne, ni = len(g.err_i), len(g.inh_i)
p = torch.full((g.n_recv_all,), 0.5); tgt = torch.zeros(g.n_recv_all)
p[g.err_ch_i[:5]] = 0.0; tgt[g.err_ch_i[:5]] = 1.0          # 来た（的 1・予測 0）⇒ 誤差の体が撃つ
p[om[:5]] = 1.0; tgt[om[:5]] = 0.0                          # 来ない（的 0・予測 1）⇒ 来ない体が撃つ
p[g.err_ch_i[5:10]] = 1.0; tgt[g.err_ch_i[5:10]] = 0.0      # 誤差の体は来ないでは撃たない
p[om[5:10]] = 0.0; tgt[om[5:10]] = 1.0                      # 来ない体は来たでは撃たない
err_sig = tgt - p
z = torch.zeros(g.n_recv_all)
out = G2._out_core(err_sig * g.fire_sgn, z, z, z, g.err_i, g.inh_i, None, torch.zeros(g.n_ch_all),
                   torch.full((g.n_ch_all,), -99.0), torch.full((ne,), -99.0), torch.full((ni,), -99.0),
                   torch.zeros(()), None, None, None)
e = out[2]
pos = {int(r): k for k, r in enumerate(g.err_i)}
f = lambda rows: [int(e[pos[int(r)]]) for r in rows]
chk(f(g.err_ch_i[:5]) == [1] * 5 and f(om[:5]) == [1] * 5, "来た ⇒ 誤差の体が撃つ ／ 来ない ⇒ 来ない体が撃つ")
chk(f(g.err_ch_i[5:10]) == [0] * 5 and f(om[5:10]) == [0] * 5, "誤差の体は来ないで撃たない ／ 来ない体は来たで撃たない")
src = inspect.getsource(G2.Group2)
chk(src.count("_out_core, err_sig * self.fire_sgn") == 3, "撃つ所（3つの回し方）は全部 向き × (的 − p)")
chk("fire_sgn" not in inspect.getsource(G2._learn_core) and "err_sig * self.fire_sgn, g" not in src,
    "学びは 的 − p のまま（来ない体も p を的に寄せる）")

print("3 線", flush=True)
orig = {"聴覚→連合": b.a2c, "連合→高次": b.c2h, "高次→前頭前野": b.p_h2pfc, "高次→嗅内浅": b.p_h2ecs,
        "前頭前野→嗅内浅": b.p_pfc2ecs, "前頭前野→前補足": b.p96["前頭前野→前補足"],
        **{k_: b.p99[k_] for k_ in ("高次→眼窩前頭", "前頭前野→眼窩前頭", "眼窩前頭→前頭前野")},
        **{k_: b.p100[k_] for k_ in ("前帯状→前頭前野", "前頭前野→前帯状", "前帯状→眼窩前頭", "眼窩前頭→前帯状", "前帯状→前補足")}}
chk(set(orig) == set(TB.OM_UP_LINES), f"上への線 {len(TB.OM_UP_LINES)}本の送り手を 誤差 ｜ 来ない に")
fr = []
for k_, (s0, t0, c0) in orig.items():
    s = b.bu[k_]; ns = b.n[k_.split("→")[0]]
    a_ = sorted(zip((s.tgt).tolist(), (s.src % ns).tolist())); b_ = sorted(zip(t0.tolist(), s0.tolist()))
    fr.append(float((s.src >= ns).float().mean()))
    chk(a_ == b_ and int(s.src.max()) < 2 * ns, f"{k_} 受け手1体の本数・帯は版100 のまま（来ない体 {fr[-1]:.0%}）")
chk(0.4 < float(np.mean(fr)) < 0.6, f"来ない体へ移った線 平均 {np.mean(fr):.0%}（半々）")
for k_ in TB.OM_STN:
    key = f"{k_}来ない→視床下核"; s = b.bu[key]
    per = torch.bincount(s.tgt, minlength=s.n_tgt); pairs = s.tgt * 10 ** 6 + s.src
    kk = min(TB.K_NUC, b.n[k_])
    chk(bool((per == kk).all()) and len(torch.unique(pairs)) == len(pairs) and int(s.src.max()) < b.n[k_] and s.n_tgt == b.n["視床下核"]
        and key in TB.NUC_BU["視床下核"], f"{key} 受け手1体 {kk}本（別々の来ない体）")
chk("前頭前野→視床下核" in TB.NUC_BU["視床下核"], "前頭前野の表現 → 視床下核 は残す（10-18 の 表現 ＋ 来ない）")

print("4 核の線・恒常性", flush=True)
inh_t = torch.cat([v[1] for v in b.bu_inh.values()])
chk(not bool(g.is_om[inh_t].any()), "核 → 領野の抑制の受け手に来ない体なし")
nt = torch.cat([grp[5] for grp in g.nucnet.groups if grp[0] < 0])
chk(not bool(g.is_om[nt].any()), "核どうしの抑制の受け手に来ない体なし")
for nm in TB.OMIT:
    i = NAMES.index(nm); lo = int(g.off_recv[i]); a = b.areas[nm]; nc = a.n_ch; o0 = lo + 2 * nc + a.n_inh
    chk(torch.allclose(g.target_recv[o0:o0 + nc], g.target_recv[lo:lo + nc]) and not bool(g.scale_skip[o0:o0 + nc].any())
        and torch.allclose(g.target_recv[lo + 2 * nc:o0], g.target_recv[lo + 2 * nc]),
        f"{nm} 来ない体の目標 ＝ 誤差の体（抑制の体の目標は抑制の体だけ）")

print("5 驚き", flush=True)
chk("om[nm] for nm in CORTEX" in inspect.getsource(BS.Brainstem.observe) if hasattr(BS, "Brainstem") else "om[nm] for nm in CORTEX" in open("brainstem.py").read(),
    "驚きの和 ＝ CORTEX の 誤差の体 ＋ 来ない体")

print("6 回る・保存と読み込み", flush=True)
b.bs.set_state(BS.WAKE)
import life as L
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
lf.run_plans(1, learn=True)
print(f"  （見るだけ）{lf.t}歩・声 {lf.stats['voices']}・最後の歩の来ない体 {{{', '.join(f'{k}: {int(v.sum())}' for k, v in b.om.items())}}}", flush=True)
chk(lf.t > 0, f"{lf.t}歩 回った")
chk(all(len(b.up[k]) == b.n[k] and len(b.om[k]) == b.n[k] for k in TB.OMIT), "up ＝ 誤差の体・om ＝ 来ない体（各 帯の数）")
d = tempfile.mkdtemp(); pth = os.path.join(d, "s.pt")
B.save(b, 1, path=pth)
b.bu_unpack()
keys = ("聴覚→連合", "前頭前野来ない→視床下核", "前帯状来ない→視床下核")
w = {k_: b.bu[k_].w.clone() for k_ in keys}
gw = g.w.clone()
b2 = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=pth)
b2.bu_unpack()
chk(all(torch.equal(b2.bu[k_].w, v) for k_, v in w.items()) and torch.equal(b2.g.w, gw), "線・網の重さが保存・読み込みで戻る")
chk(all(torch.equal(b2.om[k], b.om[k]) for k in TB.OMIT), "来ない体の発火（1歩前）も戻る")
chk(B.VERSION == 102 and B.SAME_NET == (), "VERSION 102・SAME_NET なし（state.pt は 0 から）")
# 辺の数（相手 200 のコンテナ）と arkhe の見込み
i_om = g.is_om[g.e_recv.long()]
print(f"  辺：来ない体の行 {int(i_om.sum()):,}（コンテナ・相手 200）", flush=True)
np_ark = 9100
add = n_om_all * np_ark
print(f"  arkhe の見込み【計算】：来ない体 {n_om_all} × 相手 {np_ark:,} ＝ {add:,}本", flush=True)
print("✅ 版102 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
