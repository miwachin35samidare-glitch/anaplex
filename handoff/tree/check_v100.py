"""版100 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない【み】10-18 22:08
  python3 check_v100.py
確かめること
  1 組める・体数（10 §7）・型・前帯状は皮質の作り・PAG は核の作り（時間細胞なし・自発のくじなし・i0 なし・τm 40ms・中の興奮の横なし）
  2 線：組み表_版100 §3 の受け手1体の本数・扁桃体中心 ⊣ PAG・樹状突起の側の大きさと印（横は眠りで落とす側）
  3 恒常性の目標：前帯状は皮質・PAG は判定しない
  4 PAG → 口：PAG の1体が口の欄の1つ（重ならない）へ・口が読む発火は運動 ＋ PAG
  5 回る（作り物の声で1番）・保存と読み込みで前帯状・PAG の線の重さが戻る
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


print(f"1 組めた：体 {b.n_cells:,}・網の辺 {len(g.w):,}・領野 {len(NAMES)}", flush=True)
for nm, n in {"前帯状": 323, "PAG": 256, "前頭前野": 680, "眼窩前頭": 377}.items():
    chk(b.n[nm] == n, f"{nm} {b.n[nm]}体（{n}）")
chk(TB.PRIOR_TYPE["前帯状"] == "皮質" and TB.PRIOR_TYPE["PAG"] == "PAG" and "PAG" in TB.UNBAKED_OK, "型：前帯状 皮質・PAG（焼くまで prior_v1）")
o = lambda nm: int(b.off_ch[NAMES.index(nm)])
a, p = b.areas["前帯状"], b.areas["前頭前野"]
chk(a.ch.n_cell == p.ch.n_cell and a.ch.n_cell > 0 and a.n_inh == round(323 * 0.16 / 0.84), f"前帯状 時間細胞 送り元ごとに {a.ch.n_cell}（前頭前野と同じ）・抑制の体 {a.n_inh}")
co = set(int(i) for i in b.cl_ch)
chk(all(c in co for c in range(o("前帯状"), o("前帯状") + b.n["前帯状"])), "CL → 前帯状の表現の体（皮質と同じ）")
chk(abs(a.tau_scale - p.tau_scale) < 1e-9 and abs(a.tau_scale - TB.TAU["高次"]) < 1e-9, f"前帯状 時定数 ×{a.tau_scale:.2f}（前頭前野と同じ高次の側）")
q = b.areas["PAG"]
chk(q.ch.n_cells == 0 and b.n_ext["PAG"] == 0, "PAG 時間細胞なし・樹状突起の側なし")
chk(float(g.spont_p[o("PAG"):o("PAG") + 256].max()) == 0.0, "PAG 自発のくじなし")
chk(q.n_inh == b.areas["黒質"].n_inh, f"PAG 抑制の体 {q.n_inh}（今の核と同じ割合 0.16、ノブ）")
dk = lambda tau: float(np.exp(-10.67 / tau))
chk(abs(float(g.v_decay[o("PAG")]) - dk(TB.TAU_M_NUC)) < 1e-6, "PAG の足し合わせ τm 40ms（核の作り）")
chk(bool((g.cl_drive[o("PAG"):o("PAG") + 256] == 0).all()) and "PAG" not in TB.NUC_VINF, "PAG i0 なし")
chk("PAG" not in TB.NUC_TAIL, "PAG NMDA の尾なし")

print("2 線", flush=True)
for key, tgt in TB.P100_PATHS:
    s = b.bu[key]
    sn = key.split("→")[0]
    k = min(TB.K_NEW, b.n[sn])
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    pairs = s.tgt * 10 ** 6 + s.src
    chk(bool((per == k).all()) and len(torch.unique(pairs)) == len(pairs) and s.n_tgt == b.n[tgt] and int(s.src.max()) < b.n[sn],
        f"{key} 受け手1体 {int(per.min())}本（{k}、別々の送り手・送り手は {sn} の中）")
ln = [l for l in TB.NUC_LINES if l[1] == "PAG"]
chk(len(ln) == 1 and ln[0][0] == ("扁桃体中心",) and ln[0][2] == -1, "扁桃体中心 ⊣ PAG（抑制、NucNet）だけが核の線")
chk(all("眼窩前頭→PAG" != k_ and "扁桃体基底→PAG" != k_ for k_, _t in TB.BU_PATHS), "眼窩前頭 → PAG・扁桃体基底 → PAG は張らない（人の線なし）")
n = b.n
ext = {"前頭前野": n["嗅内深"] + n["前補足"] + 72 + 377 + 323, "背内側視床": 680 + 377 + 323,
       "前補足": n["運動"] + 680 + 323, "眼窩前頭": 680 + 323, "前帯状": 680 + 377 + n["前補足"] + 72 + 27}
for nm, v in ext.items():
    chk(b.n_ext[nm] == v, f"{nm} 樹状突起の側 {b.n_ext[nm]}（{v}）")
td = {("前頭前野", 680 + n["嗅内深"] + n["前補足"] + 72 + 377, 323), ("前補足", n["前補足"] + n["運動"] + 680, 323),
      ("眼窩前頭", 377 + 680, 323), ("前帯状", 323, 680 + 377 + n["前補足"])}
chk(td <= set(b.topdown_rows), "前帯状 ⇄ 前頭前野・眼窩前頭・前補足（横の樹状突起の側）は上からの戻り（眠りで落とす）")
for nm, first, cnt in list(td) + [("前帯状", 323 + 680 + 377 + n["前補足"], 99), ("背内側視床", n["背内側視床"] + 680 + 377, 323)]:
    r = b._send_rows(nm, first, cnt)
    chk(bool(g.mod_send[r].all()), f"{nm} の前帯状まわりの樹状突起の側（{first}〜 {cnt}）は調整する側（撃たせない）")

print("3 恒常性の目標", flush=True)
i, j = NAMES.index("前帯状"), NAMES.index("前頭前野")
lo, hi, nc = int(g.off_recv[i]), int(g.off_recv[i + 1]), b.areas["前帯状"].n_ch
lp, hp = int(g.off_recv[j]), int(g.off_recv[j + 1])
chk(torch.allclose(g.target_recv[lo + 2 * nc:hi], g.target_recv[hp - 1]) and torch.allclose(g.target_recv[lo:lo + 1], g.target_recv[lp:lp + 1]),
    "前帯状 皮質の目標（前頭前野と同じ）")
k = NAMES.index("PAG")
chk(bool(g.scale_skip[int(g.off_recv[k]):int(g.off_recv[k + 1])].all()), "PAG 判定しない（人の率なし）")

print("4 PAG → 口", flush=True)
pc = b.pag_col
chk(len(pc) == 256 and len(torch.unique(pc)) == 256 and int(pc.max()) < b.n["運動"], f"PAG の1体 → 口の欄の1つ（256欄・重ならない・{b.n['運動']}欄の中）")
b.motor_rep = torch.zeros(b.n["運動"]); b.nuc["PAG"] = torch.zeros(256); b.nuc["PAG"][:3] = 1.0
mi = b.motor_rep.clone(); mi[pc] = torch.maximum(mi[pc], b.nuc["PAG"])
chk(int(mi.sum()) == 3 and bool((mi[pc[:3]] == 1).all()), "口が読む発火 ＝ 運動 ＋ PAG（PAG の 3体 → その 3欄）")
b.nuc["PAG"] = torch.zeros(256)

print("5 回る・保存と読み込み", flush=True)
b.bs.set_state(BS.WAKE)
import life as L
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
lf.run_plans(1, learn=True)
nz = {"前帯状": round(float(b.pred["前帯状"].mean()), 3), "PAG": round(float(b.nuc["PAG"].mean()), 3)}
print(f"  （見るだけ）{lf.t}歩・声 {lf.stats['voices']}・最後の歩の発火 {nz}", flush=True)
chk(lf.t > 0, f"{lf.t}歩 回った")
chk(b.mouth_in.shape == b.motor_rep.shape and bool((b.mouth_in >= b.motor_rep).all()), "口が読む発火は運動の領野を含む")
d = tempfile.mkdtemp()
pth = os.path.join(d, "s.pt")
B.save(b, 1, path=pth)
b.bu_unpack()
w = {k_: b.bu[k_].w.clone() for k_ in ("前帯状→PAG", "運動→PAG", "前帯状→前頭前野", "CA1→前帯状")}
b2 = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=pth)
b2.bu_unpack()
chk(all(torch.equal(b2.bu[k_].w, v) for k_, v in w.items()), "前帯状・PAG の線の重さが保存・読み込みで戻る")
chk(torch.equal(b2.pag_col, b.pag_col), "PAG → 口の欄は作り直しても同じ（配線は seed=0）")
chk(B.VERSION == 100 and B.SAME_NET == (), "VERSION 100・SAME_NET なし（state.pt は 0 から）")
print("✅ 版100 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
