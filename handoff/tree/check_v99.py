"""版99 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない【み】10-18 22:08
  python3 check_v99.py
確かめること
  1 組める・体数（10 §7）・型・抑制の体（基底外側は霊長類の GABA の割合・中心核と介在塊は線条体の型）・時間細胞なし・自発のくじなし・眼窩前頭は皮質の作り
  2 線：組み表_版99 §3 の受け手1体の本数・嗅内浅へは層 III の体だけ・介在塊の抑制の線・中心核の棘どうし・樹状突起の側の大きさと調整する側の印
  3 恒常性の目標：基底外側 2.0・3.35Hz・中心核 1.9Hz・介在塊は判定しない・眼窩前頭は皮質
  4 回る（作り物の声で1番）・保存と読み込みで扁桃体の線の重さが戻る
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
want = {"扁桃体外側": 86, "扁桃体基底": 72, "扁桃体副基底": 27, "扁桃体中心": 7, "介在塊": 35, "眼窩前頭": 377, "前頭前野": 816}
for nm, n in want.items():
    chk(b.n[nm] == n, f"{nm} {b.n[nm]}体（{n}）")
typ = {"扁桃体外側": "海馬", "扁桃体基底": "海馬", "扁桃体副基底": "海馬", "扁桃体中心": "線条体", "介在塊": "線条体", "眼窩前頭": "皮質"}
chk(all(TB.PRIOR_TYPE[k] == v for k, v in typ.items()), "型：基底外側 海馬・中心核と介在塊 線条体・眼窩前頭 皮質")
for nm, f in TB.AMY_INH_FRAC.items():
    a = b.areas[nm]
    chk(a.n_inh == max(1, round(a.n_ch * f / (1 - f))), f"{nm} 抑制の体 {a.n_inh}（割合 {f:.3f}）")
for nm in TB.AMY_STR:
    a = b.areas[nm]
    chk(a.n_inh == max(1, round(a.n_ch * TB.INH_FRAC_STR / (1 - TB.INH_FRAC_STR))), f"{nm} 抑制の体 {a.n_inh}（線条体の型 {TB.INH_FRAC_STR}）")
o = lambda nm: int(b.off_ch[NAMES.index(nm)])
for nm in TB.AMY:
    chk(b.areas[nm].ch.n_cells == 0 and b.n_ext[nm] == 0, f"{nm} 時間細胞なし・樹状突起の側なし")
    chk(float(g.spont_p[o(nm):o(nm) + b.n[nm]].max()) == 0.0, f"{nm} 自発のくじなし")
a, p = b.areas["眼窩前頭"], b.areas["前頭前野"]
chk(a.ch.n_cell == p.ch.n_cell and a.ch.n_cell > 0 and a.n_inh == round(377 * 0.16 / 0.84), f"眼窩前頭 時間細胞 送り元ごとに {a.ch.n_cell}（前頭前野と同じ）・抑制の体 {a.n_inh}")
co = set(int(i) for i in b.cl_ch)
chk(all(c in co for c in range(o("眼窩前頭"), o("眼窩前頭") + b.n["眼窩前頭"])), "CL → 眼窩前頭の表現の体（皮質と同じ）")
dk = lambda tau: float(np.exp(-10.67 / tau))
chk(all(abs(float(g.v_decay[o(nm)]) - dk(TB.SUM_TAU_HIPPO_MS)) < 1e-6 for nm in TB.BLA), "基底外側の足し合わせ 30ms（海馬の型）")

print("2 線", flush=True)
for key, tgt in TB.P99_PATHS:
    s = b.bu[key]
    sn = key.split("→")[0]
    k = min(TB.K_P99.get(key, TB.K_NEW), b.n[sn])
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    if tgt == "嗅内浅":
        okk = bool((per[:TB.EC_II] == 0).all()) and bool((per[TB.EC_II:] == k).all())
        chk(okk and s.n_tgt == b.n["嗅内浅"], f"{key} 層 III の体だけに {k}本（層 II の {TB.EC_II}体は 0）")
    else:
        pairs = s.tgt * 10 ** 6 + s.src
        chk(bool((per == k).all()) and len(torch.unique(pairs)) == len(pairs) and s.n_tgt == b.n[tgt],
            f"{key} 受け手1体 {int(per.min())}本（{k}、別々の送り手）")
    chk(int(s.src.max()) < b.n[sn], f"  {key} 送り手は {sn} の {b.n[sn]}体の中")
for key in ("介在塊→扁桃体中心", "介在塊→扁桃体基底", "介在塊→扁桃体外側"):
    s, rows = b.bu_inh[key]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    chk(bool((per == b.n["介在塊"]).all()) and bool((s.w == TB.W_INH_BU).all()), f"{key}（抑制）受け手1体 {int(per.min())}本（介在塊の全部）")
ln = [l for l in TB.NUC_LINES if l[0] == ("扁桃体中心",)]
chk(len(ln) == 1 and ln[0][1] == "扁桃体中心" and ln[0][2] == -1, "中心核の棘どうし（抑制の側枝、NucNet）")
n = b.n
ext = {"連合": 263 + n["嗅内深"] + 72 + 27, "高次": n["運動"] + 816 + 72 + 27 + 377, "前頭前野": n["嗅内深"] + n["前補足"] + 72 + 377,
       "背内側視床": 816 + 377, "CA1": n["嗅内浅"] - TB.EC_II + 27, "CA3": n["歯状回"] + n["CA3"] + TB.EC_II + 27, "眼窩前頭": 816}
for nm, v in ext.items():
    chk(b.n_ext[nm] == v, f"{nm} 樹状突起の側 {b.n_ext[nm]}（{v}）")
rows = (("連合", n["連合"] + n["高次"] + n["嗅内深"], 99), ("高次", n["高次"] + n["運動"] + 816, 99 + 377),
        ("CA3", n["CA3"] + n["歯状回"] + n["CA3"] + TB.EC_II, 27), ("CA1", n["CA1"] + n["嗅内浅"] - TB.EC_II, 27),
        ("前頭前野", 816 + n["嗅内深"] + n["前補足"], 72 + 377), ("背内側視床", n["背内側視床"] + 816, 377), ("眼窩前頭", 377, 816))
for nm, first, cnt in rows:
    r = b._send_rows(nm, first, cnt)
    chk(bool(g.mod_send[r].all()), f"{nm} の扁桃体・眼窩前頭の樹状突起の側は調整する側（撃たせない）")
want_td = {("高次", n["高次"] + n["運動"] + 816 + 72 + 27, 377), ("前頭前野", 816 + n["嗅内深"] + n["前補足"] + 72, 377), ("眼窩前頭", 377, 816)}
chk(want_td <= set(b.topdown_rows), "眼窩前頭 → 高次・前頭前野 ⇄ 眼窩前頭（樹状突起の側）は上からの戻り（眠りで落とす。前頭前野と同じ）")

print("3 恒常性の目標", flush=True)
hz = lambda v: v * 10.67 / 1000.0
for nm in NAMES:
    if nm not in TB.AV:
        continue
    i = NAMES.index(nm)
    lo, hi, nc = int(g.off_recv[i]), int(g.off_recv[i + 1]), b.areas[nm].n_ch
    if nm in TB.BLA:
        chk(torch.allclose(g.target_recv[lo:lo + 2 * nc], torch.tensor(hz(2.0))) and torch.allclose(g.target_recv[lo + 2 * nc:hi], torch.tensor(hz(3.35))),
            f"{nm} 表現・誤差 2.0Hz・抑制 3.35Hz")
    elif nm == "扁桃体中心":
        chk(torch.allclose(g.target_recv[lo:lo + 2 * nc], torch.tensor(hz(1.9))) and bool(g.scale_skip[lo + 2 * nc:hi].all()), f"{nm} 1.9Hz・抑制は判定しない")
    elif nm == "介在塊":
        chk(bool(g.scale_skip[lo:hi].all()), f"{nm} 判定しない")
    else:
        j = NAMES.index("前頭前野")
        lp, hp = int(g.off_recv[j]), int(g.off_recv[j + 1])
        chk(torch.allclose(g.target_recv[lo + 2 * nc:hi], g.target_recv[hp - 1]) and torch.allclose(g.target_recv[lo:lo + 1], g.target_recv[lp:lp + 1]),
            f"{nm} 皮質の目標（前頭前野と同じ）")

print("4 回る・保存と読み込み", flush=True)
b.bs.set_state(BS.WAKE)
import life as L
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
lf.run_plans(1, learn=True)
nz = {k: round(float(b.nuc[k].mean()), 3) for k in TB.AMY}
nz["眼窩前頭"] = round(float(b.pred["眼窩前頭"].mean()), 3)
print(f"  （見るだけ）{lf.t}歩・声 {lf.stats['voices']}・最後の歩の発火 {nz}", flush=True)
chk(lf.t > 0, f"{lf.t}歩 回った")
d = tempfile.mkdtemp()
pth = os.path.join(d, "s.pt")
B.save(b, 1, path=pth)
b.bu_unpack()
w = {k_: b.bu[k_].w.clone() for k_ in ("高次→扁桃体外側", "扁桃体基底→眼窩前頭", "眼窩前頭→扁桃体中心")}
b2 = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=pth)
b2.bu_unpack()
chk(all(torch.equal(b2.bu[k_].w, v) for k_, v in w.items()), "扁桃体・眼窩前頭の線の重さが保存・読み込みで戻る")
chk(B.VERSION == 99 and B.SAME_NET == (), "VERSION 99・SAME_NET なし（state.pt は 0 から）")
print("✅ 版99 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
