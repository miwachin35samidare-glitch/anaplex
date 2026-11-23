"""版96 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない【み】10-18 22:08
  python3 check_v96.py [歩=150]                    （周ごと：_pass の形）
  ANAPLEX_GRAPH=eager python3 check_v96.py [歩]   （グラフの形の式をそのまま）
確かめること
  1 組める・体数（補足 81・前補足 146・VA 56・VLa 54・CM 19・Pf 4・CL 5）・型・時間細胞・抑制の体
  2 線：新しい下からの道の本数・線条体の 2,700本の割り振り・核 → 領野の抑制（内節 → VA・VLa・網様部 → VA）
  3 樹状突起の側（下り・横）：受け手の ext の数と、調整する側の印（mod_send）が付いているか
  4 蹴りが無い・合図が補足の CUE_FRAC の体へ CUE_ON〜CUE_OFF の間だけ入る（形：山で一番多い）
  5 回る（作り物の声で喃語）・新しい領野に下からの入力が届いている・保存と読み込み
"""
import sys, os, tempfile, numpy as np, torch
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

B.DEV = "cpu"
NSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev="cpu")
b = Brain(prior, n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


print(f"1 組めた：体 {b.n_cells:,}・網の辺 {len(g.w):,}・領野 {len(NAMES)}", flush=True)
for nm, n in {"補足": 81, "前補足": 146, "VA": 56, "VLa": 54, "CM": 19, "Pf": 4, "CL": 5}.items():
    chk(b.n[nm] == n, f"{nm} {b.n[nm]}体（{n}）")
chk(TB.N_CL == 5, "N_CL 5")
for nm in ("補足", "前補足"):
    a = b.areas[nm]
    chk(TB.PRIOR_TYPE[nm] == "皮質" and a.ch.n_cells > 0, f"{nm} 型 皮質・時間細胞 {a.ch.n_cells}")
for nm in ("VA", "VLa"):
    a = b.areas[nm]
    chk(TB.PRIOR_TYPE[nm] == "視床" and a.ch.n_cells == 0 and abs(a.n_inh / (a.n_ch + a.n_inh) - 0.275) < 0.02,
        f"{nm} 型 視床・時間細胞なし・抑制の体 {a.n_inh / (a.n_ch + a.n_inh):.1%}")
    chk(b.n_ext[nm] == 0, f"{nm} 樹状突起の側なし（細胞体へだけ）")

print("2 線", flush=True)
K = TB.K_NEW
for key in ("補足→前補足", "補足→運動", "前補足→運動", "運動→前補足", "前補足→前頭前野", "前頭前野→前補足",
            "VLa→補足", "VA→前補足", "VA→前頭前野", "背内側視床→補足", "背内側視床→前補足",
            "補足→VLa", "前補足→VA", "前頭前野→VA", "補足→CM", "前補足→Pf"):
    s = b.bu[key]
    ns = b.n[key.split("→")[0]]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    k = min(K, ns)
    chk(bool((per == k).all()) and int(s.src.max()) < ns, f"{key} 受け手1体 {int(per.min())}〜{int(per.max())}本（{k}）")
for key in ("補足→視床下核", "前補足→視床下核"):
    s = b.bu[key]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    chk(bool((per == min(TB.K_NUC, b.n[key.split("→")[0]])).all()), f"{key} 受け手1体 {int(per.min())}〜{int(per.max())}本")
for tgt, keys in (("被殻", ("運動→被殻", "補足→被殻", "前補足→被殻")), ("尾状核", ("前頭前野→尾状核", "前補足→尾状核", "補足→尾状核"))):
    tot = 0
    for key in keys:
        per = torch.bincount(b.bu[key].tgt, minlength=b.n[tgt])
        chk(bool((per == TB.K_CX[key]).all()), f"{key} 受け手1体 {int(per[0])}本（{TB.K_CX[key]}）")
        tot += TB.K_CX[key]
    chk(tot == TB.K_CX_STR, f"{tgt} の皮質の線の和 {tot}（{TB.K_CX_STR}）")
chk(set(b.bu_inh) == set(TB.INH_TO_AREA) and {"内節→VLa", "内節→VA", "網様部→VA"} <= set(b.bu_inh), f"核 → 領野の抑制 {sorted(b.bu_inh)}")
chk(set(TB.NUC_BU["視床下核"]) == {"前頭前野→視床下核", "補足→視床下核", "前補足→視床下核"}, "視床下核へ 前頭前野・補足・前補足")
chk("補足" in TB.CL_CORTEX and "前補足" in TB.CL_CORTEX, "CL → 補足・前補足")

print("3 樹状突起の側（下り・横）", flush=True)
nn = b.n
want_ext = {"補足": nn["前補足"] + nn["運動"], "前補足": nn["運動"] + nn["前頭前野"],
            "運動": nn["高次"] + nn["前頭前野"] + nn["前補足"], "前頭前野": nn["嗅内深"] + nn["前補足"]}
for nm, n in want_ext.items():
    chk(b.n_ext[nm] == n, f"{nm} ext {b.n_ext[nm]}（{n}）")
for nm, first, cnt in (("運動", nn["運動"] + nn["高次"] + nn["前頭前野"], nn["前補足"]), ("前頭前野", nn["前頭前野"] + nn["嗅内深"], nn["前補足"]),
                       ("補足", nn["補足"], nn["前補足"] + nn["運動"]), ("前補足", nn["前補足"], nn["運動"] + nn["前頭前野"])):
    rows = b._send_rows(nm, first, cnt)
    chk(bool(g.mod_send[rows].all()), f"{nm} の ext の行 {cnt} が調整する側")
chk((("補足", nn["補足"], nn["前補足"] + nn["運動"]) in b.topdown_rows), "眠りで落とす線に入っている")

print("4 蹴りが無い・合図が補足へ", flush=True)
chk(not hasattr(b, "kick_abs") and not hasattr(b, "kick_left"), "蹴りが無い")
o = int(b.off_ch[NAMES.index("補足")])
chk(bool(((b.cue_abs >= o) & (b.cue_abs < o + nn["補足"])).all()) and len(b.cue_abs) == max(1, round(nn["補足"] * TB.CUE_FRAC)),
    f"合図の体 {len(b.cue_abs)}（補足 {nn['補足']} × {TB.CUE_FRAC:.3f}）")
# 合図の形：x に足されるものを、step を回さずに式で数える（くじの期待値）
ps = []
for t_ in range(110):
    if TB.CUE_ON <= t_ < TB.CUE_OFF:
        f_ = (t_ - TB.CUE_ON) / (TB.CUE_PEAK - TB.CUE_ON) if t_ < TB.CUE_PEAK else (TB.CUE_OFF - t_) / (TB.CUE_OFF - TB.CUE_PEAK)
        ps.append(TB.CUE_HZ * f_)
    else:
        ps.append(0.0)
chk(int(np.argmax(ps)) == TB.CUE_PEAK and ps[TB.CUE_ON - 1] == 0 and ps[TB.CUE_OFF] == 0, f"形：{TB.CUE_ON}歩目から上がり {TB.CUE_PEAK}歩目で山 {max(ps):.1f}Hz・{TB.CUE_OFF}歩目で 0")

print(f"5 回す（作り物の声、{NSTEP}歩ぶん）", flush=True)
b.bs.set_state(BS.WAKE)
_, say, _ = B.voices(b, True)
rng = np.random.default_rng(0)
seen = {k: 0 for k in ("補足", "前補足", "VA", "VLa")}
cue_seen, cue_steps = 0, 0
orig_step = b.g.step


def spy(x, e, learn):
    global cue_seen, cue_steps
    for nm in seen:
        o_ = int(b.off_ch[NAMES.index(nm)])
        seen[nm] += int(x[o_:o_ + b.n[nm]].abs().sum() > 0)
    if b.cue_t is not None:
        cue_steps += 1
    return orig_step(x, e, learn)


b.g.step = spy
t0 = g.t
while g.t - t0 < NSTEP:
    say(rng, True)
b.g.step = orig_step
print(f"  {g.t - t0}歩・下から入力が来た歩：" + "・".join(f"{k} {v}" for k, v in seen.items()) + f" ／ 合図の入った歩 {cue_steps}", flush=True)
chk(all(v > 0 for v in seen.values()), "新しい領野に入力が届いている")
for nm in ("補足", "前補足", "VA", "VLa"):
    print(f"  （見るだけ）{nm} の表現の発火 {float(b.pred[nm].mean()) if nm in b.pred else float(b.mth[nm].mean()):.3f}", flush=True)

print("6 保存・読み込み", flush=True)
p = os.path.join(tempfile.mkdtemp(), "state.pt")
B.save(b, 1, path=p)
b.bu_unpack()
w_before = {k: b.bu[k].w.clone() for k in ("補足→前補足", "VLa→補足", "補足→被殻")}
b2 = Brain(prior, n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=p)
b2.bu_unpack()
chk(all(torch.equal(b2.bu[k].w, v) for k, v in w_before.items()), "新しい線の重さが一致")
chk(torch.equal(b2.g.nucnet.w, b.g.nucnet.w), "NucNet の重さが一致")
print("✅ 版96 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
