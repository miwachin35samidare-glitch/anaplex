"""版103（103b：縮める側 × 0.55）の確かめ（組み表_版103：統語の部位・皮質 → 線条体を人の比に）。コンテナ・CPU・相手 200・作り物の声。数値（率）は読まない【み】10-18 22:08
  python3 check_v103.py
確かめること
  1 組める・体数（組み表 §1）・型（皮質／視床）・来ない体は 44・45 にも（10 §10 の置く所の決めを当てた）・CL は統語の皮質にも
  2 皮質どうしの線（§2・§2b）：向きは型の表どおり（上り ＝ 細胞体だけ・下り ＝ 樹状突起だけ・横 ＝ 両方）・受け手1体 K_NEW 本
  3 樹状突起の側：ext の並び・眠りで落とす行（topdown_rows）・調整する側の印
  4 視床 ⇄ 皮質（§3）・皮質 → 線条体の本数が人の比（Tziortzi 2014 の帯の合計 × 送り手の体数の比）
  5 来ない体：44・45 から上・横への線は送り手が 誤差 ｜ 来ない の半々
  6 回る・保存と読み込み・辺の数
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
want = {"44": 99, "45": 133, "STS1": 99, "STS2": 177, "中側頭後": 240, "TPJ": 54, "視床枕": 139}   # 版103b：組み表 §1 × 0.55（視床枕は変えない）
for nm, v in want.items():
    chk(b.n[nm] == v, f"{nm} {b.n[nm]}体（組み表 §1 {v}）")
chk(NAMES[-7:] == TB.SYN + TB.PUL, "新しい領野は NAMES の最後（前の並びは変えない）")
for nm in ("連合", "高次", "前頭前野", "眼窩前頭", "前帯状"):
    chk(b.n[nm] == {"連合": 112, "高次": 123, "前頭前野": 318, "眼窩前頭": 176, "前帯状": 151}[nm], f"{nm} {b.n[nm]}体（版102 × 0.55）")
chk(all(TB.PRIOR_TYPE[k] == "皮質" for k in TB.SYN) and TB.PRIOR_TYPE["視床枕"] == "視床", "型：統語の 6つは皮質・視床枕は視床")
a_va, a_pu = b.areas["VA"], b.areas["視床枕"]
chk(a_pu.ch.n_cell == a_va.ch.n_cell == 0 and a_pu.n_ext == 0 and abs(a_pu.n_inh / a_pu.n_ch - a_va.n_inh / a_va.n_ch) < 0.02,
    f"視床枕は VA と同じ作り（時間細胞なし・ext なし・抑制 {a_pu.n_inh}/{a_pu.n_ch}）")
chk(all(b.areas[k].ch.n_cell == b.areas["高次"].ch.n_cell for k in TB.SYN), "統語の皮質は今の皮質と同じ作り（時間細胞・傾きの細胞）")
chk(TB.OMIT[-2:] == ("44", "45") and all(b.areas[k].n_om == b.n[k] for k in ("44", "45"))
    and all(b.areas[k].n_om == 0 for k in ("STS1", "STS2", "中側頭後", "TPJ", "視床枕")), "来ない体：44・45 に帯ごとに1体・ほかの新しい領野には無い")
cl_set = set(b.cl_ch.tolist())
for k in TB.SYN:
    o = int(b.off_ch[NAMES.index(k)])
    chk(set(range(o, o + b.n[k])) <= cl_set, f"{k} CL → 皮質の受け手")
i_ = NAMES.index("44"); lo = int(g.off_recv[i_]); nc = b.n["44"]
chk(torch.allclose(g.target_recv[lo + 2 * nc:lo + 2 * nc + b.areas["44"].n_inh], torch.tensor(4.70 * 10.67 / 1000.0)),
    "統語の皮質の抑制の体の目標 4.70Hz（今の皮質と同じ）")

print("2 皮質どうしの線（型の表で向き）", flush=True)
TYPE = {"44": 4, "45": 5, "STS1": 4, "STS2": 5, "中側頭後": 5, "TPJ": 5, "運動": 4, "高次": 5, "連合": 5}
PAIRS = [("44", "45"), ("44", "運動"), ("44", "TPJ"), ("44", "中側頭後"), ("45", "高次"), ("連合", "中側頭後"), ("45", "STS1"),
         ("45", "STS2"), ("45", "中側頭後"),                                                       # §2
         ("高次", "STS1"), ("高次", "STS2"), ("STS1", "STS2"), ("STS1", "中側頭後"), ("STS2", "中側頭後"), ("STS1", "TPJ"),
         ("STS2", "TPJ"), ("高次", "TPJ"), ("中側頭後", "TPJ"), ("高次", "中側頭後")]                # §2b
want_bu, want_ext = set(), set()
for x_, y_ in PAIRS:
    for s_, t_ in ((x_, y_), (y_, x_)):
        if TYPE[s_] > TYPE[t_]:
            want_bu.add((s_, t_))            # 上り：誤差 → 細胞体
        elif TYPE[s_] < TYPE[t_]:
            want_ext.add((s_, t_))           # 下り：表現 → 樹状突起
        else:
            want_bu.add((s_, t_)); want_ext.add((s_, t_))   # 横：両方
got_bu = {tuple(k_.split("→")) for k_, _t in TB.P103_CX}
got_ext = {(s_, t_) for t_, ss in TB.EXT103.items() for s_ in ss}
chk(got_bu == want_bu, f"細胞体の側 {len(got_bu)}本 ＝ 型の表の上り・横（{len(want_bu)}）")
chk(got_ext == want_ext, f"樹状突起の側 {len(got_ext)}本 ＝ 型の表の下り・横（{len(want_ext)}）")
chk(all(t_ == k_.split("→")[1] for k_, t_ in TB.P103_ALL), "線の名前の受け手 ＝ 受け手の領野")
for k_, t_ in TB.P103_CX + TB.P103_TH:
    s = b.bu[k_]; ns = b.n[k_.split("→")[0]]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    kk = min(TB.K_NEW, ns)
    chk(bool((per == kk).all()) and s.n_tgt == b.n[t_], f"{k_} 受け手1体 {kk}本")
src = inspect.getsource(Brain.step)
chk("self.up[k_.split(\"→\")[0]] if (k_, t_) in P103_CX else self._rep103" in src, "皮質どうしは誤差を送る・視床と皮質 → 視床・線条体は表現を送る")

print("3 樹状突起の側", flush=True)
ext_old = {"運動": 123 + 318 + 146, "連合": 123 + 116 + 72 + 27, "高次": 840 + 318 + 72 + 27 + 176}   # 版103b
for nm, ss in TB.EXT103.items():
    base = ext_old.get(nm, 0)
    chk(b.n_ext[nm] == base + sum(b.n[s_] for s_ in ss), f"{nm} ext {b.n_ext[nm]}（今の {base} ＋ {'・'.join(ss) or 'なし'}）")
    first = b.n[nm] + base
    for s_ in ss:
        row = (nm, first, b.n[s_])
        rows = b._send_rows(nm, first, b.n[s_])
        chk(row in b.topdown_rows and bool(g.mod_send[rows].all()), f"  {s_} → {nm}：眠りで落とす行・調整する側（撃たせない）")
        first += b.n[s_]
chk(all(b.n_ext[k] == 0 for k in ("STS1", "視床枕")), "STS1・視床枕は ext なし（下り・横の相手がいない／視床）")

print("4 視床 ⇄ 皮質・皮質 → 線条体", flush=True)
k = b.k_cx
for r_, bands in TB.STR_BAND.items():
    for b_, tot in bands.items():
        ks = [k[f"{a_}→{r_}"] for a_ in TB.BAND_AREAS[b_]]
        chk(sum(ks) == tot, f"{r_} ← {b_} {sum(ks)}本（2,700 × 帯の割合 ＝ {tot}）")
    s_all = sum(v for kk_, v in k.items() if kk_.endswith("→" + r_))
    print(f"    {r_}：皮質から受け手1体 {s_all:,}本（{'・'.join(f'{kk_.split(chr(8594))[0]} {v}' for kk_, v in k.items() if kk_.endswith(chr(8594) + r_))}）", flush=True)
nn = b.n
for r_ in ("被殻", "尾状核", "側坐核"):
    for kk_, v in k.items():
        if kk_.endswith("→" + r_):
            s = b.bu[kk_]
            per = torch.bincount(s.tgt, minlength=s.n_tgt)
            if not (bool((per == v).all()) and int(s.src.max()) < nn[kk_.split("→")[0]]):
                chk(False, f"{kk_} 受け手1体 {v}本")
chk(True, "皮質 → 線条体の線は全部 受け手1体 k_cx 本")
chk(k["運動→被殻"] == 335 and k["前頭前野→尾状核"] == 1436 and k["眼窩前頭→側坐核"] == 1731,   # 帯に1つの送り手の線は体数によらない
    f"今ある線も人の比：運動 → 被殻 1,962 → {k['運動→被殻']}・前頭前野 → 尾状核 1,953 → {k['前頭前野→尾状核']}・眼窩前頭 → 側坐核 8 → {k['眼窩前頭→側坐核']}")
chk("K_CX[" not in inspect.getsource(Brain.__init__), "旧 K_CX（マカクの比）を組むところで使っていない")

print("5 来ない体（44・45）", flush=True)
om_new = [k_ for k_, _t in TB.P103_CX if k_.split("→")[0] in TB.OMIT]
chk(set(om_new) <= set(TB.OM_UP_LINES), f"送り手が来ない体を持つ線 {len(om_new)}本を 誤差 ｜ 来ない に")
for k_ in om_new:
    s = b.bu[k_]; ns = b.n[k_.split("→")[0]]
    fr = float((s.src >= ns).float().mean())
    chk(0.3 < fr < 0.7 and int(s.src.max()) < 2 * ns, f"{k_} 来ない体 {fr:.0%}")

print("6 回る・保存と読み込み", flush=True)
b.bs.set_state(BS.WAKE)
import life as L
lf = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
lf.run_plans(1, learn=True)
fired = {k_: int(b.pred[k_].sum()) for k_ in TB.SYN}
print(f"  （見るだけ）{lf.t}歩・声 {lf.stats['voices']}・最後の歩の表現 {fired}・視床枕 {int(b.mth['視床枕'].sum())}", flush=True)
chk(lf.t > 0, f"{lf.t}歩 回った")
chk(all(len(b.pred[k_]) == b.n[k_] for k_ in TB.SYN) and len(b.mth["視床枕"]) == b.n["視床枕"], "新しい領野の表現の発火（1歩前）")
d = tempfile.mkdtemp(); pth = os.path.join(d, "s.pt")
B.save(b, 1, path=pth)
b.bu_unpack()
keys = ("45→44", "高次→STS1", "VA→44", "眼窩前頭→被殻", "運動→被殻")
w = {k_: b.bu[k_].w.clone() for k_ in keys}
gw = g.w.clone()
b2 = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=pth)
b2.bu_unpack()
chk(all(torch.equal(b2.bu[k_].w, v) for k_, v in w.items()) and torch.equal(b2.g.w, gw), "線・網の重さが保存・読み込みで戻る")
chk(B.VERSION == 103 and B.SAME_NET == (), "VERSION 103・SAME_NET なし（state.pt は 0 から）")
nbu = sum(len(s.src) for s in b.bu.values())
print(f"  辺（コンテナ・相手 200）：網 {len(g.w):,}・下からの線 {nbu:,}", flush=True)
print("✅ 版103 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
