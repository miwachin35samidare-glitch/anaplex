"""版95 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない【み】10-18 22:08
  python3 check_v95.py [歩=60]                    （周ごと：_pass の形）
  ANAPLEX_GRAPH=eager python3 check_v95.py [歩]   （グラフの形の式をそのまま：_fast_body・_gp_body）
確かめること
  1 当たる（組める）・領野の体数・素子の型・時間細胞なし・線条体の抑制の体 22.5%・受け手の相手 3,800（線条体）
  2 NucNet の線：線ごとに 受け手1体の本数・遅れ・符号・送り手が「線を出す割合」の中か（組み表 §2）
  3 下からの道（運動 → 被殻 2,700 ほか）・核 → 領野の抑制の4本・PPN のくじが無い
  4 回る（作り物の声で喃語）・核どうしの線が周ごとに届いている（受け手の nuc_x・nuc_ihn が 0 でない周がある）
  5 保存・読み込みで NucNet の重さが一致
"""
import sys, os, tempfile, numpy as np, torch
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

B.DEV = "cpu"
NSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
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
want = {"尾状核": 986, "被殻": 1302, "外節": 256, "内節": 256, "網様部": 256, "腹側淡蒼球": 256, "PPN": 512, "CM": 64, "Pf": 64, "側坐核": 146}
for nm, n in want.items():
    chk(b.n[nm] == n, f"{nm} {b.n[nm]}体（{n}）")
for nm in TB.STR_ALL:
    a = b.areas[nm]
    chk(abs(a.n_inh / (a.n_ch + a.n_inh) - 0.225) < 0.01, f"{nm} 抑制の体 {a.n_inh}（{a.n_inh / (a.n_ch + a.n_inh):.1%}）")
    chk(a.ch.n_cells == 0, f"{nm} 時間細胞なし")
    chk(a.no_lat_exc, f"{nm} 中の興奮の横なし")
chk(TB.PRIOR_TYPE["境"] == "内節" and TB.PRIOR_TYPE["側坐核"] == "線条体" and TB.PRIOR_TYPE["尾状核"] == "線条体", "素子の型（境 ＝ 内節・側坐核 ＝ 線条体）")
for nm in ("外節", "内節", "網様部", "腹側淡蒼球", "PPN", "CM", "Pf"):
    chk(b.areas[nm].ch.n_cells == 0, f"{nm} 時間細胞なし")

print("2 NucNet の線", flush=True)
nn_ = g.nucnet
n_ch = g.n_ch_all
off = {nm: int(b.off_ch[NAMES.index(nm)]) for nm in NAMES}
area_of = torch.zeros(n_ch, dtype=torch.long)
for i, nm in enumerate(NAMES):
    area_of[off[nm]:off[nm] + b.n[nm]] = i
src_area = area_of[nn_.src % n_ch]; dst_area = area_of[nn_.dst]
seen = 0
for srcs, tgt, sg, k, frac, dl, kind in TB.NUC_LINES:
    si = torch.tensor([NAMES.index(s) for s in srcs])
    m = torch.isin(src_area, si) & (dst_area == NAMES.index(tgt)) & (nn_.sign == sg)
    seen += int(m.sum())
    per = torch.bincount(nn_.dst[m] - off[tgt], minlength=b.n[tgt])
    pool_n = sum(max(1, int(round(b.n[s] * frac))) for s in srcs) * (2 if kind == "表誤" else 1)
    k_exp = min(k, pool_n)
    self_ = srcs == (tgt,)
    good_k = bool(((per == k_exp) | (self_ & (per >= k_exp - 1))).all())
    n_src_cells = len(torch.unique(nn_.src[m] % n_ch))
    good_frac = n_src_cells <= sum(max(1, int(round(b.n[s] * frac))) for s in srcs)
    good_d = bool((nn_.delay[m] == dl).all())
    has_err = bool((nn_.src[m] >= n_ch).any())
    chk(good_k and good_frac and good_d and has_err == (kind == "表誤"),
        f"{'・'.join(srcs)} → {tgt}（{'+' if sg > 0 else '−'}）受け手1体 {int(per.min())}〜{int(per.max())}本（{k_exp}）・送り手 {n_src_cells}体・遅れ {dl}周・{kind}")
chk(seen == len(nn_.src), f"線の数 {len(nn_.src):,}（表の線の和 {seen:,}）")

print("3 下からの道・核 → 領野の抑制・PPN", flush=True)
for key, k in (("運動→被殻", 2700), ("前頭前野→尾状核", 2700), ("CM→被殻", 1100), ("Pf→尾状核", 1100), ("運動→CM", 8), ("前頭前野→Pf", 8)):
    s = b.bu[key]
    per = torch.bincount(s.tgt, minlength=s.n_tgt)
    chk(bool((per == k).all()), f"{key} 受け手1体 {int(per.min())}〜{int(per.max())}本（{k}）")
for key in ("視床下核→境", "境→外側手綱核", "外側手綱核→吻内側被蓋核", "視床下核→黒質", "PPN→黒質"):
    chk(key not in b.bu, f"{key} は下からの道に無い（NucNet へ）")
chk(set(b.bu_inh) == set(TB.INH_TO_AREA), f"核 → 領野の抑制 {sorted(b.bu_inh)}")
chk(not hasattr(b, "_ppn10"), "PPN のくじが無い（組んだ直後）")

print(f"4 回す（作り物の声、{NSTEP}歩ぶん）", flush=True)
b.bs.set_state(BS.WAKE)
_, say, _ = B.voices(b, True)
rng = np.random.default_rng(0)
seen_x = seen_i = 0
orig = nn_.after


def spy(S, rep, err):
    global seen_x, seen_i
    orig(S, rep, err)
    seen_x += int(S["nuc_x"].abs().sum() > 0)
    seen_i += int(S["nuc_ihn"].abs().sum() > 0)


nn_.after = spy
t0 = g.t
while g.t - t0 < NSTEP:
    say(rng, True)
print(f"  {g.t - t0}歩・NucNet が届けた周：興奮 {seen_x}・抑制 {seen_i}", flush=True)
chk(seen_x > 0 and seen_i > 0, "核どうしの線が周ごとに届いている")
chk(not hasattr(b, "_ppn10"), "PPN のくじが無い（回したあと）")
rates = b.mb.rates_hz()
print("  （率は読まない。見るだけ）" + "・".join(f"{k} {v:.0f}" for k, v in rates.items()), flush=True)
w_changed = float((nn_.w != nn_.w0).float().mean())
print(f"  NucNet の重さが動いた線の割合 {w_changed:.2%}（門が開いた歩だけ動く）", flush=True)

print("5 保存・読み込み", flush=True)
p = os.path.join(tempfile.mkdtemp(), "state.pt")
B.save(b, 1, path=p)
w_before = nn_.w.clone()
b2 = Brain(prior, n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
B.load(b2, path=p)
chk(torch.equal(b2.g.nucnet.w, w_before), "NucNet の重さが一致")
print("✅ 版95 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
