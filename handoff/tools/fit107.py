"""版107e：学ぶ前（0番）の率を人の値に合わせるノブを探して knobs.json に書く（arkhe の ~/anaplex で。網のファイルは変えない）
  【み】11-05 07:30「この回しの結果を読んだら、私に聞かずに調整できる範囲の数値は調整しておいて」
  python3 fit107.py [回=14] [1回の歩=40]        （コンテナ：python3 fit107.py 2 3 --test）
  合わせるノブ（30 §5 のノブ・10 の決めの中だけ）と的
    核の自分で撃つ分 i0（NUC_VINF は teacher_v95 の写し「合わせ直す」）：
      外節 65.7・内節 73.5【マカク】Galvan & Wichmann 2023 ／ 網様部 60.5【マカク】Elias 2008 ／ 視床下核 19.3【ヒト】Steigerwald 2008 ／
      境 33【マカク】Hong & Hikosaka 2008 ／ 外側手綱核 26【マカク】Matsumoto 2007 ／ 吻内側被蓋核 17.8【マカク】Hong 2011 ／ PPN 23.2【ヒト】
    黒質へ入る抑制の線の重さ（10「黒質の入力の割り振りは試し台で測って探す（率 4.56Hz）」。本数は変えない）：黒質 4.56【ヒト】Ramayya 2014
    皮質の受け手の盛り xcx（版105b のノブ）：盛りを振る皮質の表現の平均 3.3Hz【ヒト】（measure_rate105 と同じ的）
  やり方：学びを止めて 40歩ずつ回し、率から ノブ ×（的 ÷ 率）^0.5（1回で 0.5〜2倍まで）を全部いっしょに。的の ±15% に入るか回が尽きたら、
    的からのずれ（対数の二乗の和）が一番小さかった回のノブを knobs.json に書く。人の値の無い領野・線条体の率は合わせない
"""
import os, sys, json, math, numpy as np, torch
# 版107h：グラフで回す（⟲ eager は取り置きが増えて 12〜15GB【測】11-05）。eager にするなら ANAPLEX_GRAPH=eager
os.environ.setdefault("ANAPLEX_KNOBS", "none")             # 前の knobs.json に引きずられない（元の値から探す）
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NIT = int(args[0]) if args else 14
NS = int(args[1]) if len(args) > 1 else 40
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
T_NUC = {"外節": 65.7, "内節": 73.5, "網様部": 60.5, "視床下核": 19.3, "境": 33.0, "外側手綱核": 26.0, "吻内側被蓋核": 17.8, "PPN": 23.2}
T_SN, T_CX = 4.56, 3.3
DT = 10.67 / 1000.0

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
kn = {"nuc_vinf": {k: 1.0 for k in T_NUC}, "sn_inh": 1.0, "xcx": float(os.environ.get("ANAPLEX_XCX", 1.7))}
os.environ.pop("ANAPLEX_XCX", None)


def rate(nm):
    i = NAMES.index(nm)
    o, n = int(b.off_ch[i]), b.n[nm]
    return float(any_r[o:o + n].mean()) / NS / DT


best = (float("inf"), None, None)
for it in range(NIT):
    b._knobs107_apply(kn)
    any_r = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    for _ in range(NS):
        life.step(learn=False)
        any_r += g._any[0][:g.n_ch_all].double()
    r_nuc = {k: rate(k) for k in T_NUC}
    r_sn = rate("黒質")
    r_cx = float(np.mean([rate(k) for k in TB.X_CX_AREAS]))
    errs = [math.log(max(r_nuc[k], 0.1) / T_NUC[k]) for k in T_NUC] + [math.log(max(r_sn, 0.1) / T_SN), math.log(max(r_cx, 0.1) / T_CX)]
    err = sum(e * e for e in errs)
    ok = all(abs(e) < math.log(1.15) for e in errs)
    print(f"\n回 {it}：ずれ {err:.3f}{' ✓ 全部 ±15%' if ok else ''} ／ 黒質 {r_sn:.1f}Hz（抑制 ×{kn['sn_inh']:.3f}）・皮質 {r_cx:.1f}Hz（盛り {kn['xcx']:.3f}）", flush=True)
    print("   " + "・".join(f"{k} {r_nuc[k]:.1f}（i0 ×{kn['nuc_vinf'][k]:.3f}）" for k in T_NUC), flush=True)
    if err < best[0]:
        best = (err, json.loads(json.dumps(kn)), it)
    if ok:
        break
    cl = lambda x, lo=0.5, hi=2.0: min(hi, max(lo, x))
    for k in T_NUC:
        kn["nuc_vinf"][k] *= cl((T_NUC[k] / max(r_nuc[k], 0.5)) ** 0.5)
    kn["sn_inh"] = cl(kn["sn_inh"] * cl((max(r_sn, 0.1) / T_SN) ** 0.5), 0.25, 16.0)
    kn["xcx"] *= cl((T_CX / max(r_cx, 0.1)) ** 0.5, 0.7, 1.4)

err, kb, it = best
kb["note"] = f"fit107：学ぶ前（0番）・学びを止めて {NS}歩ずつ・{it}回目（ずれ {err:.3f}）"
with open("knobs.json", "w") as f:
    json.dump(kb, f, ensure_ascii=False, indent=1)
print(f"\n⇒ knobs.json に書いた（{it}回目・ずれ {err:.3f}）：黒質の抑制 ×{kb['sn_inh']:.3f}・盛り {kb['xcx']:.3f}・"
      + "・".join(f"{k} ×{v:.3f}" for k, v in kb["nuc_vinf"].items()), flush=True)
