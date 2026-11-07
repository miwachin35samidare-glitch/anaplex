"""版107e：学ぶ前（0番）の率を人の値に合わせるノブを探して knobs.json に書く（arkhe の ~/anaplex で。網のファイルは変えない）
  版107m：盛り xcx は「的」（下からの入力 × 盛りだけで閾値を越える率＝網の中の辺が学んで寄せる先）を 3.3Hz に合わせる。
    ⟲ 前は表現の率に合わせたので、学びを止めた熱い網に合わせて盛りが 1.7 → 0.63 に下がり、的が 0.3〜0.5Hz になって皮質がそこへ下がった（quiet107l_tgt、11-07 17:01）
  版107l：VTA は自分で撃つ分の倍率（nuc_vinf["VTA"]）で合わせる（107k で抑制を 0.25倍まで下げても 1.2〜1.7Hz ⇒ 抑えられているのでなく駆動が無い）。
    書く値は後ろ半分の回のノブの幾何平均（黒質の率の揺れが大きく、一番ずれの小さい回は運の当たり：107k で 12.6倍を選び、回すと門 0.58）
  版107k：VTA へ入る抑制の線の重さ（vta_inh、的は黒質と同じ 4.56Hz＝門の分母 p0）を足した。組んだ直後の熱い所を捨てる（先に WARM 歩）・1回の歩 40 → 120（率の揺れが大きい：measure_sn107 で黒質は1番ごとに 1〜9Hz）
  【み】11-05 07:30「この回しの結果を読んだら、私に聞かずに調整できる範囲の数値は調整しておいて」
  python3 fit107.py [回=14] [1回の歩=120] [先に捨てる歩=200]        （コンテナ：python3 fit107.py 2 3 3 --test）
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
NS = int(args[1]) if len(args) > 1 else 120
WARM = int(args[2]) if len(args) > 2 else 200
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
kn = {"nuc_vinf": {**{k: 1.0 for k in T_NUC}, "VTA": 1.0}, "sn_inh": 1.0, "vta_inh": 1.0, "xcx": float(os.environ.get("ANAPLEX_XCX", 1.7))}
os.environ.pop("ANAPLEX_XCX", None)


def rate(nm):
    i = NAMES.index(nm)
    o, n = int(b.off_ch[i]), b.n[nm]
    return float(any_r[o:o + n].mean()) / NS / DT


_rx = torch.cat([torch.arange(int(g.off_recv[NAMES.index(k)]), int(g.off_recv[NAMES.index(k) + 1]), device=dev) for k in TB.X_CX_AREAS])
RCX = _rx[~g.is_inh_r[_rx]]                                   # 版107m：盛りを振る皮質の興奮の行
best = (float("inf"), None, None)
hist = []
kn["nuc_vinf"]["黒質"] = kn["nuc_vinf"]["VTA"]
b._knobs107_apply(kn)
for _ in range(WARM):                                        # 版107k：組んだ直後の熱い所を捨てる
    life.step(learn=False)
for it in range(NIT):
    kn["nuc_vinf"]["黒質"] = kn["nuc_vinf"]["VTA"]             # 版107n：黒質の i0 は VTA と同じ倍率（同じ作り）
    b._knobs107_apply(kn)
    any_r = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    tg = torch.zeros((), device=dev, dtype=torch.float64)
    for _ in range(NS):
        life.step(learn=False)
        any_r += g._any[0][:g.n_ch_all].double()
        tg += g._tgt[RCX].double().mean()
    r_tg = float(tg) / NS / DT                               # 版107m
    r_nuc = {k: rate(k) for k in T_NUC}
    r_sn = rate("黒質")
    r_vta = rate("VTA")
    r_cx = float(np.mean([rate(k) for k in TB.X_CX_AREAS]))
    errs = [math.log(max(r_nuc[k], 0.1) / T_NUC[k]) for k in T_NUC] + [math.log(max(r_sn, 0.1) / T_SN), math.log(max(r_vta, 0.1) / T_SN), math.log(max(r_tg, 0.1) / T_CX)]
    err = sum(e * e for e in errs)
    ok = all(abs(e) < math.log(1.15) for e in errs)
    print(f"\n回 {it}：ずれ {err:.3f}{' ✓ 全部 ±15%' if ok else ''} ／ 黒質 {r_sn:.1f}Hz（抑制 ×{kn['sn_inh']:.3f}）・VTA {r_vta:.1f}Hz（i0 ×{kn['nuc_vinf']['VTA']:.3f}）・皮質 {r_cx:.1f}Hz・的 {r_tg:.2f}Hz（盛り {kn['xcx']:.3f}）", flush=True)
    print("   " + "・".join(f"{k} {r_nuc[k]:.1f}（i0 ×{kn['nuc_vinf'][k]:.3f}）" for k in T_NUC), flush=True)
    if err < best[0]:
        best = (err, json.loads(json.dumps(kn)), it)
    hist.append(json.loads(json.dumps(kn)))
    if ok:
        break
    cl = lambda x, lo=0.5, hi=2.0: min(hi, max(lo, x))
    for k in T_NUC:
        kn["nuc_vinf"][k] *= cl((T_NUC[k] / max(r_nuc[k], 0.5)) ** 0.5)
    kn["sn_inh"] = cl(kn["sn_inh"] * cl((max(r_sn, 0.1) / T_SN) ** 0.5), 0.25, 16.0)
    kn["nuc_vinf"]["VTA"] = cl(kn["nuc_vinf"]["VTA"] * cl((T_SN / max(r_vta, 0.1)) ** 0.5), 0.1, 64.0)   # 版107l
    kn["xcx"] *= cl((T_CX / max(r_tg, 0.1)) ** 0.5, 0.7, 1.4)   # 版107m：的に合わせる

err, kb, it = best
if not ok and len(hist) >= 4:                                 # 版107l：揺れで当たった回でなく、後ろ半分の平均（幾何）
    hh = hist[len(hist) // 2:]
    gm = lambda xs: float(math.exp(sum(math.log(max(x, 1e-6)) for x in xs) / len(xs)))
    kb = {"nuc_vinf": {k: gm([h["nuc_vinf"][k] for h in hh]) for k in hh[0]["nuc_vinf"]},
          "sn_inh": gm([h["sn_inh"] for h in hh]), "vta_inh": gm([h["vta_inh"] for h in hh]), "xcx": gm([h["xcx"] for h in hh])}
    it = f"{len(hist) - len(hh)}〜{len(hist) - 1} の平均"
kb["nuc_vinf"]["黒質"] = kb["nuc_vinf"]["VTA"]                 # 版107n
kb["note"] = f"fit107：学ぶ前（0番）・先に {WARM}歩捨てて・学びを止めて {NS}歩ずつ・{it}回目（一番ずれの小さい回 {best[2]}：{err:.3f}）"
with open("knobs.json", "w") as f:
    json.dump(kb, f, ensure_ascii=False, indent=1)
print(f"\n⇒ knobs.json に書いた（{it}回目・ずれ {err:.3f}）：黒質の抑制 ×{kb['sn_inh']:.3f}・VTA の抑制 ×{kb['vta_inh']:.3f}・盛り {kb['xcx']:.3f}・"
      + "・".join(f"{k} ×{v:.3f}" for k, v in kb["nuc_vinf"].items()), flush=True)
