"""版107l：学びながらの網で、黒質・VTA の率を 4.56Hz（門の分母）に合わせ直す（arkhe の ~/anaplex で。網のファイルは変えない）
  python3 fit107b.py [回=12] [1回の歩=200] [先に学びながら回す歩=1000]      （コンテナ：python3 fit107b.py 4 20 20 --test）
  【み】11-05 07:30「私に聞かずに調整できる範囲の数値は調整しておいて」
  measure_quiet107b --win=200（107l、11-07 00:18）：学びを止めて合わせた VTA 4.56Hz が、学びながらでは 2.9〜3.2Hz ⇒ VTA の門 0.58〜0.96（1 の少し下）
    ⇒ 皮質・海馬・扁桃体・側坐核は弱まる側に少しずつ残り、皮質が下がり続ける（50「合わせるのは落ち着いた所で」）
  動かすノブは 黒質の抑制 sn_inh と VTA の自分で撃つ分 nuc_vinf["VTA"] の 2つだけ（ほかは今の knobs.json のまま）
  版107n：黒質の自分で撃つ分の倍率は VTA と同じにする（同じ作り。撃ち方 B＝自分のリズム＋入力【み】10-15、リズムの値は霊長類の生体が無ければノブ）。
    ⟲ 107m までは黒質の i0 は ×1（入力なしの行き着く先 0.35 で閾値に届かず、入力だけで撃つ）⇒ 皮質 → 視床下核 → 黒質 の輪にそのまま引きずられた
  書く値は後ろ半分の回の幾何平均。今の knobs.json は knobs_fit107b_前.json に残す
"""
import os, sys, json, math, shutil, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NIT = int(args[0]) if args else 12
NS = int(args[1]) if len(args) > 1 else 200
WARM = int(args[2]) if len(args) > 2 else 1000
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
T_DA = 4.56
DT = 10.67 / 1000.0
kn = json.loads(json.dumps(TB.KNOBS)) if TB.KNOBS else {"nuc_vinf": {}, "sn_inh": 1.0, "vta_inh": 1.0}
kn.setdefault("nuc_vinf", {}).setdefault("VTA", 1.0)
kn.setdefault("sn_inh", 1.0)
kn.pop("note", None)

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
for _ in range(WARM):
    life.step(learn=True)


def rate(acc, nm):
    i = NAMES.index(nm)
    o, n = int(b.off_ch[i]), b.n[nm]
    return float(acc[o:o + n].mean()) / NS / DT


hist = []
cl = lambda x, lo, hi: min(hi, max(lo, x))
for it in range(NIT):
    kn["nuc_vinf"]["黒質"] = kn["nuc_vinf"]["VTA"]                 # 版107n
    b._knobs107_apply(kn)
    acc = torch.zeros(g.n_ch_all, device=dev, dtype=torch.float64)
    for _ in range(NS):
        life.step(learn=True)
        acc += g._any[0][:g.n_ch_all].double()
    r_sn, r_vta = rate(acc, "黒質"), rate(acc, "VTA")
    r_cx = float(np.mean([rate(acc, k) for k in TB.X_CX_AREAS]))
    print(f"回 {it}（{life.t}歩）：黒質 {r_sn:.2f}Hz（抑制 ×{kn['sn_inh']:.3f}）・VTA {r_vta:.2f}Hz（i0 ×{kn['nuc_vinf']['VTA']:.3f}）・"
          f"門 {float(b.gate[0]):.2f}・{float(b.gate[1]):.2f}・皮質 {r_cx:.2f}Hz", flush=True)
    hist.append((kn["sn_inh"], kn["nuc_vinf"]["VTA"]))
    kn["sn_inh"] = cl(kn["sn_inh"] * cl((max(r_sn, 0.1) / T_DA) ** 0.5, 0.5, 2.0), 0.25, 32.0)
    kn["nuc_vinf"]["VTA"] = cl(kn["nuc_vinf"]["VTA"] * cl((T_DA / max(r_vta, 0.1)) ** 0.5, 0.5, 2.0), 0.1, 64.0)

hh = hist[len(hist) // 2:]
gm = lambda xs: float(math.exp(sum(math.log(x) for x in xs) / len(xs)))
kn["sn_inh"], kn["nuc_vinf"]["VTA"] = gm([h[0] for h in hh]), gm([h[1] for h in hh])
kn["nuc_vinf"]["黒質"] = kn["nuc_vinf"]["VTA"]                     # 版107n
kn["note"] = f"fit107b：学びながら（先に {WARM}歩）・{NS}歩ずつ・回 {len(hist) - len(hh)}〜{len(hist) - 1} の平均（黒質の抑制・VTA の i0 だけ）"
if os.path.exists("knobs.json"):
    shutil.copy("knobs.json", "knobs_fit107b_前.json")
with open("knobs.json", "w") as f:
    json.dump(kn, f, ensure_ascii=False, indent=1)
print(f"\n⇒ knobs.json に書いた：黒質の抑制 ×{kn['sn_inh']:.3f}・VTA と黒質の i0 ×{kn['nuc_vinf']['VTA']:.3f}（前の knobs.json は knobs_fit107b_前.json）", flush=True)
