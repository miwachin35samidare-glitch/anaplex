"""版107：線条体の率が入力で決まるか、線条体の中の抑制の輪で決まるかを切り分ける（arkhe の ~/anaplex で。保存しない）
  python3 measure_strscale107.py [先に回す番=100] [1区切りの歩=200]   （グラフを撮らずに回す）
  measure_firepath107（11-09 01:17）：尾状核は撃った歩に Pf・前頭前野の押しが 7〜8倍乗る／被殻は撃った歩に 1＋抑制 が 0.33倍に下がる
  ⇒ 先に回したあと、線条体（尾状核・被殻・側坐核・ストリオソーム）へ入る下からの線の重さを ×1 → ×0.5 → ×0.25 → ×1（戻す）と区切りごとに掛けて、
     区切りごとの率・1＋抑制 を出す。学びは止めない（10 §1）。⚠️ 切り分けで、直しではない（50 §4）
  あわせて、道ごとの重さ（平均・90%・最大）を 島前／島後 → 視床 ほかで出す（firepath107：視床が撃つ歩に 島 → 視床 の押しが 45 × 閾値）
"""
import os, sys, collections, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 100
NS = int(args[1]) if len(args) > 1 else 200
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
TGT = TB.NUC + TB.STR + ("側坐核", "ストリオソーム", "分界条床核", "視床網様核")   # 版107u：目標を置いたのに下からの線が弱まっても率が下がらない核
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
if WARM > 0:
    life.run_plans(WARM, learn=True)          # measure_drive103g と同じ回し方（学びながら・眠りなし）
g = b.g
TB_ = TB._BUTable
t_ = b.but
STRN = ("尾状核", "被殻", "側坐核", "ストリオソーム")
ch0n = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
idx = []
for i, k in enumerate(t_.keys):
    if ch0n.get(int(t_.ch0[i])) in STRN:
        idx.append(torch.arange(int(t_.eo[i]), int(t_.eo[i + 1]), device=dev))
idx = torch.cat(idx)
print(f"版 {B.VERSION} ／ 先に {WARM}番（{life.t}歩）・1区切り {NS}歩 ／ 線条体へ入る下からの線 {len(idx):,}本", flush=True)
print("道ごとの重さ（平均・90%・最大）", flush=True)
for i, k in enumerate(t_.keys):
    if k in ("島前→視床", "島後→視床", "下丘→視床", "Pf→尾状核", "CM→被殻", "前頭前野→尾状核", "眼窩前頭→側坐核", "島前→ストリオソーム"):
        w = t_.w[int(t_.eo[i]):int(t_.eo[i + 1])].float()
        print(f"   {k:<16s} 本数 {len(w):>9,} 平均 {float(w.mean()):8.3f} 90% {float(w.quantile(0.9)) if len(w) < 16000000 else float('nan'):8.3f} 最大 {float(w.max()):9.3f}", flush=True)
g = b.g
ihsum = {}
R0 = G._run
on = [False]


def run(fn, *a):
    out = R0(fn, *a)
    if fn is G._drive_core and on[0]:
        ihn, rep_i = out[1], a[6]
        ih = (1.0 + ihn[rep_i]).double()
        ihsum["s"] = ihsum.get("s", 0) + ih
        ihsum["n"] = ihsum.get("n", 0) + 1
    return out


G._run = run
for fac in (1.0, 0.5, 0.25, 1.0):
    with torch.no_grad():
        w0 = t_.w[idx].clone()
        if fac != 1.0:
            t_.w[idx] = w0 * fac
    rep_n = torch.zeros(g.n_ch_all, device=dev)
    ihsum.clear(); on[0] = True
    t1 = life.t
    for _ in range(NS):
        life.step()
        rep_n.add_(g._any[0][:g.n_ch_all].float())
    on[0] = False
    ns = max(1, life.t - t1)
    with torch.no_grad():
        if fac != 1.0:
            t_.w[idx] = t_.w[idx] / fac          # 区切りのあいだに学んだ分は残し、掛けた分だけ戻す
    ih = ihsum["s"] / max(1, ihsum["n"])
    cells = []
    for nm in STRN + ("黒質", "外節", "内節", "前頭前野", "Pf", "CM"):
        o, n = int(b.off_ch[NAMES.index(nm)]), b.n[nm]
        hz = float(rep_n[o:o + n].mean()) / ns / (10.67 / 1000.0)
        cells.append(f"{nm} {hz:5.1f}Hz（1＋抑制 {float(ih[o:o + n].mean()):5.2f}）")
    print(f"× {fac:4.2f}：" + "・".join(cells), flush=True)
G._run = R0
