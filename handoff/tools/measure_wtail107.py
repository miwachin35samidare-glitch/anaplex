"""版107：下からの線の重さの尾（人の測った幅を越える線）を道ごとに数える（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_wtail107.py [先に回す番=100]
  strscale107x（11-09 07:20）：CM → 被殻 の最大 410・下丘 → 視床 140・前頭前野 → 尾状核 90（重さ 1 ＝ 1.12mV【ヒト】Hunt 2023）。
    人の皮質の錐体どうしの EPSP は 0.1〜6mV（【ヒト】Geiger 2024、台帳 92）⇒ 重さ 5.4 を越える線は人の測った幅の外
  道ごとに：本数・重さ 5.4 を越える本数と割合・20・100 を越える本数・越えた線の重さが道の重さの和に占める割合・その線の1本で閾値の何倍か
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
t_ = b.but
ch0n = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
print(f"版 {B.VERSION} ／ 先に {WARM}番（{life.t}歩）／ 重さ 1 ＝ 1.12mV ・ 人の測った幅の上 6mV ＝ 重さ 5.4", flush=True)
print(f"   {'道':<18s} {'受け手':<8s} {'本数':>10s} {'>5.4':>9s} {'割合':>7s} {'>20':>7s} {'>100':>6s} {'>5.4 の重さの割合':>10s} {'最大':>8s} {'最大÷閾値':>9s}", flush=True)
th = {}
for nm in NAMES:
    a = b.areas[nm]
    th[nm] = float(getattr(a, "fire_th", float("nan")) or float("nan"))
rows = []
for i, k in enumerate(t_.keys):
    w = t_.w[int(t_.eo[i]):int(t_.eo[i + 1])].float()
    if len(w) == 0:
        continue
    big = w > 5.4
    nb = int(big.sum())
    if nb == 0:
        continue
    nm = ch0n.get(int(t_.ch0[i]), "?")
    rows.append((nb, k, nm, len(w), nb / len(w), int((w > 20).sum()), int((w > 100).sum()), float(w[big].sum() / w.sum()), float(w.max())))
for nb, k, nm, n, fr, n20, n100, wf, mx in sorted(rows, key=lambda r: -r[0]):
    tq = th.get(nm, float("nan"))
    print(f"   {k:<18s} {nm:<8s} {n:10,d} {nb:9,d} {fr:7.2%} {n20:7,d} {n100:6,d} {wf:10.1%} {mx:8.1f} {mx / tq if tq == tq and tq > 0 else float('nan'):9.2f}", flush=True)
tot = sum(r[3] for r in rows)
print(f"越えた線のある道 {len(rows)}・その本数の和 {tot:,}・越えた線 {sum(r[0] for r in rows):,}", flush=True)
