"""版107：辺の重さの跳ねと、恒常性の下げ（scale）が追いかけ合っているかを見る（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_wscale107.py [番=100] [何番ごと=10]
  見立て（まだ確かめていない）：網の中の学びは 誤差 ＝ 的 − 見込み p。p ＝ sigmoid(… + 興奮 × scale × scale_up)。
    撃ちすぎ ⇒ scale が下がる ⇒ p が下がる ⇒ 誤差が正のまま ⇒ w が上がる ⇒ また scale が下がる … の輪なら、
    w の和は跳ねても「w × scale × scale_up」（効いている重さ）はほぼ平らのはず。
  wgrow107y_cu60（11-09）：CA1 ← CA1 は 0番 560 → 10番 1,450万 → 60番 116万 → 100番 3,749万。最大 4,773。
  何番ごとに、領野ごと（表現・誤差の行＝抑制の体を除く受け手）：
    受け手1つあたりの w 和（平均）・scale の中央値・scale_up の中央値・効き（受け手1つあたり w 和 × scale × scale_up の平均）・rate_ema ÷ 目標 の中央値・的が 1 の割合
  ⚠️ 表現・誤差の行への興奮の辺は 0 から始まる（開通は 0）。wgrow の「0番比」の 0番 の値は、抑制の体への辺の分だった【測】test 11-09
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NP = int(args[0]) if args else 100
EV = int(args[1]) if len(args) > 1 else 10
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
WANT = ["CA1", "眼窩前頭", "嗅内深", "前補足", "視床", "聴覚"]
AREAS = [(nm, NAMES.index(nm)) for nm in WANT if nm in NAMES]
rows = {}
for nm, i in AREAS:
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    r = torch.arange(lo, hi, device=dev)
    rows[nm] = r[~g.is_inh_r[r]]


def snap():
    g.flush_all()
    ws = torch.zeros(g.n_recv_all, device=dev, dtype=torch.float64)
    E, st = len(g.w), 1 << 24
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        ex = ~g.e_inh[sl].bool()
        ws.index_add_(0, g.e_recv[sl].long()[ex], g.w[sl].double()[ex])
    out = {}
    sc, su = g.scale.double(), g.scale_up.double()
    q = (g.rate_ema / g.target_recv.clamp(min=1e-9)).double()
    tg = getattr(g, "_tgt", None)
    for nm, r in rows.items():
        out[nm] = (float(ws[r].mean()), float(sc[r].median()), float(su[r].median()),
                   float((ws[r] * sc[r] * su[r]).mean()), float(q[r].median()),
                   (float((tg[r] > 0.5).double().mean()) if tg is not None and len(tg) == g.n_recv_all else float("nan")))
    return out


done = 0
print(f"版 {B.VERSION} ／ {NP}番まで {EV}番ごと ／ 領野（表現・誤差の受け手）：受け手1つあたり w 和・scale・scale_up・効き（w 和×scale×scale_up）・率÷目標・的1の割合", flush=True)
while True:
    S = snap()
    print(f"{done:3d}番 {life.t:6d}歩", flush=True)
    for nm, _ in AREAS:
        w_, sc, su, ef, q, tg = S[nm]
        print(f"      {nm}：w {w_:.4g}・scale {sc:.3g}・scale_up {su:.3g}・効き {ef:.4g}・率÷目標 {q:.2f}・的1 {tg:.2f}", flush=True)
    if done >= NP:
        break
    life.run_plans(EV, learn=True)
    done += EV
