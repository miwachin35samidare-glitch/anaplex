"""版107：線条体ほかを、撃った歩と撃たなかった歩に分けて、道ごとの押しを比べる（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_firepath107.py [先に回す番=100] [測る歩=200]   （グラフを撮らずに回す。値は同じ）
  measure_src107u（11-08）は歩の平均だけ：尾状核は 下から ÷(1＋抑制) で閾値の 0.42 しかないのに 23Hz ／ 版107w（CM・Pf × 64）でも 22Hz のまま
  ⇒ 撃った歩にどの道の押しが乗っているかを見る。道ごとに：撃った歩の押し・撃たなかった歩の押し（÷閾値・÷(1＋抑制) の前）とその比
  比が大きい道 ＝ 撃つ歩にそろって来る道（撃たせている道）。1＋抑制 も撃った歩・撃たなかった歩で出す
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
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
on = [False]
TGT = ("尾状核", "被殻", "側坐核", "ストリオソーム", "分界条床核", "視床網様核", "視床")
offs = {nm: (int(b.off_ch[NAMES.index(nm)]), b.n[nm]) for nm in TGT}
last = {}
D0 = TB._BUTable.drive


def drive_w(self, gi, vs):
    out = D0(self, gi, vs)
    if on[0]:
        i0, i1 = self._span(gi)
        for j, i in enumerate(range(i0, i1)):
            k = self.keys[i]
            nm = ch0_name.get(int(self.ch0[i]))
            if nm in offs:
                last[(nm, k)] = out[j].detach().double()
    return out


TB._BUTable.drive = drive_w
ihs = {}
R0 = G._run


def run(fn, *a):
    out = R0(fn, *a)
    if fn is G._drive_core and on[0]:
        (both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry, fire_th, th_gain,
         complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_recv) = a
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        th = (fire_th * th_gain).double().expand(g.n_ch_all)
        xg = torch.as_tensor(x_gain, device=dev).double().expand(g.n_ch_all)
        ih = (1.0 + ihn[rep_i]).double()
        ihs["th"], ihs["xg"] = th, xg
        ihs["ih"] = ih if "ih" not in ihs else torch.maximum(ihs["ih"], ih)
        ihs["nx"] = ((nuc_x * x_gain).double() if nuc_x is not None else torch.zeros_like(ih)) + ihs.get("nx", 0)
        ihs["np"] = ihs.get("np", 0) + 1
    return out


G._run = run
S = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])   # (nm,k) → 撃った歩の和・その数・撃たなかった歩の和・その数
on[0] = True
t0 = life.t
for _ in range(NS):
    last.clear(); ihs.clear()
    life.step()
    f_all = g._any[0][:g.n_ch_all] > 0
    th, xg = ihs["th"], ihs["xg"]
    for nm, (o, n) in offs.items():
        f = f_all[o:o + n].double()
        thn = th[o:o + n] / xg[o:o + n]
        items = [(k, v) for (nm2, k), v in last.items() if nm2 == nm]
        items.append(("核どうし（nuc_x、周の和）", ihs["nx"][o:o + n] / max(1, ihs["np"]) / xg[o:o + n]))
        items.append(("1＋抑制（周の最大）", ihs["ih"][o:o + n] * thn))
        for k, v in items:
            q = v / thn
            r = S[(nm, k)]
            r[0] += float((q * f).sum()); r[1] += float(f.sum())
            r[2] += float((q * (1 - f)).sum()); r[3] += float((1 - f).sum())
on[0] = False
G._run = R0
TB._BUTable.drive = D0
print(f"版 {B.VERSION} ／ 先に {WARM}番（{t0}歩）・測る {NS}歩 ／ 押しは 届いた重さ × 放出 × x_gain ÷ 閾値（÷(1＋抑制) の前）", flush=True)
for nm in TGT:
    rows = []
    for (nm2, k), r in S.items():
        if nm2 != nm:
            continue
        a = r[0] / r[1] if r[1] else float("nan")
        c = r[2] / r[3] if r[3] else float("nan")
        rows.append((k, a, c, r[1]))
    if not rows:
        continue
    nf = max(r[3] for r in rows)
    print(f"\n■ {nm}（{b.n[nm]}体）撃った 体×歩 {int(rows[0][3]):,} ／ 測った 体×歩 {b.n[nm] * NS:,}", flush=True)
    print(f"   {'道':<26s} {'撃った歩':>8s} {'撃たない歩':>9s} {'比':>6s}", flush=True)
    for k, a, c, _ in sorted(rows, key=lambda r: -r[1]):
        if a < 0.005 and c < 0.005 and not k.startswith("1＋"):
            continue
        ratio = a / c if c > 1e-9 else float("inf")
        print(f"   {k:<26s} {a:8.3f} {c:9.3f} {ratio:6.2f}", flush=True)
