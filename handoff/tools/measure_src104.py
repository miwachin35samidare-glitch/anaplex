"""版104：核・線条体を押している入力を送り手ごとに分ける（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_src104.py [先に回す番=0] [測る歩=40]     （グラフを撮らずに回す。値は同じ）
  【み】11-03 21:48「93.7って異様に高いよね？直せそうなの？」⇒ 変える前に測る
    見込み【計算】：黒質の下からの押しの約89% は 上丘 → 黒質 64本 × X_NEW 7.14（X_NEW は 8本の道のための盛り、30 §5）
    measure_drive103（版103f・0から）：黒質は学ぶ前から下から 3.1 × 閾値・視床下核 10.1 × 閾値
  受け手の領野ごとに（その歩の平均、撃ったかどうかに依らない）：
    率      表現の体が撃った率（1歩に1回まで数える）
    下から（道ごと） 届いた重さ × 放出（受け手1体あたり）× x_gain ÷ 閾値 ／ 線の送り手が撃った割合（Hz）／ 1体の線の数
    抑制の線（道ごと）届いた重さ × 放出（受け手1体あたり）。⚠️ これは割り算の側（÷(1＋抑制)）に入る
    核どうし  核どうしの興奮の線（nuc_x）× x_gain ÷ 閾値
    i0・CL   自分で撃つ分・CL（cl_drive）÷ 閾値
    1＋抑制  割り算の側の平均（核どうしの抑制の線・外からの抑制の線・領野の中の抑制の体）
  ⚠️ 下から・核どうし・i0 は ÷(1＋抑制) の前の値（どれが大きいかを見るため）。撃つかは (足したもの) ÷ (1＋抑制) ＋ 持ち越し ＞ 閾値
  ⚠️ 持ち越し ＝ 撃たなかった体の前の周の残り（核は τm 40ms で足し合わせる ⇒ 同じ入力が続くと 1周の入力の 約38倍まで積もる）
"""
import os, sys, collections, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 0
NS = int(args[1]) if len(args) > 1 else 40
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
TGT = TB.NUC + TB.STR + ("側坐核",)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
if WARM > 0:
    life.run_plans(WARM, learn=True)          # measure_drive103g と同じ回し方（学びながら・眠りなし）
g = b.g
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
on = [False]
acc_del = collections.defaultdict(float)      # 道 → 受け手1体あたりの届いた重さ × 放出（歩の和）
acc_fire = collections.defaultdict(float)     # 道 → 線の送り手が撃った割合（歩の和）
inh_keys = set(getattr(b, "bu_inh", {}).keys())
D0 = TB._BUTable.drive


def drive_w(self, gi, vs):
    out = D0(self, gi, vs)
    if on[0]:
        i0, i1 = self._span(gi)
        x = torch.cat(vs)
        fired = (x[self._gsrc[gi]] > 0).float()
        a0 = self.eo[i0]
        for j, i in enumerate(range(i0, i1)):
            k = self.keys[i]
            acc_del[k] += float(out[j].mean())
            acc_fire[k] += float(fired[self.eo[i] - a0:self.eo[i + 1] - a0].mean())
    return out


TB._BUTable.drive = drive_w
n_all = g.n_ch_all
cap = {"x_gain": None, "th": None}
acc = {k: torch.zeros(n_all, device=dev, dtype=torch.float64) for k in ("核どうし", "i0・CL", "1＋抑制", "核どうしの抑制", "外からの抑制", "持ち越し")}
npass = [0]
R0 = G._run


def run(fn, *a):
    out = R0(fn, *a)
    if fn is G._drive_core and on[0]:
        (both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry, fire_th, th_gain,
         complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_recv) = a
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        th = (fire_th * th_gain).double()
        xg = torch.as_tensor(x_gain, device=dev).double().expand(n_all)
        cap["x_gain"], cap["th"] = xg, th.expand(n_all)
        if nuc_x is not None:
            acc["核どうし"] += (nuc_x * x_gain).double() / th
        if cl_drive is not None:
            acc["i0・CL"] += cl_drive.double() / th
        acc["1＋抑制"] += (1.0 + ihn[rep_i]).double()
        if v_carry is not None:
            acc["持ち越し"] += v_carry.double() / th
        if nuc_ihn is not None:
            acc["核どうしの抑制"] += nuc_ihn[rep_i].double()
        if ext_ihn is not None:
            acc["外からの抑制"] += ext_ihn[rep_i].double()
        npass[0] += 1
    return out


G._run = run
rep_n = torch.zeros(n_all, device=dev)
on[0] = True
t0 = life.t
for _ in range(NS):
    life.step()
    rep_n.add_(g._any[0][:n_all].float())
on[0] = False
ns = max(1, life.t - t0)
G._run = R0
TB._BUTable.drive = D0
DT = 10.67 / 1000.0
print(f"版 {B.VERSION}（{'版104' if hasattr(b, 'occ') else '版103'}）／ 先に {WARM}番（{t0}歩）・測る {ns}歩 ／ 値は ÷(1＋抑制) の前・閾値 ＝ 1", flush=True)
keys_by_tgt = collections.defaultdict(list)
for tab in [t for t in (getattr(b, "but", None), getattr(b, "but_inh", None)) if t is not None]:
    for i, k in enumerate(tab.keys):
        keys_by_tgt[ch0_name.get(int(b.bu[k].ch0) if k in b.bu else int(b.bu_inh[k][0].ch0), "?")].append((k, tab, i))
for nm in TGT:
    i = NAMES.index(nm)
    oc, n = int(b.off_ch[i]), b.n[nm]
    sl = slice(oc, oc + n)
    hz = float(rep_n[sl].mean()) / ns / DT
    xg = float(cap["x_gain"][oc]); th = float(cap["th"][oc])
    pp = max(1, npass[0])
    nucx = float(acc["核どうし"][sl].mean()) / pp
    i0 = float(acc["i0・CL"][sl].mean()) / pp
    ih = float(acc["1＋抑制"][sl].mean()) / pp
    ihn_n = float(acc["核どうしの抑制"][sl].mean()) / pp
    ihn_e = float(acc["外からの抑制"][sl].mean()) / pp
    car = float(acc["持ち越し"][sl].mean()) / pp
    rows, rows_i = [], []
    for k, tab, j in keys_by_tgt.get(nm, []):
        nl = (tab.eo[j + 1] - tab.eo[j]) / max(1, tab.nt[j])
        dv = acc_del[k] / ns
        fr = acc_fire[k] / ns / DT
        if k in inh_keys:
            rows_i.append((dv, k, nl, fr))
        else:
            rows.append((dv * xg / th, k, nl, fr))
    tot = sum(r[0] for r in rows) + nucx + i0
    print(f"\n■ {nm}（{n}体）率 {hz:.1f}Hz ／ x_gain {xg:.2f}・閾値 {th:.1f} ／ 1＋抑制 {ih:.2f}（核どうし {ihn_n:.2f}・外から {ihn_e:.2f}・領野の中 {ih - 1 - ihn_n - ihn_e:.2f}）／ 足したもの {tot:.2f}（÷(1＋抑制) で {tot / ih:.2f}）＋ 持ち越し {car:.2f}", flush=True)
    print(f"   {'入力':<22s} {'÷閾値':>8s} {'割合':>6s} {'線/体':>7s} {'送り手Hz':>8s}", flush=True)
    for v, k, nl, fr in sorted(rows, key=lambda r: -r[0]):
        print(f"   {k:<22s} {v:8.3f} {v / max(tot, 1e-12):6.1%} {nl:7.1f} {fr:8.1f}", flush=True)
    print(f"   {'核どうし（nuc_x）':<22s} {nucx:8.3f} {nucx / max(tot, 1e-12):6.1%}", flush=True)
    print(f"   {'i0・CL':<22s} {i0:8.3f} {i0 / max(tot, 1e-12):6.1%}", flush=True)
    for v, k, nl, fr in sorted(rows_i, key=lambda r: -r[0]):
        print(f"   抑制 {k:<19s} 届いた {v:8.3f}（割り算の側） {nl:7.1f} {fr:8.1f}", flush=True)
