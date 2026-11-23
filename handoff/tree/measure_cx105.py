"""版105：皮質の体を何が撃たせているか（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_cx105.py [先に回す番=0] [測る歩=40]     （グラフを撮らずに回す。値は同じ）
  【み】11-03 23:51「いいよ」。版105 で核の盛りを外した後も、視床下核・線条体を押す皮質の送り手が速い
    （前頭前野 9.2・運動 21・前帯状の来ない体 27Hz ／ 皮質の人 3.3Hz）⇒ 皮質の上流を分ける
  1 表現の体（撃つ ＝ (網 ＋ 下から × x_gain ＋ CL) ÷ (1＋抑制) ＋ 持ち越し ＞ 閾値）：撃ったときの内訳（閾値 ＝ 1）・下からは道ごと（撃つかに依らない平均）
  2 誤差の体（撃つ ＝ 的 − p ＞ 0.5）・来ない体（撃つ ＝ p − 的 ＞ 0.5）：率・p の平均・p の中身（素子の見込み prior と 入力 exc ÷(1＋抑制) の logit）・的が 1 の割合
     的 ＝ その帯の下からの入力だけで閾値を越えるか（tgt_by_fire）
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
CX = TB.CL_CORTEX                     # 新皮質 16
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
if WARM > 0:
    life.run_plans(WARM, learn=True)
g = b.g
n_all, n_recv = g.n_ch_all, g.n_recv_all
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
on = [False]
# ── 下からの道ごと ──
acc_del, acc_fire = collections.defaultdict(float), collections.defaultdict(float)
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
# ── 表現の体の撃ったときの内訳 ／ 誤差・来ない体の p ──
z = lambda n: torch.zeros(n, device=dev, dtype=torch.float64)
R = {k: z(n_all) for k in ("撃った", "網", "下から", "CL", "持ち越し", "1＋抑制")}
E = {k: z(n_recv) for k in ("p", "prior", "入力", "的", "越え", "p越え", "prior越え", "入力越え")}
cap = {}
npass = [0]
R0 = G._run


def run(fn, *a):
    out = R0(fn, *a)
    if not on[0]:
        return out
    if fn is G._drive_core:
        (both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry, fire_th, th_gain,
         complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_r) = a
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        ih = (1.0 + ihn[rep_i]).double()
        th = (fire_th * th_gain).double()
        xx = x_all if nuc_x is None else x_all + nuc_x
        f = rep > 0
        parts = {"網": (exc - exc_mod2)[rep_i].double() / ih, "下から": (xx * x_gain).double() / ih,
                 "CL": (cl_drive.double() / ih) if cl_drive is not None else torch.zeros_like(ih),
                 "持ち越し": v_carry.double() if v_carry is not None else torch.zeros_like(ih)}
        R["撃った"] += f.double()
        for k_, v in parts.items():
            R[k_] += torch.where(f, v / th.clamp(min=1e-9), torch.zeros_like(v))
        R["1＋抑制"] += ih
        # 誤差・来ない体：p ＝ σ(prior ＋ exc ÷ (1＋抑制))（_drive_core と同じ式）
        E["p"] += p.double()
        E["prior"] += prior_l.double()
        E["入力"] += (exc / (1.0 + ihn)).double()
        x_t = torch.where(g.tgt_by_fire, (x_all * x_gain > fire_th).float(), x_all)
        tg_r = x_t[g.x_index].double()
        E["的"] += tg_r
        cond = ((g.fire_sgn.double() * (tg_r - p.double())) > G.THETA).double()   # 誤差：的 − p ＞ θ ／ 来ない：p − 的 ＞ θ
        E["越え"] += cond
        E["p越え"] += cond * p.double()
        E["prior越え"] += cond * prior_l.double()
        E["入力越え"] += cond * (exc / (1.0 + ihn)).double()
        npass[0] += 1
    return out


G._run = run
any_r = z(n_all)
any_e = z(len(g.err_i))
on[0] = True
t0 = life.t
for _ in range(NS):
    life.step()
    any_r += g._any[0][:n_all].double()
    any_e += g._any[1].double()
on[0] = False
ns = max(1, life.t - t0)
G._run = R0
TB._BUTable.drive = D0
DT = 10.67 / 1000.0
pp = max(1, npass[0])
print(f"版 {B.VERSION} ／ 先に {WARM}番（{t0}歩）・測る {ns}歩 ／ 閾値 ＝ 1", flush=True)
print("1 表現の体：撃ったときの内訳（÷(1＋抑制) のあと）と、下からの道（撃つかに依らない平均・÷(1＋抑制) の前 × x_gain ÷ 閾値）", flush=True)
keys_by_tgt = collections.defaultdict(list)
for i, k in enumerate(b.but.keys):
    keys_by_tgt[ch0_name.get(int(b.bu[k].ch0), "?")].append((k, i))
err_area = g.area_of[g.err_i]
err_om = g.is_om[g.err_i]
for nm in CX:
    i = NAMES.index(nm)
    oc, n = int(b.off_ch[i]), b.n[nm]
    sl = slice(oc, oc + n)
    hz = float(any_r[sl].mean()) / ns / DT
    fr = float(R["撃った"][sl].sum())
    m = {k_: (float(R[k_][sl].sum()) / fr if fr > 0 else 0.0) for k_ in ("網", "下から", "CL", "持ち越し")}
    ih = float(R["1＋抑制"][sl].mean()) / pp
    xg = float(g.x_gain[oc]); th = float(g.fire_th[oc]) if torch.is_tensor(g.fire_th) else float(g.fire_th)
    print(f"\n■ {nm}（{n}体）表現 {hz:.1f}Hz ／ 撃ったとき 網 {m['網']:.2f}・下から {m['下から']:.2f}・CL {m['CL']:.2f}・持ち越し {m['持ち越し']:.2f}"
          f" ／ 1＋抑制 {ih:.2f}（x_gain {xg:.2f}）", flush=True)
    rows = []
    for k, j in keys_by_tgt.get(nm, []):
        tab = b.but
        nl = (tab.eo[j + 1] - tab.eo[j]) / max(1, tab.nt[j])
        rows.append((acc_del[k] / ns * xg / th, k, nl, acc_fire[k] / ns / DT))
    tot = sum(r[0] for r in rows)
    for v, k, nl, f_ in sorted(rows, key=lambda r: -r[0])[:8]:
        print(f"   下から {k:<20s} {v:8.3f} {v / max(tot, 1e-12):6.1%}  線/体 {nl:6.1f}  送り手 {f_:6.1f}Hz", flush=True)
    # 誤差・来ない体
    for om, lab in ((False, "誤差の体"), (True, "来ない体")):
        sel = (err_area == i) & (err_om == om)
        if not bool(sel.any()):
            continue
        rows_e = g.err_i[sel]
        hz_e = float(any_e[sel].mean()) / ns / DT
        pm = float(E["p"][rows_e].mean()) / pp
        pr = float(E["prior"][rows_e].mean()) / pp if torch.is_tensor(E["prior"]) else float("nan")
        inp = float(E["入力"][rows_e].mean()) / pp
        tg = float(E["的"][rows_e].mean()) / pp
        nc = float(E["越え"][rows_e].sum())
        pc_ = (float(E["p越え"][rows_e].sum()) / nc) if nc > 0 else float("nan")
        prc = (float(E["prior越え"][rows_e].sum()) / nc) if nc > 0 else float("nan")
        inc = (float(E["入力越え"][rows_e].sum()) / nc) if nc > 0 else float("nan")
        print(f"   {lab}：{hz_e:.1f}Hz ／ 平均 p {pm:.3f}（logit：素子の見込み {pr:+.2f}・入力 {inp:+.2f}）・的が 1 の割合 {tg:.1%}"
              f" ／ 撃つ条件を満たした周 {nc / pp / len(rows_e):.1%}・そのとき p {pc_:.3f}（見込み {prc:+.2f}・入力 {inc:+.2f}）", flush=True)
