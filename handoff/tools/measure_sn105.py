"""版105b：黒質を押している入力（核どうしの線を送り手ごとに）と、率（1行1領野）。音あり／無音（arkhe の ~/anaplex で。保存しない）
  ANAPLEX_XCX=1.7 python3 measure_sn105.py [先に回す番=0] [測る歩=40] [--silent]
    --silent：測る歩は耳・口の指令を 0 で回す（補助輪・合図・返し手・声なし）。先に回す番は音ありのまま
  【み】11-04 00:31「いいよ」。盛り 1.7 で皮質は人の近く・視床下核 25Hz でも、黒質は 30Hz のまま（measure_rate105）
    ⇒ 黒質を押しているものを、核どうしの線（視床下核・PPN の興奮・吻内側被蓋核の抑制）・下からの線・i0 に分ける
    ⇒ 【み】「間を11倍早めてるのが悪さしてるところはないの？」：音あり と 無音 で率を比べる（同じなら出来事の詰めに依らない）
  値は 受け手1体あたり・その周の平均（撃つかに依らない）÷ 閾値。興奮は × x_gain（版105 で核は 1）、抑制は 届いた重さ（割り算の側）
"""
import os, sys, collections, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES
from mouth import N_CMD

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 0
NS = int(args[1]) if len(args) > 1 else 40
SILENT = "--silent" in sys.argv
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
if WARM > 0:
    life.run_plans(WARM, learn=True)
g = b.g
n_all = g.n_ch_all
ch_area = torch.cat([torch.full((b.n[nm],), i, dtype=torch.long, device=dev) for i, nm in enumerate(NAMES)])
TGT = ("黒質", "視床下核", "外節", "PPN", "吻内側被蓋核")
tgt_idx = {NAMES.index(nm) for nm in TGT}
on = [False]
# ── 核どうしの線：送り手の領野 × 受け手の領野 × 符号 ──
nn_ = g.nucnet
src_all, dch_all, sg_all, _dl = nn_.flat()
s_area = ch_area[src_all % n_all]
s_kind = (src_all >= n_all).long()            # 0 表現 ／ 1 誤差
d_area = ch_area[dch_all]
keep = torch.zeros(len(src_all), dtype=torch.bool, device=dev)
for i in tgt_idx:
    keep |= d_area == i
key_id = (d_area * 1000 + s_area) * 4 + s_kind * 2 + (sg_all > 0).long()
uk, inv = torch.unique(key_id[keep], return_inverse=True)
kidx = torch.nonzero(keep).squeeze(1)
acc_nn = torch.zeros(len(uk), device=dev, dtype=torch.float64)
A0 = G.NucNet.after


def after_w(self, S, rep, err):
    if on[0]:
        v = torch.cat([rep, err])
        c = (self.w[kidx] * v[self.src[kidx]]).double()
        acc_nn.index_add_(0, inv, c)
    return A0(self, S, rep, err)


G.NucNet.after = after_w
# ── 下からの線（道ごと）──
acc_del = collections.defaultdict(float)
D0 = TB._BUTable.drive


def drive_w(self, gi, vs):
    out = D0(self, gi, vs)
    if on[0]:
        i0_, i1_ = self._span(gi)
        for j, i in enumerate(range(i0_, i1_)):
            acc_del[self.keys[i]] += float(out[j].mean())
    return out


TB._BUTable.drive = drive_w
# ── i0・抑制（_drive_core）──
acc = {k: torch.zeros(n_all, device=dev, dtype=torch.float64) for k in ("i0", "1＋抑制")}
npass = [0]
R0 = G._run


def run(fn, *a):
    out = R0(fn, *a)
    if fn is G._drive_core and on[0]:
        (both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry, fire_th, th_gain,
         complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_r) = a
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        th = (fire_th * th_gain).double()
        if cl_drive is not None:
            acc["i0"] += cl_drive.double() / th
        acc["1＋抑制"] += (1.0 + ihn[rep_i]).double()
        npass[0] += 1
    return out


G._run = run
any_r = torch.zeros(n_all, device=dev, dtype=torch.float64)
any_e = torch.zeros(len(g.err_i), device=dev, dtype=torch.float64)
on[0] = True
t0 = life.t
z_ear, z_cmd = torch.zeros(B.N_EAR, device=dev), torch.zeros(N_CMD, device=dev)
for _ in range(NS):
    if SILENT:
        b.step(z_ear, z_cmd, learn=True)
    else:
        life.step()
    any_r += g._any[0][:n_all].double()
    any_e += g._any[1].double()
on[0] = False
G._run = R0
TB._BUTable.drive = D0
G.NucNet.after = A0
ns = NS if SILENT else max(1, life.t - t0)
DT = 10.67 / 1000.0
pp = max(1, npass[0])
xcx = os.environ.get("ANAPLEX_XCX", "なし（X_NEW）")
print(f"\n=== ANAPLEX_XCX={xcx} ／ 先に {WARM}番（{t0}歩）・測る {ns}歩 {'無音（耳・口 0）' if SILENT else '音あり（暮らし）'}", flush=True)
err_area, err_om = g.area_of[g.err_i], g.is_om[g.err_i]
AREAS = TB.CL_CORTEX + ("背内側視床", "VA", "視床枕", "扁桃体基底", "視床下核", "黒質", "外節", "内節", "網様部", "PPN", "吻内側被蓋核",
                        "尾状核", "被殻", "側坐核")
cx = []
for nm in AREAS:
    i = NAMES.index(nm)
    oc, n = int(b.off_ch[i]), b.n[nm]
    r = float(any_r[oc:oc + n].mean()) / ns / DT
    e_sel, o_sel = (err_area == i) & ~err_om, (err_area == i) & err_om
    e = float(any_e[e_sel].mean()) / ns / DT if bool(e_sel.any()) else float("nan")
    o = float(any_e[o_sel].mean()) / ns / DT if bool(o_sel.any()) else float("nan")
    if nm in TB.X_CX_AREAS:
        cx.append(r)
    print(f"  {nm:<8s} 表現 {r:6.1f}  誤差 {e:6.1f}  来ない {o:6.1f}", flush=True)
print(f"  ⇒ 盛りを振った 12領野の表現の平均 {np.mean(cx):.1f}Hz", flush=True)
ukc = uk.cpu().tolist()
vals = (acc_nn / pp).cpu().tolist()
for nm in TGT:
    i = NAMES.index(nm)
    oc, n = int(b.off_ch[i]), b.n[nm]
    th = float(g.fire_th[oc]) if torch.is_tensor(g.fire_th) else float(g.fire_th)
    xg = float(g.x_gain[oc])
    i0 = float(acc["i0"][oc:oc + n].mean()) / pp
    ih = float(acc["1＋抑制"][oc:oc + n].mean()) / pp
    print(f"\n■ {nm} 受け手1体あたり（÷閾値 {th:.1f}・x_gain {xg:.2f}）／ 1＋抑制 {ih:.2f} ／ i0 {i0:.3f}", flush=True)
    for kv, v in sorted(zip(ukc, vals), key=lambda t: -abs(t[1])):
        da, rest = divmod(kv, 4000)
        if da != i:
            continue
        sa, sk = divmod(rest, 4)
        kind, sgn = divmod(sk, 2)
        per = v / n
        lab = f"{NAMES[sa]}（{'誤差' if kind else '表現'}）→ {nm}"
        if sgn:
            print(f"   核どうし 興奮 {lab:<26s} {per * xg / th:8.3f}", flush=True)
        else:
            print(f"   核どうし 抑制 {lab:<26s} 届いた重さ {per:8.3f}（割り算の側）", flush=True)
    for k_, s_ in b.bu.items():
        if int(s_.ch0) == oc and k_ in acc_del:
            print(f"   下から {k_:<24s} {acc_del[k_] / ns * xg / th:8.3f}", flush=True)
