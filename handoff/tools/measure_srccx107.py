"""版107z（measure_src107u を皮質の受け手へ）：皮質が時間とともに黙っていく元を、駆動の項ごとに分けて番ごとに見る
  （arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_srccx107.py [測る番の並び=1,10,30,60,100] [窓の歩=100]
  問い：wscale107z（11-10）で 前補足・聴覚・45・STS1 は時間とともに静まり、眼窩前頭は上がった。
        網の中の辺の重さの和は前補足でほぼ平ら（300〜440）なのに率は 0.54 → 0 ⇒ どの項が落ちたかを分ける
  恒常性の上げ（scale_up）は rate_long（1時間）と K_WAKE_UP（2日で 2.5倍）で、この長さでは動かない作り【計算】⇒ 見ない
  表現の体が撃つか（torch_group2._drive_core）：
     駆動 ＝ (網の中の興奮 − 上からの調整)[表現] ÷ (1＋抑制) ＋ 下からの線 × x_gain ÷ (1＋抑制) ＋ CL ÷ (1＋抑制) ＋ 持ち越し  ＞ 閾値
  窓ごと・領野ごとに（歩の平均、撃ったかどうかに依らない。値は ÷閾値。網の中・調整・下から・CL は ÷(1＋抑制) の前）：
     表現Hz・誤差Hz・率÷目標（rate_ema の中央値）
     網中   (exc − exc_mod)[表現]（scale × scale_up 込み）
     調整   exc_mod[表現]（駆動からは引かれる側）
     下から 下からの線の合計（x_all × x_gain）
     CL     cl_drive
     持越   v_carry
     1＋抑  1＋抑制（割り算の側）。領野の中の抑制の体・外からの線・核どうし を分ける
     駆動   ÷(1＋抑制) のあとの駆動の平均 ÷ 閾値 ／ 越え：駆動が閾値を越えた周の割合
  詳しく見る領野（WANT）は、さらに
     下からの線（道ごと）：÷閾値・送り手Hz・1体の線の数・線の重さの平均
     網の中の辺の重さ（表現の行へ・興奮の辺）：受け手1体あたりの和を送り手の種類ごと（時間細胞・自分の表現・外から・自分の誤差）
       ＋ 抑制の辺の和（抑制の送り手）
  ⚠️ 平均の押しで見る（撃った歩の押しではない）。元の見立てを版にする前に、撃った歩でも確かめる（50 §9 11-09）
  ⚠️ グラフを撮らずに回す（ANAPLEX_GRAPH=eager。値は同じ・遅くなる）
"""
import os, sys, collections, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WIN = [int(x) for x in (args[0] if args else "1,10,30,60,100").split(",")]
NS = int(args[1]) if len(args) > 1 else 100
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 107:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版107（107z）の形で読む")
CX = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質"]
WANT = [n for n in ("前補足", "45", "聴覚", "STS1", "眼窩前頭", "前頭前野", "補足") if n in NAMES]
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
n_all = g.n_ch_all
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
inh_keys = set(getattr(b, "bu_inh", {}).keys())
DT = 10.67 / 1000.0

# ── 送り手の種類（網の中の辺の e_src の並び。torch_group2 の remap と同じ）──
KIND = ("時間細胞", "自分の表現", "外から", "抑制の送り手")
areas = g.areas
ch_off = np.cumsum([0] + [a.ch.n_cells for a in areas])
buf_base = int(ch_off[-1])
buf_off = np.cumsum([0] + [a.n_src * a.D for a in areas])
kind = torch.full((g.n_send_all,), -1, dtype=torch.int8, device=dev)
for i, a in enumerate(areas):
    kind[int(ch_off[i]):int(ch_off[i + 1])] = 0
    s = torch.arange(a.n_src, device=dev).repeat_interleave(a.D)
    k = torch.where(s < a.n_ch, 1, torch.where(s < a.n_ch + a.n_ext, 2, 3)).to(torch.int8)
    kind[buf_base + int(buf_off[i]):buf_base + int(buf_off[i + 1])] = k
assert int((kind < 0).sum()) == 0, "送り手の種類が埋まらない（並びが版107 と違う）"
is_rep_r = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev)
is_rep_r[g.rep_i.long()] = True


def wsnap():
    """表現の行へ入る網の中の辺：受け手ごとに 送り手の種類 × 興奮／抑制 の重さの和"""
    g.flush_all()
    out = torch.zeros(5, g.n_recv_all, device=dev, dtype=torch.float64)
    E, st = len(g.w), 1 << 22   # 区切りの写しを小さく（GPU の山を上げない）
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        r = g.e_recv[sl].long()
        m = is_rep_r[r]
        inh = g.e_inh[sl].bool()
        kk = kind[g.e_src[sl].long()].long()
        w = g.w[sl].double()
        for j in range(4):
            mm = m & ~inh & (kk == j)
            out[j].index_add_(0, r[mm], w[mm])
        mm = m & inh
        out[4].index_add_(0, r[mm], w[mm])
    return out


# ── 駆動の項（_drive_core を周ごとに拾う）──
D0 = TB._BUTable.drive
R0 = G._run
on = [False]
acc_del, acc_fire = collections.defaultdict(float), collections.defaultdict(float)
NAMES_ACC = ("網中", "調整", "下から", "CL", "持越", "1＋抑", "外からの抑", "核どうしの抑", "駆動", "越え")
acc = {k: torch.zeros(n_all, device=dev, dtype=torch.float64) for k in NAMES_ACC}
npass = [0]
cap = {}


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


def run(fn, *a):
    out = R0(fn, *a)
    if fn is G._drive_core and on[0]:
        (both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry, fire_th, th_gain,
         complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_recv) = a
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        th = (fire_th * th_gain).double().expand(n_all)
        xg = torch.as_tensor(x_gain, device=dev).double().expand(n_all)
        cap["x_gain"], cap["th"] = xg, th
        acc["網中"] += (exc - exc_mod2)[rep_i].double() / th
        acc["調整"] += exc_mod2[rep_i].double() / th
        xx = x_all if nuc_x is None else x_all + nuc_x
        acc["下から"] += (xx * x_gain).double() / th
        if cl_drive is not None:
            acc["CL"] += cl_drive.double() / th
        if v_carry is not None:
            acc["持越"] += v_carry.double() / th
        acc["1＋抑"] += (1.0 + ihn[rep_i]).double()
        if ext_ihn is not None:
            acc["外からの抑"] += ext_ihn[rep_i].double()
        if nuc_ihn is not None:
            acc["核どうしの抑"] += nuc_ihn[rep_i].double()
        acc["駆動"] += drive.double() / th
        acc["越え"] += (drive > fire_th * th_gain).double()
        npass[0] += 1
    return out


keys_by_tgt = collections.defaultdict(list)
for tab in [t for t in (getattr(b, "but", None), getattr(b, "but_inh", None)) if t is not None]:
    for i, k in enumerate(tab.keys):
        keys_by_tgt[ch0_name.get(int(b.bu[k].ch0) if k in b.bu else int(b.bu_inh[k][0].ch0), "?")].append((k, tab, i))


def window(label):
    for d in (acc_del, acc_fire):
        d.clear()
    for v in acc.values():
        v.zero_()
    npass[0] = 0
    rep_n = torch.zeros(n_all, device=dev)
    err_n = torch.zeros(n_all, device=dev)
    TB._BUTable.drive, G._run = drive_w, run
    on[0] = True
    t0 = life.t
    for _ in range(NS):
        life.step()
        rep_n.add_(g._any[0][:n_all].float())
        err_n.add_(g._any[1][:n_all].float())
    on[0] = False
    TB._BUTable.drive, G._run = D0, R0
    ns, pp = max(1, life.t - t0), max(1, npass[0])
    q = (g.rate_ema / g.target_recv.clamp(min=1e-9))
    W = wsnap()
    print(f"\n━━ {label}（{t0}〜{life.t}歩・{pp}周）値は ÷閾値。網中・調整・下から・CL は ÷(1＋抑制) の前 ━━", flush=True)
    print(f"  {'領野':<8s}{'表現Hz':>7s}{'誤差Hz':>7s}{'率÷目標':>8s}{'網中':>8s}{'調整':>8s}{'下から':>8s}{'CL':>7s}{'持越':>7s}"
          f"{'1＋抑':>7s}{'駆動':>7s}{'越え':>7s}", flush=True)
    for nm in CX:
        i = NAMES.index(nm)
        oc, n = int(b.off_ch[i]), b.n[nm]
        sl = slice(oc, oc + n)
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        rr = torch.arange(lo, hi, device=dev)
        rr = rr[is_rep_r[rr]]
        m = lambda k: float(acc[k][sl].mean()) / pp
        print(f"  {nm:<8s}{float(rep_n[sl].mean()) / ns / DT:7.2f}{float(err_n[sl].mean()) / ns / DT:7.2f}"
              f"{float(q[rr].median()):8.2f}{m('網中'):8.3f}{m('調整'):8.3f}{m('下から'):8.3f}{m('CL'):7.3f}{m('持越'):7.3f}"
              f"{m('1＋抑'):7.2f}{m('駆動'):7.3f}{m('越え'):7.4f}", flush=True)
    for nm in WANT:
        i = NAMES.index(nm)
        oc, n = int(b.off_ch[i]), b.n[nm]
        xg, th = float(cap["x_gain"][oc]), float(cap["th"][oc])
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        rr = torch.arange(lo, hi, device=dev)
        rr = rr[is_rep_r[rr]]
        ws = "・".join(f"{KIND[j]} {float(W[j][rr].mean()):.4g}" for j in range(4))
        sl = slice(oc, oc + n)
        print(f"  ▸ {nm}：網の中の辺（表現の行1体あたり w 和）{ws} ／ 抑制の辺 {float(W[4][rr].mean()):.4g}"
              f" ／ 1＋抑制のうち 外から {float(acc['外からの抑'][sl].mean()) / pp:.3f}・核どうし {float(acc['核どうしの抑'][sl].mean()) / pp:.3f}", flush=True)
        for k, tab, j in keys_by_tgt.get(nm, []):
            nl = (tab.eo[j + 1] - tab.eo[j]) / max(1, tab.nt[j])
            dv = acc_del[k] / ns
            fr = acc_fire[k] / ns / DT
            wm = float(tab.w[tab.eo[j]:tab.eo[j + 1]].mean()) if tab.eo[j + 1] > tab.eo[j] else float("nan")
            if k in inh_keys:
                print(f"      抑制 {k:<20s} 届いた {dv:8.3f}（割り算の側）線/体 {nl:6.1f} 送り手 {fr:6.2f}Hz 重さ {wm:.3f}", flush=True)
            else:
                print(f"      {k:<25s} {dv * xg / th:8.3f} 線/体 {nl:6.1f} 送り手 {fr:6.2f}Hz 重さ {wm:.3f}", flush=True)


print(f"版 {B.VERSION} ／ 窓 {WIN} 番・各 {NS}歩 ／ 皮質 {len(CX)} 領野（詳しく：{'・'.join(WANT)}）", flush=True)
done = 0
for k in WIN:
    if k > done:
        life.run_plans(k - done, learn=True)
        done = k
    window(f"{k}番")
print("\n（終わり）", flush=True)
