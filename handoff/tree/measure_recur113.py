"""版113：皮質の 表現 → 表現 を人の本数・重さ・下からの線と同じ学びにした網で、その辺が育つか・止まるか・皮質のそろいが変わるかを見る
  （arkhe の ~/anaplex で。網は変えない・保存しない。measure_recur112 に版113 の行を足した）
  python3 measure_recur113.py [歩=15000]          ← 版113
  ANAPLEX_RR=0 python3 measure_recur113.py 15000  ← 同じ道具で版112 と同じ網（比べる元）
  版113 で足した読み（組み表_版113_案 §13.7）
    rr の辺（表現 → 表現）：領野ごとの本数（刈り込みで減っていないか）・平均 w・最大 w・押し÷閾値・学びの跡の最大（0 のはず ⇒ GPU の Triton の道の確かめ）
    網の中の興奮の辺で w 1.62（6mV ÷ 1.12 ÷ 接触 3.3）を越える数（上限の単位の誤りが効いているか。rr とそれ以外）
    区間ごと（前の読みから）：当てる領野の表現の率・その発火のうち「当てる領野の 20% 以上が同じ歩に撃った歩」に入る割合・その歩の割合
  毎歩に足す仕事：当てる領野の表現の発火（3,256体）を GPU の上で足すだけ（少ない箱への足し込みなし・CPU を待たない）
  何のため：data/arkhe_srccx109_1111.txt（版109）で、皮質の駆動のうち「網の中」が閾値の 0.001 以下、表現の行1体あたりの重さの和は
    自分の表現から 0〜0.09（時間細胞から 32〜263）だった。版110〜112 の網でも同じかを、撃たせる側の辺だけ数える
  読むもの（1000・5000・10000・15000歩＝終わりの歩で。歩のあいだには何も足さない）
    皮質の領野ごと、表現の行1体あたり：送り手の種類（時間細胞・自分の表現・外から）の辺の本数・重さの和
    自分の表現から：撃たせる側の本数・重さ 0.05 以上の本数・平均の重さ・和 × 接触 ÷ 閾値（全部が同時に届いたときの押し）
  毎歩に足す仕事：なし。読む歩だけ、辺を 4M 本ずつ CPU へ写して numpy で数える（GPU で足し込まない。区切りごとに beat）
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 15000
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 113:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版113 で回す（ANAPLEX_RR=0 で版112 と同じ網）")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
CX = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質"]
KIND = ("時間細胞", "自分の表現", "外から")
areas = g.areas
ch_off = np.cumsum([0] + [a.ch.n_cells for a in areas])
buf_base = int(ch_off[-1])
buf_off = np.cumsum([0] + [a.n_src * a.D for a in areas])
kind = np.full(g.n_send_all, -1, dtype=np.int8)
for i, a in enumerate(areas):
    kind[int(ch_off[i]):int(ch_off[i + 1])] = 0
    s = np.repeat(np.arange(a.n_src), a.D)
    kind[buf_base + int(buf_off[i]):buf_base + int(buf_off[i + 1])] = np.where(s < a.n_ch, 1, np.where(s < a.n_ch + a.n_ext, 2, 3))
assert int((kind < 0).sum()) == 0, "送り手の種類が埋まらない（並びが違う）"
mod = g.mod_send.cpu().numpy().astype(bool)
row_area = np.full(g.n_recv_all, -1, dtype=np.int16)
rep_rows = g.rep_i.long().cpu().numpy()
off_r = [int(x) for x in g.off_recv]
for j, nm in enumerate(CX):
    i = NAMES.index(nm)
    r = rep_rows[(rep_rows >= off_r[i]) & (rep_rows < off_r[i + 1])]
    row_area[r] = j
n_rep = np.array([b.n[nm] for nm in CX], dtype=np.float64)
c_exc = float(areas[NAMES.index("聴覚")].contacts_exc)
th = float(TB.HUMAN_TH)
nc = len(CX)
RRN = [nm for nm in NAMES if areas[NAMES.index(nm)].rr_k > 0]
if TB.RR_ON:
    Lc = g.rr_L_ch
    L_area = torch.cat([torch.full((areas[NAMES.index(nm)].n_ch,), j, dtype=torch.long) for j, nm in enumerate(RRN)]).numpy()
else:                                                   # 版112 と同じ網：同じ 20領野の表現の体を同じ並びで
    RRN = [nm for nm in CX if nm != "運動"]
    Lc = torch.cat([torch.arange(areas[NAMES.index(nm)].n_ch, device=g.rr_L_ch.device) + int(g.off_ch[NAMES.index(nm)]) for nm in RRN])
    L_area = torch.cat([torch.full((areas[NAMES.index(nm)].n_ch,), j, dtype=torch.long) for j, nm in enumerate(RRN)]).numpy()
nL = len(Lc)
nL_a = np.bincount(L_area, minlength=len(RRN)).astype(np.float64)
CAP = NS + 1
step_n = torch.zeros(CAP, device=g.w.device)            # 歩ごとの当てる領野の表現の発火の数（GPU の上に置くだけ）
acc = torch.zeros(nL, device=g.w.device)
last = {"t": 0, "acc": np.zeros(nL)}
W6 = 6.0 / 1.12 / c_exc                                  # 6mV（皮質の辺は 接触 3.3 × w で効く）


def snap_rr(t):
    L.beat()
    a_ = acc.cpu().numpy(); dn = a_ - last["acc"]; dt = t - last["t"]
    sn = step_n[last["t"]:t].cpu().numpy()
    last["acc"], last["t"] = a_.copy(), t
    hz = np.bincount(L_area, weights=dn, minlength=len(RRN)) / nL_a / max(dt, 1) / 0.01067
    tot = float(sn.sum())
    syn = float(sn[sn >= 0.2 * nL].sum()) / tot if tot > 0 else float("nan")
    print(f"  ［区間 {t - dt}〜{t} 歩］当てる領野の表現 {float(dn.sum()) / nL / max(dt, 1) / 0.01067:.2f}Hz ／ 発火のうち 20% 以上がそろう歩に入る割合 {syn:.1%}"
          f"（その歩 {float((sn >= 0.2 * nL).mean()):.1%}）／ 1体も撃たない歩 {float((sn == 0).mean()):.1%}", flush=True)
    print("    領野ごとの率 Hz：" + "  ".join(f"{nm} {hz[j]:.2f}" for j, nm in enumerate(RRN)), flush=True)
    E, st = len(g.w), 1 << 22
    n6, n6i, ncap = 0, 0, 0
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        r_ = g.e_recv[sl].long()
        ex = ~g.e_inh[sl]
        mv = g.w[sl] * g.c_exc[r_] * 1.12 > 6.0             # 1本が 6mV を越える（受け手ごとの接触で）
        ti = g.is_inh_r[r_]
        n6 += int((mv & ex & ~ti).sum()); n6i += int((mv & ex & ti).sum())
        ncap += int(((g.w[sl] >= 5.35) & ex & ~ti).sum())   # 上限 W_E_MAX 5.36 で止まっている（⚠️ 単位の誤りなら 19.8mV）
        L.beat()
    if TB.RR_ON:
        if g._rr_idx is None or g._rr_src is not g.e_src:
            g.rr_rebuild()
        idx = g._rr_idx
        w = g.w[idx].double().cpu().numpy()
        el = float(g.elig[idx].abs().max())
        ar = L_area[g._rr_sp.cpu().numpy()]
        n_ = np.bincount(ar, minlength=len(RRN)); s_ = np.bincount(ar, weights=w, minlength=len(RRN))
        mx = np.zeros(len(RRN)); np.maximum.at(mx, ar, w)
        n6r = int((w > W6).sum())
        print(f"  ［rr の辺］{len(w):,} 本・学びの跡の最大 {el:g}（0 のはず）・w 中央 {np.median(w):.4f}・平均 {w.mean():.4f}・最大 {w.max():.4f}・6mV（w {W6:.2f}）越え {n6r:,} 本", flush=True)
        print(f"    {'領野':<8s}{'本/体':>7s}{'平均 w':>9s}{'最大 w':>9s}{'押し÷閾値':>10s}", flush=True)
        for j, nm in enumerate(RRN):
            p = nL_a[j]
            print(f"    {nm:<8s}{n_[j] / p:7.1f}{s_[j] / max(n_[j], 1):9.4f}{mx[j]:9.4f}{s_[j] / p * c_exc / th:10.4f}", flush=True)
    else:
        n6r = 0
        print("  ［rr の辺なし（ANAPLEX_RR=0）］", flush=True)
    print(f"  ［網の中の興奮の辺で1本 6mV 越え（受け手の接触で）］表現・誤差の受け手 {n6 - n6r:,} 本（rr を除く）・抑制の体の受け手 {n6i:,} 本"
          f" ／ 上限 5.36 で止まる 表現・誤差の受け手 {ncap:,} 本（⚠️ 上限は単位の誤りで 19.8mV。30 §4）", flush=True)
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    L.beat()


def snap(t):
    t0 = time.perf_counter()
    L.beat(); g.flush_all(); L.beat()
    N, W = np.zeros((3, nc)), np.zeros((3, nc))
    drv, big = np.zeros(nc), np.zeros(nc)
    E, st = len(g.w), 1 << 22
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        ra = row_area[g.e_recv[sl].long().cpu().numpy()]
        keep = (ra >= 0) & ~g.e_inh[sl].bool().cpu().numpy()
        src = g.e_src[sl].long().cpu().numpy()[keep]
        w = g.w[sl].double().cpu().numpy()[keep]
        ra = ra[keep].astype(np.int64)
        kk = kind[src]
        for j in range(3):
            m = kk == j
            N[j] += np.bincount(ra[m], minlength=nc)
            W[j] += np.bincount(ra[m], weights=w[m], minlength=nc)
        m = (kk == 1) & ~mod[src]
        drv += np.bincount(ra[m], weights=w[m], minlength=nc)
        big += np.bincount(ra[m & (w >= 0.05)], minlength=nc)
        L.beat()
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()                       # 読みの GPU の仕事を、次の歩の時間（守りの 0.5秒）に入れない（11-18 の落ち方）
    L.beat()
    print(f"\n━━ {t} 歩（読むのに {time.perf_counter() - t0:.1f} 秒）表現の行1体あたり ━━", flush=True)
    print(f"  {'領野':<8s}｜ {'時間細胞 本':>10s}{'w 和':>9s} ｜ {'自分の表現 本':>12s}{'w 和':>9s}{'w≧0.05 本':>10s}{'平均 w':>9s}{'押し÷閾値':>10s} ｜ {'外から 本':>9s}{'w 和':>8s}", flush=True)
    for j, nm in enumerate(CX):
        p = n_rep[j]
        print(f"  {nm:<8s}｜ {N[0][j] / p:10.0f}{W[0][j] / p:9.2f} ｜ {N[1][j] / p:12.0f}{W[1][j] / p:9.4f}{big[j] / p:10.2f}{W[1][j] / max(N[1][j], 1.0):9.5f}{drv[j] / p * c_exc / th:10.4f} ｜ {N[2][j] / p:9.0f}{W[2][j] / p:8.2f}", flush=True)
    print(f"  全部の領野の平均：自分の表現の w 和 {float((W[1] / n_rep).mean()):.4f}・押し÷閾値 {float((drv / n_rep).mean()) * c_exc / th:.4f}", flush=True)


print(f"版 {B.VERSION}（ANAPLEX_RR={'1' if TB.RR_ON else '0'}）／ {NS} 歩 ／ 皮質 {nc} 領野 ／ 当てる領野 {len(RRN)}・表現の体 {nL:,} ／ 接触 {c_exc}・閾値 {th:.1f}", flush=True)
marks = {m for m in (1000, 3000, 6000, 10000) if m < NS} | {NS}
for i in range(1, NS + 1):
    life.step(learn=True)
    a_ = g._any[0][Lc]
    acc += a_
    step_n[i - 1] = a_.sum()
    if i % 1000 == 0:
        L.beat()
    if i in marks:
        snap(life.t)
        snap_rr(i)
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
