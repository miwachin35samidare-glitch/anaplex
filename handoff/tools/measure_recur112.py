"""版112：皮質の中の 表現 → 表現 の興奮の辺（撃たせる側）が、どれだけ育っているかを見る（arkhe の ~/anaplex で。網は変えない・保存しない）
  python3 measure_recur112.py [歩=15000]
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
if B.VERSION != 112:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版112 で回す")
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


print(f"版 {B.VERSION} ／ {NS} 歩 ／ 皮質 {nc} 領野 ／ 接触 {c_exc}・閾値 {th:.1f}", flush=True)
marks = {m for m in (1000, 5000, 10000) if m < NS} | {NS}
for i in range(1, NS + 1):
    life.step(learn=True)
    if i % 1000 == 0:
        L.beat()
    if i in marks:
        snap(life.t)
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
