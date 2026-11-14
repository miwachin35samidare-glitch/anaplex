"""版110：刈り込み（Group2.rewire、1000歩ごと）の中で、何が時間と VRAM を使うかを測る（arkhe の ~/anaplex で。保存しない・式は変えない）
  python3 measure_rewire110.py [回=3] [あいだの歩=300]
  何のため：刈り込みの歩が 1.77秒・PyTorch の取り置きの山 11.98GB（ふだん約 8GB）で、守りが止め・GPU 全体が 12GB を越える（30 §1、11-14）
  やり方：ふだんの暮らしを「あいだの歩」だけ回してから g.rewire() を1回呼ぶ、を「回」だけ繰り返す（1000歩を待たない。刈り込みの中身は同じ式）
    rewire の中の呼び（flush_all・ふさがりの前後・一番弱い辺の選び 2回・差し込みの並び・短期の並べ直し・引き直し・始まりの表）を包んで、
    呼びごとに 時間（前後で GPU を待つ）と、その呼びのあいだの「使用の山 − 呼ぶ前の使用」（torch.cuda の数え）を出す。
    包んでいない所（辺を選ぶ・生やす先を引く・5つの配列の写し・重さの入れ直し）は「残り」に入る
  ⚠️ 包むたびに GPU を待つので、合計はふだんより少し長く出る。刈り込み1回は 2秒ほどの GPU の仕事（ふだんの 1000歩ごとと同じ長さ）
  毎歩に足す仕事：なし（歩は life.step のまま）
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB, life as L
from torch_brain import Brain

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 3
GAP = int(args[1]) if len(args) > 1 else 300
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
CUDA = str(dev).startswith("cuda")
if B.VERSION != 110:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：版110 で確かめた道具")
GB = 1024 ** 3

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
rows = []          # （名前, 秒, 山の増え GB）
PK = {"rest": 0.0, "rs": 0.0, "base": 0.0}     # 包んでいない所の使用の山・取り置きの山（刈り込み1回の中）


def mark():
    """山の数えを 0 に戻す前に、ここまで（包んでいない所）の山を覚える"""
    if CUDA:
        PK["rest"] = max(PK["rest"], torch.cuda.max_memory_allocated() / GB - PK["base"])
        PK["rs"] = max(PK["rs"], torch.cuda.max_memory_reserved() / GB)


def sync():
    if CUDA:
        torch.cuda.synchronize()


def used():
    return torch.cuda.memory_allocated() / GB if CUDA else 0.0


def wrap(obj, name, label):
    f0 = getattr(obj, name)

    def f(*a, **k):
        sync()
        mark()
        u0 = used()
        if CUDA:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = f0(*a, **k)
        sync()
        dt = time.perf_counter() - t0
        pk = (torch.cuda.max_memory_allocated() / GB - u0) if CUDA else 0.0
        if CUDA:
            PK["rs"] = max(PK["rs"], torch.cuda.max_memory_reserved() / GB)
            torch.cuda.reset_peak_memory_stats()
        rows.append((label, dt, pk))
        return r

    setattr(obj, name, f)


wrap(g, "flush_all", "学びの分を足す flush_all")
wrap(g, "_first_per_recv", "受け手ごとの一番小さい辺 _first_per_recv")
wrap(g, "_order_after_cut", "差し込みの並び _order_after_cut")
wrap(g, "_starts", "始まりの表 _starts")
if getattr(g, "_occ", None) is not None:
    for nm in ("rewire_pre", "rewire_map", "rewire_done"):
        wrap(g._occ, nm, f"ふさがり {nm}")
if g.stp is not None:
    wrap(g.stp, "reorder", "短期の並べ直し stp.reorder")
    wrap(g.stp, "redraw", "短期の引き直し stp.redraw")

E = len(g.w)
print(f"版 {B.VERSION} ／ 辺 {E:,} 本・受け手の行 {g.n_recv_all:,} ／ 刈り込み {N} 回・あいだ {GAP} 歩 ／ ふさがり {'あり' if getattr(g, '_occ', None) is not None else 'なし'}"
      f"・短期 {'あり' if g.stp is not None else 'なし'}", flush=True)
for k in range(N):
    for _ in range(GAP):
        life.step(learn=True)
    L.beat()
    sync()
    rows.clear()
    cut0 = getattr(g, "n_cut", 0)
    u0 = used()
    PK.update(rest=0.0, rs=0.0, base=u0)
    rs0 = torch.cuda.memory_reserved() / GB if CUDA else 0.0
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g.rewire()
    sync()
    tot = time.perf_counter() - t0
    L.beat()
    mark()
    rs1 = torch.cuda.memory_reserved() / GB if CUDA else 0.0
    print(f"\n━━ 刈り込み {k + 1} 回目（{life.t} 歩のあと）：合計 {tot:.3f} 秒・切った辺 {getattr(g, 'n_cut', 0) - cut0:,} 本"
          f"・呼ぶ前の使用 {u0:.2f}GB・取り置き 前 {rs0:.2f} → 後 {rs1:.2f}GB ━━", flush=True)
    print(f"  {'呼び':<40s}{'秒':>8s}{'山の増え GB':>12s}", flush=True)
    for label, dt, pk in rows:
        print(f"  {label:<40s}{dt:8.3f}{pk:12.2f}", flush=True)
    rest = tot - sum(r[1] for r in rows)
    print(f"  {'残り（包んでいない所）':<40s}{rest:8.3f}{PK['rest']:12.2f}（呼ぶ前の使用からの山）", flush=True)
    print(f"  刈り込みの中の取り置きの山 {PK['rs']:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
