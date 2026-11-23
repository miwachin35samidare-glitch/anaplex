"""版81 の足し込み（_scatter3）が、網の中では 1.73ms・作り物の番号では 0.75ms。違いは番号の並びかを見る（arkhe で。何も書かない）
  python3 probe_scatter_real.py
【測】probe_atomic2（10-14）：足し込みの命令は使っている（buffer_atomic_add_f32）／ 作り物の番号 230万本で _scatter3 0.749ms
やること
  網を組んで 30歩まわし、1周目の学びの足し込みに渡る番号・値を1回だけ写し取る（網の値は変えない）
  その番号で、並びを変えて速さを並べる：そのまま ／ でたらめに並べ替え ／ 受け手の順
  並びの様子：となり合う64本（GPU が一度に扱う束）の中で、同じ受け手が何本あるか
"""
import sys, statistics, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain

if B.VERSION < 81 or not getattr(G, "_TRITON", False):
    sys.exit("⚠️ 版81（Triton あり）で回してください")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
box = []
S0 = G._scatter3


def grab(r, ie, amp, mm, n_recv, do_cnt):
    if do_cnt and not box:
        box.append((r.clone(), ie.clone(), amp.clone(), mm.clone(), n_recv))
    return S0(r, ie, amp, mm, n_recv, do_cnt)


k = 0
while k < 31:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(len(ear)):
        if k == 30:
            G._scatter3 = grab
        b.step(torch.tensor(ear[t], device=B.DEV), torch.tensor(cmd[t], device=B.DEV), learn=True)
        k += 1
        if k >= 31:
            break
G._scatter3 = S0
r, ie, amp, mm, nr = box[0]
n = len(r)


def tm(fn, rep=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rep):
        a, c = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); c.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(c))
    return statistics.median(out)


tgt = (r.long() + nr * ie.long())
w = tgt[: (n // 64) * 64].view(-1, 64)
ws = torch.sort(w, 1).values
dup = (ws[:, 1:] == ws[:, :-1]).sum(1).float()
line = torch.sort(w // 32, 1).values          # 128バイト（32個）ごとの置き場所
dl = (line[:, 1:] == line[:, :-1]).sum(1).float()
print(f"辺 {n:,} ／ 受け手 {nr:,} ／ 調整する側の印 {int(mm.sum()):,}本", flush=True)
print(f"64本の束の中で同じ受け手：平均 {dup.mean():.1f}本・最大 {int(dup.max())} ／ 同じ置き場所の塊（128バイト）：平均 {dl.mean():.1f}本", flush=True)
cnt_r = torch.bincount(tgt, minlength=nr * 2)
print(f"受け手1つあたりの本数：平均 {float(cnt_r[cnt_r > 0].float().mean()):.1f} ／ 最大 {int(cnt_r.max())}", flush=True)

perm = torch.randperm(n, device=r.device)
srt = torch.argsort(tgt)
for name, o in (("そのまま", None), ("でたらめに並べ替え", perm), ("受け手の順", srt)):
    rr, ii, aa, mmm = (r, ie, amp, mm) if o is None else (r[o], ie[o], amp[o], mm[o])
    t3 = tm(lambda: S0(rr, ii, aa, mmm, nr, True))
    tb = tm(lambda: torch.zeros(nr * 2, device=r.device).scatter_add_(0, rr.long() + nr * ii.long(), aa))
    print(f"  {name:<12} _scatter3 {t3:.3f}ms ／ aten の足し込み（both だけ）{tb:.3f}ms", flush=True)
