"""網の本当の番号で、足し込みの写しを K 個に分けると速くなるかを見る（arkhe で。何も書かない）
  python3 probe_scatter_real2.py
【測】probe_scatter_real（10-14）：辺 292万本が受け手 9,736 に集まる（1つあたり平均 166本・最大 1469）⇒ 同じ所への足し込みが順番待ち
  でたらめに並べ替えても 2.19 → 1.86ms どまり ⇒ 並びではなく、1か所に来る数が多すぎる
やり方：束ごとに K 個の写しのどれかへ足し、最後に写しを足し合わせる（1か所に来る数が 1/K に）
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


import triton, triton.language as tl


@triton.jit
def _rk(r_p, ie_p, amp_p, mm_p, both_p, mod_p, cnt_p, n, n_recv, K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    rep = pid % K
    r = tl.load(r_p + o, mask=m, other=0)
    ie = tl.load(ie_p + o, mask=m, other=0).to(tl.int32)
    a = tl.load(amp_p + o, mask=m, other=0.0)
    mm = tl.load(mm_p + o, mask=m, other=0).to(tl.int32)
    tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie, a, mask=m, sem="relaxed")
    tl.atomic_add(mod_p + rep * n_recv + r, a, mask=m & (mm != 0), sem="relaxed")
    tl.atomic_add(cnt_p + rep * n_recv + r, tl.full([BLOCK], 1, tl.int32), mask=m, sem="relaxed")


def rk(K, BLOCK=1024):
    both = torch.zeros(K, 2 * nr, device=r.device)
    mod = torch.zeros(K, nr, device=r.device)
    cnt = torch.zeros(K, nr, device=r.device, dtype=torch.int32)
    _rk[(triton.cdiv(n, BLOCK),)](r, ie, amp, mm, both, mod, cnt, n, nr, K=K, BLOCK=BLOCK)
    return both.sum(0), mod.sum(0), cnt.sum(0).float()


ref = S0(r, ie, amp, mm, nr, True)
print(f"  版81 _scatter3（写し1つ）{tm(lambda: S0(r, ie, amp, mm, nr, True)):.3f}ms", flush=True)
for K in (4, 16, 64, 256):
    for BL in (256, 1024):
        out = rk(K, BL)
        d = max(float((x - y).abs().max() / (y.abs().max() + 1e-12)) for x, y in zip(out, ref))
        print(f"  写し {K:4d} 個・束 {BL:5d}：{tm(lambda: rk(K, BL)):.3f}ms ／ 値 差の最大（割合）{d:.1e}", flush=True)
