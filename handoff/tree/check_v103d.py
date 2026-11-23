"""版103d の確かめ（arkhe・GPU。保存しない）：放出のくじの新しい引き方が torch.binomial と同じ分布か・速さ
  python3 check_v103d.py
  コンテナ（CPU）で 7 × 9 通りの (N, p) を 40万標本ずつ、二項分布の確率と比べて揺らぎの範囲（最大のずれは 400万標本で目安以内）
"""
import math, time, torch
import torch_area as T
from torch_group2 import _run

ok = True
def chk(c, m):
    global ok
    print(("  ✓ " if c else "  ✗ ") + m, flush=True); ok = ok and bool(c)

dev = "cuda"
g = torch.Generator(device=dev); g.manual_seed(0)
M = 2_000_000
worst = (0.0, None)
for n in (1, 2, 3, 6, 12, 24, 32):
    for p in (0.0098, 0.03, 0.2, 0.5, 0.51, 0.76, 0.99, 1.0):
        N = torch.full((M,), float(n), device=dev); P = torch.full((M,), p, device=dev)
        k = T.binom(N, P, g)
        emp = torch.bincount(k.long(), minlength=n + 1)[:n + 1].float().cpu() / M
        pmf = [math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n + 1)]
        d = max(abs(float(emp[j]) - pmf[j]) for j in range(n + 1))
        if d > worst[0]:
            worst = (d, (n, p))
        if not (int(k.max()) <= n and int(k.min()) >= 0):
            chk(False, f"N {n}・p {p} の出目が 0〜N の外")
chk(worst[0] < 6 * math.sqrt(0.25 / M), f"二項分布の確率との最大のずれ {worst[0]:.5f}（{worst[1]}。揺らぎの目安 {math.sqrt(0.25 / M):.5f} の 6倍まで）")
# 速さ（撃った線くらいの数で）
N = torch.randint(1, 33, (1_000_000,), device=dev).float(); P = torch.rand(1_000_000, device=dev)
for name, f in (("torch.binomial", lambda: torch.binomial(N, P, generator=g)), ("新しい引き方", lambda: T.binom(N, P, g))):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    print(f"  {name}：100万本 {(time.time() - t0) / 20 * 1000:.2f}ms/回", flush=True)
print("✅ 版103d の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
