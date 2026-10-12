"""1歩を丸ごと覚えさせる（HIP Graph）・命令をまとめる（torch.compile／Triton）が、arkhe の網の作りで使えるか（網は使わない。1分ほど）
  python3 probe_gpu2.py
【測】probe_gpu（10-10）：HIP Graph は動く・同じ値。nonzero_static は ROCm で15倍遅い
網の1歩には、脳の中の乱数（自分の乱数の種を持つ torch.Generator）で撃つ所と、短期可塑性の放出（binomial）がある
  ⇒ 覚えさせた1歩を出し直すたびに、乱数が進むか（毎回同じ乱数にならないか）を確かめる
確かめるもの
  1 自分の Generator の torch.rand を覚えさせて出し直す（register_generator_state）。出し直すたびに違う値か
  2 binomial（短期可塑性の放出）を覚えさせられるか
  3 数を読まない詰め方（累積和で番号を出して決まった大きさの箱へ）が nonzero と同じ並びになるか・速さ
  4 torch.compile（Triton）が gfx1201 で動くか。小さい命令を並べた形の速さ（ふだん ／ まとめた）
"""
import time, torch

dev = "cuda"
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
sync = torch.cuda.synchronize

# 1・2
try:
    gen = torch.Generator(device=dev); gen.manual_seed(1)
    out = torch.zeros(5, device=dev); cnt = torch.full((5,), 10.0, device=dev); pr = torch.full((5,), 0.5, device=dev)
    def f():
        out.copy_(torch.rand(5, device=dev, generator=gen))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    if hasattr(g, "register_generator_state"):
        g.register_generator_state(gen)
    with torch.cuda.graph(g):
        f()
    vals = []
    for _ in range(3):
        g.replay(); sync(); vals.append(out.clone())
    print(f"1 自分の Generator の rand：覚えさせられた。出し直すたびに違う値 {not torch.equal(vals[0], vals[1]) and not torch.equal(vals[1], vals[2])}"
          f"（register_generator_state {'あり' if hasattr(g, 'register_generator_state') else '無し'}）", flush=True)
except Exception as ex:
    print(f"1 自分の Generator の rand：だめ {type(ex).__name__}: {str(ex)[:200]}", flush=True)
try:
    out2 = torch.zeros(5, device=dev)
    def f2():
        out2.copy_(torch.binomial(cnt, pr))
    with torch.cuda.stream(s):
        f2()
    torch.cuda.current_stream().wait_stream(s)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        f2()
    v = []
    for _ in range(3):
        g2.replay(); sync(); v.append(out2.clone())
    print(f"2 binomial：覚えさせられた。出し直すたびに違う値 {not torch.equal(v[0], v[1]) or not torch.equal(v[1], v[2])}", flush=True)
except Exception as ex:
    print(f"2 binomial：だめ {type(ex).__name__}: {str(ex)[:200]}", flush=True)

# 3
x = torch.rand(400000, device=dev) < 0.03
CAP = 20000
def compact(mask):
    pos = torch.cumsum(mask.to(torch.int32), 0) - 1
    out = torch.full((CAP + 1,), -1, dtype=torch.long, device=dev)
    out.scatter_(0, torch.where(mask, pos.long().clamp(max=CAP - 1), torch.full_like(pos.long(), CAP)),
                 torch.arange(len(mask), device=dev))
    return out[:CAP], mask.sum()
a, n = compact(x); b = torch.nonzero(x).squeeze(1)
same = torch.equal(a[:len(b)], b) and bool((a[len(b):] == -1).all())
sync(); t = time.time()
for _ in range(200):
    compact(x)
sync(); tc = (time.time() - t) / 200 * 1e3
t = time.time()
for _ in range(200):
    torch.nonzero(x)
sync(); tn = (time.time() - t) / 200 * 1e3
print(f"3 数を読まない詰め方：nonzero と同じ並び {same} ／ {tc:.3f}ms/回（nonzero {tn:.3f}ms、こちらは CPU を待たない）", flush=True)

# 4
def small(a, b, c):
    for _ in range(20):
        a = torch.sigmoid(a * b + c) - 0.5
        a = torch.where(a > 0, a, a * 0.1)
    return a
A = torch.rand(100000, device=dev); Bv = torch.rand(100000, device=dev); C = torch.rand(100000, device=dev)
small(A, Bv, C); sync(); t = time.time()
for _ in range(100):
    small(A, Bv, C)
sync(); te = (time.time() - t) / 100 * 1e3
try:
    cs = torch.compile(small)
    r1 = cs(A, Bv, C); sync(); t = time.time()
    for _ in range(100):
        cs(A, Bv, C)
    sync(); tcmp = (time.time() - t) / 100 * 1e3
    print(f"4 torch.compile（Triton）：動く。{te:.3f} → {tcmp:.3f}ms（{te/tcmp:.1f}倍）／ 同じ値（丸めの範囲）{torch.allclose(r1, small(A, Bv, C), atol=1e-5)}", flush=True)
except Exception as ex:
    print(f"4 torch.compile（Triton）：だめ {type(ex).__name__}: {str(ex)[:300]}", flush=True)
