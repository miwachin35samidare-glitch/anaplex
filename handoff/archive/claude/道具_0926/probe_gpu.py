"""arkhe の GPU（ROCm・gfx1201）で、1歩を縮める手法が使えるかを確かめる（網は使わない。数十秒）
  python3 probe_gpu.py
【み】10-10「体数が増えると遅くなるってのがいけない」「広範に調べてみてほしい」
確かめるもの
  1 torch.nonzero_static（数の上限を決めた nonzero）が GPU で動くか。CPU を待たずに済むか
  2 HIP Graph（torch.cuda.CUDAGraph）で、小さい命令を並べた1歩の形を覚えて一度に出せるか。速さ（ふだん ／ 覚えて出す）
     1歩の形の真似：撃った送り手 → 辺の索引 → 強さを集めて足し込む（scatter_add）→ シグモイド → 閾値 を 10周
"""
import time, torch

dev = "cuda"
print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0), flush=True)
g = torch.Generator(device=dev).manual_seed(0)
x = (torch.rand(100000, device=dev, generator=g) < 0.03)
# 1
try:
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200):
        nz = torch.nonzero_static(x, size=8000, fill_value=-1)
    torch.cuda.synchronize()
    print(f"1 nonzero_static：動く（{(time.time()-t)/200*1e3:.3f}ms/回、形 {tuple(nz.shape)}）", flush=True)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200):
        nz2 = torch.nonzero(x)
    torch.cuda.synchronize()
    print(f"  参考 nonzero：{(time.time()-t)/200*1e3:.3f}ms/回（数を CPU が知るために待つ）", flush=True)
except Exception as ex:
    print(f"1 nonzero_static：動かない {type(ex).__name__}: {ex}", flush=True)

# 2
N_SEND, N_RECV, FAN, K = 200000, 20000, 50, 6000
src_e = torch.randint(0, N_SEND, (N_SEND * FAN // 10,), device=dev, generator=g)
recv_e = torch.randint(0, N_RECV, (N_SEND * FAN // 10,), device=dev, generator=g)
w = torch.rand(len(recv_e), device=dev, generator=g) * 0.01
start = torch.arange(0, len(recv_e) + 1, FAN // 10, device=dev)       # 送り手ごとに 5本
fired_buf = torch.full((K,), -1, dtype=torch.long, device=dev)
spk = (torch.rand(N_SEND, device=dev, generator=g) < 0.02)
out = torch.zeros(N_RECV, device=dev)


def one_step():
    f = torch.nonzero_static(spk, size=K, fill_value=0).squeeze(1)
    ok = (torch.arange(K, device=dev) < spk.sum()).float()
    e = (start[f].unsqueeze(1) + torch.arange(5, device=dev)).reshape(-1)
    amp = w[e] * ok.repeat_interleave(5)
    acc = torch.zeros(N_RECV, device=dev)
    for _ in range(10):
        acc2 = acc.scatter_add(0, recv_e[e], amp)
        p = torch.sigmoid(acc2 - 1.0)
        acc = acc2 * 0.0 + (p > 0.5).float() * 0.0 + acc2
    out.copy_(acc)


for _ in range(5):
    one_step()
torch.cuda.synchronize(); t = time.time()
for _ in range(100):
    one_step()
torch.cuda.synchronize(); eager = (time.time() - t) / 100 * 1e3
print(f"2 ふだん：{eager:.3f}ms/歩", flush=True)
try:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        one_step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize(); gt = (time.time() - t) / 100 * 1e3
    ref = out.clone(); one_step(); same = torch.allclose(ref, out)
    print(f"  HIP Graph：動く。{gt:.3f}ms/歩（ふだんの {eager/gt:.1f}倍速）／ ふだんと同じ値 {same}", flush=True)
except Exception as ex:
    print(f"  HIP Graph：動かない {type(ex).__name__}: {ex}", flush=True)
