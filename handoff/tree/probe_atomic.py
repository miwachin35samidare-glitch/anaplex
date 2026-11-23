"""足し込み（受け手へばらばらに足す）が GPU でどのくらいかかるか、作り方ごとに（arkhe で。網は使わない・何も書かない。数十秒）
  python3 probe_atomic.py
【測】measure_gpu80（10-14）：1歩ぶんの辺 平均 2,313,190本 ／ 学ぶ＋足し込み 2.67ms ／ 足す＋足し込み 2.08ms
  読む量から見ると 0.3ms 前後で済むはず【計算】⇒ 受け手へばらばらに足す所（float の足し込み）が遅いのかを見る
並べるもの（同じ 230万本・受け手 240万の所へ。30回の中央）
  0 読むだけ（帯域の目安）
  1 aten scatter_add_（float）／ 2 aten index_add_（float）／ 3 aten scatter_add_（int32）
  4 受け手の順に並べた番号で aten scatter_add_（float）
  5 torch.compile（今の _sum_core と同じ作り）
  6 Triton で直に書いた足し込み（float の atomic_add）
"""
import torch, statistics
import torch._inductor.config as ic
ic.assert_indirect_indexing = False

dev = "cuda"
N, R = 2_313_190, 2_405_872
g = torch.Generator(device=dev); g.manual_seed(0)
idx = torch.randint(0, R, (N,), device=dev, generator=g)
val = torch.rand(N, device=dev, generator=g)
idx32 = idx.int()
ival = torch.ones(N, dtype=torch.int32, device=dev)
sidx = torch.sort(idx).values


def t(fn, rep=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def core(idx, val, R: int):
    o = torch.zeros(R, device=val.device, dtype=val.dtype)
    o.scatter_add_(0, idx, val * 2.0)
    return o


cc = torch.compile(core, dynamic=True)

rows = [("0 読むだけ（val×2）", lambda: val * 2.0),
        ("1 aten scatter_add_ float", lambda: torch.zeros(R, device=dev).scatter_add_(0, idx, val)),
        ("2 aten index_add_ float", lambda: torch.zeros(R, device=dev).index_add_(0, idx, val)),
        ("3 aten scatter_add_ int32", lambda: torch.zeros(R, dtype=torch.int32, device=dev).scatter_add_(0, idx, ival)),
        ("4 受け手の順の番号で scatter_add_ float", lambda: torch.zeros(R, device=dev).scatter_add_(0, sidx, val)),
        ("5 torch.compile（今の作り）", lambda: cc(idx, val, R))]
try:
    import triton, triton.language as tl

    @triton.jit
    def _k(idx_p, val_p, out_p, n, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        i = tl.load(idx_p + o, mask=m, other=0)
        v = tl.load(val_p + o, mask=m, other=0.0)
        tl.atomic_add(out_p + i, v, mask=m, sem="relaxed")

    def tri():
        out = torch.zeros(R, device=dev)
        _k[(triton.cdiv(N, 1024),)](idx32, val, out, N, BLOCK=1024)
        return out
    rows.append(("6 Triton で直に（float atomic_add・relaxed）", tri))
except Exception as ex:          # noqa
    print(f"（Triton を直に使えない：{type(ex).__name__}: {ex}）")

print(f"torch {torch.__version__} ／ {torch.cuda.get_device_name(0)} ／ 辺 {N:,} ／ 受け手 {R:,}", flush=True)
ref = torch.zeros(R, device=dev).scatter_add_(0, idx, val)
for name, fn in rows:
    try:
        ms = t(fn)
        r = fn()
        ok = ""
        if name.startswith(("1", "2", "6")):
            ok = f" ／ 値 差の最大 {float((r - ref).abs().max()):.1e}"
        print(f"  {ms:7.3f}ms  {name}{ok}", flush=True)
    except Exception as ex:      # noqa
        print(f"  ✗ {name}：{type(ex).__name__}: {str(ex)[:120]}", flush=True)
