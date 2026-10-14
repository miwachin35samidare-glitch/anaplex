"""小数の足し込みが GPU の「足し込みの命令」で出ているか、やり直しの輪（読む→足す→書けたか確かめる）になっているかを見る（arkhe で。網は使わない・何も書かない）
  python3 probe_atomic2.py
【測】probe_atomic（10-14）：230万本 小数 0.67ms ／ 整数 0.158ms（4倍の差）／ 版81 の _scatter3_k 1.73ms/回
見るもの
  1 Triton の小数の足し込みの機械語に、足し込みの命令（global_atomic_add_f32）があるか、やり直しの輪（cmpswap）か
  2 HIP で直に書いた足し込み（unsafeAtomicAdd ＝ 足し込みの命令を必ず使う）の速さ ⇒ 作れるか（hipcc があるか）も見る
  3 版81 の _scatter3_k と同じ形（3つの足し込み・印つき）の速さ
"""
import os, re, statistics, torch
import torch._inductor.config as ic
ic.assert_indirect_indexing = False
import triton, triton.language as tl

dev = "cuda"
N, R = 2_313_190, 2_405_872
g = torch.Generator(device=dev); g.manual_seed(0)
idx = torch.randint(0, R, (N,), device=dev, generator=g)
idx32 = idx.int()
val = torch.rand(N, device=dev, generator=g)
ie = torch.rand(N, device=dev, generator=g) < 0.2
mm = torch.rand(N, device=dev, generator=g) < 0.02
ref = torch.zeros(R, device=dev).scatter_add_(0, idx, val)


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


@triton.jit
def _k(idx_p, val_p, out_p, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    i = tl.load(idx_p + o, mask=m, other=0)
    v = tl.load(val_p + o, mask=m, other=0.0)
    tl.atomic_add(out_p + i, v, mask=m, sem="relaxed")


print(f"torch {torch.__version__} ／ triton {triton.__version__} ／ {torch.cuda.get_device_name(0)}", flush=True)
out = torch.zeros(R, device=dev)
ck = _k[(triton.cdiv(N, 1024),)](idx32, val, out, N, BLOCK=1024)
asm = ck.asm.get("amdgcn", "") if hasattr(ck, "asm") else ""
ops = sorted(set(re.findall(r"(global_atomic_\w+|buffer_atomic_\w+|flat_atomic_\w+)", asm)))
print(f"1 Triton の小数の足し込みの命令：{ops if ops else '（機械語を読めない）'}", flush=True)
print("   ⇒ " + ("足し込みの命令" if any("add_f32" in o for o in ops) else
                ("やり直しの輪（cmpswap）" if any("cmpswap" in o for o in ops) else "わからない")), flush=True)
print(f"   速さ {t(lambda: _k[(triton.cdiv(N, 1024),)](idx32, val, torch.zeros(R, device=dev), N, BLOCK=1024)):.3f}ms", flush=True)

# 2 HIP で直に
SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
__global__ void k_add(const int* idx, const float* val, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) unsafeAtomicAdd(out + idx[i], val[i]);
}
__global__ void k_cas(const int* idx, const float* val, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out + idx[i], val[i]);
}
torch::Tensor add_u(torch::Tensor idx, torch::Tensor val, int64_t R) {
  auto out = torch::zeros({R}, val.options());
  int n = idx.size(0);
  hipLaunchKernelGGL(k_add, dim3((n + 255) / 256), dim3(256), 0, at::cuda::getCurrentCUDAStream(),
                     idx.data_ptr<int>(), val.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
torch::Tensor add_a(torch::Tensor idx, torch::Tensor val, int64_t R) {
  auto out = torch::zeros({R}, val.options());
  int n = idx.size(0);
  hipLaunchKernelGGL(k_cas, dim3((n + 255) / 256), dim3(256), 0, at::cuda::getCurrentCUDAStream(),
                     idx.data_ptr<int>(), val.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
'''
try:
    from torch.utils.cpp_extension import load_inline
    m = load_inline(name="probe_atomic2_ext", cpp_sources="torch::Tensor add_u(torch::Tensor, torch::Tensor, int64_t);\n"
                    "torch::Tensor add_a(torch::Tensor, torch::Tensor, int64_t);",
                    cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + SRC, functions=["add_u", "add_a"],
                    extra_cuda_cflags=["-O3"], verbose=False)
    for name, f in (("unsafeAtomicAdd（足し込みの命令を必ず使う）", m.add_u), ("atomicAdd（ふつう）", m.add_a)):
        r = f(idx32, val, R)
        print(f"2 HIP {name}：{t(lambda: f(idx32, val, R)):.3f}ms ／ 値 差の最大 {float((r - ref).abs().max()):.1e}", flush=True)
except Exception as ex:          # noqa
    print(f"2 HIP で直に：作れない（{type(ex).__name__}: {str(ex).splitlines()[0][:150]}）", flush=True)
    print(f"   ROCM_HOME={os.environ.get('ROCM_HOME')} ／ /opt/rocm {'あり' if os.path.isdir('/opt/rocm') else 'なし'}", flush=True)

# 3 版81 と同じ形
try:
    import sys
    sys.path.insert(0, ".")
    import torch_group2 as G
    if getattr(G, "_TRITON", False):
        r2 = idx32 // 2
        f3 = lambda: G._scatter3(r2, ie, val, mm, R // 2, True)
        print(f"3 版81 の _scatter3（3つ・印つき）：{t(f3):.3f}ms", flush=True)
        f3b = lambda: G._scatter3(r2, ie, val, mm, R // 2, False)
        print(f"   cnt なし：{t(f3b):.3f}ms", flush=True)
except Exception as ex:          # noqa
    print(f"3 ✗ {type(ex).__name__}: {ex}", flush=True)
