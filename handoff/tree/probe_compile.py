"""版76 の torch.compile が GPU で止まった訳を分ける（arkhe の ~/anaplex で。網は使わない。ファイルは書き換えない。数分）
  python3 probe_compile.py
【測】10-14 measure_speed75（版76）：_flush_core の Triton の核を「速さを比べて選ぶ」ところで起動に失敗（driver error 401 ＝ この状態ではできない）
  そのあと GPU が使えない状態になり、まとめない形にも戻れなかった ⇒ 1つずつ別のプロセスで試す
試すもの（版76 の3つの関数と同じ式・同じ型。大きさは網に近い）
  A そのまま（dynamic）
  B 核の起動を Triton 側の起動に（use_static_cuda_launcher を外す）
  C B ＋ 速さを比べて選ぶのをやめる（autotune_pointwise を外す）
  D 大きさを決めて作る（dynamic を外す。大きさが変わるたびに作り直す）
それぞれ：動くか ／ まとめない形と同じ値か ／ 速さ
"""
import subprocess, sys

CODE = r'''
import sys, time, torch
v = sys.argv[1]
import torch._inductor.config as cfg
if v in ("B", "C"):
    cfg.use_static_cuda_launcher = False
if v == "C":
    cfg.triton.autotune_pointwise = False
dyn = v != "D"
dev = "cuda"
g = torch.Generator(device=dev); g.manual_seed(0)
NE, NR, NS = 8_000_000, 400_000, 1_200_000
w = torch.rand(NE, device=dev, generator=g); elig = torch.rand(NE, device=dev, generator=g) * 1e-3
k_start = torch.randint(1, 50, (NE,), device=dev, generator=g).to(torch.int16)
A = torch.rand(129, device=dev, generator=g, dtype=torch.float64).cumsum(0)
e_recv = torch.randint(0, NR, (NE,), device=dev, generator=g).int(); e_src = torch.randint(0, NS, (NE,), device=dev, generator=g).int()
e_inh = torch.rand(NE, device=dev, generator=g) < 0.2
c_exc = torch.full((NR,), 3.3, device=dev); is_inh_r = torch.rand(NR, device=dev, generator=g) < 0.2
send_gain = torch.ones(NS, device=dev); mod_send = torch.rand(NS, device=dev, generator=g) < 0.1; mod_drive_inh = mod_send & (torch.rand(NS, device=dev, generator=g) < 0.5)
amp_s = torch.rand(NE, device=dev, generator=g); hist_n = torch.randint(0, 10, (NS,), device=dev, generator=g).to(torch.int16)
err_sig = torch.randn(NR, device=dev, generator=g)
import torch_group2 as G
fns = [("_flush_core", G._flush_core), ("_sum_core", G._sum_core), ("_learn_core", G._learn_core)]
for name, fn in fns:
    cf = torch.compile(fn, dynamic=dyn)
    try:
        for n in (1_100_000, 900_000, 1_300_000):
            idx = torch.randperm(NE, device=dev, generator=g)[:n].sort().values
            if name == "_flush_core":
                w1, w2 = w.clone(), w.clone()
                fn(idx, k_start, elig, A, A[60], w1); cf(idx, k_start, elig, A, A[60], w2); torch.cuda.synchronize()
                same = float((w1 - w2).abs().max())
                args = (idx, k_start, elig, A, A[60], w2)
            elif name == "_sum_core":
                a1 = fn(idx, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r, NR)
                a2 = cf(idx, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r, NR)
                torch.cuda.synchronize(); same = max(float((x - y).abs().max() / (x.abs().max() + 1e-9)) for x, y in zip(a1, a2))
                args = (idx, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r, NR)
            else:
                g_ = torch.tensor(0.3, device=dev); pw = torch.tensor(1.01, device=dev)
                el1, el2, w1, w2 = elig.clone(), elig.clone(), w.clone(), w.clone()
                c1 = fn(idx, e_recv, e_src, e_inh, is_inh_r, err_sig, g_, pw, el1, hist_n, w1, c_exc, NR)
                c2 = cf(idx, e_recv, e_src, e_inh, is_inh_r, err_sig, g_, pw, el2, hist_n, w2, c_exc, NR)
                torch.cuda.synchronize(); same = max(float((el1 - el2).abs().max()), float((w1 - w2).abs().max()), float((c1 - c2).abs().max()))
                args = (idx, e_recv, e_src, e_inh, is_inh_r, err_sig, g_, pw, el2, hist_n, w2, c_exc, NR)
        def t(f):
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20): f(*args)
            torch.cuda.synchronize(); return (time.time() - t0) / 20 * 1000
        print(f"  {v} {name}：○ 動く ／ 差 {same:.1e} ／ まとめない {t(fn):.2f}ms → まとめた {t(cf):.2f}ms", flush=True)
    except Exception as ex:
        print(f"  {v} {name}：✗ {type(ex).__name__}: {str(ex).splitlines()[0][:150]}", flush=True)
        break
'''

print("torch.compile の試し（版76 の3つの関数）", flush=True)
for v in ("A", "B", "C", "D"):
    r = subprocess.run([sys.executable, "-c", CODE, v], capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.strip()]
    print("\n".join(out) if out else f"  {v}：出力なし（止まった）", flush=True)
    if r.returncode != 0 and not out:
        err = [l for l in r.stderr.splitlines() if "Error" in l or "error" in l]
        print("    " + (err[-1][:200] if err else "（理由が出ていない）"), flush=True)
