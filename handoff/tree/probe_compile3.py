"""torch.compile の「番号で引く」核を、番号の範囲の確かめ（核の中の assert）を外して試す（arkhe の ~/anaplex で。網は使わない。数分）
  python3 probe_compile3.py
【測】probe_compile2（10-14）：番号で引く・書く計算は全部 ✗（driver error 401）、番号なしの 64bit の計算は ○
【見】核に入る「番号が外に出ていないか」の assert が、RX 9070 XT で起動できない ⇒ assert_indirect_indexing を外す（値は変わらない）
  あわせて dynamic（大きさが変わっても組み直さない）と、大きさを決める形の両方を見る
【測】probe_compile（10-14）：版76 の _flush_core は A〜D の4つとも起動できない（driver error 401）
  probe_gpu2（10-14）：32bit の小数の単純な計算の列は動く（9.6倍）
⇒ _flush_core の中の計算を1つずつ、別のプロセスで試す（1つが GPU を壊しても次に響かない）
"""
import subprocess, sys

CASES = {
    "1 32bit 小数を番号で引いて足す":        "lambda idx: (w[idx] + elig[idx]).clamp(min=0.0)",
    "2 16bit 整数を番号で引く":              "lambda idx: k_start[idx].long() + 1",
    "3 64bit 小数の足し算（番号なし）":       "lambda idx: A * 2.0 + 1.0",
    "4 64bit 小数を番号で引く":              "lambda idx: A[k_start[idx].long() - 1]",
    "5 64bit → 32bit に直す":               "lambda idx: (A[60] - A[k_start[idx].long() - 1]).float()",
    "6 重さへ書き戻す（番号で書く）":          "lambda idx: w2.index_put_((idx,), w2[idx] * 1.5)",
    "7 _flush_core そのもの":               "lambda idx: G._flush_core(idx, k_start, elig, A, A[60], w2)",
    "8 _sum_core そのもの":                 "lambda idx: G._sum_core(idx, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r, NR)",
    "9 _learn_core そのもの":               "lambda idx: G._learn_core(idx, e_recv, e_src, e_inh, is_inh_r, err_sig, g_, pw, elig2, hist_n, w2, c_exc, NR)",
}

CODE = r'''
import sys, torch
import torch._inductor.config as cfg
cfg.assert_indirect_indexing = False
dev = "cuda"
g = torch.Generator(device=dev); g.manual_seed(0)
NE, NR, NS = 8_000_000, 400_000, 1_200_000
w = torch.rand(NE, device=dev, generator=g); w2 = w.clone(); elig = torch.rand(NE, device=dev, generator=g) * 1e-3; elig2 = elig.clone()
k_start = torch.randint(1, 50, (NE,), device=dev, generator=g).to(torch.int16)
A = torch.rand(129, device=dev, generator=g, dtype=torch.float64).cumsum(0)
e_recv = torch.randint(0, NR, (NE,), device=dev, generator=g).int(); e_src = torch.randint(0, NS, (NE,), device=dev, generator=g).int()
e_inh = torch.rand(NE, device=dev, generator=g) < 0.2
c_exc = torch.full((NR,), 3.3, device=dev); is_inh_r = torch.rand(NR, device=dev, generator=g) < 0.2
send_gain = torch.ones(NS, device=dev); mod_send = torch.rand(NS, device=dev, generator=g) < 0.1; mod_drive_inh = mod_send & (torch.rand(NS, device=dev, generator=g) < 0.5)
amp_s = torch.rand(NE, device=dev, generator=g); hist_n = torch.randint(0, 10, (NS,), device=dev, generator=g).to(torch.int16)
err_sig = torch.randn(NR, device=dev, generator=g); g_ = torch.tensor(0.3, device=dev); pw = torch.tensor(1.01, device=dev)
import torch_group2 as G
f = eval(sys.argv[2])
cf = torch.compile(f, dynamic=(sys.argv[3] == "1"))
try:
    for n in (1_100_000, 900_000):
        idx = torch.randperm(NE, device=dev, generator=g)[:n].sort().values
        cf(idx); torch.cuda.synchronize()
    print("○", flush=True)
except Exception as ex:
    print(f"✗ {type(ex).__name__}: {str(ex).splitlines()[0][:120]}", flush=True)
'''

print("assert を外して（左：dynamic ／ 右：大きさを決める）", flush=True)
for name, expr in CASES.items():
    res = []
    for dyn in ("1", "0"):
        r = subprocess.run([sys.executable, "-c", CODE, name, expr, dyn], capture_output=True, text=True)
        out = [l for l in r.stdout.splitlines() if l.strip()]
        res.append(out[-1] if out else "出力なし")
    print(f"  {name}：{res[0]} ／ {res[1]}", flush=True)
