"""版107：門 2つの GPU の確かめ（check_v107 の 4 が GPU で ✗ だったので、どこがずれるかを分ける）。率は読まない
  python3 check_v107_gpu.py
  A 網を回して比べる：同じ網を2回（門を同じに）回した差 ＝ GPU の足し込みの順の揺れ。これより「組 0 と門を同じ」の差が大きいかを見る
  B 核を1つずつ比べる：書き換えた Triton の核 3つ（_p1_flush_k・_p1_learn_k・_pass_edges_k）を、torch の式（コンテナで確かめた方）と
    同じ入力で比べる。A は 2列を別の値にし、受け手の組はでたらめ ⇒ 列の選び方が違えば重さがずれる
"""
import sys, math, torch
sys.path.insert(0, ".")
import babble as B, torch_area as T, brainstem as BS
import torch_group2 as G
from torch_brain import Brain
from mouth import N_CMD

assert torch.cuda.is_available(), "GPU で回す"
DEV = "cuda"
B.DEV = DEV
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


_obs = BS.Brainstem.observe


def build(mode):
    def obs(self, b, *a, **k):
        if mode in ("eq", "zero"):
            b.gate = b.gate[:1].expand(2).clone()
        return _obs(self, b, *a, **k)
    BS.Brainstem.observe = obs
    torch.manual_seed(0)
    b = Brain(T.Prior.load("prior_v1.npz", dev=DEV), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
    if mode == "zero":
        b.g.recv_grp.zero_()
    return b


def run(b, n):
    gen = torch.Generator().manual_seed(1)
    for _ in range(n):
        b.step((torch.rand(B.N_EAR, generator=gen) < 0.2).float().to(DEV), torch.zeros(N_CMD, device=DEV), learn=True)
    b.g.flush_all()
    torch.cuda.synchronize()


print("A 網を回して比べる（6歩）", flush=True)
res = {}
for nm in ("eq", "eq2", "zero"):
    b = build("eq" if nm == "eq2" else nm)
    run(b, 6)
    res[nm] = b.g.w.clone()
    del b
    torch.cuda.empty_cache()
d_same = float((res["eq"] - res["eq2"]).abs().max())
d_zero = float((res["zero"] - res["eq"]).abs().max())
print(f"     同じ網を2回：最大の差 {d_same:.4g}（重さが違う辺 {int((res['eq'] != res['eq2']).sum())} 本）", flush=True)
print(f"     組 0 と門を同じ：最大の差 {d_zero:.4g}（重さが違う辺 {int((res['zero'] != res['eq']).sum())} 本）", flush=True)
del res
torch.cuda.empty_cache()

print("B 核を1つずつ比べる", flush=True)
b = build("new")
run(b, 3)
g = b.g
ne_all = len(g.w)
gen = torch.Generator(device=DEV); gen.manual_seed(7)
KC = 20
At = torch.zeros(G.LAZY_EPOCH + 1, 2, dtype=torch.float64, device=DEV)
At[1:KC + 1] = torch.rand(KC, 2, dtype=torch.float64, device=DEV, generator=gen).cumsum(0)
At[1:KC + 1, 1] *= 3.0                                     # 2列を別の値に
rg = (torch.rand(g.n_recv_all, device=DEV, generator=gen) < 0.5).int()   # 受け手の組をでたらめに
a_now = At[KC].clone()
ks0 = torch.randint(1, KC + 1, (ne_all,), device=DEV, generator=gen).to(g.k_start.dtype)
el0 = torch.rand(ne_all, device=DEV, generator=gen) * 0.01
w0 = g.w.clone()
amp = torch.ones(ne_all, device=DEV)
kc1 = torch.full((), KC + 1, dtype=g.k_start.dtype, device=DEV)
E = torch.randperm(ne_all, device=DEV, generator=gen)[:min(ne_all, 300000)].sort().values
ne = torch.full((), len(E), dtype=torch.long, device=DEV)
K = G._SC_K


def close(a, b_, tol=1e-5):
    return bool(torch.allclose(a.float(), b_.float(), rtol=tol, atol=1e-6))


# 1 _p1_flush_k ↔ _flush_sum_core
w1, k1, w2, k2 = w0.clone(), ks0.clone(), w0.clone(), ks0.clone()
b0 = torch.zeros(K, g.n_recv_all * 2, device=DEV); m0 = torch.zeros(K, g.n_recv_all, device=DEV)
G._p1_flush_k[(1024,)](E, ne, k1, el0, At, a_now, rg, kc1, w1, g.e_recv, g.e_src, g.e_inh, g.c_exc, amp, g.send_gain,
                       g.mod_send, g.mod_drive_inh, g.is_inh_r, b0, m0, g.n_recv_all, HAS_AMP=True, K=K, BLOCK=256)
bb, mb = G._flush_sum_core(E, k2, el0, At, a_now, w2, kc1, g.e_recv, g.e_src, g.e_inh, g.c_exc, amp, g.send_gain,
                           g.mod_send, g.mod_drive_inh, g.is_inh_r, g.n_recv_all, rg)
torch.cuda.synchronize()
chk(close(w1, w2) and bool((k1 == k2).all()) and close(b0.sum(0), bb, 1e-3), f"_p1_flush_k（重さの最大の差 {float((w1 - w2).abs().max()):.3g}）")

# 2 _p1_learn_k ↔ _learn_sum_core
err = torch.rand(g.n_recv_all, device=DEV, generator=gen) - 0.5
gb = torch.full((), 1e-3, device=DEV); pw = torch.full((), 1.5, device=DEV)
w1, k1, e1, w2, k2, e2 = w0.clone(), ks0.clone(), el0.clone(), w0.clone(), ks0.clone(), el0.clone()
b2 = torch.zeros(K, g.n_recv_all * 2, device=DEV); m2 = torch.zeros(K, g.n_recv_all, device=DEV)
c2 = torch.zeros(K, g.n_recv_all, device=DEV, dtype=torch.int32)
G._p1_learn_k[(1024,)](E, ne, g.e_recv, g.e_src, g.e_inh, g.is_inh_r, err, gb, pw, e1, g.hist_n, w1, g.c_exc, k1,
                       At, a_now, rg, kc1, amp, g.send_gain, g.mod_send, g.mod_drive_inh, b2, m2, c2, g.n_recv_all,
                       G.INH_LTD_N, G.INH_LTD_TH, G.INH_LTD_F, HAS_AMP=True, K=K, BLOCK=256)
cnt, bb, mb = G._learn_sum_core(E, g.e_recv, g.e_src, g.e_inh, g.is_inh_r, err, gb, pw, e2, g.hist_n, w2, g.c_exc, k2,
                                At, a_now, kc1, amp, g.send_gain, g.mod_send, g.mod_drive_inh, g.n_recv_all, rg)
torch.cuda.synchronize()
chk(close(w1, w2) and close(e1, e2) and bool((k1 == k2).all()) and close(b2.sum(0), bb, 1e-3),
    f"_p1_learn_k（重さの最大の差 {float((w1 - w2).abs().max()):.3g}）")

# 3 _pass_edges_k ↔ _edges_core ＋ _flush_vals
W = g._gW
pick = torch.rand(len(W), device=DEV, generator=gen) < 0.3
lens = torch.where(pick, g.start[W + 1] - g.start[W], torch.zeros_like(W))
csum = lens.cumsum(0)
tot = int(csum[-1])
nW = len(W)
w1, k1, w2, k2 = w0.clone(), ks0.clone(), w0.clone(), ks0.clone()
b3 = torch.zeros(K, g.n_recv_all * 2, device=DEV); m3 = torch.zeros(K, g.n_recv_all, device=DEV)
G._pass_edges_k[(512,)](csum, lens, W, g.start, k1, el0, At, a_now, rg, kc1, w1, g.e_recv, g.e_src, g.e_inh, g.c_exc,
                        amp, g.send_gain, g.mod_send, g.mod_drive_inh, g.is_inh_r, b3, m3, nW, g.n_recv_all,
                        HAS_AMP=True, LOG=max(1, int(math.ceil(math.log2(nW + 1)))) + 1, K=K, BLOCK=256)
e, valid = G._edges_core(lens, csum, W, g.start, tot)
w2[e] = G._flush_vals(e, k2, el0, At, a_now, w2, g.e_recv, rg)
k2[e] = kc1
torch.cuda.synchronize()
chk(tot > 0 and close(w1, w2) and bool((k1 == k2).all()), f"_pass_edges_k（辺 {tot} 本・重さの最大の差 {float((w1 - w2).abs().max()):.3g}）")
print("全部 ✓" if ok else "✗ がある", flush=True)
sys.exit(0 if ok else 1)
