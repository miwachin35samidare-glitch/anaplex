"""版103c の確かめ（arkhe・GPU で。保存しない）：速さの直し3つが同じ式か
  python3 check_v103c.py
  1 焼いた素子の見込み：Triton の核（_prior_fast）と前の形（_prior_core）の差（丸めの範囲か）
  2 下書きの枚数 16・見込みの核が使われている
  3 回る（暮らし 60歩）
  下からの表の数え（searchsorted）は、コンテナで版103b と 80歩 ビット一致を確かめた（網の重さ・線・誤差の発火）
"""
import numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
from torch_brain import Brain

ok = True
def chk(c, m):
    global ok
    print(("  ✓ " if c else "  ✗ ") + m, flush=True); ok = ok and bool(c)

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(60):
    life.step()
chk(life.t >= 60, f"暮らし {life.t}歩 回った")
g = b.g
chk(G._SC_K == 16 and G.PRIOR_K and G._TRITON, f"下書き {G._SC_K}枚・見込みの核 {G.PRIOR_K}・Triton {G._TRITON}")
ref = G._prior_core(g.tb, *g._pri_stack, g._pri_segs)
out = G._prior_fast(g.tb, g._pri_seg_id, *g._pri_stack)
d = float((out - ref).abs().max())
chk(d < 1e-4, f"見込み：核と前の形の差 最大 {d:.2e}（受け手 {len(ref):,}・値の大きさ {float(ref.abs().max()):.2f}）")
torch.cuda.synchronize()
import time
for name, f in (("前の形", lambda: G._prior_core(g.tb, *g._pri_stack, g._pri_segs)),
                ("核", lambda: G._prior_fast(g.tb, g._pri_seg_id, *g._pri_stack))):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    print(f"  見込み {name}：{(time.time() - t0) / 20 * 1000:.2f}ms/回", flush=True)
print("✅ 版103c の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
