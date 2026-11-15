"""版110b の確かめ（容器・CPU・小さい網）：同じ状態から、前の刈り込み（版110）と新しい刈り込み（版110b）を回して、全部の配列が同じか
  python3 check_v110b.py <前の torch_group2.py> <前の occlusion.py>
"""
import sys, importlib.util, copy, numpy as np, torch
import babble as B
B.DEV = "cpu"
import brainstem as BS, torch_area as T, torch_group2 as G, occlusion as O, torch_brain as TB
from torch_brain import Brain


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


G0 = load("g2_old", sys.argv[1])
O0 = load("occ_old", sys.argv[2])
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(20):
    life.step(learn=True)
g = b.g
occ = g._occ
assert occ is not None, "ふさがりが無い"
gq = torch.Generator().manual_seed(1)
occ.grp.q = torch.rand(len(occ.grp.q), generator=gq)                 # 痕跡に値を入れておく（動かす値が 0 ばかりだと確かめにならない）
occ.grp.m = torch.rand(len(occ.grp.m), generator=gq).to(occ.grp.m.dtype)
g.rate_long.copy_(torch.rand(len(g.rate_long), generator=gq) * 2 * g.target_recv)   # 静か・撃ちすぎの両方が出るように
G_NAMES = ("e_src", "e_recv", "w", "elig", "k_start", "e_inh", "start")
S_NAMES = ("row", "state", "fac", "last", "deplete")
O_NAMES = ("q", "m", "prev", "idx")


def snap():
    d = {("g", k): getattr(g, k).clone() for k in G_NAMES}
    d.update({("s", k): getattr(g.stp, k).clone() for k in S_NAMES})
    d.update({("o", k): getattr(occ.grp, k).clone() for k in O_NAMES})
    d["gen"] = g.gen.get_state()
    d["rng"] = torch.get_rng_state()
    return d


def restore(d):
    for (w, k), v in ((kk, vv) for kk, vv in d.items() if isinstance(kk, tuple)):
        if w == "g":
            getattr(g, k).copy_(v)
        elif w == "s":
            getattr(g.stp, k).copy_(v)
        else:
            setattr(occ.grp, k, v.clone())
    g.gen.set_state(d["gen"])
    torch.set_rng_state(d["rng"])
    g._amp_buf = None
    g._amp_prev = None


ok = True
for trial in range(3):
    s0 = snap()
    # 前の刈り込み（版110 の文）
    occ.__class__ = O0.Occ
    G0.Group2.rewire(g)
    r_old = snap()
    n_old = g.n_cut
    restore(s0)
    occ.__class__ = O.Occ
    G.Group2.rewire(g)
    r_new = snap()
    cut = g.n_cut - n_old
    bad = [k for k in r_old if isinstance(k, tuple) and not (r_old[k].shape == r_new[k].shape and r_old[k].dtype == r_new[k].dtype and bool((r_old[k] == r_new[k]).all()))]
    moved = int((r_new[("g", "e_src")] != s0[("g", "e_src")]).sum())
    print(f"{'✓' if not bad else '✗'} {trial + 1} 回目：切った辺 {cut} 本・送り元の変わった位置 {moved}・ふさがりの辺 {len(r_new[('o', 'idx')]):,}"
          f"・痕跡 q の 0 でない数 {int((r_new[('o', 'q')] != 0).sum()):,}" + (f"・違う配列 {bad}" if bad else "・全部の配列が同じ"), flush=True)
    ok &= not bad
    st = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(g.e_src, minlength=g.n_send_all).cumsum(0)])
    if not bool((st == g.start).all()):
        print("✗ 始まりの表が数え直しと違う"); ok = False
    for _ in range(3):
        life.step(learn=True)
    occ.grp.q = torch.rand(len(occ.grp.q), generator=gq)
    occ.grp.m = torch.rand(len(occ.grp.m), generator=gq).to(occ.grp.m.dtype)
print("\n全部 ✓" if ok else "\n✗ あり")
