"""版91 の前に：刈り込みの「受け手ごとに一番小さい辺」を、前（版90・並べ替え）と後（版91・塊ごと）で選んだ辺が全部同じか（arkhe で。版90 のまま回す）
  python3 check_v91_gpu.py [歩=300]
  （コンテナ：python3 check_v91_gpu.py 30 --test ⇒ CPU・相手 200）
【み】10-15「予算12gb」⇒ 版91。同じ値が並ぶとき前は「並べ替えで前に来た辺」⇒ GPU の並べ替えが番号順に安定なら同じになるはず。それを確かめる
比べる key（rewire と同じ作り方）
  1 一番弱い興奮：where(抑制, inf, w)
  2 でたらめな抑制1本：where(抑制, 乱数, inf)
  3 同じ値をわざと増やした興奮：where(抑制, inf, round(w × 10) ／ 10)（同じ値の並びの扱いを強く試す）
  4 3 の 0 の半分を −0.0 に
出すもの：選んだ辺が違う受け手の数 ／ どちらも −1 の数 ／ GPU のメモリの山（前・後）
"""
import sys, types, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
GB = 1024 ** 3
if B.VERSION != 90:
    sys.exit(f"⚠️ babble.py が版90 ではない（版 {B.VERSION}）。版91 を当てる前に回す")


def new_first(self, key):
    """apply_v91 と同じ式"""
    n, m, step = self.n_recv_all, len(key), 1 << 24
    mn = torch.full((n,), float("inf"), dtype=key.dtype, device=self.dev)
    for a in range(0, m, step):
        b = min(m, a + step)
        mn.scatter_reduce_(0, self.e_recv[a:b].long(), key[a:b], reduce="amin")
    big = torch.iinfo(torch.long).max
    pick = torch.full((n,), big, dtype=torch.long, device=self.dev)
    for a in range(0, m, step):
        b = min(m, a + step)
        r = self.e_recv[a:b].long()
        k = key[a:b]
        hit = (k == mn[r]) & torch.isfinite(k)
        idx = torch.arange(a, b, device=self.dev)
        pick.scatter_reduce_(0, r[hit], idx[hit], reduce="amin")
    pick[pick == big] = -1
    return pick


torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
g = b.g
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
up = (lambda x: TB.to_dev(x, dev)) if hasattr(TB, "to_dev") else (lambda x: torch.tensor(x, device=dev))
n = 0
while n < NS:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(min(len(ear), NS - n)):
        b.step(up(np.asarray(ear[t], np.float32)), up(np.asarray(cmd[t], np.float32)), learn=True)
    n += len(ear)
g.flush_all()
print(f"版 {B.VERSION} ／ {'GPU' if gpu else 'CPU'} ／ 辺 {len(g.w):,} ／ 受け手 {g.n_recv_all:,} ／ {NS}歩のあと", flush=True)
with torch.no_grad():
    inh = g.e_inh
    INF = float("inf")

    def k1():
        return torch.where(inh, INF, g.w)

    def k2():
        gen = torch.Generator(device=dev); gen.manual_seed(1)
        return torch.where(inh, torch.rand(len(g.w), device=dev, generator=gen), INF)

    def k3():
        return torch.where(inh, INF, torch.round(g.w * 10) / 10)

    def k4():
        gen = torch.Generator(device=dev); gen.manual_seed(2)
        wr = torch.round(g.w * 10) / 10
        neg = (wr == 0) & (torch.rand(len(wr), device=dev, generator=gen) < 0.5)
        wr[neg] = -0.0
        return torch.where(inh, INF, wr)

    keys = {"1 一番弱い興奮": k1, "2 でたらめな抑制": k2, "3 同じ値を増やした": k3, "4 3 ＋ −0.0": k4}
    old_f = G.Group2._first_per_recv
    bad_all = 0
    for name, mk in keys.items():
        key = mk()
        nties = int(len(key) - torch.unique(key[torch.isfinite(key)]).numel() - int((~torch.isfinite(key)).sum()))
        res = {}
        for tag, f in (("前", old_f), ("後", new_first)):
            if gpu:
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_allocated()
            p = f(g, key)
            if gpu:
                torch.cuda.synchronize()
                res[tag + "山"] = (torch.cuda.max_memory_allocated() - m0) / GB
            res[tag] = p
        d = int((res["前"] != res["後"]).sum())
        bad_all += d
        neg1 = int(((res["前"] == -1) & (res["後"] == -1)).sum())
        mem = f" ／ 山の増え 前 {res['前山']:.2f}GB・後 {res['後山']:.2f}GB" if gpu else ""
        print(f"  {name}：選んだ辺が違う受け手 {d:,} ／ どちらも −1 {neg1:,} ／ 同じ値で重なった辺 {nties:,}{mem}", flush=True)
        if d:
            i = torch.nonzero(res["前"] != res["後"])[:3].flatten()
            for r in i.tolist():
                a, c = int(res["前"][r]), int(res["後"][r])
                ka = float(key[a]) if a >= 0 else None
                kc = float(key[c]) if c >= 0 else None
                print(f"     受け手 {r}：前 辺 {a}（key {ka!r}）／ 後 辺 {c}（key {kc!r}）", flush=True)
        del key, res
print("✅ 全部同じ ⇒ apply_v91.py を当ててよい" if bad_all == 0 else "⚠️ 違いあり ⇒ 当てずに結果を Claude へ", flush=True)
