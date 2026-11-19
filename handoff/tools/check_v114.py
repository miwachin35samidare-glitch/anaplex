"""版114 の確かめ（容器・CPU・小さい網）：網の中の興奮の辺の上限を受け手ごとに 6mV（W_E_MAX ÷ 接触）に（組み表_版114_案）
  python3 check_v114.py
  ⚠️ ANAPLEX_WCAP=107z で版113 と同じになることは dump113.py の2回で見る（この道具の外）
"""
import numpy as np, torch
import babble as B
B.DEV = "cpu"
import torch_area as T, torch_brain as TB, torch_group2 as G, brainstem as BS
from torch_brain import Brain, NAMES

ok = True


def ck(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'✓' if cond else '✗'} {name}{extra}", flush=True)


ck("1 VERSION 114・SAME_NET 113（網の形は同じ）・既定は新しい上限", B.VERSION == 114 and B.SAME_NET == (113,) and not G.WCAP_107Z)
x = torch.tensor([0.5, 3.0, 9.0]); o = torch.tensor([0.0, 0.0, 9.0])
c = torch.tensor([3.3, 3.3, 3.3])
got = G._wcap(x, o, c)
ck("2 _wcap：学んで足す分は 6mV（w 1.623）で止まり、もとから上の辺は削らない", torch.allclose(got, torch.tensor([0.5, 6 / 1.12 / 3.3, 9.0])),
   f"：{[round(float(v), 4) for v in got]}")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
ex0 = ~g.e_inh
mv0 = g.w * g.c_exc[g.e_recv.long()] * 1.12
pre_over = int(((mv0 > 6.0 + 1e-4) & ex0).sum())
ck("3 初めから 6mV を越える興奮の辺は無い（初めの重さは人の幅の中）", pre_over == 0, f"：{pre_over}")
ca3 = NAMES.index("CA3")
ck("4 CA3 の接触は 1（上限は 5.36 のまま）", float(g.c_exc[int(g.off_recv[ca3])]) == 1.0)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for k in range(40):
    life.step()
g.flush_all()
er = g.e_recv.long()
mv = g.w * g.c_exc[er] * 1.12
ex = ~g.e_inh
rr = g.rr_send[g.e_src.long()] & g.rr_recv[er]             # 版113 の rr の辺は跡 0 で上限を通らない（下からの線と同じく上限なし）
over = int(((mv > 6.0 + 1e-4) & ex & ~rr).sum())
over_rr = int(((mv > 6.0 + 1e-4) & rr).sum())
at = int(((mv > 5.99) & ex).sum())
ar = g.area_of[er][(mv > 5.99) & ex]
cnt = torch.bincount(ar, minlength=len(NAMES))
top = torch.argsort(-cnt)[:6]
ck("5 40歩のあと：予測誤差で学ぶ網の中の興奮の辺で 6mV を越えるもの 0（版113 の容器では 38,731本）", over == 0,
   f"：rr の辺（上限の外）の 6mV 越え {over_rr} 本 ／ 上限で止まる {at:,} 本（" + "・".join(f"{NAMES[i]} {int(cnt[i]):,}" for i in top if cnt[i]) + "）")
ck("6 NaN なし", not bool(torch.isnan(g.w).any()))
print("\n全部 ✓" if ok else "\n✗ あり")
