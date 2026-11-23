"""版112 の確かめ（容器・CPU・小さい網）：CL 256・別々の体から 40本・視床網様核の抑制の体なし・橋核 5%。ほかの抑制の体は版111 のまま
  python3 check_v112.py
"""
import numpy as np, torch
import babble as B
B.DEV = "cpu"
import torch_area as T, torch_brain as TB, brainstem as BS
from torch_brain import Brain, NAMES

ok = True


def ck(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'✓' if cond else '✗'} {name}{extra}", flush=True)


ck("1 VERSION 112・SAME_NET なし（state.pt は 0 から）", B.VERSION == 112 and B.SAME_NET == ())
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
nr = len(b.cl_ch)
src = b.cl_src.view(nr, TB.K_CL)
s_ = src.sort(1).values
ck("2 CL は 256 体", b.n["CL"] == 256 and len(b.cl_spk) == 256 and TB.N_NEW["CL"] == 256)
ck("3 皮質の体1つは別々の CL の体から 40 本（重なりなし）", bool((s_[:, 1:] != s_[:, :-1]).all()) and len(b.cl_w) == nr * TB.K_CL,
   f"：受け手 {nr:,}")
use = torch.bincount(b.cl_src, minlength=256)
ck("4 CL の体はどれも使われる", int(use.min()) > 0, f"：1体あたり {int(use.min())}〜{int(use.max())} 本（平均 {nr * TB.K_CL / 256:.0f}）")
ck("5 視床網様核の抑制の体なし", b.areas["視床網様核"].n_inh == 0, f"：体 {b.areas['視床網様核'].n_ch}")
a_ = b.areas["橋核"]
ck("6 橋核の抑制の体は約 5%", a_.n_inh == max(1, int(round(a_.n_ch * 0.05 / 0.95))),
   f"：{a_.n_inh}／{a_.n_ch + a_.n_inh} ＝ {a_.n_inh / (a_.n_ch + a_.n_inh):.1%}")
ex = {"外節", "内節", "境", "網様部"}
fr = {nm: b.areas[nm].n_inh / (b.areas[nm].n_ch + b.areas[nm].n_inh) for nm in ex}
ck("7 外節・内節・境・網様部は 16% のまま（保留）", all(abs(v - 0.16) < 0.01 for v in fr.values()),
   "：" + "・".join(f"{k} {v:.1%}" for k, v in fr.items()))
# 網様核の中：行に来る辺（体の行）が 0 本でも壊れないか・目標
i = NAMES.index("視床網様核")
lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
ck("8 視床網様核の行は 表現・誤差だけ", hi - lo == 2 * b.areas["視床網様核"].n_ch + b.areas["視床網様核"].n_om, f"：{hi - lo} 行")
print("一覧（抑制の体の数）：" + "  ".join(f"{nm} {b.areas[nm].n_inh}" for nm in NAMES if nm in TB.NUC + TB.NC107 + ("橋核", "CL")), flush=True)
# 歩ける・NaN なし
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
n_ok = True
for k in range(300):
    life.step()
    if k % 50 == 49:
        n_ok &= not bool(torch.isnan(g.w).any()) and not bool(torch.isnan(b.cl_w).any())
spk = float(b.cl_spk.sum())
ck("9 300 歩 歩けて重さに NaN なし", n_ok, f"：最後の歩の CL {int(spk)} 体")
print("\n全部 ✓" if ok else "\n✗ あり")
