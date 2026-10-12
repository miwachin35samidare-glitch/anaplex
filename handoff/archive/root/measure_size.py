"""領野ごとの大きさとメモリの内訳（arkhe の ~/anaplex で  python3 measure_size.py [比] ）⚠️ 保存しない

12GB に収める割り振りを決めるための道具（09-28【み】「12GB に収まるよう全体の体数を調節」）
  比：比で縮める側（連合・高次・前頭前野）に掛ける倍率。既定 1.0（版43a の 240・263・946）
      例 python3 measure_size.py 0.7 ⇒ 168・184・662。下限の側（海馬・報酬の輪・視床・聴覚・運動・視覚）は変えない
出すもの
  領野ごと  入口の数・受け手の数・辺の数・辺の割合
  網の中の大きな配列（上位15）と合計
  GPU  確保の山（allocated）／ 取り置きの山（reserved：実際に GPU から取った量）／ 端末全体の空き
       組んだ直後 ／ 喃語3回（学びあり）と刈り込み1回の後
  ⟲ 09-29 1時：前は allocated だけ出していて、11.99GB と出たのに実際は 16GB を使い切って溢れた【み】
"""
import sys
import numpy as np
import torch
import torch_area as T
import babble as B
import brainstem as BS
import torch_brain as TB
from torch_brain import Brain, NAMES

r = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
TB.N_NEW["前頭前野"] = int(round(946 * r))
n_assoc, n_high = int(round(240 * r)), int(round(263 * r))
GB = 2 ** 30


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  {tag}: 確保の山 {torch.cuda.max_memory_allocated() / GB:.2f}GB ／ 取り置きの山 "
          f"{torch.cuda.max_memory_reserved() / GB:.2f}GB ／ 端末の空き {free / GB:.2f} / {total / GB:.2f}GB", flush=True)


free0, total = torch.cuda.mem_get_info()
print(f"比 {r}（連合 {n_assoc}・高次 {n_high}・前頭前野 {TB.N_NEW['前頭前野']}）  組む前の端末の空き {free0 / GB:.2f} / {total / GB:.2f}GB"
      f"（ほかのプログラムが {(total - free0) / GB:.2f}GB 使っている）", flush=True)
torch.cuda.reset_peak_memory_stats()
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV, n_assoc=n_assoc, n_high=n_high)
b.bs.set_state(BS.WAKE)
g = b.g
torch.cuda.synchronize()
print("GPU")
mem("組んだ直後")

off = g.off_recv.cpu().numpy()
er = g.e_recv
tot_e = len(er)
print(f"体 {b.n_cells:,}  辺 {tot_e:,}")
print(f"{'領野':6s} {'入口':>6s} {'受け手':>8s} {'辺':>13s} {'辺の割合':>8s}")
for i, nm in enumerate(NAMES):
    lo, hi = int(off[i]), int(off[i + 1])
    ne = int(((er >= lo) & (er < hi)).sum())
    print(f"{nm:6s} {b.n[nm]:6d} {hi - lo:8d} {ne:13,d} {ne / tot_e:8.1%}")

sizes = []
for k, v in vars(g).items():
    if torch.is_tensor(v) and v.is_cuda:
        sizes.append((v.numel() * v.element_size(), k, tuple(v.shape), str(v.dtype)))
stp = getattr(g, "stp", None)
if stp is not None:
    for k, v in vars(stp).items():
        if torch.is_tensor(v) and v.is_cuda:
            sizes.append((v.numel() * v.element_size(), "stp." + k, tuple(v.shape), str(v.dtype)))
sizes.sort(reverse=True)
print("\n網の中の大きな配列（上位15）")
for nb, k, sh, dt in sizes[:15]:
    print(f"  {k:20s} {nb / GB:7.3f} GB  {sh}  {dt}")
print(f"  合計 {sum(s[0] for s in sizes) / GB:.2f} GB（網の配列だけ）")

from voice import Voice
from mouth import Mouth
vo, mo = Voice(), Mouth()
rng = np.random.default_rng(0)
for _ in range(3):
    morae, vol = vo.random_morae(rng)
    mel = vo.say(morae, vol)
    B.run(b, vo.cmd_stream(morae, vol, len(mel), mo), vo.encode(mel), True, True)
g.rewire()                               # 刈り込みの一時的な分も必ず山に入れる
torch.cuda.synchronize()
print("\nGPU")
mem("喃語3回と刈り込みの後")
print(f"⇒ 12GB の目安：取り置きの山 ＋ ほかのプログラム ≦ 端末の {total / GB:.1f}GB。取り置きの山は 12GB 以下に", flush=True)
