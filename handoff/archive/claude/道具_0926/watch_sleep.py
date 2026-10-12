"""版39b の眠りで、下方調整がどう動くかを見る（arkhe の ~/anaplex で  python3 watch_sleep.py [歩数] [何歩ごと] ）

state.pt（喃語のあと）を読んで眠らせる。⚠️ 保存しない（state.pt は書き換えない）
既定は1回の眠りの長さ（脳幹の SLEEP_STEPS、10万歩）を 5000歩ごとに出す。短く見るなら  python3 watch_sleep.py 20000 2000
出すもの（領野ごと）
  発火率 ÷ 目標   表現／誤差／抑制。目標は版39b の受け手ごとの【ヒト】の値。下げない受け手は「－」
  scale          表現／誤差／抑制ごとの平均と、領野の中の最小（0 へ向かって下がり続けるかを見る）
見どころ（コードから出る見込み。確かめるのはこれ）
  皮質の誤差    scale を下げると p が下がり、的が1の歩で誤差が増える ⇒ 率は下がらず scale だけ下がるかもしれない
  視床の表現    下からの入力で撃つ（眠りは耳に0）。目標 20% より低ければ scale は上がる側
  抑制          p > 0.5 で撃つので、scale を下げれば減るはず
"""
import sys
import time
import numpy as np
import torch
import torch_area as T
import babble as B
import brainstem as BS
import sleep as S
from torch_brain import Brain, NAMES

DEV = "cuda"
n_step = int(sys.argv[1]) if len(sys.argv) > 1 else BS.SLEEP_STEPS
every = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
prior = T.Prior.load("prior_v1.npz", dev=DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
done, log = B.load(b)
g = b.g
if not hasattr(g, "target_recv"):
    sys.exit("⚠️ 版39b になっていない（先に apply_v39b.py）")
hip = "is_hippo, torch.ones_like" in open("torch_group2.py", encoding="utf-8").read()
print(f"喃語 {done} 回の網（版 {B.VERSION}）を {n_step} 歩 眠らせる（保存しない）  海馬を下げない行: {'ある' if hip else '無い'}",
      flush=True)


def show(k, st, dt):
    print(f"  ── 眠り {k} 歩（{st}）  1歩 {dt*1000:.0f}ms", flush=True)
    print(f"    {'領野':4} {'率÷目標 表現／誤差／抑制':>24} {'scale 表現／誤差／抑制':>24} {'最小':>6}", flush=True)
    for i, nm in enumerate(NAMES):
        lo, hi, nc = int(g.off_recv[i]), int(g.off_recv[i + 1]), b.areas[nm].n_ch
        parts = (slice(lo + nc, lo + 2 * nc), slice(lo, lo + nc), slice(lo + 2 * nc, hi))   # 表現・誤差・抑制
        r, s = [], []
        for sl in parts:
            if bool(g.scale_skip[sl].all()):
                r.append("    －")
            else:
                r.append(f"{float((g.rate_ema[sl] / g.target_recv[sl]).mean()):6.2f}")
            s.append(f"{float(g.scale[sl].mean()):6.3f}")
        print(f"    {nm:4} {'／'.join(r):>24} {'／'.join(s):>24} {float(g.scale[lo:hi].min()):6.3f}", flush=True)


show(0, b.bs.state, 0.0)
orig = b.step
cnt = [0, time.time()]


def step(*a, **kw):
    out = orig(*a, **kw)
    cnt[0] += 1
    if cnt[0] % every == 0:
        dt = (time.time() - cnt[1]) / every
        cnt[1] = time.time()
        show(cnt[0], b.bs.state, dt)
    return out


b.step = step
S.sleep(b, n_step=n_step, verbose=False)
print(f"眠った歩数 {b.bs.last_slept}（{'脳幹が途中で起こした' if b.bs.last_slept < n_step else '最後まで'}）", flush=True)
if cnt[0] % every:
    show(cnt[0], "終わり", (time.time() - cnt[1]) / (cnt[0] % every))
