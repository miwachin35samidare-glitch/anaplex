"""眠りの1歩が何 ms かかり、どこに時間を使っているかを測る（arkhe の ~/anaplex で  python3 measure_sleep.py ）

中身は何も変えない。網を作って、N3 と REM を少しずつ回し、時間を測るだけ（state.pt も書かない）
  python3 measure_sleep.py           0から作った網で測る
  python3 measure_sleep.py --load    state.pt の網で測る（育った網は撃つ数が違うので、かかる時間も違う）
出すもの
  1) N3 ／ REM の1歩の ms（2000歩ずつ）と、10万歩・90万歩に直した時間
  2) 1歩の中の内訳（torch の profiler。上位 20）
"""
import sys
import time
import torch

import torch_area as T
import babble as B
import sleep as S
import brainstem as BS
from torch_brain import Brain

DEV = "cuda"
N = 2000


def sync():
    torch.cuda.synchronize()


prior = T.Prior.load("prior_v1.npz", dev=DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
if "--load" in sys.argv:
    done, _ = B.load(b)
    print(f"state.pt の網（喃語 {done} 回）")
else:
    print("0から作った網")
print(f"体 {b.n_cells:,}  辺 {len(b.g.w):,}  周回 {b.g.recur}", flush=True)

# 眠りの関数をそのまま使い、N3 と REM を別々に測る（CYCLE を一時的に縮めて、前半 N3・後半 REM にする）
cyc = BS.CYCLE_STEPS
res = {}
for st in (BS.N3, BS.REM):
    BS.CYCLE_STEPS = 2 * N
    k0 = 0 if st == BS.N3 else N
    orig = b.bs.sleep_state
    b.bs.sleep_state = lambda k, k0=k0: orig(k + k0)
    S.sleep(b, n_step=50, verbose=False)            # 温める
    sync(); t0 = time.time()
    S.sleep(b, n_step=N, verbose=False)
    sync(); res[st] = (time.time() - t0) / N * 1000
    b.bs.sleep_state = orig
    BS.CYCLE_STEPS = cyc
    print(f"{st:4s}  1歩 {res[st]:.2f} ms", flush=True)

ms = (res[BS.N3] + res[BS.REM]) / 2
print(f"半分ずつなら 10万歩 {ms * 1e5 / 1000 / 60:.0f}分 ／ 90万歩 {ms * 9e5 / 1000 / 3600:.1f}時間", flush=True)

# 1歩のまわりの手間（眠りの輪の外側）と、b.step の中身
zero_ear = torch.zeros(b.n["聴覚"], device=DEV)
zero_cmd = torch.zeros(b.n["運動"], device=DEV)
b.bs.set_state(BS.N3)
sync(); t0 = time.time()
for _ in range(N):
    b.step(zero_ear, zero_cmd, learn=True)
sync(); step_only = (time.time() - t0) / N * 1000
print(f"b.step だけ（輪の外の手間なし） 1歩 {step_only:.2f} ms", flush=True)

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(200):
        b.step(zero_ear, zero_cmd, learn=True)
    sync()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
b.bs.set_state(BS.WAKE)
