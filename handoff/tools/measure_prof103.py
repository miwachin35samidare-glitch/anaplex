"""版103 の速さの段：1歩の中で GPU の時間をどこが使っているか（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_prof103.py [歩=20]
  ① まず measure_speed98.py（部分ごとの内訳。版103 で回せる）を回し、② これで網の領野（Group2）の中を見る
  HIP グラフの中は1つの命令に見えるので、ここではグラフを撮らずに回す（ANAPLEX_GRAPH=eager：グラフの形のまま撮らない。値は同じ）
    ⇒ 1歩の時間はグラフありより長く出る。見るのは「どの命令が GPU の時間を使うか」の割合
  出すもの
    1 GPU の時間の合計 ／ CPU の時間の合計 ／ 1歩あたりの命令の数（GPU が待っているのか、CPU の呼び出しが律速かを分ける）
    2 GPU の時間の多い命令 上位 30（1歩あたり ms・回数）
    3 呼び出しの多い所（torch_brain・torch_group2 の行ごと、GPU の時間）上位 20
  出力は data_prof103.txt にも書く
"""
import os, sys
os.environ.setdefault("ANAPLEX_GRAPH", "eager")   # 版103e：ANAPLEX_GRAPH=1 python3 measure_prof103.py でグラフありのまま（グラフの中の核が見えるかを見る）
import time, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB
from torch_brain import Brain
from torch.profiler import profile, ProfilerActivity

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 20
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(60):                       # 温め（compile・最初の取り置き）
    life.step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(NS):
    life.step()
torch.cuda.synchronize()
wall = (time.time() - t0) / NS * 1000
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    for _ in range(NS):
        life.step()
    torch.cuda.synchronize()
ka = prof.key_averages()
dev_t = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
from torch.autograd import DeviceType
is_k = lambda e: getattr(e, "device_type", None) == DeviceType.CUDA      # 版103e：核の行だけ（⟲ aten の行と核の行を2回数えていた）
tot_dev = sum(dev_t(e) for e in ka if is_k(e)) / 1000 / NS
tot_cpu = sum(e.self_cpu_time_total for e in ka) / 1000 / NS
n_k = sum(e.count for e in ka if is_k(e)) / NS
out = [f"版 {B.VERSION} ／ {torch.cuda.get_device_name(0)} ／ グラフ {os.environ['ANAPLEX_GRAPH']} ／ 体 {b.n_cells:,}・辺 {len(b.g.w):,}",
       f"1 グラフなしの暮らしの1歩 {wall:.1f}ms ／ GPU の時間の合計 {tot_dev:.1f}ms ／ CPU の時間の合計 {tot_cpu:.1f}ms ／ GPU の命令 {n_k:.0f}回/歩",
       "2 GPU の時間の多い核（1歩あたり。核の行だけ）"]
rows = sorted((e for e in ka if is_k(e)), key=lambda e: -dev_t(e))[:40]
for e in rows:
    out.append(f"  {dev_t(e) / 1000 / NS:7.2f}ms  {e.count / NS:6.1f}回  {e.key[:90]}")
out.append("3 呼んだ所（自分のコードの行）ごとの GPU の時間（1歩あたり）")
by = collections.defaultdict(float)
for e in prof.key_averages(group_by_stack_n=6):
    if dev_t(e) <= 0:
        continue
    st = [s for s in (e.stack or []) if any(k in s for k in ("torch_brain", "torch_group2", "torch_area", "life", "babble"))]
    by[st[0] if st else "(自分のコードの外)"] += dev_t(e)
for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:20]:
    out.append(f"  {v / 1000 / NS:7.2f}ms  {k[:110]}")
txt = "\n".join(out)
print(txt, flush=True)
open("data_prof103.txt", "w").write(txt + "\n")
print("（data_prof103.txt に書いた）", flush=True)
