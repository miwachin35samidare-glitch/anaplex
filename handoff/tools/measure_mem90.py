"""今の網（版90〜）の GPU のメモリの山を測る（arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_mem90.py [歩=300]
  （コンテナで道具を確かめるとき：python3 measure_mem90.py 30 --test ⇒ CPU・相手 200。数は読まない）
【み】10-15「メモリ計算してから」⇒ 基底核・小脳を足す余裕を見る。前の測りは版67 の 10.35GB（版79〜90 のグラフ・下書きの後は測っていない）
出すもの：GPU 全体 ／ 組んだ直後 ／ 慣らし後 ／ 歩を回したあとの山（使った・取り置き）／ 網の辺の数 ／ 1辺あたり
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
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


def mem(tag):
    if not gpu:
        print(f"  {tag}（CPU）", flush=True)
        return
    torch.cuda.synchronize()
    fr, to = torch.cuda.mem_get_info()
    print(f"  {tag}：GPU から見た使用（ほかのプロセス・HIP の土台・核のコードも含む）{(to - fr) / GB:6.2f}GB", flush=True)
    print(f"  {tag}：使用 {torch.cuda.memory_allocated() / GB:6.2f}GB ／ 取り置き {torch.cuda.memory_reserved() / GB:6.2f}GB ／ "
          f"山 使用 {torch.cuda.max_memory_allocated() / GB:6.2f}GB・取り置き {torch.cuda.max_memory_reserved() / GB:6.2f}GB", flush=True)


print(f"版 {B.VERSION} ／ torch {torch.__version__} ／ {torch.cuda.get_device_name(0) if gpu else 'CPU'}", flush=True)
if gpu:
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU 全体 {total / GB:.2f}GB ／ 始める前の空き {free / GB:.2f}GB", flush=True)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
ne = len(b.g.w)
print(f"体 {b.n_cells:,} ／ 網の辺 {ne:,}", flush=True)
mem("組んだ直後")
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
up = (lambda x: TB.to_dev(x, dev)) if hasattr(TB, "to_dev") else (lambda x: torch.tensor(x, device=dev))
n = 0
while n < NS:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(min(len(ear), NS - n)):
        b.step(up(np.asarray(ear[t], np.float32)), up(np.asarray(cmd[t], np.float32)), learn=True)
    n += len(ear)
    if n >= 30 and n - len(ear) < 30:
        mem("慣らし後")
mem(f"{NS}歩のあと")
if gpu:
    print(f"  1辺あたり（山の取り置き ÷ 網の辺）{torch.cuda.max_memory_reserved() / ne:.0f}バイト", flush=True)
