"""版99 の VRAM の山がどこで立つか（arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_mem99.py [歩=60]
  （コンテナで道具を確かめるとき：python3 measure_mem99.py 10 --test ⇒ CPU・相手 200。数は読まない）
版99 の measure_mem90：山の取り置き 11.20GB（版98 7.64GB）・慣らしで 6.52 → 11.20GB。measure_life97 で 979MB を取れずに OOM の警告
⇒ どの歩・どの行で大きく取るかを出す（数値・網は変えない）
出すもの
  1 歩ごとに取り置き・使用が増えた歩
  2 100MB 以上を一度に取った所（大きい順 20、torch_brain・torch_group2・torch_area の行）
"""
import sys, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 60
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
GB, MB = 1024 ** 3, 1024 ** 2
hist = False
if gpu:
    try:
        torch.cuda.memory._record_memory_history(max_entries=200000)
        hist = True
    except Exception as e:
        print(f"⚠️ 取った所の記録が使えない（{e}）⇒ 歩ごとの増えだけ出す", flush=True)

print(f"版 {B.VERSION} ／ torch {torch.__version__} ／ {torch.cuda.get_device_name(0) if gpu else 'CPU'}", flush=True)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
print(f"体 {b.n_cells:,} ／ 網の辺 {len(b.g.w):,}", flush=True)


def now():
    if not gpu:
        return 0.0, 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / GB, torch.cuda.max_memory_allocated() / GB


r0, a0 = now()
print(f"1 組んだ直後：取り置き {r0:.2f}GB ／ 山 使用 {a0:.2f}GB", flush=True)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
up = lambda x: TB.to_dev(x, dev)
n, last = 0, r0
while n < NS:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(min(len(ear), NS - n)):
        b.step(up(np.asarray(ear[t], np.float32)), up(np.asarray(cmd[t], np.float32)), learn=True)
        n += 1
        r, a = now()
        if r - last > 0.05:
            print(f"  {n:4d}歩目：取り置き {last:.2f} → {r:.2f}GB（+{r - last:.2f}）／ 山 使用 {a:.2f}GB", flush=True)
            last = r
r, a = now()
print(f"  {n}歩のあと：取り置き {r:.2f}GB ／ 山 使用 {a:.2f}GB", flush=True)

if hist:
    snap = torch.cuda.memory._snapshot()
    big = collections.Counter()
    where = {}
    for tr in snap.get("device_traces", []):
        for e in tr:
            if e.get("action") not in ("alloc", "segment_alloc") or e.get("size", 0) < 100 * MB:
                continue
            fr = [f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in e.get("frames", [])
                  if any(k in f["filename"] for k in ("torch_brain", "torch_group2", "torch_area", "babble"))]
            key = (e["action"], fr[0] if fr else "(ほかの所)")
            big[key] += 1
            if e["size"] > where.get(key, (0, None))[0]:
                where[key] = (e["size"], fr[:3])
    print("2 100MB 以上を一度に取った所（大きい順）", flush=True)
    for (act, k), (sz, fr) in sorted(where.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"  {sz / MB:7.0f}MB × {big[(act, k)]}回  {'区画' if act == 'segment_alloc' else '使用'}  {' ← '.join(fr) or k}", flush=True)
    torch.cuda.memory._record_memory_history(enabled=None)
