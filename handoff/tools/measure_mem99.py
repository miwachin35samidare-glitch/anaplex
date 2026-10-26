"""版99 の VRAM の山がどこで立つか（arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_mem99.py [歩=1100] [--hist]
  （コンテナで道具を確かめるとき：python3 measure_mem99.py 10 --test ⇒ CPU・相手 200。数は読まない）
1回目（10-26 21:51、--hist あり・60歩）：取り置き 5.55 → 7.36GB（11〜13歩目）で止まり、measure_mem90 の 11.20GB は出なかった
  ⇒ 記録（--hist）を付けると振る舞いが変わるのかを分けるため、2回目は記録なしを既定にして、刈り込み（1000歩ごと）まで回す
出すもの
  1 取り置きが 0.05GB 以上増えた歩・そのときのグラフを撮った回数（g.n_capture）と、刈り込みの歩か
  2 終わりの 取り置き ／ 使っていない取り置きを返したあと（empty_cache）／ 使用 ⇒ 使わず持っている分と、グラフが持つ分を分ける
  3（--hist のとき）100MB 以上を一度に取った所（大きい順 20）
"""
import sys, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
HIST = "--hist" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 1100
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
GB, MB = 1024 ** 3, 1024 ** 2
if gpu and HIST:
    torch.cuda.memory._record_memory_history(max_entries=200000)

print(f"版 {B.VERSION} ／ torch {torch.__version__} ／ {torch.cuda.get_device_name(0) if gpu else 'CPU'} ／ 記録 {'あり' if HIST else 'なし'}", flush=True)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
g = b.g
print(f"体 {b.n_cells:,} ／ 網の辺 {len(g.w):,} ／ 刈り込み {g.rewire_every}歩ごと", flush=True)


def now():
    if not gpu:
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / GB, torch.cuda.memory_allocated() / GB, torch.cuda.max_memory_reserved() / GB


r0, a0, _ = now()
print(f"1 組んだ直後：取り置き {r0:.2f}GB ／ 使用 {a0:.2f}GB", flush=True)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
up = lambda x: TB.to_dev(x, dev)
n, last = 0, r0
while n < NS:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(min(len(ear), NS - n)):
        cap0 = getattr(g, "n_capture", 0)
        b.step(up(np.asarray(ear[t], np.float32)), up(np.asarray(cmd[t], np.float32)), learn=True)
        n += 1
        r, a, pk = now()
        if r - last > 0.05:
            rw = "・刈り込みの歩" if g.rewire_every and g.t % g.rewire_every == 0 else ""
            print(f"  {n:4d}歩目：取り置き {last:.2f} → {r:.2f}GB（+{r - last:.2f}）／ 使用 {a:.2f} ／ "
                  f"グラフを撮った {getattr(g, 'n_capture', 0)}回（この歩 +{getattr(g, 'n_capture', 0) - cap0}）{rw}", flush=True)
            last = r
r, a, pk = now()
print(f"2 {n}歩のあと：取り置き {r:.2f}GB ／ 使用 {a:.2f}GB ／ 取り置きの山 {pk:.2f}GB ／ 撮ったグラフ {len(getattr(g, '_graphs', {}))}つ（{getattr(g, 'n_capture', 0)}回）", flush=True)
if gpu:
    torch.cuda.empty_cache()
    r2, a2, _ = now()
    print(f"  使っていない取り置きを返したあと：取り置き {r2:.2f}GB（返せた {r - r2:.2f}GB）／ 使用 {a2:.2f}GB ⇒ グラフなどが持つ分 {r2 - a2:.2f}GB", flush=True)

if gpu and HIST:
    snap = torch.cuda.memory._snapshot()
    big, where = collections.Counter(), {}
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
    print("3 100MB 以上を一度に取った所（大きい順）", flush=True)
    for (act, k), (sz, fr) in sorted(where.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"  {sz / MB:7.0f}MB × {big[(act, k)]}回  {'区画' if act == 'segment_alloc' else '使用'}  {' ← '.join(fr) or k}", flush=True)
    torch.cuda.memory._record_memory_history(enabled=None)
