"""版103b：VRAM を GPU 全体の使用（ドライバの見る量）で測る
  【み】11-01 15:44「まず実使用で10gbに到達している　以前と同じく正確に使用量を測れていない」
  python3 measure_mem103.py [比=1] [歩=1100]      （版103b のまま測るなら引数なし）
  ⟲ 今まで道具が「使用」と書いていたのは torch.cuda.memory_allocated（PyTorch が今持っている箱の合計）で、GPU 全体の使用ではない
     （HIP の文脈・核のコード・グラフの分・ほかの取り方が入らない）。「取り置き」は memory_reserved
  出すもの：GPU 全体の使用 ＝ 全体 − 空き（torch.cuda.mem_get_info ＝ ドライバの値）を毎歩
    0 torch を起こした直後（網を組む前。画面などほかが使っている分）
    1 組んだ直後 ／ 0.1GB 以上増えた歩 ／ 2 終わりと山（毎歩の一番上）
    いっしょに PyTorch の取り置き・箱の合計を出して、GPU 全体との差（PyTorch の外の分）を見る
  ファイルは書き換えない・保存しない（コンテナ：python3 measure_mem103.py 1 10 --test）
"""
import sys, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
HIST = "--hist" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
RATIO = float(args[0]) if args else 1.0
NS = int(args[1]) if len(args) > 1 else 1100
SHR = {nm: max(1, int(round(TB.N_NEW[nm] * RATIO))) for nm in ("前頭前野", "眼窩前頭", "前帯状") + TB.SYN}
TB.N_NEW.update(SHR)                      # 版103 の体数 × 比
n_assoc, n_high = max(1, int(round(112 * RATIO))), max(1, int(round(123 * RATIO)))   # 版103b の 112・123 が 1
if B.VERSION not in (103,):
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
GB, MB = 1024 ** 3, 1024 ** 2
if gpu and HIST:
    torch.cuda.memory._record_memory_history(max_entries=200000)

print(f"版 {B.VERSION} ／ torch {torch.__version__} ／ {torch.cuda.get_device_name(0) if gpu else 'CPU'} ／ 記録 {'あり' if HIST else 'なし'}", flush=True)


def dev_used():
    """GPU 全体の使用 GB（ドライバの値。ほかのプログラム・画面の分も入る）"""
    if not gpu:
        return 0.0, 0.0
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GB, total / GB


if gpu:
    torch.zeros(1, device=dev)
d0, tot = dev_used()
print(f"0 torch を起こした直後：GPU 全体の使用 {d0:.2f}GB ／ 全体 {tot:.2f}GB（網を組む前。ほかのプログラム・画面の分）", flush=True)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev,
          n_assoc=n_assoc, n_high=n_high)
print(f"比 {RATIO}：連合 {b.n['連合']}・高次 {b.n['高次']}・前頭前野 {b.n['前頭前野']}・眼窩前頭 {b.n['眼窩前頭']}・前帯状 {b.n['前帯状']}・"
      + "・".join(f"{k} {b.n[k]}" for k in TB.SYN) + f"（係数 {0.340 * RATIO:.3f}/100万。版103b が比 1）", flush=True)
b.bs.set_state(BS.WAKE)
g = b.g
print(f"体 {b.n_cells:,} ／ 網の辺 {len(g.w):,} ／ 刈り込み {g.rewire_every}歩ごと", flush=True)


def now():
    if not gpu:
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / GB, torch.cuda.memory_allocated() / GB, torch.cuda.max_memory_reserved() / GB


r0, a0, _ = now()
du, _t = dev_used()
print(f"1 組んだ直後：GPU 全体 {du:.2f}GB（網の分 {du - d0:.2f}）／ 取り置き {r0:.2f}GB ／ 箱の合計 {a0:.2f}GB ⇒ PyTorch の外 {du - d0 - r0:.2f}GB", flush=True)
d_last, d_pk, d_pk_at = du, du, 0
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
        du, _t = dev_used()
        if du > d_pk:
            d_pk, d_pk_at = du, n
        if du - d_last > 0.1 or r - last > 0.1:
            rw = "・刈り込みの歩" if g.rewire_every and g.t % g.rewire_every == 0 else ""
            print(f"  {n:4d}歩目：GPU 全体 {d_last:.2f} → {du:.2f}GB ／ 取り置き {r:.2f} ／ 箱 {a:.2f} ／ PyTorch の外 {du - d0 - r:.2f} ／ "
                  f"グラフを撮った {getattr(g, 'n_capture', 0)}回（この歩 +{getattr(g, 'n_capture', 0) - cap0}）{rw}", flush=True)
            d_last, last = du, r
r, a, pk = now()
du, _t = dev_used()
print(f"2 {n}歩のあと：GPU 全体 {du:.2f}GB ／ GPU 全体の山 {d_pk:.2f}GB（{d_pk_at}歩目。網の分 {d_pk - d0:.2f}）／ 全体 {tot:.2f}GB", flush=True)
print(f"  PyTorch の側：取り置き {r:.2f}GB ／ 箱の合計 {a:.2f}GB ／ 取り置きの山 {pk:.2f}GB ／ 撮ったグラフ {len(getattr(g, '_graphs', {}))}つ（{getattr(g, 'n_capture', 0)}回）", flush=True)
if gpu:
    torch.cuda.empty_cache()
    r2, a2, _ = now()
    d2, _t = dev_used()
    print(f"  使っていない取り置きを返したあと：GPU 全体 {d2:.2f}GB ／ 取り置き {r2:.2f}GB（返せた {r - r2:.2f}GB）／ 箱 {a2:.2f}GB ⇒ PyTorch の外 {d2 - d0 - r2:.2f}GB", flush=True)

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
