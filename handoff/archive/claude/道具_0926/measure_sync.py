"""1歩の中で CPU が GPU を待つ場所と、撃った辺の数（版54。arkhe の ~/anaplex で。保存しない・網は書き換えない。数十秒）
  python3 measure_sync.py [歩=20]
【測】measure_speed2（10-10、版54）：1歩 37.4ms ／ GPU から読み出し 149回/歩 ／ 命令の発行 2,545回/歩 ／ nonzero 132回/歩
【測】probe_gpu（10-10）：nonzero_static は ROCm で 0.95ms/回（nonzero 0.06ms の15倍遅い）⇒ そのまま置き換えると遅くなる
  ⇒ どこで待っているかを行ごとに数え、1つずつ待たない書き方に替える（意味は同じ）
並べるもの
  1 待つ場所（ファイル:行・その行の中身・1歩あたりの回数）多い順
  2 撃った辺の数：1周目（学ぶ辺）／ 辺の総数 ／ 割合（学びを触ったときにまとめる形が、どれだけ辺を減らしているか）
"""
import sys, collections, linecache, warnings, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 20
if B.VERSION not in (54,):
    sys.exit(f"⚠️ babble.py が版54 ではない（版 {B.VERSION}）")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
b.new_utterance(True)
for t in range(10):
    b.step(torch.tensor(ear[t], device=B.DEV), torch.tensor(cmd[t], device=B.DEV), learn=True)

ne = []
P0 = G.Group2._pass
def pas(self, x_all, ext_all, learn, advance, first):
    if first:
        ne.append(len(self._fired_edges()))
    return P0(self, x_all, ext_all, learn, advance, first)
G.Group2._pass = pas

cnt = collections.Counter()
torch.cuda.set_sync_debug_mode(1)
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    for t in range(10, 10 + NS):
        b.step(torch.tensor(ear[t % len(ear)], device=B.DEV), torch.tensor(cmd[t % len(cmd)], device=B.DEV), learn=True)
    for w in ws:
        cnt[(w.filename, w.lineno)] += 1
torch.cuda.set_sync_debug_mode(0)
print(f"待つ場所（{NS}歩、多い順）合計 {sum(cnt.values())/NS:.0f}回/歩", flush=True)
for (f, ln), c in cnt.most_common(40):
    src = linecache.getline(f, ln).strip()
    print(f"  {c/NS:6.1f}回/歩  {f.split('/')[-1]}:{ln}  {src[:110]}", flush=True)
E = len(b.g.w)
print(f"撃った辺（1周目）1歩 平均 {np.mean(ne):,.0f}本（最大 {max(ne):,}）／ 辺の総数 {E:,} ／ 割合 {np.mean(ne)/E:.1%}", flush=True)
