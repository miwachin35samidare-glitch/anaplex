"""版95c の速さの内訳（グラフの形のまま測る。arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_speed95.py [歩=300]
  （コンテナで道具を確かめるとき：python3 measure_speed95.py 20 --test ⇒ CPU・相手 200。時間は読まない）
【み】10-22 21:15（版95c 1歩 21.9ms）
measure_speed75 の「区切って測った」は _pass を差し替えるので、グラフの形（版79〜90）を通らない ⇒ 実際の回し方の内訳ではない
ここでは グラフの形のまま：
  1 1歩の全体（ふだん）
  2 網（Group2.step）・下からのシナプス（表の drive・learn）・NucNet の学び を、前後で GPU を待って測る（グラフはそのまま）
  3 外した形と比べる（1つずつ外して、ふだんの1歩の差）：NucNet（核どうしの線）／ 下からの 線条体・CM・Pf の6本の道
     ⚠️ 外すと網のふるまいが変わる（撃つ数が変わる）⇒ 差は目安
"""
import sys, time, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
import torch_brain as TB
from torch_brain import Brain

if B.VERSION not in (95,):
    sys.exit(f"⚠️ babble.py が版95 ではない（版 {B.VERSION}）")
TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
NP = 200 if TEST else 9100


def build():
    torch.manual_seed(0)
    b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=NP, hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
    b.bs.set_state(BS.WAKE)
    return b


def stream(b, n, seed=0):
    inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(seed)
    while n > 0:
        mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
        b.new_utterance(bool(np.any(cmd)))
        for t in range(min(len(ear), n)):
            yield torch.tensor(ear[t], device=dev), torch.tensor(cmd[t], device=dev)
        n -= len(ear)


def plain(b, n):
    for e, c in stream(b, 40, 1):
        b.step(e, c, learn=True)
    sync(); t0 = time.time(); k = 0
    for e, c in stream(b, n):
        b.step(e, c, learn=True); k += 1
    sync()
    return (time.time() - t0) * 1000 / k


b = build()
print(f"体 {b.n_cells:,} ／ 網の辺 {len(b.g.w):,} ／ 下からのシナプス {sum(len(s.w) for s in b.bu.values()):,} ／ NucNet {len(b.g.nucnet.w):,}", flush=True)
p0 = plain(b, NS)
print(f"\n1 1歩 ふだん {p0:.1f}ms（実時間 {T.DT_MS}ms の {p0 / T.DT_MS:.2f}倍）", flush=True)

# 2 グラフの形のまま、前後で待って測る
acc = collections.defaultdict(float); n_st = [0]
STEP0, D0, L0, NL0, BST0 = G.Group2.step, TB._BUTable.drive, TB._BUTable.learn, G.NucNet.learn, Brain.step


def timed(key, f):
    def w(*a, **kw):
        sync(); t = time.time(); out = f(*a, **kw); sync(); acc[key] += (time.time() - t) * 1000
        return out
    return w


def bst(self, *a, **kw):
    n_st[0] += 1
    return timed("歩の全体", BST0)(self, *a, **kw)


G.Group2.step, TB._BUTable.drive, TB._BUTable.learn, G.NucNet.learn, Brain.step = (
    timed("網（Group2.step）", STEP0), timed("下からの表 drive", D0), timed("下からの表 learn", L0), timed("NucNet の学び", NL0), bst)
fast0 = getattr(b.g, "fast_steps", 0)
for e, c in stream(b, NS):
    b.step(e, c, learn=True)
G.Group2.step, TB._BUTable.drive, TB._BUTable.learn, G.NucNet.learn, Brain.step = STEP0, D0, L0, NL0, BST0
n = max(n_st[0], 1)
print(f"\n2 グラフの形のまま区切って（{n}歩・グラフで回った歩 {getattr(b.g, 'fast_steps', 0) - fast0}）", flush=True)
tot = acc["歩の全体"] / n
for k in ("網（Group2.step）", "下からの表 drive", "下からの表 learn", "NucNet の学び"):
    print(f"   {acc[k] / n:6.2f}ms  {k}", flush=True)
rest = tot - sum(acc[k] for k in ("網（Group2.step）", "下からの表 drive", "下からの表 learn", "NucNet の学び")) / n
print(f"   {rest:6.2f}ms  網の外の残り（脳幹・CL・入口・核への抑制の線ほか）", flush=True)
print(f"   {tot:6.2f}ms  合計（区切った分だけ遅い）", flush=True)

# 3 外した形と比べる
del b
if dev == "cuda":
    torch.cuda.empty_cache()
b = build(); b.g.nucnet = None
for k_ in list(b.g.__dict__):
    if k_ in ("_gS", "_gX", "_graphs"):
        delattr(b.g, k_)
p1 = plain(b, NS)
print(f"\n3 NucNet を外す：1歩 {p1:.1f}ms（差 {p0 - p1:+.1f}ms）", flush=True)
del b
if dev == "cuda":
    torch.cuda.empty_cache()
b = build()
NEW = ("運動→被殻", "前頭前野→尾状核", "CM→被殻", "Pf→尾状核", "運動→CM", "前頭前野→Pf")
for k_ in NEW:
    s = b.bu[k_]
    keep = torch.zeros(len(s.w), dtype=torch.bool, device=dev); keep[::len(s.w) // 1000 + 1] = True   # 1000本ほど残す（形は保つ）
    s.src, s.tgt, s.w = s.src[keep], s.tgt[keep], s.w[keep]
    for kk in ("row", "state", "fac", "last", "deplete"):
        setattr(s.stp, kk, getattr(s.stp, kk)[keep])
    s.post_age = s.post_age
b.bu_pack()
p2 = plain(b, NS)
print(f"   線条体・CM・Pf への下からの6本の道を 1000本ほどに：1歩 {p2:.1f}ms（差 {p0 - p2:+.1f}ms）", flush=True)
