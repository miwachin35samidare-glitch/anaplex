"""実時間とのずれ：1歩（10.67ms）の中で何に時間を使っているか（版79〜。arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_speed79.py [歩=300]
  ANAPLEX_GRAPH=0 python3 measure_speed79.py   ⇒ グラフなし（版78 と同じ周ごと）で同じ測り。並べて差を見る
  （コンテナで道具を確かめるとき：python3 measure_speed79.py 60 --test ⇒ CPU・相手 200。時間は読まない）
  measure_speed75 との違い：周（_pass）を差し替えない（差し替えるとグラフを使わなくなる）。1周目 ＝ 網 − 2周目から − 刈り込み
【み】10-14「今の環境を測るところからじゃない？」
前の測り【測】
  版53 1歩 56.2ms ／ 版54b 25.0ms（CPU が GPU を待つ所 230 → 22回/歩）／ 版55 29.5 ／ 版59 31.0 ／ 版68 28.2 ／ 版70 32.6 ／ 版71 37.8 ／ 版73 49.4
  版69〜 の下からのシナプス（_BUSyn）は、drive のたびに torch.nonzero（撃った線を数える ⇒ CPU が GPU を待つ）
並べるもの
  0 環境（PyTorch・HIP・GPU）
  1 1歩の全体（区切りなし。ふだんの速さ）
  2 区切りごと（その区切りの前後で GPU を待って測る ⇒ 区切りの時間は正しいが、全体は待つ分だけ遅くなる）
     網（Group2.step）：焼いた網 ／ 周回（1周目＝学び込み と 2周目から）／ 刈り込み・歩の終わり
     下からのシナプス（_BUSyn）：drive（足し込み）・learn（学び）、道ごとの多い順
     網の外の残り（脳幹・中脳・CL・入口ほか）
  3 CPU が GPU を待つ場所（行ごと・1歩あたり。GPU のときだけ）
  4 torch.profiler の多い順（50歩。GPU のときだけ）
"""
import sys, os, time, collections, linecache, warnings, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION < 79:
    sys.exit(f"⚠️ babble.py が版79 より前（版 {B.VERSION}）。版78 までは measure_speed75.py")
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
sync = (lambda: torch.cuda.synchronize()) if gpu else (lambda: None)
print(f"torch {torch.__version__} ／ hip {torch.version.hip} ／ "
      f"{torch.cuda.get_device_name(0) if gpu else 'CPU'} ／ ANAPLEX_GRAPH={G.GRAPH}", flush=True)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
n_bu = sum(len(s.w) for s in b.bu.values()) if b.bu_on else 0
print(f"体 {b.n_cells:,} ／ 網の辺 {len(b.g.w):,} ／ 下からのシナプス {len(b.bu)}本の道・{n_bu:,}本", flush=True)


def stream(n):
    """作り物の声を続けて n 歩ぶん（発話ごとに new_utterance）"""
    while n > 0:
        mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
        b.new_utterance(bool(np.any(cmd)))
        for t in range(min(len(ear), n)):
            yield torch.tensor(ear[t], device=dev), torch.tensor(cmd[t], device=dev)
        n -= len(ear)


for e, c in stream(30):          # 慣らし
    b.step(e, c, learn=True)

# 1 ふだんの速さ（区切りなし）
sync(); t0 = time.time(); k = 0
for e, c in stream(NS):
    b.step(e, c, learn=True); k += 1
sync(); plain = (time.time() - t0) * 1000 / k
print(f"\n1 1歩 ふだん {plain:.1f}ms（実時間 {T.DT_MS}ms の {plain / T.DT_MS:.1f}倍）", flush=True)

# 2 区切りごと
acc = collections.defaultdict(float)
cnt = collections.Counter()
per_path = collections.defaultdict(float)
P0, GP0, STEP0, RW0, BSTEP0 = T.Prior.logit, G.Group2._gp_passes, G.Group2.step, G.Group2.rewire, Brain.step
TAB = getattr(TB, "_BUTable", None)                 # 版75：道をまとめた表
KL = TAB if TAB is not None else TB._BUSyn
D0, L0 = KL.drive, KL.learn
name_of = {id(s): k for k, s in b.bu.items()}


def timed(key, f):
    def w(*a, **kw):
        sync(); t = time.time(); out = f(*a, **kw); sync(); acc[key] += (time.time() - t) * 1000
        return out
    return w


def logit(self, f):
    cnt["見込み"] += 1
    return timed("見込み", P0)(self, f)


def gpp(self, x_all, ext_all):
    sync(); t = time.time(); out = GP0(self, x_all, ext_all); sync()
    acc["2周目から"] += (time.time() - t) * 1000; cnt["2周目から"] += 1
    return out


def drv(self, *a):
    sync(); t = time.time(); out = D0(self, *a); sync()
    dt = (time.time() - t) * 1000; acc["シナプス drive"] += dt; cnt["drive"] += 1
    per_path[(f"表の{a[0]}回目" if TAB is not None else name_of.get(id(self), "?")) + "（drive）"] += dt
    return out


def lrn(self, *a):
    sync(); t = time.time(); out = L0(self, *a); sync()
    dt = (time.time() - t) * 1000; acc["シナプス learn"] += dt; cnt["learn"] += 1
    per_path[("表" if TAB is not None else name_of.get(id(self), "?")) + "（learn）"] += dt
    return out


def bstep(self, *a, **kw):
    cnt["歩"] += 1
    return timed("歩の全体", BSTEP0)(self, *a, **kw)


cap0 = getattr(b.g, "n_capture", 0); hist0 = dict(getattr(b.g, "gp_hist", {}) or {})
T.Prior.logit, G.Group2._gp_passes, G.Group2.rewire, G.Group2.step = logit, gpp, timed("刈り込み", RW0), timed("網の全体", STEP0)
KL.drive, KL.learn, Brain.step = drv, lrn, bstep
for e, c in stream(NS):
    b.step(e, c, learn=True)
T.Prior.logit, G.Group2._gp_passes, G.Group2.rewire, G.Group2.step = P0, GP0, RW0, STEP0
KL.drive, KL.learn, Brain.step = D0, L0, BSTEP0

n = max(cnt["歩"], 1)
syn = acc["シナプス drive"] + acc["シナプス learn"]
print(f"\n2 区切って測った1歩 {acc['歩の全体']/n:.1f}ms（区切りで待つ分だけ遅い）", flush=True)
print(f"   網（Group2.step） {acc['網の全体']/n:6.1f}ms", flush=True)
gp_on = cnt["2周目から"] > 0
if gp_on:
    rec = b.g.recur
    print(f"     1周目（学び込み・刈り込みを除く）{(acc['網の全体'] - acc['2周目から'] - acc['刈り込み'])/n:6.2f}ms ／ "
          f"2周目から（グラフ・歩の終わり込み）{acc['2周目から']/n:.2f}ms（1周 {acc['2周目から']/n/(rec-1):.2f}ms）", flush=True)
    hist = {k: v - hist0.get(k, 0) for k, v in b.g.gp_hist.items() if v - hist0.get(k, 0)}
    print(f"     撮ったグラフ {getattr(b.g, 'n_capture', 0)}個（この測りの中で {getattr(b.g, 'n_capture', 0) - cap0}個）／ 箱の大きさごとの周：" +
          "  ".join(f"{k:,}:{v}" for k, v in sorted(hist.items())), flush=True)
else:
    print("     ⚠️ 2周目からのグラフを通っていない（ANAPLEX_GRAPH=0 か CPU）", flush=True)
print(f"     焼いた網 {acc['見込み']/n:6.2f}ms（{cnt['見込み']/n:.0f}回/歩）", flush=True)
print(f"     刈り込み・歩の終わり {acc['刈り込み']/n:6.2f}ms", flush=True)
print(f"   下からのシナプス {syn/n:6.1f}ms：drive {acc['シナプス drive']/n:.1f}ms（{cnt['drive']/n:.0f}回/歩）／ "
      f"learn {acc['シナプス learn']/n:.1f}ms（{cnt['learn']/n:.0f}回/歩）", flush=True)
for kk, v in sorted(per_path.items(), key=lambda x: -x[1])[:10]:
    print(f"       {v/n:6.2f}ms  {kk}", flush=True)
print(f"   網の外の残り（脳幹・中脳・CL・入口ほか）{(acc['歩の全体'] - acc['網の全体'] - syn)/n:6.1f}ms", flush=True)

if gpu:
    # 3 CPU が GPU を待つ場所
    NW = 20
    wc = collections.Counter()
    st = stream(NW)
    torch.cuda.set_sync_debug_mode(1)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        for e, c in st:
            b.step(e, c, learn=True)
        for w in ws:
            wc[(w.filename, w.lineno)] += 1
    torch.cuda.set_sync_debug_mode(0)
    print(f"\n3 CPU が GPU を待つ場所（{NW}歩、多い順）合計 {sum(wc.values())/NW:.0f}回/歩", flush=True)
    for (f, ln), c in wc.most_common(25):
        print(f"  {c/NW:6.1f}回/歩  {f.split('/')[-1]}:{ln}  {linecache.getline(f, ln).strip()[:100]}", flush=True)

    # 4 profiler
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for e, c in stream(50):
            b.step(e, c, learn=True)
        sync()
    print("\n4 50歩の内訳（GPU の時間の多い順）", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), flush=True)
    print("── 50歩の内訳（CPU の時間の多い順）", flush=True)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
