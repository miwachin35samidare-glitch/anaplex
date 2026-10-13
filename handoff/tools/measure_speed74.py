"""実時間とのずれ：1歩（10.67ms）の中で何に時間を使っているか（版74。arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_speed74.py [歩=300]
  ANAPLEX_BU=0 python3 measure_speed74.py   ⇒ 下からの道をシナプスにする前（版68 と同じ網）で同じ測り。並べて差を見る
  （コンテナで道具を確かめるとき：python3 measure_speed74.py 60 --test ⇒ CPU・相手 200。時間は読まない）
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
if B.VERSION not in (74,):
    sys.exit(f"⚠️ babble.py が版74 ではない（版 {B.VERSION}）")
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
sync = (lambda: torch.cuda.synchronize()) if gpu else (lambda: None)
print(f"torch {torch.__version__} ／ hip {torch.version.hip} ／ "
      f"{torch.cuda.get_device_name(0) if gpu else 'CPU'} ／ ANAPLEX_BU={os.environ.get('ANAPLEX_BU', '1')}", flush=True)
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
P0, PASS0, STEP0, RW0, BSTEP0 = T.Prior.logit, G.Group2._pass, G.Group2.step, G.Group2.rewire, Brain.step
D0, L0 = TB._BUSyn.drive, TB._BUSyn.learn
name_of = {id(s): k for k, s in b.bu.items()}


def timed(key, f):
    def w(*a, **kw):
        sync(); t = time.time(); out = f(*a, **kw); sync(); acc[key] += (time.time() - t) * 1000
        return out
    return w


def logit(self, f):
    cnt["見込み"] += 1
    return timed("見込み", P0)(self, f)


first_t = [0.0, 0]


def pas(self, x_all, ext_all, learn, advance, first):
    sync(); t = time.time(); out = PASS0(self, x_all, ext_all, learn, advance, first); sync()
    dt = (time.time() - t) * 1000; acc["周回"] += dt; cnt["周"] += 1
    if first:
        first_t[0] += dt; first_t[1] += 1
    return out


def drv(self, v):
    sync(); t = time.time(); out = D0(self, v); sync()
    dt = (time.time() - t) * 1000; acc["シナプス drive"] += dt; cnt["drive"] += 1
    per_path[name_of.get(id(self), "?") + "（drive）"] += dt
    return out


def lrn(self, post, d):
    sync(); t = time.time(); out = L0(self, post, d); sync()
    dt = (time.time() - t) * 1000; acc["シナプス learn"] += dt; cnt["learn"] += 1
    per_path[name_of.get(id(self), "?") + "（learn）"] += dt
    return out


def bstep(self, *a, **kw):
    cnt["歩"] += 1
    return timed("歩の全体", BSTEP0)(self, *a, **kw)


T.Prior.logit, G.Group2._pass, G.Group2.rewire, G.Group2.step = logit, pas, timed("刈り込み", RW0), timed("網の全体", STEP0)
TB._BUSyn.drive, TB._BUSyn.learn, Brain.step = drv, lrn, bstep
for e, c in stream(NS):
    b.step(e, c, learn=True)
T.Prior.logit, G.Group2._pass, G.Group2.rewire, G.Group2.step = P0, PASS0, RW0, STEP0
TB._BUSyn.drive, TB._BUSyn.learn, Brain.step = D0, L0, BSTEP0

n = max(cnt["歩"], 1)
first_avg = first_t[0] / max(first_t[1], 1)
other = (acc["周回"] - first_t[0]) / max(cnt["周"] - first_t[1], 1)
syn = acc["シナプス drive"] + acc["シナプス learn"]
print(f"\n2 区切って測った1歩 {acc['歩の全体']/n:.1f}ms（区切りで待つ分だけ遅い）", flush=True)
print(f"   網（Group2.step） {acc['網の全体']/n:6.1f}ms", flush=True)
print(f"     周回 {cnt['周']/n:.0f}回/歩：1周目（学び込み）{first_avg:.2f}ms ／ 2周目から {other:.2f}ms", flush=True)
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
