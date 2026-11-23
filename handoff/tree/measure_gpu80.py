"""1歩の中で GPU と CPU がどこに時間を使っているか、計算の場所ごとに（版80〜。arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_gpu80.py [歩=50]
  （コンテナで道具を確かめるとき：python3 measure_gpu80.py 5 --test ⇒ CPU・相手 200）
やり方：torch.profiler に場所の名札を付ける（周・足し込みの関数は差し替えない ⇒ グラフ・合わせた形のまま測る）
  名札：compile した核ごと（_run の関数の名前）・時間細胞・撃った辺を集める・放出・2周目から・歩の終わり・下からの表・脳の残り
並べるもの
  1 場所ごとの GPU の時間・CPU の時間（1歩あたり。GPU の時間は、その場所で呼んだ核の合計）
  2 辺の数（1歩ぶんの辺・放出した辺・2周目からの箱）
"""
import sys, time, collections, numpy as np, torch
from torch.profiler import profile, ProfilerActivity, record_function
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION < 80:
    sys.exit(f"⚠️ babble.py が版80 より前（版 {B.VERSION}）")
NS = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
print(f"torch {torch.__version__} ／ {torch.cuda.get_device_name(0) if gpu else 'CPU'} ／ ANAPLEX_GRAPH={G.GRAPH} ／ "
      f"網の辺 {len(b.g.w):,}", flush=True)


def stream(n):
    while n > 0:
        mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
        b.new_utterance(bool(np.any(cmd)))
        for t in range(min(len(ear), n)):
            yield torch.tensor(ear[t], device=dev), torch.tensor(cmd[t], device=dev)
        n -= len(ear)


for e, c in stream(30):          # 慣らし（compile・グラフを撮り終える）
    b.step(e, c, learn=True)

# ── 名札を付ける（中身は変えない）──
N = collections.defaultdict(list)
RUN0 = G._run


def run(fn, *a):
    with record_function("核 " + fn.__name__):
        return RUN0(fn, *a)


def lab(name, f, count=None):
    def w(*a, **kw):
        with record_function(name):
            out = f(*a, **kw)
        if count:
            count(a, out)
        return out
    return w


PATCH = [(G, "_run", run),
         (G.TimeCellsGroup, "spikes", lab("時間細胞", G.TimeCellsGroup.spikes)),
         (G.Group2, "_expand", lab("撃った辺を集める", G.Group2._expand, lambda a, o: N["撃った辺"].append(len(o)))),
         (T.STP, "release_at", lab("放出", T.STP.release_at, lambda a, o: N["放出した辺"].append(len(a[1])))),
         (G.Group2, "_gp_passes", lab("2周目から（全体）", G.Group2._gp_passes)),
         (G.Group2, "_advance_tail", lab("歩の終わり", G.Group2._advance_tail)),
         (G.Group2, "rewire", lab("刈り込み", G.Group2.rewire)),
         (G.Group2, "step", lab("網（全体）", G.Group2.step)),
         (TB._BUTable, "drive", lab("下からの表 drive", TB._BUTable.drive)),
         (TB._BUTable, "learn", lab("下からの表 learn", TB._BUTable.learn)),
         (T.Prior, "logit", lab("焼いた網", T.Prior.logit)),
         (Brain, "step", lab("歩（全体）", Brain.step))]
orig = [(o, k, getattr(o, k)) for o, k, _ in PATCH]
for o, k, f in PATCH:
    setattr(o, k, f)
G2O = dict(G._G2_ORIG)              # 差し替えた関数も「元のまま」と見なさせる（でないと周ごとの形に戻ってしまう）
for k in G._G2_ORIG:
    G._G2_ORIG[k] = getattr(G.Group2, k)
# _run は torch_group2 の中で名前で引くので、モジュールの値を差し替えれば効く（torch_area は呼ぶときに import する）
with profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if gpu else [])) as prof:
    t0 = time.time()
    for e, c in stream(NS):
        b.step(e, c, learn=True)
        N["1歩ぶんの辺（時間細胞と buf）"].append(len(b.g._base_e))
    if gpu:
        torch.cuda.synchronize()
    wall = (time.time() - t0) * 1000 / NS
for o, k, f in orig:
    setattr(o, k, f)
G._G2_ORIG.update(G2O)

def is_k(ev):
    return str(getattr(ev, "device_type", "")).endswith("CUDA")
ka = {ev.key: ev for ev in prof.key_averages() if not is_k(ev)}     # 名札は CPU の側（その中で呼んだ核の GPU の時間の合計を持つ）
def dev_t(ev):
    for nm in ("device_time_total", "cuda_time_total", "self_device_time_total"):
        v = getattr(ev, nm, None)
        if v:
            return v / 1000.0
    return 0.0
names = ["歩（全体）", "網（全体）", "2周目から（全体）", "歩の終わり", "時間細胞", "撃った辺を集める", "放出", "焼いた網",
         "下からの表 drive", "下からの表 learn", "刈り込み"] + sorted(k for k in ka if k.startswith("核 "))
LAB = set(names)
kern = [ev for ev in prof.key_averages() if is_k(ev) and ev.key not in LAB and not ev.key.startswith("核 ")]
gpu_all = sum(ev.self_device_time_total for ev in kern) / 1000.0 if gpu else 0
print(f"\n（名札を付けた分だけ少し遅い）1歩 {wall:.1f}ms" + (f" ／ GPU の核の合計 {gpu_all / NS:.2f}ms/歩" if gpu else ""), flush=True)
print("\n1 場所ごと（1歩あたり）      GPU ms    CPU ms   回/歩", flush=True)
for nm in names:
    ev = ka.get(nm)
    if ev is None:
        continue
    print(f"  {nm:<28} {dev_t(ev) / NS:8.2f}  {ev.cpu_time_total / 1000.0 / NS:8.2f}  {ev.count / NS:6.1f}", flush=True)
print("  ⚠️ グラフの中の核は名札に入らないことがある（2周目から（全体）が小さく出る）。GPU の核の合計は全部入る", flush=True)
if gpu:
    print("\n── 名札の外も含めた GPU の多い順（核の名前・1歩あたり）", flush=True)
    rows = sorted((ev for ev in kern if ev.self_device_time_total > 0),
                  key=lambda ev: -ev.self_device_time_total)[:15]
    for ev in rows:
        print(f"  {ev.self_device_time_total / 1000.0 / NS:7.2f}ms  {ev.count / NS:6.1f}回  {ev.key[:90]}", flush=True)
print("\n2 辺の数（1回あたりの平均・最大）", flush=True)
for k, v in N.items():
    if v:
        print(f"  {k}：平均 {np.mean(v):,.0f} ／ 最大 {max(v):,} ／ {len(v) / NS:.1f}回/歩", flush=True)
h = getattr(b.g, "gp_hist", None)
if h:
    tot = sum(h.values())
    print(f"  2周目からの箱（周の数で重みづけた平均の大きさ）{sum(k * v for k, v in h.items()) / tot:,.0f}", flush=True)
