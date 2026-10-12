"""実時間とのずれ：1歩（10.67ms）の中で何に時間を使っているか（版53。arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_speed2.py [歩=300]
  （コンテナで道具を確かめるとき：python3 measure_speed2.py 60 --test ⇒ CPU・相手 200。時間は読まない）
【み】10-10「素子から触るならさ、実時間とのズレも解消しておきたい」
前の内訳（10-07 measure_speed、版51、もと 50歩）：1歩 66.6ms ／ GPU 39ms ／ CPU 65ms（GPU から読み出し 1歩 135回・命令 2,410回）
  ⇒ 版52 で 60.5ms、版53 で 63.7ms（健診）。実時間の 約6倍
並べるもの（1歩あたり ms。その区切りの前後で GPU を待って測る ⇒ 区切りごとの時間は正しいが、全体は待つ分だけ遅くなる）
  1歩の全体（区切りなし。ふだんの速さ）
  網（Group2.step）の中：焼いた網（見込み）／ 周回1回の残り（辺を集める・足し込む・撃つ）／ 学び（1周目）／ 刈り込み・恒常性（歩の終わり）
  網の外：脳幹・中脳・耳の入口・ほか（Brain.step から網を引いた残り）
  焼いた網を呼ぶ回数（1歩あたり）。tb は歩の終わりにしか変わらない ⇒ 1歩の中の10周は同じ値【コード】
あわせて torch.profiler の GPU・CPU の多い順（50歩）
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (53,):
    sys.exit(f"⚠️ babble.py が版53 ではない（版 {B.VERSION}）")
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)


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

# 2 区切りごと
acc = {"見込み": 0.0, "周回の残り": 0.0, "学び（1周目の差）": 0.0, "刈り込み": 0.0, "網の全体": 0.0, "歩の全体": 0.0}
cnt = {"見込み": 0, "周": 0, "歩": 0}
P0, PASS0, STEP0, RW0, BSTEP0 = T.Prior.logit, G.Group2._pass, G.Group2.step, G.Group2.rewire, Brain.step


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
    dt = (time.time() - t) * 1000; acc["周回の残り"] += dt; cnt["周"] += 1
    if first:
        first_t[0] += dt; first_t[1] += 1
    return out


T.Prior.logit = logit
G.Group2._pass = pas
G.Group2.rewire = timed("刈り込み", RW0)
G.Group2.step = timed("網の全体", STEP0)


def bstep(self, *a, **kw):
    cnt["歩"] += 1
    return timed("歩の全体", BSTEP0)(self, *a, **kw)


Brain.step = bstep
for e, c in stream(NS):
    b.step(e, c, learn=True)
n = max(cnt["歩"], 1)
pass_total = acc["周回の残り"]
other_pass = (pass_total - first_t[0]) / max(cnt["周"] - first_t[1], 1)
first_avg = first_t[0] / max(first_t[1], 1)
print(f"1歩 ふだん {plain:.1f}ms（実時間 {T.DT_MS}ms の {plain / T.DT_MS:.1f}倍）", flush=True)
print(f"区切って測った1歩 {acc['歩の全体']/n:.1f}ms（区切りで待つ分だけ遅い）", flush=True)
print(f"   網（Group2.step） {acc['網の全体']/n:6.1f}ms", flush=True)
print(f"     周回 {cnt['周']/n:.0f}回/歩：1周目（学び込み）{first_avg:.2f}ms ／ 2周目から {other_pass:.2f}ms ⇒ 学びの分 約 {first_avg-other_pass:.2f}ms", flush=True)
print(f"     そのうち 焼いた網 {acc['見込み']/n:6.2f}ms（{cnt['見込み']/n:.0f}回/歩）", flush=True)
print(f"     刈り込み・歩の終わり {acc['刈り込み']/n:6.2f}ms", flush=True)
print(f"   網の外（脳幹・中脳・入口ほか）{(acc['歩の全体'] - acc['網の全体'])/n:6.1f}ms", flush=True)
T.Prior.logit, G.Group2._pass, G.Group2.rewire, G.Group2.step, Brain.step = P0, PASS0, RW0, STEP0, BSTEP0

if not TEST:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for e, c in stream(50):
            b.step(e, c, learn=True)
        sync()
    print("\n── 50歩の内訳（GPU の時間の多い順）", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), flush=True)
    print("── 50歩の内訳（CPU の時間の多い順）", flush=True)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
