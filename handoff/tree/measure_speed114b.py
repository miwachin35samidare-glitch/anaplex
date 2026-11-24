"""版114：1歩の時間を「撃った辺に比べる分」と「比べない分（固定の手間）」に分ける（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_speed114b.py [温め=6000] [歩=1000] [見る歩=30]
  （コンテナ：python3 measure_speed114b.py 30 20 5 --test ⇒ CPU・相手 200）
  何のため：measure_speed114（11-22 09:02）で 1歩ぜんぶ 31.7ms のうち 網の領野 18.7ms・どこにも数えていない分 8.4ms。
    数値合わせで作りを変えても残る手（呼び出しの回数・待ち）がどれだけあるかを見てから、時間短縮を先にするか決める【み】11-22 09:08
  R 歩ごとの記録（歩 回）：部分ごとに GPU を待って時間を取り、その歩の撃った辺の数と並べる。
      歩どうしの撃った辺の違い（そろう歩・黙る歩）から、時間 ＝ 固定 ＋ 傾き × 撃った辺 を最小二乗で当てる（刈り込みの歩は除く）
      部分：暮らしの1歩 ／ 1歩ぜんぶ（b.step）／ 網の領野（g.step）／ その中の 撃った辺を作る・見込み（焼いた網）・放出・グラフの再生・周回（グラフでない歩）
  P 見る歩（torch.profiler、CPU と GPU）：1歩あたりの GPU の核の数・核の時間の多い順・CPU の手間の多い順
      ⚠️ 見ているあいだは遅くなる（数倍）。動かない・止まるときは R だけ残る（例外は書いて続ける）
  毎歩に足す仕事：GPU を待つことだけ（部分の前後）。少ない箱への足し込み・網と同じ大きさの写しなし。撃った辺の数は網が持つ並びの長さを読む
"""
import sys, time, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB, life as L
from torch_brain import Brain, NAMES

if B.VERSION not in (114, 115, 116):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版114 の網のため")
TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 6000
NS = int(args[1]) if len(args) > 1 else 1000
NP = int(args[2]) if len(args) > 2 else 30
if TEST:
    B.DEV = "cpu"
dev = B.DEV
cuda = str(dev).startswith("cuda")


def sync():
    if cuda and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
print(f"版 {B.VERSION} ／ 温め {WARM} 歩・記録 {NS} 歩・見る {NP} 歩 ／ 網の中の辺 {len(g.w):,}・体 {g.n_ch_all:,}", flush=True)
t0 = time.time()
for i in range(WARM):
    life.step()
    if (i + 1) % 1000 == 0:
        L.beat()
        print(f"  温め {i + 1} 歩（{(time.time() - t0) / (i + 1) * 1000:.1f}ms/歩）", flush=True)

# ── R 歩ごとの記録 ──
cur = collections.defaultdict(float)
ne_now = [None]


def timed(name, f, ne=False):
    def w(*a, **k):
        sync(); t1 = time.perf_counter()
        r = f(*a, **k)
        sync(); cur[name] += (time.perf_counter() - t1) * 1000
        if ne and ne_now[0] is None and r is not None:
            ne_now[0] = len(r)
        return r
    return w


orig = {}
PARTS = [("1歩ぜんぶ", b, "step"), ("網の領野", g, "step"), ("撃った辺を作る", g, "_fired_edges"), ("見込み（焼いた網）", g, "_prior_now"),
         ("放出", g.stp, "release_at"), ("グラフの再生", g, "_fast_run"), ("周回（グラフでない歩）", g, "_pass"),
         ("周回の続き（グラフ）", g, "_gp_passes"), ("刈り込み", g, "rewire"), ("下からの線（足す）", b.but, "drive"),
         ("下からの線（学ぶ）", b.but, "learn")]
for name, obj, attr in PARTS:
    if obj is not None and hasattr(obj, attr):
        orig[(id(obj), attr)] = (obj, attr, getattr(obj, attr))
        setattr(obj, attr, timed(name, getattr(obj, attr), ne=(attr == "_fired_edges")))
rec = []
fast0 = getattr(g, "fast_steps", 0)
sync()
for i in range(NS):
    cur.clear(); ne_now[0] = None
    fs0 = getattr(g, "fast_steps", 0)
    t1 = time.perf_counter()
    life.step()
    sync()
    tl = (time.perf_counter() - t1) * 1000
    rec.append((tl, ne_now[0] or 0, int(getattr(g, "fast_steps", 0) - fs0), dict(cur)))
    if (i + 1) % 200 == 0:
        L.beat()
for (obj, attr, f) in orig.values():
    setattr(obj, attr, f)
L.beat()
names = ["暮らしの1歩"] + [n for n, _, _ in PARTS]
tab = np.array([[r[0]] + [r[3].get(n, 0.0) for n, _, _ in PARTS] for r in rec])
ne = np.array([r[1] for r in rec], dtype=np.float64)
fast = np.array([r[2] for r in rec])
rw = tab[:, names.index("刈り込み")] > 0
keep = ~rw
print(f"\nR 歩ごと（{NS} 歩・刈り込みの歩 {int(rw.sum())} を除く {int(keep.sum())} 歩）：撃った辺 平均 {ne[keep].mean():,.0f}・"
      f"幅 {np.percentile(ne[keep], 5):,.0f}〜{np.percentile(ne[keep], 95):,.0f}（5〜95%）・グラフで回った歩 {int(fast[keep].sum())}", flush=True)
print(f"  {'部分':<14s} {'平均ms':>8s} {'固定ms':>8s} {'傾き ms/100万本':>14s} {'辺の分ms':>9s} {'合い方R²':>8s}", flush=True)
X = np.stack([np.ones(int(keep.sum())), ne[keep] / 1e6], 1)
for j, nm in enumerate(names):
    if nm == "刈り込み":
        continue
    y = tab[keep, j]
    if y.mean() < 0.005:
        continue
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ coef
    r2 = 1 - ((y - yh) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())
    print(f"  {nm:<14s} {y.mean():8.2f} {coef[0]:8.2f} {coef[1]:14.3f} {coef[1] * ne[keep].mean() / 1e6:9.2f} {r2:8.2f}", flush=True)
if rw.any():
    print(f"  （刈り込みの歩：刈り込み {tab[rw, names.index('刈り込み')].mean():.0f}ms・暮らしの1歩 {tab[rw, 0].mean():.0f}ms）", flush=True)
inner = ["撃った辺を作る", "見込み（焼いた網）", "放出", "グラフの再生", "周回（グラフでない歩）", "周回の続き（グラフ）"]
rest_g = tab[keep, names.index("網の領野")] - sum(tab[keep, names.index(n)] for n in inner)
rest_b = tab[keep, names.index("1歩ぜんぶ")] - tab[keep, names.index("網の領野")] - tab[keep, names.index("下からの線（足す）")] - tab[keep, names.index("下からの線（学ぶ）")]
print(f"  網の領野のうち上の部分の外：{rest_g.mean():.2f}ms ／ 1歩ぜんぶのうち 網の領野・下からの線の外：{rest_b.mean():.2f}ms ／ 暮らしの1歩のうち 1歩ぜんぶの外：{(tab[keep, 0] - tab[keep, 1]).mean():.2f}ms", flush=True)
q = np.argsort(ne[keep])
lo_, hi_ = q[: len(q) // 10], q[-len(q) // 10:]
print(f"  撃った辺の少ない 1割の歩：辺 {ne[keep][lo_].mean():,.0f}・暮らしの1歩 {tab[keep][lo_, 0].mean():.1f}ms ／ 多い 1割：辺 {ne[keep][hi_].mean():,.0f}・{tab[keep][hi_, 0].mean():.1f}ms", flush=True)

# ── P 見る歩（profiler）──
L.beat()
try:
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    sync()
    with profile(activities=acts, record_shapes=False) as prof:
        for i in range(NP):
            life.step()
            sync()
            if (i + 1) % 5 == 0:
                L.beat()
    L.beat()
    ev = prof.key_averages()
    L.beat()
    dev_attr = "self_device_time_total" if hasattr(ev[0], "self_device_time_total") else "self_cuda_time_total"
    kern = [e for e in ev if getattr(e, dev_attr, 0) > 0]
    n_k = sum(e.count for e in kern)
    t_k = sum(getattr(e, dev_attr) for e in kern) / 1000.0
    print(f"\nP 見る歩（{NP} 歩・torch.profiler。見ているあいだは遅い）：GPU の核 1歩 {n_k / NP:,.0f} 回・核の時間の和 1歩 {t_k / NP:.2f}ms", flush=True)
    print("  GPU の核の時間の多い順（1歩あたり）", flush=True)
    for e in sorted(kern, key=lambda e: -getattr(e, dev_attr))[:20]:
        print(f"    {getattr(e, dev_attr) / 1000.0 / NP:7.3f}ms  {e.count / NP:7.1f}回  {e.key[:70]}", flush=True)
    L.beat()
    cpu = [e for e in ev if e.self_cpu_time_total > 0]
    print(f"  CPU の手間の多い順（1歩あたり。呼び出しの数（入れ子込み） 合わせて {sum(e.count for e in cpu) / NP:,.0f} 回）", flush=True)
    for e in sorted(cpu, key=lambda e: -e.self_cpu_time_total)[:20]:
        print(f"    {e.self_cpu_time_total / 1000.0 / NP:7.3f}ms  {e.count / NP:7.1f}回  {e.key[:70]}", flush=True)
except Exception as ex:                                     # 見る道具が動かないときも R は残す
    print(f"\nP は回らなかった：{type(ex).__name__}: {str(ex)[:300]}", flush=True)
if cuda:
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
