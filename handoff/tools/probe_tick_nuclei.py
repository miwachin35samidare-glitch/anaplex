"""基底核・中脳の核を 1歩（10.67ms）ごとに回すか、周回（1.1ms）ごとに回すかを、かかる時間で決めるための測り（arkhe の ~/anaplex で。何も書き換えない）
  python3 probe_tick_nuclei.py [歩=300]
  （コンテナ：python3 probe_tick_nuclei.py 20 --test ⇒ CPU・相手 200。時間は読まない）
【み】10-16 0時33分「それは測って決める約束でしょ」（01_現在地 10-14：測って 1.1ms 刻みの時間を見積もる → 乗る見込みが立ってから）
置いたもの（形だけ。値は決まったもの・決まりかけのもの）
  核：外節・内節・網様部・視床下核・境・外側手綱核・吻内側被蓋核 各64体 ＋ 黒質 56体（計 504体）
  撃ち方 B：v ← v·α ＋ i0 ＋ w·（興奮 − 抑制）、閾値 1 で撃って 0 へ（α は τm 40ms を刻みに合わせる）
  線（受け手1体の本数）：線条体 → 外節・内節・網様部 100本（【マカク】Percheron）／ 核どうし 8本 ／ 前頭前野 → 視床下核 100本
    核どうしの遅れ：周回なら 1〜3周（1〜3ms、模型の論文の置き方）／ 1歩なら 1歩
  外（線条体 2,288体 ＋ 前頭前野 946体）からは1歩に1回渡す（09-11 の決まり）
並べるもの（1歩あたり）
  A 核を1歩ごとに1回（今の midbrain と同じ回し方）
  B 核を周回ごとに10回（1.1ms 刻み）
  それぞれ：そのまま ／ HIP グラフに撮って再生
  ⇒ 今の網（版92）の1歩と並べて、B の分を足すと実時間 10.67ms に入るか
  版2：網だけ・＋A・＋B を 50歩ずつ交互にくり返して、差の中央と幅を出す（前の版は順に1回ずつで差を測れていなかった）
"""
import sys, time, math, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
gpu = dev == "cuda"
sync = (lambda: torch.cuda.synchronize()) if gpu else (lambda: None)
torch.manual_seed(0)
g = torch.Generator(device=dev); g.manual_seed(1)

NUC = {"外節": 64, "内節": 64, "網様部": 64, "視床下核": 64, "境": 64, "外側手綱核": 64, "吻内側被蓋核": 64, "黒質": 56}
HZ = {"外節": 69.7, "内節": 82.5, "網様部": 60.5, "視床下核": 19.3, "境": 33.0, "外側手綱核": 30.0, "吻内側被蓋核": 17.8, "黒質": 4.56}
N_STR, N_PFC = 2288, 946
names = list(NUC); off = np.cumsum([0] + [NUC[k] for k in names]); N = int(off[-1])
sl = {k: slice(int(off[i]), int(off[i + 1])) for i, k in enumerate(names)}
# 線：（送り手, 受け手, 本数, 興奮か, 遅れの周）
LINES = [("線条体", "外節", 100, False, 2), ("線条体", "内節", 100, False, 2), ("線条体", "網様部", 100, False, 2),
         ("前頭前野", "視床下核", 100, True, 1), ("外節", "視床下核", 8, False, 1), ("視床下核", "外節", 8, True, 1),
         ("外節", "内節", 8, False, 1), ("視床下核", "内節", 8, True, 1), ("外節", "網様部", 8, False, 1),
         ("視床下核", "網様部", 8, True, 1), ("外節", "外節", 8, False, 1), ("視床下核", "境", 8, True, 1),
         ("境", "外側手綱核", 8, True, 1), ("外側手綱核", "吻内側被蓋核", 8, True, 1), ("吻内側被蓋核", "黒質", 8, False, 1),
         ("網様部", "黒質", 8, False, 1)]
DMAX = 3
# 送り手の通し番号：核 0..N-1 ／ 線条体 N..N+N_STR-1 ／ 前頭前野 その先
SRC_OFF = {"線条体": N, "前頭前野": N + N_STR}
NSRC = N + N_STR + N_PFC
rows = [[] for _ in range(N)]                 # 受け手ごとに（送り手, 符号, 遅れ）
for a, b, k, exc, d in LINES:
    for r in range(sl[b].start, sl[b].stop):
        if a in SRC_OFF:
            s = SRC_OFF[a] + torch.randint(0, N_STR if a == "線条体" else N_PFC, (k,), generator=torch.Generator().manual_seed(r)).numpy()
        else:
            s = sl[a].start + torch.randint(0, NUC[a], (k,), generator=torch.Generator().manual_seed(r + 7)).numpy()
        rows[r] += [(int(x), 1.0 if exc else -1.0, d) for x in s]
KMAX = max(len(x) for x in rows)
idx = torch.zeros(N, KMAX, dtype=torch.long); sg = torch.zeros(N, KMAX); dl = torch.zeros(N, KMAX, dtype=torch.long)
for r, x in enumerate(rows):
    for j, (s, v, d) in enumerate(x):
        idx[r, j], sg[r, j], dl[r, j] = s, v, d
idx, sg, dl = idx.to(dev), sg.to(dev), dl.to(dev)
flat = (dl * NSRC + idx)                          # 遅れの輪 [DMAX+1, NSRC] を平らにした番号
hz = torch.cat([torch.full((NUC[k],), HZ[k]) for k in names]).to(dev)


def make(dt_ms):
    a = math.exp(-dt_ms / 40.0)
    p = hz * dt_ms / 1000.0                         # 1刻みに撃つ割合（i0 は入力なしでこの率になる値）
    steps = 1.0 / p.clamp(max=0.999)
    i0 = (1 - a) / (1 - a ** steps)
    return a, i0


S = {"v": torch.rand(N, device=dev, generator=g), "ring": torch.zeros(DMAX + 1, NSRC, device=dev),
     "ext": torch.zeros(N_STR + N_PFC, device=dev)}
W = 0.02


def tick(a, i0):
    ring = S["ring"]
    x = (ring.view(-1)[flat] * sg).sum(1)          # 遅れごとの送り手の発火 × 符号
    v = S["v"] * a + i0 + W * x
    f = (v >= 1.0).float()
    S["v"].copy_(v * (1 - f))
    ring.copy_(torch.roll(ring, 1, 0))             # 遅れを1周進める
    ring[0, :N].copy_(f)
    ring[0, N:].copy_(S["ext"])
    return f


def run_step(per_pass):
    if per_pass:
        a, i0 = AB10
        for _ in range(10):
            tick(a, i0)
    else:
        a, i0 = AB1
        tick(a, i0)


AB1, AB10 = make(10.67), make(1.067)


def timeit(fn, n):
    sync(); t = time.perf_counter()
    for _ in range(n):
        S["ext"].copy_((torch.rand(N_STR + N_PFC, device=dev, generator=g) < 0.02).float())   # 外の発火（1歩に1回）
        fn()
    sync()
    return (time.perf_counter() - t) / n * 1000


print(f"{'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'} ／ 核 {N}体 ／ 受け手1体の本数 最大 {KMAX} ／ 歩 {NS}", flush=True)
res = {}
for name, pp in (("A 1歩ごと", False), ("B 周回ごと（10回）", True)):
    for _ in range(20):
        run_step(pp)
    res[name + " そのまま"] = timeit(lambda: run_step(pp), NS)
    if gpu:
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                run_step(pp)
        torch.cuda.current_stream().wait_stream(st)
        G = torch.cuda.CUDAGraph()
        with torch.cuda.graph(G):
            run_step(pp)
        res[name + " グラフ"] = timeit(G.replay, NS)
for k, v in res.items():
    print(f"  {k:<22} 1歩あたり {v:7.3f}ms", flush=True)

# 今の網（版92）の1歩と、核を足したとき
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
up = (lambda x: TB.to_dev(x, dev)) if hasattr(TB, "to_dev") else (lambda x: torch.tensor(x, device=dev))


def net(n, extra=None):
    k = 0; t_all = 0.0
    while k < n:
        mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
        b.new_utterance(bool(np.any(cmd)))
        for t in range(min(len(ear), n - k)):
            e, c = up(np.asarray(ear[t], np.float32)), up(np.asarray(cmd[t], np.float32))
            sync(); t0 = time.perf_counter()
            b.step(e, c, learn=True)
            if extra is not None:
                extra()
            sync(); t_all += time.perf_counter() - t0
        k += min(len(ear), n - k)
    return t_all / n * 1000


net(60)
# 版2（10-16 0時40分【み】「今のモデルの方は測ってないんじゃないのか？」）：
#   前の版は 網だけ → ＋A → ＋B を1回ずつ順に測った ⇒ 後ほど速く出る（足したのに速い）。揺れ ±0.4ms が差より大きく、差を測れていなかった
#   ⇒ 網だけ・＋A・＋B を 50歩ずつ交互に R 回まわし、組ごとの差の中央と幅を出す
graphs = {"網だけ": None}
if gpu:
    for name, pp in (("＋A 1歩ごと", False), ("＋B 周回ごと", True)):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            run_step(pp)
        torch.cuda.current_stream().wait_stream(st)
        G = torch.cuda.CUDAGraph()
        with torch.cuda.graph(G):
            run_step(pp)
        graphs[name] = G.replay
R = max(1, NS // 50)
rec = {k: [] for k in graphs}
for r in range(R):
    order = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
    for k in order:
        rec[k].append(net(50, graphs[k]))
base = np.array(rec["網だけ"])
print(f"\n今の網（版 {B.VERSION}）1歩 中央 {np.median(base):.2f}ms（{base.min():.2f}〜{base.max():.2f}、50歩 × {R}回。歩ごとに待つので、ふだんの測りより少し遅い）", flush=True)
for k in graphs:
    if k == "網だけ":
        continue
    x = np.array(rec[k]); d = x - base
    print(f"  {k}：1歩 中央 {np.median(x):.2f}ms ／ 網だけとの差 中央 {np.median(d):+.2f}ms（{d.min():+.2f}〜{d.max():+.2f}）／ 実時間まで {10.67 - np.median(x):+.2f}ms", flush=True)
