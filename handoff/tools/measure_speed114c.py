"""版114：1歩の固定の手間が、コードのどの行・どの関数から出ているか（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_speed114c.py [温め=6000] [行を測る歩=50] [見る歩=5]
  ANAPLEX_GRAPH=eager python3 measure_speed114c.py …   ⇒ グラフを撮らずに回す（グラフの中身 _fast_body の行も測れる。値は同じ・遅い）
  （コンテナ：python3 measure_speed114c.py 30 3 2 --test ⇒ CPU・相手 200）
  何のため：measure_speed114b（11-22 21:44）で 1歩の固定の手間 約25ms（網の領野の外 約9ms・網の領野 11.6ms）・GPU の仕事 1歩 4,664回。
    回数を減らす直し（束ねる）は網の値や線を変えても残る手 ⇒ どこから束ねるのが効くかを見る【み】11-22 21:47「すすめて」
  L 行ごとの時間（行を測る歩）：Brain.step・Group2.step・_fast_step・_fast_body・_advance_tail の行ごとに、
      行の終わりで GPU を待って時間を取る（その行が出した GPU の仕事の時間 ＋ CPU の手間。行の中で呼んだ関数の分も入る）
      ⚠️ 行ごとに待つので全体は遅く出る。グラフを撮っている最中は待たない
  P 関数ごとの命令の数（見る歩、torch.profiler・呼び出し元つき）：GPU の仕事を出した命令を、この網のファイルの一番内側の関数に結ぶ
      ⚠️ torch の profiler の呼び出し元は関数の頭の行（行までは出ない）。torch.compile の塊は1命令で中に幾つも
  毎歩に足す仕事：L は行ごとの待ち（GPU の仕事は足さない）／ P は記録だけ
"""
import sys, os, time, inspect, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, life as L, torch_group2 as G2
from torch_brain import Brain

if B.VERSION not in (114,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版114 の網のため")
TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 6000
NL = int(args[1]) if len(args) > 1 else 50
NP = int(args[2]) if len(args) > 2 else 5
if TEST:
    B.DEV = "cpu"
dev = B.DEV
cuda = str(dev).startswith("cuda")


def sync():
    if cuda and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


OURS = ("torch_brain.py", "torch_group2.py", "torch_area.py", "life.py", "babble.py", "brainstem.py", "midbrain.py",
        "voice.py", "ear.py", "mouth.py", "day.py", "sleep.py", "occlusion.py", "voicevox.py", "config.py")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
print(f"版 {B.VERSION} ／ ANAPLEX_GRAPH={os.environ.get('ANAPLEX_GRAPH', '1')} ／ 温め {WARM} 歩・行を測る {NL} 歩・見る {NP} 歩", flush=True)
t0 = time.time()
for i in range(WARM):
    life.step()
    if (i + 1) % 1000 == 0:
        L.beat()
        print(f"  温め {i + 1} 歩（{(time.time() - t0) / (i + 1) * 1000:.1f}ms/歩）", flush=True)
sync()
ts = []
for i in range(50):
    a0 = time.perf_counter(); life.step(); sync(); ts.append((time.perf_counter() - a0) * 1000)
L.beat()
print(f"  測る前の 50歩：1歩 中央 {np.median(ts):.1f}ms", flush=True)

# ── L 行ごとの時間 ──
TARGETS = {}
for nm, fn in (("Brain.step", Brain.step), ("Group2.step", G2.Group2.step), ("_fast_step", G2.Group2._fast_step),
               ("_fast_body", G2.Group2._fast_body), ("_advance_tail", G2.Group2._advance_tail)):
    TARGETS[inspect.unwrap(fn).__code__] = nm
SRC = {}
line_t = collections.defaultdict(float)      # (名, 行) → ms
calls = collections.Counter()
state = {}


def make_local(code):
    nm = TARGETS[code]

    def tr(frame, event, arg):
        if event == "line" or event == "return":
            sync()
            now = time.perf_counter()
            key = id(frame)
            last = state.get(key)
            if last is not None:
                line_t[(nm, last[0])] += (now - last[1]) * 1000
            if event == "line":
                state[key] = (frame.f_lineno, time.perf_counter())
            else:
                state.pop(key, None)
        return tr
    return tr


LOCALS = {c: make_local(c) for c in TARGETS}


def glob_tracer(frame, event, arg):
    if event == "call" and frame.f_code in LOCALS:
        calls[TARGETS[frame.f_code]] += 1
        if frame.f_code not in SRC:
            try:
                lines, lo = inspect.getsourcelines(frame.f_code)
                SRC[frame.f_code] = (lo, lines)
            except Exception:
                SRC[frame.f_code] = (0, [])
        return LOCALS[frame.f_code]
    return None


sync()
t1 = time.perf_counter()
sys.settrace(glob_tracer)
try:
    for i in range(NL):
        life.step()
        if (i + 1) % 10 == 0:
            L.beat()
finally:
    sys.settrace(None)
sync()
tl = (time.perf_counter() - t1) / NL * 1000
L.beat()
code_of = {v: k for k, v in TARGETS.items()}
print(f"\nL 行ごとの時間（{NL} 歩・行ごとに待つので遅い：この間の暮らしの1歩 {tl:.1f}ms）。呼ばれた回数：{dict(calls)}", flush=True)
for nm in ("Brain.step", "Group2.step", "_fast_step", "_fast_body", "_advance_tail"):
    rows = [(t, ln) for (n_, ln), t in line_t.items() if n_ == nm]
    if not rows:
        continue
    tot = sum(t for t, _ in rows)
    lo, lines = SRC.get(code_of[nm], (0, []))
    print(f"\n  {nm}：1歩あたり 合わせて {tot / NL:.2f}ms（行の多い順・上位 25）", flush=True)
    for t, ln in sorted(rows, reverse=True)[:25]:
        txt = lines[ln - lo].strip() if lines and 0 <= ln - lo < len(lines) else ""
        print(f"    {t / NL:7.3f}ms  {ln:5d}  {txt[:95]}", flush=True)
L.beat()

# ── P 関数ごとの命令の数 ──
from torch.profiler import profile, ProfilerActivity
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
try:
    _cfg = dict(experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True))
except Exception:
    _cfg = {}
try:
    with profile(activities=acts, record_shapes=False, with_stack=True, **_cfg) as prof:
        for i in range(NP):
            life.step()
            sync()
            L.beat()
    L.beat()
    evs = prof.events()
    L.beat()
    dattr = "self_device_time_total" if hasattr(evs[0], "self_device_time_total") else "self_cuda_time_total"
    tattr = dattr if cuda else "self_cpu_time_total"

    def ours(e):
        out = []
        for fr in (e.stack or []):
            f = fr.split("(")[0]
            if os.path.basename(f) in OURS:
                out.append(os.path.basename(f) + fr[len(f):])
                if len(out) == 2:
                    break
        return out

    by_pair = collections.defaultdict(lambda: [0, 0.0])
    n_all = t_all = n_none = 0
    for e in evs:
        if not str(getattr(e, "device_type", "")).endswith("CPU"):
            continue
        t = getattr(e, tattr, 0) or 0
        if t <= 0 or (cuda and e.name.startswith(("hip", "cuda"))):
            continue
        o = ours(e)
        n_all += 1; t_all += t
        if not o:
            n_none += 1
            continue
        k = o[0] + "  ←  " + (o[1] if len(o) > 1 else "")
        by_pair[k][0] += 1; by_pair[k][1] += t
    unit = "GPU" if cuda else "CPU（容器）"
    print(f"\nP {NP} 歩：{unit} の時間を出した命令 1歩 {n_all / NP:,.0f} 回・{t_all / 1000 / NP:.2f}ms（呼び出し元が網のファイルに無いもの {n_none / NP:,.0f} 回）", flush=True)
    print(f"  関数 ← 呼んだ関数（回数の多い順・1歩あたり。行の数字は関数の頭）", flush=True)
    for k, (n, t) in sorted(by_pair.items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"    {n / NP:7.1f}回 {t / 1000 / NP:7.3f}ms  {k[:110]}", flush=True)
except Exception as ex:
    print(f"\nP は回らなかった：{type(ex).__name__}: {str(ex)[:300]}", flush=True)
if cuda:
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
