"""喃語の最中の VRAM の山が「どこで」出るかを見る（arkhe の ~/anaplex で。state.pt は書き換えない）
  python3 watch_vram2.py [回数=10] [babble.py に渡す引数…]
【み】10-15 watch_vram：喃語の最中に 11.24GB の山（20〜30秒の一度だけ。ふだん 6.4GB）「やっぱ言えてないからじゃないの？」
  ⇒ 山が撃つ量（言えていない網）のせいか、ときどきの処理（刈り込み 1000歩ごと・学びの区切り 1024刻みごと）のせいかを分ける
やり方：babble.main() を同じプロセスで回し、0.05秒ごとに Linux 側（sysfs）の VRAM を読む。
  今までの山を 0.2GB 越えるたびに、その時の本体の呼び出しの列（どの関数の中か）を残す
  網の歩の数（g.t）と、刈り込み・区切りが呼ばれた歩も記録する（中身は変えない。呼ばれたことを数えるだけ）
"""
import glob, os, shutil, sys, tempfile, threading, time, traceback

HERE = os.path.dirname(os.path.realpath(__file__))
GB = 1024 ** 3


def card():
    best = None
    for d in glob.glob("/sys/class/drm/card*/device"):
        u, t = os.path.join(d, "mem_info_vram_used"), os.path.join(d, "mem_info_vram_total")
        if os.path.exists(u) and os.path.exists(t):
            tot = int(open(t).read())
            if best is None or tot > best[1]:
                best = (u, tot)
    return best


cd = card()
if cd is None:
    sys.exit("⚠️ mem_info_vram_used が無い")
USED, TOTAL = cd
read = lambda: int(open(USED).read())
base = max(read() for _ in range(10))
print(f"VRAM 全体 {TOTAL / GB:.2f}GB ／ 始める前 {base / GB:.2f}GB", flush=True)

work = tempfile.mkdtemp(prefix="anaplex_vram_")
for f in os.listdir(HERE):
    if f in ("state.pt", "state.pt.tmp") or f.startswith("."):
        continue
    os.symlink(os.path.join(HERE, f), os.path.join(work, f))
if os.path.exists(os.path.join(HERE, "state.pt")):
    shutil.copy2(os.path.join(HERE, "state.pt"), os.path.join(work, "state.pt"))
    print("state.pt を作業場へ写した（続きから）", flush=True)
else:
    print("state.pt なし ⇒ 0 から（学んでいない網）", flush=True)
os.chdir(work)
sys.path.insert(0, HERE)

import torch_group2 as G
import babble as B

events = []           # (秒, 何, 歩)
t0 = time.time()
cur = {"g": None}


def wrap(name):
    f0 = getattr(G.Group2, name)

    def f(self, *a, **k):
        cur["g"] = self
        s = time.time() - t0
        v0 = read()
        r = f0(self, *a, **k)
        events.append((s, name, int(self.t), v0, read()))
        return r
    setattr(G.Group2, name, f)


for nm in ("rewire", "_new_epoch", "flush_all"):
    if hasattr(G.Group2, nm):
        wrap(nm)
# 差し替えるのは rewire・_new_epoch・flush_all・step だけ（版90 の速い形が見る _pass・_sum・_flush・_fired_edges・_expand には触れない）

main_id = threading.get_ident()
peaks = []            # (秒, バイト, 歩, 呼び出しの列)
samples = []
stop = threading.Event()


def sampler():
    top = 0
    while not stop.is_set():
        v = read(); s = time.time() - t0
        samples.append((s, v))
        if v > top + 0.2 * GB:
            top = v
            fr = sys._current_frames().get(main_id)
            st = [f"{os.path.basename(x.filename)}:{x.lineno} {x.name}" for x in traceback.extract_stack(fr)][-7:] if fr else []
            g = cur["g"]
            peaks.append((s, v, int(g.t) if g is not None else -1, st))
        time.sleep(0.05)


th = threading.Thread(target=sampler, daemon=True)
th.start()
sys.argv = ["babble.py", sys.argv[1] if len(sys.argv) > 1 else "10"] + sys.argv[2:]
_step0 = G.Group2.step


def _step(self, *a, **k):
    cur["g"] = self
    return _step0(self, *a, **k)


G.Group2.step = _step
try:
    B.main()
finally:
    stop.set(); th.join()
    os.chdir(HERE)
    shutil.rmtree(work, ignore_errors=True)

print("\n山が 0.2GB 越えるたび（秒 ／ VRAM ／ 網の歩 ／ 本体の呼び出しの列 新しい方が下）", flush=True)
for s, v, t, st in peaks:
    print(f"  {s:7.1f}s  {v / GB:6.2f}GB  歩 {t}", flush=True)
    for x in st:
        print(f"           {x}", flush=True)
print("\n刈り込み・区切りが呼ばれた所（秒 ／ 名前 ／ 歩 ／ 前 → 後 の VRAM）", flush=True)
for s, nm, t, v0, v1 in events:
    print(f"  {s:7.1f}s  {nm:<11} 歩 {t:6d}  {v0 / GB:6.2f} → {v1 / GB:6.2f}GB", flush=True)
run = [v for s, v in samples]
print(f"\n山 {max(run) / GB:.2f}GB ／ 平均 {sum(run) / len(run) / GB:.2f}GB ／ 最後の歩 {int(cur['g'].t) if cur['g'] is not None else -1}", flush=True)
