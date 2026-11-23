"""喃語を回している間の VRAM を Linux 側（amdgpu の sysfs）から測る（arkhe の ~/anaplex で）
  python3 watch_vram.py [回数=10] [babble.py に渡す引数…]     例：python3 watch_vram.py 10 --fake
【み】10-15「喃語10回回す間のVRAM負荷をAnaPlex側からじゃなくてLinux側から測るツールを回せばいいんじゃないの？」
  measure_mem90 は PyTorch の数（と mem_get_info）。モニタの約10GB は喃語の最中 ⇒ 外から GPU 全体の使用を 0.1秒ごとに読む
⚠️ state.pt は書き換えない：使い捨ての作業場（/tmp）に ~/anaplex のファイルを繋ぎ、state.pt だけ写して、そこで babble.py を回す
   （今の state.pt の続きから回す。回したあとの保存は作業場ごと消す）
出すもの：始める前 ／ 組み終わり（「体 …」の行）までの山 ／ 喃語の最中の山・平均 ／ 10秒ごとの山 ／ babble の行（経過秒つき）
"""
import glob, os, shutil, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.realpath(__file__))
N = sys.argv[1] if len(sys.argv) > 1 else "10"
EXTRA = sys.argv[2:]
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
    sys.exit("⚠️ /sys/class/drm/card*/device/mem_info_vram_used が無い（amdgpu でない？）")
USED, TOTAL = cd
read = lambda: int(open(USED).read())
print(f"VRAM 全体 {TOTAL / GB:.2f}GB（{USED}）", flush=True)
base = [read() for _ in range(10)]
print(f"始める前 {max(base) / GB:.2f}GB", flush=True)

work = tempfile.mkdtemp(prefix="anaplex_vram_")
for f in os.listdir(HERE):
    if f in ("state.pt", "state.pt.tmp") or f.startswith("."):
        continue
    os.symlink(os.path.join(HERE, f), os.path.join(work, f))
if os.path.exists(os.path.join(HERE, "state.pt")):
    shutil.copy2(os.path.join(HERE, "state.pt"), os.path.join(work, "state.pt"))
    print("state.pt を作業場へ写した（続きから回す。~/anaplex の state.pt は触らない）", flush=True)

samples = []          # (秒, バイト)
mark = {}             # 組み終わりの秒
t0 = time.time()
stop = threading.Event()


def sampler():
    while not stop.is_set():
        samples.append((time.time() - t0, read()))
        time.sleep(0.1)


th = threading.Thread(target=sampler, daemon=True)
th.start()
p = subprocess.Popen([sys.executable, "-u", "babble.py", N] + EXTRA, cwd=work,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    s = time.time() - t0
    if "built" not in mark and line.startswith("体 "):
        mark["built"] = s
    print(f"  [{s:7.1f}s ／ {read() / GB:5.2f}GB] {line.rstrip()}", flush=True)
p.wait()
stop.set(); th.join()
shutil.rmtree(work, ignore_errors=True)

b0 = max(base)
tb = mark.get("built", 0.0)
pre = [v for t, v in samples if t <= tb] or [0]
run = [v for t, v in samples if t > tb] or [0]
print(f"\nbabble の終わり（返り値 {p.returncode}）／ 測った点 {len(samples)}", flush=True)
print(f"始める前        {b0 / GB:6.2f}GB", flush=True)
print(f"組み終わりまで  山 {max(pre) / GB:6.2f}GB（{tb:.0f}秒まで）", flush=True)
print(f"喃語の最中      山 {max(run) / GB:6.2f}GB ／ 平均 {sum(run) / len(run) / GB:6.2f}GB", flush=True)
print(f"⇒ この網の分（山 − 始める前）{(max(max(pre), max(run)) - b0) / GB:.2f}GB ／ 全体の残り {(TOTAL - max(max(pre), max(run))) / GB:.2f}GB", flush=True)
print("10秒ごとの山", flush=True)
for k in range(int(samples[-1][0] // 10) + 1 if samples else 0):
    w = [v for t, v in samples if k * 10 <= t < (k + 1) * 10]
    if w:
        print(f"  {k * 10:5d}〜{(k + 1) * 10:5d}秒  {max(w) / GB:6.2f}GB", flush=True)
