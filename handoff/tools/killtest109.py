"""11-12：プロセスを落としたとき、もう GPU で走っている仕事をドライバが止めるかを確かめる（網は使わない・保存しない）
  【み】11-12「どちらも防ぐ必要があるだけでしょ」：見張りが落とせても、走っている仕事が止まらなければ画面は守れない
  python3 killtest109.py      （arkhe の ~/anaplex で）
やること（GPU の仕事は 1つずつ・どれも 1.5秒まで ⇒ 画面の待てる 10秒より十分短い）
  1 時計合わせ：torch.cuda._sleep（GPU の上で数を数えて待つだけの仕事・1つの波）を短い数から倍にして、0.2秒を越えたら止める
  2 空のとき：何も走らせずに GPU の働きを 1秒見る
  3 落とさない：1.5秒の仕事を1つ走らせ、GPU の働きが高いのが何秒続くかを見る（働きの値がこの仕事で上がるかの確かめ）
  4 落とす：同じ仕事を走らせ、0.3秒で SIGKILL。落としてから働きが下がるまでの秒を見る
     ⇒ 0.3秒ほどで下がる：ドライバが走っている仕事を止める（見張りで画面を守れる）
     ⇒ 1.2秒ほど（仕事の残り）続く：止めない（長い GPU の仕事を作らないことだけが守り）
  見る親のプロセスは GPU を使わない。GPU の働き（gpu_busy_percent）は 0.1秒ごと・合わせて約5秒だけ読む
  life.py は通らない（守りは無い）。仕事の長さはこの道具の中で 1.5秒に切ってある
  容器で道具を確かめるとき：python3 killtest109.py --fake（GPU を使わない。働きの値は「子が生きているか」で作る）
"""
import os, signal, subprocess, sys, time, glob

FAKE = "--fake" in sys.argv
RUN_S, KILL_AT, HIGH = 1.5, 0.3, 50

CAL = r"""
import sys, time, torch
assert torch.cuda.is_available()
torch.cuda._sleep(1000); torch.cuda.synchronize()
c = 1 << 16
for _ in range(40):
    t = time.perf_counter(); torch.cuda._sleep(c); torch.cuda.synchronize(); el = time.perf_counter() - t
    if el > 1.0:
        print("OVER", c, el, flush=True); sys.exit(0)
    if el >= 0.2:
        print("RATE", c / el, el, flush=True); sys.exit(0)
    c *= 2
print("NOEFFECT", c, el, flush=True)
"""
RUN = r"""
import sys, time, torch
n = int(sys.argv[1])
torch.cuda._sleep(1000); torch.cuda.synchronize()
torch.cuda._sleep(n)
print("GO", flush=True)
torch.cuda.synchronize()
print("DONE", flush=True)
"""
FAKE_CAL = 'print("RATE", 1e9, 0.2, flush=True)'
FAKE_RUN = r"""
import sys, time
print("GO", flush=True); time.sleep(int(sys.argv[1]) / 1e9); print("DONE", flush=True)
"""


def busy_path():
    for p in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        try:
            int(open(p).read().strip()); return p
        except (OSError, ValueError):
            pass
    return None


PATH = None if FAKE else busy_path()
if not FAKE and PATH is None:
    sys.exit("⚠️ GPU の働きが読めない（gpu_busy_percent が無い）⇒ 確かめられない")
CHILD = {"p": None}


def busy():
    if FAKE:
        p = CHILD["p"]
        return 100 if (p is not None and p.poll() is None) else 0
    try:
        return int(open(PATH).read().strip())
    except (OSError, ValueError):
        return -1


def child(code, *args):
    return subprocess.Popen([sys.executable, "-I", "-c", code, *map(str, args)], stdout=subprocess.PIPE, text=True)


def watch(sec):
    out, t0 = [], time.monotonic()
    while time.monotonic() - t0 < sec:
        out.append((time.monotonic(), busy())); time.sleep(0.1)
    return out


def high_until(samples, t_from):
    """t_from より後で、働きが HIGH% 未満になった最初の見本の時刻（ずっと高いなら None）"""
    for t, b in samples:
        if t >= t_from and 0 <= b < HIGH:
            return t
    return None


# 1 時計合わせ
p = child(FAKE_CAL if FAKE else CAL); line = p.communicate(timeout=60)[0].split()
if not line or line[0] != "RATE":
    sys.exit(f"⚠️ 時計合わせができない：{' '.join(line)}（_sleep が効かない・または 1回で 1秒を越えた）⇒ ここで止める")
rate = float(line[1]); n = int(rate * RUN_S)
print(f"1 時計合わせ：{float(line[2]):.2f}秒 ⇒ 1.5秒の仕事 ＝ 数 {n:,}", flush=True)

# 2 空のとき
s = watch(1.0); idle = [b for _, b in s if b >= 0]
print(f"2 空のとき：GPU の働き {min(idle)}〜{max(idle)}%（見本 {len(idle)}）", flush=True)

# 3 落とさない
CHILD["p"] = p = child(FAKE_RUN if FAKE else RUN, n)
assert p.stdout.readline().strip() == "GO"; t_go = time.monotonic()
s = watch(RUN_S + 1.5); p.wait(timeout=30)
hi = [b for t, b in s if t - t_go < RUN_S - 0.2 and b >= 0]
end = high_until(s, t_go)
frac = sum(b >= HIGH for b in hi) / max(len(hi), 1)
print(f"3 落とさない：走っているあいだ 働き {HIGH}% 以上の見本 {frac:.0%}・下がったのは走り出して "
      f"{'下がらない' if end is None else f'{end - t_go:.1f}秒'}（仕事は {RUN_S}秒）", flush=True)
if frac < 0.8:
    sys.exit(f"⚠️ この仕事で GPU の働きの値が上がらない ⇒ 4 の落とす試しは読めない（ここで止める）")

# 4 落とす
CHILD["p"] = p = child(FAKE_RUN if FAKE else RUN, n)
assert p.stdout.readline().strip() == "GO"; t_go = time.monotonic()
s = watch(KILL_AT)
p.send_signal(signal.SIGKILL); t_kill = time.monotonic(); p.wait(timeout=30)
t_dead = time.monotonic()
s += watch(RUN_S + 1.0)
end = high_until(s, t_kill)
if end is None:
    v = "下がらない（見た 2.5秒のあいだずっと高い）"
else:
    v = f"{end - t_kill:.1f}秒（走り出して {end - t_go:.1f}秒）"
print(f"4 落とす：走り出して {t_kill - t_go:.1f}秒で SIGKILL・プロセスが消えるまで {t_dead - t_kill:.1f}秒・働きが下がるまで {v}", flush=True)
if end is not None and end - t_go < RUN_S - 0.5:
    print("⇒ 落とすと、走っている GPU の仕事も止まる（見張りで画面を守れる）", flush=True)
elif end is None or end - t_go >= RUN_S - 0.3:
    print("⇒ 落としても、走っている GPU の仕事は最後まで走る（見張りだけでは画面を守れない・長い GPU の仕事を作らないことが守り）", flush=True)
else:
    print("⇒ どちらとも言い切れない（下がった時刻が仕事の終わりに近い）", flush=True)
