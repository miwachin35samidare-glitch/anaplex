"""11-11：どの測りの道具も、この包みを通して回す（網・コードは変えない。版に依らない）
  python3 run_safe.py measure_xxx.py 引数…

  11-11 22:25〜22:35 に落ちた記録（data/arkhe_crash_1111.txt）：
    ・落ちる直前の暮らしの 100歩は 1歩の中央 5,763ms・最大 37,435ms（ふだんは 23ms）⇒ 1歩が画面の待てる 10秒を越えていた
    ・はじめから遅かったので、「はじめの 2倍で止める」（apply_duty107）は効かなかった（はじめ自体が遅い）
    ・VRAM の記録：11-11 07:43 measure_srccx107 で 15.92GB（全部）・19:37 measure_nan109 で 13.54GB。決めの 12GB を越える回が出ている
      VRAM が尽きると、画面の側（Xorg・chrome）の置き場が追い出され、計算も画面も遅くなる【見】（確かめていない）
  守り方（2つとも、越えたら網の側を止める。画面は絞らない）：
    1 VRAM：PyTorch が取れるのを 12GB までにする（越える取り方は追い出しにならず、その場で止まる）。ANAPLEX_VRAM_GB で変える
    2 1歩：1歩が 2秒（画面の待てる 10秒の 1/5）を越えたら、その歩のあとで止める。ANAPLEX_STEP_MAX_S で変える
      ⚠️ 1歩の中は割り込めない ⇒ 1歩目から 10秒を越える歩は守れない（それは 1 の VRAM で防ぐ）
    3 見張り：歩が 20歩 済んだあと、次の歩が 8秒 済まなければ（1歩の中・測りの道具が歩のあいだにする重い計算のどちらでも）、
      網のプロセスを落とす（GPU の仕事を手放させ、画面の 10秒の時間切れより先に空ける）。ANAPLEX_WATCH_S で変える・0 で外す
"""
import os, sys, time, runpy

if len(sys.argv) < 2:
    sys.exit("使い方：python3 run_safe.py measure_xxx.py 引数…")

import torch

GB = float(os.environ.get("ANAPLEX_VRAM_GB", "12"))
STEP_MAX_S = float(os.environ.get("ANAPLEX_STEP_MAX_S", "2.0"))
WATCH_S = float(os.environ.get("ANAPLEX_WATCH_S", "8.0"))

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(GB * 1024 ** 3 / total, 0.95)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"［守り］PyTorch の VRAM は {GB:g}GB まで（全体 {total / 1024 ** 3:.2f}GB の {frac:.3f}）・1歩 {STEP_MAX_S:g}秒 を越えたら止める・{WATCH_S:g}秒 進まなければ落とす", flush=True)

    import life, threading
    _step = life.Life.step
    _last = {"t": None, "n": 0}

    def _watch():
        while True:
            time.sleep(0.5)
            t = _last["t"]
            if t is not None and _last["n"] >= 20 and time.perf_counter() - t > WATCH_S:
                print(f"🛑 落とした（画面を守る）：最後の歩から {time.perf_counter() - t:.1f}秒 進まない ＞ {WATCH_S:g}秒"
                      f"（{_last['n']}歩 済み）", flush=True)
                os._exit(5)

    if WATCH_S > 0:
        threading.Thread(target=_watch, daemon=True).start()

    def _guarded(self, *a, **k):
        t0 = time.perf_counter()
        r = _step(self, *a, **k)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        _last["t"] = time.perf_counter()
        _last["n"] += 1
        if el > STEP_MAX_S:
            print(f"🛑 止めた（画面を守る）：{getattr(self, 't', '?')}歩・1歩 {el:.2f}秒 ＞ {STEP_MAX_S:g}秒"
                  f"・PyTorch の取り置き {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
            os._exit(3)
        return r

    life.Life.step = _guarded

sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except torch.cuda.OutOfMemoryError as e:
    print(f"🛑 止めた（画面を守る）：VRAM が {GB:g}GB を越える取り方をした ⇒ {str(e).splitlines()[0][:160]}", flush=True)
    os._exit(4)
