"""11-11：どの測りの道具も、この包みを通して回す（網・コードは変えない。版に依らない）
  python3 run_safe.py measure_xxx.py 引数…

  11-11 22:25〜22:35 に落ちた記録（data/arkhe_crash_1111.txt）：
    ・落ちる直前の暮らしの 100歩は 1歩の中央 5,763ms・最大 37,435ms（ふだんは 23ms）⇒ 1歩が画面の待てる 10秒を越えていた
    ・はじめから遅かったので、「はじめの 2倍で止める」（apply_duty107）は効かなかった（はじめ自体が遅い）
    ・VRAM の記録：11-11 07:43 measure_srccx107 で 15.92GB（全部）・19:37 measure_nan109 で 13.54GB。決めの 12GB を越える回が出ている
      VRAM が尽きると、画面の側（Xorg・chrome）の置き場が追い出され、計算も画面も遅くなる【見】（確かめていない）
  守り方（どれも、越えたら網の側を止める。画面・ゲームは絞らない）：
    1 VRAM：PyTorch が取れるのを 12GB までにする（越える取り方はその場で止まる）。ANAPLEX_VRAM_GB で変える
    2 1歩：21歩目からは、1歩が 2秒を越えたらその歩のあとで止める。ANAPLEX_STEP_MAX_S で変える
      はじめの 20歩は torch.compile・グラフ作りで長い歩がありうる（そのあいだ GPU は空いている）ので 2 は掛けず、3 に任せる
      ふだん 1歩だけ 1.7〜1.8秒の歩が出る（step_ms.log の最大 1,689・1,841ms）⇒ 2秒は近い。誤って止まったら ANAPLEX_STEP_MAX_S=4 など
    3 見張り：歩が 8秒 進まない（1歩の中・測りの道具の歩のあいだの計算のどちらでも）うえに、そのあいだ GPU が
      ずっと 90% 以上働いているなら、網のプロセスを落とす（GPU を手放させ、画面の 10秒の時間切れより先に空ける）
      GPU が働いているかは /sys/class/drm/card*/device/gpu_busy_percent（ドライバの値）を 0.5秒ごとに読む ⇒ 作りの最中（CPU だけ）では落とさない
      読めないときは、21歩目から「8秒 進まない」だけで落とす。ANAPLEX_WATCH_S で変える・0 で外す
    ⚠️ 1歩の中の GPU の仕事は割り込めない。3 でプロセスを落としても、ドライバが仕事を片付けるまで画面は待つ（GPU では試していない）
"""
import os, sys, time, glob, runpy

if len(sys.argv) < 2:
    sys.exit("使い方：python3 run_safe.py measure_xxx.py 引数…")

import torch

GB = float(os.environ.get("ANAPLEX_VRAM_GB", "12"))
STEP_MAX_S = float(os.environ.get("ANAPLEX_STEP_MAX_S", "2.0"))
WATCH_S = float(os.environ.get("ANAPLEX_WATCH_S", "8.0"))
WARM = 20          # はじめのこの歩数は 2 を掛けない
BUSY = 90          # 3：GPU がこれ以上働いているとき（%）

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(GB * 1024 ** 3 / total, 0.95)
    torch.cuda.set_per_process_memory_fraction(frac, 0)

    busy_path = None
    for p_ in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        try:
            int(open(p_).read().strip())
            busy_path = p_
            break
        except (OSError, ValueError):
            pass
    print(f"［守り］PyTorch の VRAM は {GB:g}GB まで（全体 {total / 1024 ** 3:.2f}GB の {frac:.3f}）"
          f"・{WARM + 1}歩目から 1歩 {STEP_MAX_S:g}秒 を越えたら止める"
          f"・{WATCH_S:g}秒 進まず GPU が {BUSY}% 以上なら落とす（GPU の働き：{busy_path or '読めない ⇒ 21歩目から時間だけで'}）", flush=True)

    import life, threading
    _step = life.Life.step
    _last = {"t": time.perf_counter(), "n": 0}

    def _busy():
        try:
            return int(open(busy_path).read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def _watch():
        low_at = time.perf_counter()        # GPU が BUSY 未満だった最後の時刻
        while True:
            time.sleep(0.5)
            now = time.perf_counter()
            b = _busy() if busy_path else None
            if b is not None and b < BUSY:
                low_at = now
            stuck = now - _last["t"]
            if b is not None:
                bad = stuck > WATCH_S and now - low_at > WATCH_S     # 進まない ＆ GPU が続けて働いている
            else:
                bad = _last["n"] >= WARM and stuck > WATCH_S
            if bad:
                print(f"🛑 落とした（画面を守る）：最後の歩から {stuck:.1f}秒 進まない ＞ {WATCH_S:g}秒"
                      f"（{_last['n']}歩 済み・GPU の働き {b if b is not None else '?'}%）", flush=True)
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
        if _last["n"] > WARM and el > STEP_MAX_S:
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
