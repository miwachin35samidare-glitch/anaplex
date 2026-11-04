"""版107e：その回の GPU 全体の最大使用（ドライバの値。ほかのプログラム・画面の分も入る）を測って、終わりに出して残す
  【み】11-05 07:30「ハード側でAnaPlex全体のその回での最大使用量がわかるようにしておいて」
  網を組むとき（Brain の __init__）に start() ⇒ 裏で 20ms ごとに torch.cuda.mem_get_info を見る
  終わり（プログラムを閉じるとき）に 1行出し、logs/vram_peak.log に 1行足す（日時・道具の名・山・山の時刻・始めの使用・全体）
  ⚠️ 20ms より短い山は取りこぼしうる。網の歩より細かい（1歩 40ms 台）
"""
import atexit, datetime, os, sys, threading, time

_state = {"on": False}


def start(dev_index=0, every_s=0.02):
    """何度呼んでも 1回だけ始める（GPU が無ければ何もしない）"""
    if _state["on"]:
        return
    import torch
    if not torch.cuda.is_available():
        return
    _state.update(on=True, peak=0.0, t_peak=0.0, t0=time.time(), first=None, total=0.0, n=0, stop=False)

    def _look():
        free, total = torch.cuda.mem_get_info(dev_index)
        used = (total - free) / 1024 ** 3
        _state["total"] = total / 1024 ** 3
        if _state["first"] is None:
            _state["first"] = used
        if used > _state["peak"]:
            _state["peak"], _state["t_peak"] = used, time.time() - _state["t0"]
        _state["n"] += 1

    def _loop():
        torch.cuda.set_device(dev_index)
        while not _state["stop"]:
            try:
                _look()
            except Exception:
                pass
            time.sleep(every_s)

    _look()
    threading.Thread(target=_loop, daemon=True, name="vram_peak").start()
    atexit.register(_report)


def peak():
    """今までの山（GB）"""
    return _state.get("peak", 0.0)


def _report():
    if not _state["on"]:
        return
    _state["stop"] = True
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024 ** 3
        if used > _state["peak"]:
            _state["peak"], _state["t_peak"] = used, time.time() - _state["t0"]
    except Exception:
        pass
    tool = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "?"
    line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {tool} {' '.join(sys.argv[1:])} ／ GPU 全体の山 {_state['peak']:.2f}GB"
            f"（始めから {_state['t_peak']:.0f}秒）／ 始めの使用 {(_state['first'] or 0):.2f}GB ／ 全体 {_state['total']:.2f}GB")
    print(f"［VRAM］この回の GPU 全体の山 {_state['peak']:.2f}GB（始めから {_state['t_peak']:.0f}秒・網を組む前 "
          f"{(_state['first'] or 0):.2f}GB・全体 {_state['total']:.2f}GB）⇒ logs/vram_peak.log", flush=True)
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "vram_peak.log"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
