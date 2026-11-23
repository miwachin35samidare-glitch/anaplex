"""版107e：その回の GPU 全体の最大使用（ドライバの値。ほかのプログラム・画面の分も入る）を測って、終わりに出して残す
  【み】11-05 07:30「ハード側でAnaPlex全体のその回での最大使用量がわかるようにしておいて」
  網を組むとき（Brain の __init__）に start() ⇒ 裏で 1秒ごとに torch.cuda.mem_get_info（ドライバ）を見る
    ⟲ 20ms ごと（版107e の初め）は、見ること自体が網の歩とドライバ・Python を取り合う【み】11-05 12:17 ⇒ 1秒
  1秒の間の短い山は、PyTorch が自分で数えている取り置きの山（max_memory_reserved、数えるのに手間なし・取りこぼしなし）で拾う
    GPU 全体の山の下限 ＝ max（1秒ごとに見た山, 取り置きの山 ＋ PyTorch の外の一番小さい値）
    ⟲ 外の一番大きい値を足した形（11-05）は、外と取り置きの山が別の時刻なので 18.51GB のような見込みすぎが出た
  終わり（プログラムを閉じるとき）に 1行出し、logs/vram_peak.log に 1行足す
"""
import atexit, datetime, os, sys, threading, time

_state = {"on": False}


def start(dev_index=0, every_s=1.0):
    """何度呼んでも 1回だけ始める（GPU が無ければ何もしない）"""
    if _state["on"]:
        return
    import torch
    if not torch.cuda.is_available():
        return
    _state.update(on=True, peak=0.0, t_peak=0.0, t0=time.time(), first=None, total=0.0, n=0, stop=False, outside=0.0, out_min=None)

    def _look():
        free, total = torch.cuda.mem_get_info(dev_index)
        used = (total - free) / 1024 ** 3
        _state["total"] = total / 1024 ** 3
        if _state["first"] is None:
            _state["first"] = used
        if used > _state["peak"]:
            _state["peak"], _state["t_peak"] = used, time.time() - _state["t0"]
        rs_ = torch.cuda.memory_reserved(dev_index) / 1024 ** 3
        _state["outside"] = max(_state["outside"], used - rs_)
        if rs_ > 0:
            _state["out_min"] = used - rs_ if _state["out_min"] is None else min(_state["out_min"], used - rs_)
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
    try:
        import torch
        rpk = torch.cuda.max_memory_reserved() / 1024 ** 3
    except Exception:
        rpk = 0.0
    omin = _state["out_min"] or 0.0
    est = max(_state["peak"], rpk + omin)
    tool = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "?"
    line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {tool} {' '.join(sys.argv[1:])} ／ GPU 全体の山 {_state['peak']:.2f}GB"
            f"（1秒ごと・始めから {_state['t_peak']:.0f}秒）／ 山の下限 {est:.2f}GB（取り置きの山 {rpk:.2f} ＋ 外の一番小さい値 {omin:.2f}）／ 始めの使用 {(_state['first'] or 0):.2f}GB ／ 全体 {_state['total']:.2f}GB")
    print(f"［VRAM］この回の GPU 全体の山は少なくとも {est:.2f}GB（PyTorch の取り置きの山 {rpk:.2f} ＋ 外の一番小さい値 {omin:.2f}）"
          f"・1秒ごとに見た山 {_state['peak']:.2f}GB（始めから {_state['t_peak']:.0f}秒）・網を組む前 "
          f"{(_state['first'] or 0):.2f}GB・全体 {_state['total']:.2f}GB）⇒ logs/vram_peak.log", flush=True)
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "vram_peak.log"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
