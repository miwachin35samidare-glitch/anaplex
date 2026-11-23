"""版97：時間を続けて流す ／ 口はいつも読む ／ 返し手と録音はどの声にも（10_決まったこと §12・§13）
【み】10-16 21:55「OK」（口はいつも読む）・「その場合自発が出ると返し手と録音をそれに対応させないといけないね」・10-24「続けて」・10-25「続きを組んで」
これまで（版96 まで）：合図 → 準備 103歩 → 口が窓の終わりに1回だけ読む → 声 → 返し手 の順番の台本。返事を聞く間・黙る間は口を読まない ⇒ 自発の声は出ようがない
これから
  時間  1歩ずつ流れ続ける。耳にはその歩に鳴っている音（自分の声・返し手・録音）を全部重ねて入れる（音の強さを足す）
  口    毎歩、運動の領野の表現の発火を筋の形でならし、力が READ_TH を越えた型を語として読む。鳴っている間は次を読まない
  返し手 口が実際に出した声に、どの声（合図・自発）にも同じ割合・遅れで返す【ヒト】REPLY_CUM（家庭の一日録音）
  番の並び（足場の入れ方）：版96 までの並びのまま（新しい値を足さない）
        1番 ＝ 補助輪がモーラを選ぶ（2割は黙る：SILENCE_PROB）。選べば合図（補足へ）と語の指令（運動の指令の体へ、声の CMD_LEAD 歩前から）
        番の終わり：自分の声（後ろの静けさ 20歩込み）が鳴り終わり、返し手が返したならその声も鳴り終わったとき。黙る番は LATENCY ＋ 20歩
        ⟲ 10-24 の版97 は合図の間を【ヒト】乳児の発声 1分に 4.5回（指数）で置いた ⇒ 誤り（それは自発の声を確かめる目安で、作りの値ではない）
  録音  口が出した声を全部録る（自発も）。健診（rec_test）では 10番ごとに3つ（2回以上前の録音からでたらめ）を番として流す
        学びは止めない【み】10-09「学ばずに聞かせるだけって反復横飛びじゃないの？」（measure_check2 と同じ）
        録音のあいだも口・返し手は回る【み】10-25「赤ちゃんに録音を聞かせるとき、赤ちゃんは黙っていてくれないよ」
        ⟲ 10-24 の版97 は健診で学びを止めていた（誤り）
"""
import os as _os, time as _time
import torch as _torch
# 11-10【み】：AnaPlex が GPU を 100% 塞ぐと画面が固まる（11-09 23:35）。「5％でいいから空けといて・95％までは使い切っていい」
#   1歩ごとに、その歩にかかった時間の (1/DUTY − 1) 倍だけ休む ⇒ GPU を使う時間の割合が DUTY 以下（ゲーム・画面は絞らない）
#   ANAPLEX_DUTY=1 で外す（休まない）。網の値は変わらない
DUTY = min(max(float(_os.environ.get("ANAPLEX_DUTY", "0.95")), 0.05), 1.0)
# 11-11【み】「穴だらけ」：止めは 1歩目から・数は短く（「一歩は本来10msなのに1.8秒あるのが異常」「8秒は長すぎる、4秒でいい」）
# 11-12【み】「1歩の中で判断できないの？」「1歩を始めてから2秒って」「どちらも防ぐ必要があるだけでしょ」
#   どの測りの道具でも、この life.py を通る限り効く（包みを付け忘れても効く）。値は【C】（画面の 10秒の時間切れより十分手前）
#   1 VRAM：PyTorch が取れるのを 12GB までに（越える取り方はその場で止まる）。ANAPLEX_VRAM_GB
#   2 1歩：1歩目から、1歩が 0.5秒を越え、その歩のあいだ GPU が働いていたら（90% 以上の見本が半分以上）止める。ANAPLEX_STEP_MAX_S
#     （GPU が空いたまま長い歩＝作りの最中の CPU だけの計算は、画面を塞がないので止めない。GPU の働きが読めなければ時間だけで止める）
#   3 見張り：別のプロセス（1歩目に life.py が自分で起こす）。最後に進んでから 2秒、そのあいだ GPU が働いていたら
#     （90% 以上の見本が半分以上）網のプロセスを SIGKILL で落とす。1歩の途中でも・測りの道具の歩のあいだの計算でも。ANAPLEX_WATCH_S
#     別のプロセスなので、網が GPU を待って Python を握ったままでも見張りは回る（⟲ 11-11 の見張りは同じプロセスの糸だった）
#     「進んだ」は 1歩の終わりに life.py が書く。道具が歩のあいだに長い計算をするときは、区切りごとに life.beat() を呼ぶ
#     GPU の働きは /sys/class/drm/card*/device/gpu_busy_percent（ドライバの値）を 0.25秒ごとに読む（11-05：細かく聞くと取り合う）
#     ⟲ 11-11 は「4秒のあいだ ずっと 90% 以上」：1回でも下がると数え直し ⇒ 半分以上に
#   ⚠️ 落としたとき、もう GPU で走っている仕事をドライバが止めるかは試していない ⇒ killtest109.py で確かめる（止まらないなら、長い GPU の仕事を作らないことだけが守り）
#   ⟲ 11-10 の止め（はじめの 2倍・61歩目から）は外した：はじめから遅いと効かず（11-11 22:25）、働き出す前に膨らむと守れない
VRAM_GB = float(_os.environ.get("ANAPLEX_VRAM_GB", "12"))
STEP_MAX_S = float(_os.environ.get("ANAPLEX_STEP_MAX_S", "0.5"))
WATCH_S = float(_os.environ.get("ANAPLEX_WATCH_S", "2.0"))
GPU_BUSY = 90
if _torch.cuda.is_available() and VRAM_GB > 0:
    _torch.cuda.set_per_process_memory_fraction(
        min(VRAM_GB * 1024 ** 3 / _torch.cuda.get_device_properties(0).total_memory, 0.95), 0)


def _busy_path():
    import glob as _glob
    for p_ in sorted(_glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        try:
            int(open(p_).read().strip())
            return p_
        except (OSError, ValueError):
            pass
    return None


_G = {"started": False, "path": None, "n": 0, "mm": None, "proc": None}
_NS = 64                                   # 共有する GPU の働きの見本の数（0.25秒ごと ⇒ 16秒ぶん）
_HB = 24 + 16 * _NS                        # 共有の帳：最後に進んだ時刻 d・歩 q・見本の番号 q・見本 (時刻 d, 働き q) × _NS

# 見張りのプロセス（引数：網の pid・帳・秒・GPU の働きのファイル・境の %）。網が先に終われば自分も終わる
_WATCH_CODE = r"""
import os, sys, time, struct, mmap, signal
pid, hb, W, path, BUSY, NS = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4] or None, int(sys.argv[5]), int(sys.argv[6])
fd = os.open(hb, os.O_RDWR); mm = mmap.mmap(fd, 24 + 16 * NS); os.unlink(hb)
k = 0
while True:
    time.sleep(0.25)
    if os.getppid() != pid:
        sys.exit(0)
    now = time.monotonic()
    b = -1
    if path:
        try:
            b = int(open(path).read().strip())
        except (OSError, ValueError):
            b = -1
    struct.pack_into("dq", mm, 24 + 16 * (k % NS), now, b); k += 1
    struct.pack_into("q", mm, 16, k)
    last, n = struct.unpack_from("dq", mm, 0)
    stuck = now - last
    if stuck <= W:
        continue
    win = [struct.unpack_from("dq", mm, 24 + 16 * (i % NS)) for i in range(max(0, k - NS), k)]
    win = [x for (t, x) in win if t >= now - W and x >= 0]
    hot = (sum(x >= BUSY for x in win) / len(win)) if win else None
    if hot is None or hot >= 0.5:
        print(f"🛑 落とした（画面を守る）：{n}歩 済み・最後に進んでから {stuck:.1f}秒 ＞ {W:g}秒"
              f"・そのあいだ GPU が {BUSY}% 以上の見本 {'?' if hot is None else f'{hot:.0%}'}（ANAPLEX_WATCH_S・見張りのプロセス）", file=sys.stderr, flush=True)
        os.kill(pid, signal.SIGKILL)
        sys.exit(0)
"""


def beat():
    """11-12：見張りに「進んだ」と伝える（1歩の終わりに life.py が呼ぶ。道具が歩のあいだに長い計算をするときは区切りごとに呼ぶ）"""
    _G["n"] += 1
    if _G["mm"] is not None:
        import struct as _st
        _st.pack_into("dq", _G["mm"], 0, _time.monotonic(), _G["n"])


def _busy_since(t0):
    """見張りのプロセスが帳に書いた GPU の働きの見本のうち、時刻 t0（monotonic）から後のもの"""
    if _G["mm"] is None:
        return []
    import struct as _st
    k = _st.unpack_from("q", _G["mm"], 16)[0]
    out = []
    for i in range(max(0, k - _NS), k):
        t_, b_ = _st.unpack_from("dq", _G["mm"], 24 + 16 * (i % _NS))
        if t_ >= t0 and b_ >= 0:
            out.append(b_)
    return out


def _start_watch():
    if _G["started"] or WATCH_S <= 0:
        return
    _G["started"] = True
    _G["path"] = _busy_path()
    import mmap as _mm, subprocess as _sp, sys as _sys, tempfile as _tf
    fd, hb = _tf.mkstemp(prefix="anaplex_hb_", dir="/dev/shm" if _os.path.isdir("/dev/shm") else None)
    _os.write(fd, b"\0" * _HB)
    _G["mm"] = _mm.mmap(fd, _HB)
    _os.close(fd)
    _G["n"] -= 1
    beat()                                                   # 1歩目の頭から数える
    _G["proc"] = _sp.Popen([_sys.executable, "-I", "-c", _WATCH_CODE, str(_os.getpid()), hb, str(WATCH_S),
                            _G["path"] or "", str(GPU_BUSY), str(_NS)], stdin=_sp.DEVNULL)
    print(f"［守り］VRAM {VRAM_GB:g}GB まで・1歩 {STEP_MAX_S:g}秒 で止める・{WATCH_S:g}秒 進まなければ別のプロセスが落とす"
          f"（GPU の働き：{'読める' if _G['path'] else '読めない ⇒ 時間だけで'}）", flush=True)   # 11-11【み】パスは出さない（ログを GitHub に上げる）
import numpy as np
import torch

import config as C
import mouth as MO
from torch_area import DT_MS
from torch_brain import to_dev
from voice import LATENCY, SILENCE_PROB

REC_EVERY, REC_N, REC_DELAY = 10, 3, 2              # 健診の録音：10番ごとに3つ・2回以上前（measure_check2 と同じ）
EPS = C.LOG_EPS


class Muscle:
    """mouth.muscle_force と同じ 2次の系を1歩ずつ（直流の利得 1：撃ち続けると発火率に寄る）。GPU の上で回す
    式は線形 ⇒ 1歩の中の 20回の刻みを前もって (y, v, u) の掛け算1回にまとめる（値は muscle_force と同じ）"""

    def __init__(self, n, dev):
        w = 2.0 * np.pi * MO.MUSCLE_FN
        self.d = int(round(MO.MUSCLE_DELAY_MS / DT_MS))
        h = DT_MS / 1000.0 / 20
        M = np.eye(3)                                    # 状態 (y, v, u) → 次の (y, v, u)
        for _ in range(20):
            S = np.array([[1.0, 0.0, 0.0], [-w * w * h, 1.0 - 2.0 * MO.MUSCLE_ZETA * w * h, w * w * h], [0.0, 0.0, 1.0]])
            S[0] = S[0] + h * S[1]                       # y ＋＝ 新しい v × h
            M = S @ M
        self.a = [float(x) for x in M[0]] + [float(x) for x in M[1]]
        self.y = torch.zeros(n, dtype=torch.float64, device=dev); self.v = torch.zeros_like(self.y)
        self.buf = [torch.zeros_like(self.y) for _ in range(self.d)]

    def step(self, spikes):
        self.buf.append(spikes.to(torch.float64))
        u = self.buf.pop(0)
        a = self.a
        y = a[0] * self.y + a[1] * self.v + a[2] * u
        self.v = a[3] * self.y + a[4] * self.v + a[5] * u
        self.y = y
        return self.y


class Mixer:
    """耳に届く音を重ねる。音ごとに（始まりの歩, メル [歩, 40], 種類）"""

    def __init__(self, floor):
        self.floor, self.src = floor, []

    def add(self, t0, mel, kind):
        self.src.append([int(t0), np.asarray(mel, np.float32), kind])

    def at(self, t):
        p = np.zeros(C.N_MELS)
        kinds = []
        keep = []
        for s in self.src:
            t0, mel, kind = s
            if t < t0:
                keep.append(s); continue
            i = t - t0
            if i >= len(mel):
                continue                               # 鳴り終わった
            keep.append(s)
            p += np.maximum(np.exp(mel[i].astype(np.float64)) - EPS, 0.0)
            if mel[i].max() > self.floor + 1e-6:
                kinds.append(kind)
        self.src = keep
        return np.log(p + EPS).astype(np.float32), kinds


class Life:
    """続けて流す暮らし。step() を呼ぶたびに1歩進む"""

    def __init__(self, b, vo, mo, rp, rng, rng_r, dev):
        self.b, self.vo, self.mo, self.rp, self.dev = b, vo, mo, rp, dev
        self.rng, self.rng_r = rng, rng_r
        self.t = 0
        self.mix = Mixer(vo.floor)
        self.mus = Muscle(mo.n_cmd, dev)
        self.n_code_all = MO.MAX_MORAE * mo.n_code     # モーラの層（型が無ければ沈黙なので、ここが越えたときだけ読む）
        self.turn = None                               # 今の番（"言う"・"黙る"・"録音"）
        self.queue = []                                # 次の番（健診の録音）
        self.speak_end = -1                            # 自分の声が鳴り終わる歩
        self.recs = []                                 # 口が出した声の録音（メル）
        self.stats = {"turns": 0, "plans": 0, "voices": 0, "spont": 0, "ok": 0, "tried": 0, "replies": 0, "recs": 0}
        self.zero = torch.zeros(mo.n_cmd, device=dev)
        self.err = {}                                  # 健診：種類 → [聴覚の誤差の和, 歩]

    # ── 番 ──
    def _new_turn(self, t, rec_test):
        if self.queue:
            k = self.queue.pop(0)
            if k == "録音" and len(self.recs) > REC_DELAY:
                mel = self.recs[int(self.rng.integers(0, len(self.recs) - REC_DELAY))]
                self.mix.add(t, mel, "録音")
                self.stats["recs"] += 1
                return {"kind": "録音", "end": t + len(mel)}
        self.stats["turns"] += 1
        if rec_test and self.stats["turns"] % REC_EVERY == 0:
            self.queue += ["録音"] * REC_N                 # この番のあとに
        morae, vol = self.vo.random_morae(self.rng)
        if not morae:                                      # 黙る番（版96 の own_voice と同じ長さ・合図なし）
            return {"kind": "黙る", "end": t + LATENCY + 20}
        self.b.go()
        self.stats["plans"] += 1
        return {"kind": "言う", "t0": t, "one": torch.tensor(self.mo.encode(morae, vol)[0], device=self.dev),
                "want": [m for m, *_ in morae], "said": None, "end": None}

    def _turn_done(self, t):
        tu = self.turn
        if tu["kind"] != "言う":
            return t >= tu["end"]
        if tu["said"] is None:                             # 口が読まなかった：版96 と同じく 準備 ＋ 静けさ 20歩 で閉じる
            if t - tu["t0"] >= C.GO_LEAD + 20:
                self.stats["tried"] += 1
                return True
            return False
        return tu["end"] is not None and t >= tu["end"]

    # ── 1歩 ──
    def step(self, learn=True, rec_test=False):
        t_wall = None
        if str(self.dev).startswith("cuda"):
            _start_watch()
            t_wall = _time.perf_counter()
            self._t_mono = _time.monotonic()
        b, mo = self.b, self.mo
        t = self.t
        if self.turn is not None and self._turn_done(t):
            self.turn = None
        if self.turn is None:
            self.turn = self._new_turn(t, rec_test)
        cmd = self.zero
        tu = self.turn
        if tu["kind"] == "言う":
            dt = t - tu["t0"]
            if dt >= C.GO_LEAD - C.CMD_LEAD and (tu["said"] is None or t < self.speak_end):
                cmd = tu["one"]                            # 語の指令（補助輪）は声の CMD_LEAD 歩前から声の終わりまで
        mel, kinds = self.mix.at(t)
        ear = to_dev(self.vo.encode(mel[None])[0], self.dev)
        _, e, _, _ = b.step(ear, cmd, learn=learn)
        # 健診：その歩に鳴っていたのが1つの種類だけなら、その種類に数える（重なった歩は数えない【C】）
        if len(kinds) == 1:
            s = self.err.setdefault(kinds[0], [0.0, 0])
            s[0] += float(e.sum()); s[1] += 1
        self._mouth(b.mouth_in, t)                         # 口：毎歩読む（版100：運動の領野 ＋ PAG）
        self.t += 1
        if t_wall is not None:                             # 11-10：GPU を AnaPlex で塞ぎきらない（画面・ゲームの分を空ける）
            _torch.cuda.synchronize()
            el = _time.perf_counter() - t_wall
            self._brake(el, t_wall)
            if DUTY < 1.0:
                _time.sleep(el * (1.0 / DUTY - 1.0))

    def _brake(self, el, t0):
        """11-11：1歩目から、1歩が STEP_MAX_S を越え、そのあいだ GPU が働いていたら止める（画面が固まる前に）"""
        beat()
        lg = self.__dict__.setdefault("_step_log", [])   # 11-10：100歩ごとに中央値を logs/step_ms.log へ（落ちても残るように毎回 fsync）
        lg.append(el)
        if len(lg) >= 100:
            try:
                _os.makedirs("logs", exist_ok=True)
                with open("logs/step_ms.log", "a") as f_:
                    f_.write(f"{_time.strftime('%m-%d %H:%M:%S')} {self.t}歩 中央 {np.median(lg) * 1000:.1f}ms 最大 {max(lg) * 1000:.1f}ms\n")
                    f_.flush(); _os.fsync(f_.fileno())
            except OSError:
                pass
            lg.clear()
        if STEP_MAX_S > 0 and el > STEP_MAX_S:
            sm = _busy_since(getattr(self, "_t_mono", 0.0))
            hot = (sum(b >= GPU_BUSY for b in sm) / len(sm)) if sm else None
            if hot is None or hot >= 0.5:
                print(f"🛑 止めた（画面を守る）：{self.t}歩・1歩 {el:.2f}秒 ＞ {STEP_MAX_S:g}秒"
                      f"・そのあいだ GPU が {GPU_BUSY}% 以上の見本 {'?' if hot is None else f'{hot:.0%}'}"
                      f"・PyTorch の取り置き {_torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB（ANAPLEX_STEP_MAX_S）", flush=True)
                try:                                   # 11-14：止めた回も GPU 全体の山を出して残す（_exit は終わりの届けを飛ばすので、ここで呼ぶ）
                    import vram_peak as _vp
                    _vp._report()
                except Exception:
                    pass
                _os._exit(3)

    def _mouth(self, spikes, t):
        """口：運動の領野の表現の発火（この歩）を筋の形でならし、力が READ_TH を越えた型を読む（版100：PAG の発火も同じ欄に）"""
        force = self.mus.step(spikes)
        if t >= self.speak_end and bool((force[:self.n_code_all] >= MO.READ_TH).any()):   # 毎歩見るのは1つの真偽だけ
            said, v = self.mo.decode((force >= MO.READ_TH).float().cpu().numpy()[None])
            if said:
                self._speak(said, v, t)

    def _speak(self, said, v, t):
        snd = self.vo.sound(said, v)                           # 声のメル（後ろに 20歩の静けさ）
        self.mix.add(t + 1, snd, "自分")
        self.speak_end = t + 1 + len(snd) - 20
        end = t + 1 + len(snd)                                 # 静けさ込みの終わり
        self.recs.append(snd)
        self.recs = self.recs[-50:]
        self.stats["voices"] += 1
        ids = [m for m, *_ in said]
        tu = self.turn
        mine = tu is not None and tu["kind"] == "言う" and tu["said"] is None and t - tu["t0"] >= C.GO_LEAD - C.CMD_LEAD
        if mine:
            tu["said"] = ids                                   # 指令（補助輪）が入り始めてからの声だけを、その番の声に数える【C】
            self.stats["tried"] += 1
            self.stats["ok"] += int(ids == tu["want"])
        else:
            self.stats["spont"] += 1                           # 番の足場の外で出た声
        if self.rp is not None:
            ans = self.rp.answer_to(said, v, self.rng_r)       # どの声にも同じ割合・遅れ（声の終わりから）
            if ans is not None:
                self.mix.add(end, ans, "返し手")
                self.stats["replies"] += 1
                end += len(ans)
        if mine:
            tu["end"] = end                                    # 番の終わり：声（と返し手）が鳴り終わったとき

    def run_plans(self, n, learn=True, rec_test=False, until=None):
        """番を n 回終えるまで回す（今の番が終わるまで）。until(life) が真なら止める"""
        goal = self.stats["turns"] + n
        while True:
            self.step(learn, rec_test)
            if until is not None and until(self):
                break
            if self.stats["turns"] >= goal and self.turn is not None and self._turn_done(self.t) and not self.queue:
                break

    def health(self, n=20, learn=True):
        """健診：自分・録音・返し手の聴覚の誤差（その種類だけが鳴っている歩 1歩あたり）。学びは止めない（measure_check2 と同じ）"""
        self.err = {}
        st0 = dict(self.stats)
        self.run_plans(n, learn=learn, rec_test=True)
        r = {k: (v[0] / max(v[1], 1), v[1]) for k, v in self.err.items()}
        own = r.get("自分", (0.0, 0))[0]
        rec = r.get("録音", (0.0, 0))[0]
        rep = r.get("返し手", (0.0, 0))[0]
        d = {k: self.stats[k] - st0[k] for k in self.stats}
        return {"自分": own, "録音": rec, "返し手": rep, "supp": rec / max(own, 1e-12), "supp_rep": rep / max(own, 1e-12),
                "歩": {k: v[1] for k, v in r.items()}, "言えた": d["ok"], "試した": d["tried"],
                "自発": d["spont"], "声": d["voices"], "録音の数": d["recs"], "番": d["turns"]}


# ── 作り物の声（VOICEVOX の代わり。コンテナで回すため）──
class FakeVoice:
    """VOICEVOX を使わない声。モーラの番号は口（mouth.py）と同じ 0〜119。音は babble.make_inventory の形を番号で回して使う
    ⚠️ 声の中身は作り物（喃語の中身は関係ない、の決めのとおり、音が返ってくればよい）"""

    def __init__(self, seed=0):
        import babble as B
        self.B = B
        self.inv = B.make_inventory(np.random.default_rng(seed))
        self.floor = float(np.log(EPS))
        self.n_mora = MO.N_INVENTORY

    def random_morae(self, rng):
        if rng.random() < SILENCE_PROB:                    # 本物の声と同じく 2割は黙る
            return [], 1.0
        n = int(rng.integers(1, MO.MAX_MORAE + 1))
        out = []
        for _ in range(n):
            out.append((int(rng.integers(0, self.n_mora)),
                        float(np.clip(rng.normal(C.CONSONANT_LENGTH_MEAN, C.CONSONANT_LENGTH_STD), *MO.RANGE["consonant_length"])),
                        float(np.clip(rng.normal(C.VOWEL_LENGTH_MEAN, C.VOWEL_LENGTH_STD), *MO.RANGE["vowel_length"])),
                        float(np.clip(rng.normal(C.BABBLE_PITCH_CENTER, 0.05), *MO.RANGE["pitch"]))))
        return out, float(np.clip(rng.normal(C.VOLUME_SCALE_MEAN, C.VOLUME_SCALE_STD), *MO.VOLUME_RANGE))

    def sound(self, morae, volume):
        segs = []
        for m, c, v, p in morae:
            cons, vow = self.inv[m % len(self.inv)]
            nc = max(1, int(round(c * 1000.0 / DT_MS))); nv = max(2, int(round(v * 1000.0 / DT_MS)))
            segs += [cons * volume] * nc + [vow * volume] * nv
        lin = np.clip(np.array(segs), 0, None) if segs else np.zeros((0, C.N_MELS))
        mel = np.log(lin + EPS) if len(lin) else np.zeros((0, C.N_MELS))
        tail = np.full((20, C.N_MELS), self.floor)
        return np.concatenate([mel, tail]).astype(np.float32)

    def encode(self, mel):
        return self.B.encode(np.exp(np.asarray(mel, np.float64)).astype(np.float32))


class FakeReplier:
    """作り物の返し手：返す割合・遅れは本物と同じ（REPLY_CUM）。中身はまね返し（同じモーラ列の作り物の音）"""

    def __init__(self, vo):
        self.vo = vo

    def answer_to(self, morae, volume, rng):
        from voice import REPLY_CUM
        if not morae:
            return None
        u = rng.random()
        if u >= REPLY_CUM[-1]:
            return None
        sec = next(i for i, c in enumerate(REPLY_CUM) if u < c)
        delay = int((sec + rng.random()) * 1000.0 / DT_MS)
        sil = np.full((max(0, delay - 20), C.N_MELS), self.vo.floor, np.float32)
        return np.concatenate([sil, self.vo.sound(morae, volume)])
