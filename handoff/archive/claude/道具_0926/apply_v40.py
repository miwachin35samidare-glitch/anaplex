"""版39c → 版40：学びの門と返し手（arkhe の ~/anaplex で  python3 apply_v40.py ）

設計と出どころは claude/設計案_脳幹.md §8・§8-1。印が1つでも見つからなければ何も書かずに止まる
  学びの門（torch_group2.py）
    その歩の学び η(的 − p)x は、辺ごとの「学んだ跡」にためる。跡は時定数 1秒（94歩）で減る
      【ラット】大人の聴覚野・生体：修飾の信号が 1秒前・同時なら効き、3秒前後では効かない（Brain Stimulation 2025）
      【ヒト】【霊長類】【オルガノイド】は見つからない ⇒ 【み】09-27「げっ歯類までなければ使ってよい」
    重みを変えるのは、脳幹が配る第三の信号 third の分だけ：w ＋＝ third ×（1 − 減り）× 跡
      third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量になる
  中脳（brainstem.py）門 ＝ ふだんの分 1 ＋ 驚きに比例する上乗せ（設計案 §8-6〜8-8）
    耳はいつも聞いていて、いつも学ぶ（【ヒト】受け身で聞くだけで学ぶ：Saffran 1996）【み】
    返りを見分ける専用の部品は作らない（人に無い）。予期しない音なら、返し手でもみわの声でも同じ道で効く
    【ヒト】黒質のドーパミン細胞は予期しない得の 150〜375ms 後に上がる（Zaghloul 2009）⇒ 驚きの 14歩後から 21歩
    【マカク】3〜5Hz → 20〜30Hz、予測誤差に比例（Glimcher 2011）⇒ 上限はふだんの 6.25倍【み】09-27 許可
    ⚠️ 驚きの入口（皮質の誤差の平均）と比例の係数 SURPRISE_K は【C】のノブのまま
    眠りのあいだは上乗せなし（ふだんの分 1。学ぶ速さは状態で変えない）
    驚きは学ぶ速さに直接掛けるのをやめた（覚醒の側へ。感度・眠気の覆いは版41）
  返し手（voice.py・babble.py・day.py）【み】09-27
    声：春日部つむぎ（style 8。規約に機械学習の禁止なし。クレジット VOICEVOX:春日部つむぎ）
    返す割合・遅れ【ヒト】家庭の一日録音（Developmental Science 2026、1〜10か月）何秒以内（累計）1秒 24% … 5秒 36%
    中身：まね返し か「どうしたの？」「かわいいね」「いい子だね」を、でたらめに（喃語で返るのは音なら何でもいい）
    写しは付けない（自分の声ではない）。0モーラ（黙っている）には返さない。--fake のときは返さない
  刈り込み（torch_group2.py）の不具合（torch 版から）：並べ直したあとに古い位置で 0 にしていて、100歩ごとに無関係の辺の重みを消していた ⇒ 直す
学び方の根元が変わるので版40。state.pt は使えない ⇒ python3 babble.py 100 --fresh
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


NAMES = ("torch_group2.py", "brainstem.py", "babble.py", "day.py", "voice.py")
F = {k: open(k, encoding="utf-8").read() for k in NAMES}
if "VERSION = 39 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版39 ではない。何も書いていない")
if "AGE_TABLE" not in F["brainstem.py"]:
    sys.exit("⚠️ 版39c になっていない（先に apply_v39c.py）。何も書いていない")
if "elig" in F["torch_group2.py"]:
    sys.exit("⚠️ もう版40 になっている。何も書いていない")

# ── torch_group2.py ──
g = F["torch_group2.py"]
g = once(g, "RAMP_P = 0.3 * DT_MS / 1000.0\n",
         "RAMP_P = 0.3 * DT_MS / 1000.0\n"
         "# 学んだ跡の長さ（版40）【ラット】大人の聴覚野・生体：修飾の信号が 1秒前・同時なら効き、\n"
         "#   3秒前後では効かない（Brain Stimulation 2025）⇒ 時定数 1秒。【ヒト】【霊長類】は見つからない【み】09-27\n"
         "ELIG_TAU = 1000.0 / DT_MS\n", "group2 の定数")
g = once(g, "import torch\n", "import math\nimport torch\n", "group2 の import")
k = "        self.scale_skip = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)   # 人の値が無い受け手は下げない\n"
g = once(g, k, k +
         "        # 学びの門（版40）：学んだ跡を辺ごとに持ち、脳幹が配る第三の信号 third の分だけ重みを変える\n"
         "        self.elig = torch.zeros_like(self.w)\n"
         "        self.elig_decay = math.exp(-1.0 / ELIG_TAU)\n"
         "        self.third = torch.ones((), device=dev)\n", "group2 の init")
g = once(g, "        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]\n",
         "        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]\n"
         "        self.elig = self.elig[o]\n", "刈り込みの並べ直し")
g = once(g, "        self.w[cuts] = 0.0\n",
         "        # ⟲ 版40 で直した：並べ直したあとに、並べ直す前の位置 cuts で 0 にしていた。\n"
         "        #    開通した辺は切った辺の重みを持ち越し、代わりに無関係の辺の重みが 100歩ごとに受け手の数だけ消えていた\n"
         "        pos = torch.empty_like(o)\n"
         "        pos[o] = torch.arange(len(o), device=self.dev)\n"
         "        cuts = pos[cuts]\n"
         "        self.w[cuts] = 0.0             # 開通の値（決まったこと 09-11：開通は w0、学びの跡は 0）\n"
         "        self.elig[cuts] = 0.0\n", "刈り込みの開通")
g = once(g,
         "            gg = self.eta * self.eta_gain * err_sig[r]   # 脳幹：驚きで学ぶ速さを上げる（版38）\n"
         "            shunt = (exc / (1.0 + ihn) ** 2)[r]\n"
         "            self.w[e] += torch.where(self.e_inh[e], -gg * shunt, gg)\n"
         "            # 興奮も抑制も 0 以上（版36。【ヒト】繋がりの符号は送り手の型で決まる。前は興奮が負になれた）\n"
         "            self.w[e] = self.w[e].clamp(min=0.0)\n",
         "            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain は版40 から 1（驚きは覚醒の側へ）\n"
         "            shunt = (exc / (1.0 + ihn) ** 2)[r]\n"
         "            # 学びの門（版40）：その歩の学びは跡にためる。重みを変えるのは脳幹の第三の信号の分だけ\n"
         "            #   【ヒト】皮質の古典的な対ではドーパミンを入れて初めて強まる（前の作り 生-217）\n"
         "            self.elig.mul_(self.elig_decay)\n"
         "            self.elig[e] += torch.where(self.e_inh[e], -gg * shunt, gg)\n"
         "            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）\n"
         "            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))\n"
         "            # 興奮も抑制も 0 以上（版36。【ヒト】繋がりの符号は送り手の型で決まる。前は興奮が負になれた）\n"
         "            self.w.clamp_(min=0.0)\n", "group2 の学び")

# ── brainstem.py ──
s = F["brainstem.py"]
k = "WAKE_CHECK = 94              # 起こすかを見るのは約1秒ごと（毎歩 GPU から値を出すと遅くなるため）\n"
s = once(s, k, k +
         "# 学びの門（版40）設計案 §8-6〜8-8。門 ＝ ふだんの分 1 ＋ 驚きに比例する上乗せ（学んだ跡に掛かる）\n"
         "#   【ヒト】黒質のドーパミン細胞は予期しない得の 150〜375ms 後に上がる（Zaghloul 2009）⇒ 遅れ 14歩・長さ 21歩\n"
         "#   【マカク】予期しない報酬で 3〜5Hz → 20〜30Hz、予測誤差に比例（Glimcher 2011）⇒ 上限はふだんの 25/4 倍【み】09-27 許可\n"
         "DA_LAT = int(round(150.0 / DT_MS))       # 14歩\n"
         "DA_DUR = int(round(225.0 / DT_MS))       # 21歩\n"
         "DA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）\n",
         "脳幹の定数")
s = once(s, "        self.hist = torch.zeros(SURPRISE_LAT + 1, device=self.dev)\n",
         "        self.hist = torch.zeros(max(SURPRISE_LAT, DA_LAT) + 1, device=self.dev)   # 版40：中脳の遅れの分まで持つ\n"
         "        self.da_buf = torch.zeros(DA_DUR, device=self.dev)   # 中脳の短い発火が続く長さ（版40）\n"
         "        self.surprise = torch.zeros((), device=self.dev)\n"
         "        self.gate_sum = torch.zeros((), device=self.dev)     # 門の開き（測る用）\n"
         "        self.gate_n = 0\n", "脳幹の init")
s = once(s, "        g.eta_gain = torch.ones((), device=self.dev)\n        self.wake_request = False\n",
         "        g.eta_gain = torch.ones((), device=self.dev)\n"
         "        g.third = torch.ones((), device=self.dev)   # 学びの門（版40）：ふだんの分。眠りは上乗せなし（学ぶ速さは状態で変えない）\n"
         "        self.da_buf.zero_()\n"
         "        self.wake_request = False\n", "脳幹の set_state")
s = once(s,
         "            ratio = self.hist[0] / self.base.clamp(min=1e-6)     # SURPRISE_LAT 歩前の誤差\n"
         "            if self.n_wake < SURPRISE_TAU:\n"
         "                self.g.eta_gain = torch.ones((), device=self.dev)\n"
         "            else:\n"
         "                self.g.eta_gain = 1.0 + SURPRISE_K * (ratio - 1.0).clamp(min=0.0)\n",
         "            ratio = self.hist[-1 - SURPRISE_LAT] / self.base.clamp(min=1e-6)     # SURPRISE_LAT 歩前の誤差（青斑核）\n"
         "            ready = self.n_wake >= SURPRISE_TAU        # 基準ができるまでは驚きを出さない（版39）\n"
         "            # 驚き（青斑核）：覚醒（感度・眠気の覆い）に使う（版41）。学ぶ速さに直接は掛けない（版40）\n"
         "            self.surprise = (ratio - 1.0).clamp(min=0.0) if ready else torch.zeros_like(ratio)\n"
         "            self.g.eta_gain = torch.ones((), device=self.dev)\n"
         "            # 学びの門（版40）：中脳は DA_LAT 歩前の驚きに比例して撃ち、DA_DUR 歩続く。門 ＝ 1 ＋ その分（上限 DA_CAP）\n"
         "            #   ⚠️ 驚きの入口（皮質の誤差の平均）と、比例の係数 SURPRISE_K は【C】のノブのまま\n"
         "            r_da = self.hist[0] / self.base.clamp(min=1e-6)\n"
         "            da = SURPRISE_K * (r_da - 1.0).clamp(min=0.0) if ready else torch.zeros_like(r_da)\n"
         "            self.da_buf = torch.cat([self.da_buf[1:], da.reshape(1)])\n"
         "            self.g.third = (1.0 + self.da_buf.max()).clamp(max=DA_CAP)\n"
         "            self.gate_sum = self.gate_sum + self.g.third\n"
         "            self.gate_n += 1\n", "脳幹の驚き")
k = "    def sleep_state(self, k):\n"
s = once(s, k,
         "    def gate_mean(self):\n"
         "        \"\"\"前に呼んでから起きていたあいだの、門の開きの平均（測る用。呼ぶと 0 に戻す。ふだんは 1）\"\"\"\n"
         "        m = float(self.gate_sum) / max(self.gate_n, 1)\n"
         "        self.gate_sum.zero_()\n"
         "        self.gate_n = 0\n"
         "        return m\n"
         "\n" + k, "脳幹の gate_mean")

# ── voice.py ──
v = F["voice.py"]
k = "        \"\"\"指令 → 波形 → メル [歩, 40]。0モーラなら沈黙（床値のまま）\"\"\"\n"
v = once(v, k, k + "        self.last_morae, self.last_volume = morae, volume     # 返し手がまね返す（版40）\n", "voice の say")
v += '''

# ── 返し手（版40）【み】09-27 ──
#   喃語で返るのは音なら何でもいい（決まったこと。Goldstein 2003：まねでない反応でも発達を導く）
#   ⇒ まね返し か 短い言葉 を、でたらめに選ぶ。中身で学びを変えない（報酬の形にしない）
REPLY_SPEAKER = 8            # 春日部つむぎ（ノーマル）。規約に機械学習の禁止なし。クレジット VOICEVOX:春日部つむぎ
REPLY_WORDS = ("どうしたの？", "かわいいね", "いい子だね")      # 【み】09-27 の例
# 【ヒト】何秒以内に母親の声が返ったか（累計。泣き以外の発声、家庭の一日録音、1〜10か月。Developmental Science 2026）
#   ⚠️ 窓の中でたまたま話していた分も含む
REPLY_CUM = (0.24, 0.30, 0.33, 0.35, 0.36)


class Replier:
    """自分の喃語のあとに、別の声で返す（写しは付かない ＝ 自分の声ではない）"""

    def __init__(self, vo):
        from voicevox import Voicevox
        from torch_area import DT_MS
        self.vo, self.dt = vo, DT_MS
        self.vv = Voicevox()
        self.vv.speaker = REPLY_SPEAKER
        self.vv.connect()
        self.vv.inventory, self.vv._index = vo.vv.inventory, vo.vv._index
        self.vv.mora_text = vo.vv.mora_text
        self.encode = vo.encode
        print(f"返し手  話者 {REPLY_SPEAKER}（VOICEVOX:春日部つむぎ）", flush=True)

    def answer(self, rng):
        """返すなら耳のメル [歩, 40]（遅れの静けさを含む）、返さないなら None"""
        morae = getattr(self.vo, "last_morae", None)
        if not morae:                                  # 黙っていた（0モーラ）には返さない
            return None
        u = rng.random()
        if u >= REPLY_CUM[-1]:
            return None
        sec = next(i for i, c in enumerate(REPLY_CUM) if u < c)
        delay = int((sec + rng.random()) * 1000.0 / self.dt)     # 喃語の終わりから
        n_sil = max(0, delay - 20)                     # 喃語の後ろに 20歩の静けさが付いている
        pick = int(rng.integers(0, len(REPLY_WORDS) + 1))
        if pick == len(REPLY_WORDS):                   # まね返し（同じモーラ列）
            q = self.vv.build_query(morae)
            q["volumeScale"] = float(self.vo.last_volume)
        else:
            q = self.vv._audio_query(REPLY_WORDS[pick])
            q["outputSamplingRate"] = C.SAMPLE_RATE
            q["outputStereo"] = False
        q["prePhonemeLength"] = 0.0
        q["postPhonemeLength"] = 0.0
        mel = self.vo.ear.hear(self.vv.synthesize(q)).astype(np.float32)
        sil = np.full((n_sil, N_MEL), self.vo.floor, np.float32)
        tail = np.full((20, N_MEL), self.vo.floor, np.float32)
        return np.concatenate([sil, mel, tail])
'''

# ── babble.py ──
b = F["babble.py"]
b = once(b, "VERSION = 39 ", "VERSION = 40 ", "VERSION")
k = "#      視床の誤差・抑制と運動は下げない）。起きているあいだは変わらないので VERSION は 39 のまま\n"
b = once(b, k, k +
         "#  40  学びの門：その歩の学びは辺ごとの跡（時定数 1秒【ラット】）にため、門の分だけ重みを変える。\n"
         "#      門 ＝ ふだんの 1 ＋ 驚きに比例する上乗せ（【ヒト】黒質 150〜375ms 後、上限【マカク】ふだんの 6.25倍）／\n"
         "#      返し手（春日部つむぎ）が喃語の36%に、【ヒト】家庭録音の遅れで、まね返しか短い言葉を返す【み】／\n"
         "#      驚きは学ぶ速さから外した（覚醒の側へ）\n", "版の履歴")
k = "def measure(b, speak, n_test=20):\n"
b = once(b, k,
         "def reply(b, rp, rng):\n"
         "    \"\"\"返し手が返すなら、それを聞く（写しなし。学ぶ。版40）\"\"\"\n"
         "    mel = rp.answer(rng)\n"
         "    if mel is not None:\n"
         "        run(b, None, rp.encode(mel), False, True)\n"
         "\n\n" + k, "babble の reply")
b = once(b, "    rng = np.random.default_rng(0)\n    if fake:\n        inv = make_inventory(np.random.default_rng(0))\n",
         "    rng = np.random.default_rng(0)\n    rp = None                                   # 返し手（版40。作り物の声では返さない）\n"
         "    if fake:\n        inv = make_inventory(np.random.default_rng(0))\n", "babble の rp")
b = once(b, "        from voice import Voice\n        from mouth import Mouth\n        vo, mo = Voice(), Mouth()\n",
         "        from voice import Voice, Replier\n        from mouth import Mouth\n        vo, mo = Voice(), Mouth()\n"
         "        rp = Replier(vo)\n", "babble の Voice")
b = once(b, "    t0 = time.time()\n    for i in range(n_add):\n",
         "    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（喃語の列とは別）\n"
         "    t0 = time.time()\n    for i in range(n_add):\n", "babble の rng_r")
b = once(b, "        run(b, cmd, enc(mel), True, True)\n        done += 1\n        if (i + 1) % 20 == 0:\n",
         "        run(b, cmd, enc(mel), True, True)\n"
         "        if rp is not None:\n"
         "            reply(b, rp, rng_r)\n"
         "        done += 1\n        if (i + 1) % 20 == 0:\n", "babble のループ")
b = once(b, "            print(f\"  喃語 {done} 回  経過 {el/60:.1f}分  残り見込み \"\n",
         "            print(f\"  喃語 {done} 回  門 {b.bs.gate_mean():.3f}  経過 {el/60:.1f}分  残り見込み \"\n",
         "babble の表示")

# ── day.py ──
d = F["day.py"]
d = once(d, "    rng = np.random.default_rng(0)\n    if fake:\n",
         "    rng = np.random.default_rng(0)\n    rp = None                                 # 返し手（版40。作り物の声では返さない）\n"
         "    if fake:\n", "day の rp")
d = once(d, "        from voice import Voice\n        from mouth import Mouth\n        vo, mo = Voice(), Mouth()\n",
         "        from voice import Voice, Replier\n        from mouth import Mouth\n        vo, mo = Voice(), Mouth()\n"
         "        rp = Replier(vo)\n", "day の Voice")
d = once(d, "    b.bs.set_state(BS.WAKE)                   # 起きている（版38：恒常性・上からの戻りは脳幹が決める）\n",
         "    b.bs.set_state(BS.WAKE)                   # 起きている（版38：恒常性・上からの戻りは脳幹が決める）\n"
         "    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（版40）\n", "day の rng_r")
d = once(d, "            B.run(b, cmd, enc(mel), True, True)\n",
         "            B.run(b, cmd, enc(mel), True, True)\n"
         "            if rp is not None:\n"
         "                B.reply(b, rp, rng_r)\n", "day のループ")
d = once(d, "                print(f\"  [{cyc+1}/{n_cycle}] 起きて {i}回  経過 {el/60:.1f}分  \"\n",
         "                print(f\"  [{cyc+1}/{n_cycle}] 起きて {i}回  門 {b.bs.gate_mean():.3f}  経過 {el/60:.1f}分  \"\n",
         "day の表示")

out = {"torch_group2.py": g, "brainstem.py": s, "babble.py": b, "day.py": d, "voice.py": v}
for x in out.values():
    ast.parse(x)
for k, x in out.items():
    shutil.copy(k, k + ".v39c")
    open(k, "w", encoding="utf-8").write(x)
print("版40 にした（元は *.v39c に残した）")
print("回すときは  python3 babble.py 100 --fresh   （表示の「門」は起きているあいだの門の開きの平均）")
