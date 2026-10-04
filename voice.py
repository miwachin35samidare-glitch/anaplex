"""VOICEVOX を繋ぐ（arkhe 用）—— 作り物の声をやめて、本物の声で喃語を回す

前の作りの決めをそのまま使う
  指令   モーラ列（インベントリ120種。話者によらない）＋ モーラごとに連続値3つ ＋ 大域1つ
  発話   0〜4モーラ・可変長。0モーラ（沈黙）を必ず含める（生物は常時発声しない）
  固定   speedScale 1.0 ／ pitchScale 0.0 ／ intonationScale 1.0（写像を単射に保つ）
  除外   pitch = 0（無声化のフラグ）。pitch の上下限は切らない
         （崩れた音が出れば予測誤差が上がる。それが学習信号）
  喃語   中身は関係ない。音が返ってくればいい ⇒ 120モーラからでたらめに選ぶ

耳
  メル40チャネル。閾値を越えた瞬間だけ撃つ（【ヒト】聴覚の皮質はまばら）

⚠️ VOICEVOX は arkhe の Docker で動いていること
   docker run -d --name anaplex_voicevox --restart unless-stopped \\
     -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:cpu-0.25.1
   docker update --cpus=7 anaplex_voicevox        # 80℃ を超えないように
   話者は ユーレイちゃん style_id=102。バージョンと話者を固定する（latest を使わない）
"""
import numpy as np

import config as C
from voicevox import Voicevox
from ear import Ear
from mouth import RANGE, VOLUME_RANGE, CONT_KEYS, MAX_MORAE

SILENCE_PROB = 0.20          # 0モーラを引く確率（前の作りの決め）
N_MEL = C.N_MELS
LATENCY = 5                  # 口から耳に届くまで（歩）

# ── 耳の形（すべて【ヒト】から）──
#   チャネル 40     聞こえる範囲に ERB（聴覚フィルタの幅）が約40個入る
#                  ERB_N = 24.673(0.004368f + 1)。50Hz〜20kHz を1刻みで並べると40個
#                  ⇒ メル40チャネルは、人の周波数の分解能と一致している
#   1チャネル 10体  1つの内有毛細胞に聴神経が 10〜20本つく
#                  （聴神経 30,000本 ÷ 内有毛細胞 3,300個 ≒ 9本）
#   体ごとの違い     同じ有毛細胞につく繊維は、自発の発火率と閾値が違う。
#                  自発が低い繊維ほど閾値が高い（小さい音では撃たない）
#                  分布は二峰性：低自発（20/秒未満）と高自発（20/秒超）の2群
N_FIBER = 10                 # 1チャネルあたりの聴神経の本数
EAR_TH_LO = -11.0            # 高自発の繊維の閾値（小さい音で撃つ）
EAR_TH_HI = -2.0             # 低自発の繊維の閾値（大きい音でしか撃たない）
EAR_HI_FRAC = 0.4            # 低自発（高閾値）の群の割合


class Voice:
    """網が出した指令 → 本物の声 → 耳のメル"""

    def __init__(self):
        self.vv = Voicevox().connect()
        self.inventory = self.vv.build_inventory()      # 話者によらない120種
        self.ear = Ear()
        self.n_mora = len(self.inventory)
        self.floor = self.ear.floor
        # 繊維ごとの閾値（低自発＝高閾値の群と、高自発＝低閾値の群。二峰性）
        n_hi = max(1, int(round(N_FIBER * EAR_HI_FRAC)))
        self.fiber_th = np.concatenate([
            np.linspace(EAR_TH_LO, (EAR_TH_LO + EAR_TH_HI) / 2, N_FIBER - n_hi),
            np.linspace((EAR_TH_LO + EAR_TH_HI) / 2, EAR_TH_HI, n_hi + 1)[1:]])
        # ⟲ 前は2群目も中間点から始まり、1群目の終わりと同じ閾値の繊維が2本あった（10本で9段）
        self.n_ear = C.N_MELS * N_FIBER
        print(f"VOICEVOX 接続  話者 {self.vv.speaker}  モーラ {self.n_mora}種", flush=True)

    def random_morae(self, rng):
        """喃語：120モーラからでたらめに 0〜4個。長さ・高さ・強さも揺らす"""
        if rng.random() < SILENCE_PROB:
            return [], 1.0
        n = int(rng.integers(1, MAX_MORAE + 1))
        out = []
        st = np.log(2.0) / 12.0                                   # 1半音（ln）
        center = rng.normal(C.BABBLE_PITCH_CENTER, C.BABBLE_PITCH_SD_ST * st)   # 版44：発話ごと
        for _ in range(n):
            mid = int(rng.integers(0, self.n_mora))
            c = float(np.clip(rng.normal(C.CONSONANT_LENGTH_MEAN, C.CONSONANT_LENGTH_STD),
                              *RANGE["consonant_length"]))
            v = float(np.clip(rng.normal(C.VOWEL_LENGTH_MEAN, C.VOWEL_LENGTH_STD),
                              *RANGE["vowel_length"]))
            p = float(np.clip(center + rng.uniform(-0.5, 0.5) * C.BABBLE_PITCH_WIDTH_ST * st, *RANGE["pitch"]))
            out.append((mid, c, v, p))
        vol = float(np.clip(rng.normal(C.VOLUME_SCALE_MEAN, C.VOLUME_SCALE_STD), *VOLUME_RANGE))
        return out, vol

    def cmd_stream(self, morae, volume, n_step, mouth):
        """指令の写しを、発話のあいだ流す [歩, n_cmd]

        【ヒト】随伴放電は、腹側中心前回 → 上側頭回へ、発話の開始 107.5ms 前にピーク
                （8名、ECoG。5つの発話課題すべてで再現）
                ⇒ 音素ごとに刻む信号ではない。発話全体に先立って届く
        前の作り（連続版）でも、指令ベクトルを丸ごと渡して予測器が時刻を学ぶ形で、
                抑制比が 1.25 → 1.84 と単調に上がった

        ⚠️ 一度は「子音・母音が始まる歩に置く」形にしたが、supp が 1.0 を割った（0.38）。
           前のモーラの長さが後ろ全部を時間方向にずらすので、指令から時刻を計算しても音と合わない。
           いつ来るかは、聴覚の側（時間に広がった受容野）が学ぶ
        """
        v = np.zeros((n_step, mouth.n_cmd), np.float32)
        if not morae:
            return v
        one = mouth.encode(morae, volume)[0]
        v[C.GO_LEAD - C.CMD_LEAD:] = one  # 版43d：声の CMD_LEAD 歩前から（蹴りが終わってから）。運動の領野に入る
        #   ⟲ 版43c までは音の LATENCY 歩前から、聴覚へ直に入る写しだった
        return v

    def say(self, morae, volume):
        """指令 → 波形 → メル [歩, 40]。0モーラなら沈黙（床値のまま）"""
        if not morae:
            self.last_morae, self.last_volume = morae, volume
            n = LATENCY + 20
            return np.full((n, N_MEL), self.floor, np.float32)
        pad = np.full((C.GO_LEAD, N_MEL), self.floor, np.float32)   # 準備（蹴りのあいだ、声は出ない）
        return np.concatenate([pad, self.sound(morae, volume)])

    def sound(self, morae, volume):
        """口が読んだモーラ → 声のメル [歩, 40]（準備の静けさを付けない。後ろに 20歩の静けさ）
        版43d：疑核と筋は置かない。VOICEVOX がその代わり【み】09-29"""
        self.last_morae, self.last_volume = morae, volume     # 返し手がまね返す（版40）。言えたものを返す
        tail = np.full((20, N_MEL), self.floor, np.float32)
        if not morae:
            return tail
        q = self.vv.build_query(morae)
        q["volumeScale"] = float(volume)
        q["speedScale"] = 1.0
        q["pitchScale"] = 0.0
        q["intonationScale"] = 1.0
        q["prePhonemeLength"] = 0.0
        q["postPhonemeLength"] = 0.0
        wav = self.vv.synthesize(q)
        mel = self.ear.hear(wav)
        return np.concatenate([mel.astype(np.float32), tail])

    def encode(self, mel):
        """メル [歩, 40] → 耳の発火 [歩, 40 × N_FIBER]（聴神経の形）

        1つのチャネル（＝内有毛細胞）に N_FIBER 本の繊維がつき、閾値がそれぞれ違う。
        小さい音では低い閾値の繊維だけが撃ち、大きい音では多くの繊維が撃つ
        ⇒ 撃つ本数が音の大きさになる（【ヒト】自発が低い繊維ほど閾値が高い）
        越えているあいだ撃つ。疎にするのは皮質の上の段（【ヒト】上側頭回）
        """
        n = mel.shape[1] * N_FIBER
        out = np.zeros((len(mel), n), np.float32)
        for i, th in enumerate(self.fiber_th):
            out[:, i::N_FIBER] = (mel > th).astype(np.float32)
        return out


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
