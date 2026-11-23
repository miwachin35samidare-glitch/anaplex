"""口を繋ぐ —— 人の形の指令を、運動の領野の集団の型で出す

人の運動前野（Neuropixels、1,794体）
  音素を符号化していたのは 聞くとき 8.2% ／ 話すとき 10.5% のニューロン
  1つの音素に専用の体があるのではなく、多くの体の「集団の型」で決まる
  発話の準備中は、次の音素だけでなく語全体の音素の位置が並列に符号化される
  位置ごとの表現は構造を共有する（別々の入口ではない）
人の運動単位
  筋あたり約100個。動員した数が出力の強さになる ⇒ 連続値はこの形（撃った体の数が値）
  版44：体の数（＝段数）は筋の数ではなく、聞き分けで決める（設計案_脳幹.md §12-63〜73）
    疑核と筋は VOICEVOX が代わる【み】09-29 ⇒ 100体の動員は筋の側の細工で、皮質に持たせる理由が無い
    段の間が聞き分けられないなら、網にとっても人にとっても無いのと同じ

これまで（人と違った）
  モーラ1種につき専用の入口を1つ持ち、その体が撃ったらそのモーラを言う
  ⇒ 1体の発火で動きが起きる。人では動きに動員が要る
  ⇒ 1音素に専用の体がある。人は集団の型で表す

これから
  モーラの層  n_code 体の集団。撃った型が、どのモーラに一番近いかで決まる
              （型はモーラごとに決まった疎な符号。位置をまたいで同じ符号を使う ＝ 構造を共有）
  位置        4つのスロットは並列のまま（人の運動前野と同じ）。位置ごとに層を持つ
  連続値      値ごとに POOL 体。撃った数が値（人の運動単位の動員）。段は体の数
  言うか黙るか  型がどのモーラにも近くなければ沈黙（0モーラ。人も常時発声しない）
"""
import numpy as np

N_INVENTORY = 120
MAX_MORAE = 4
N_CODE = 128               # モーラの層の体数（集団の型を作る）
CODE_SPARSE = 0.10         # 1つのモーラの型で撃つ割合（人：音素を符号化するのは約1割）
MATCH_MIN = 0.5            # 型がどれだけ合えばそのモーラと見なすか（下回れば沈黙）
# 版49：口は運動の領野の発火を筋の形でならして読む
#   【ヒト】Ito 2004 J Appl Physiol：電気刺激1発への力を 2次の系で当てはめ
#     上唇 6.10Hz・ζ 0.714・15ms（8人）／ 下唇 6.05Hz・0.682・16ms（8人）／ 舌 6.11Hz・0.675・17ms（2人）
#   喉頭の筋は人の値が無い（【ヒト】発火から声の高さの山まで TA 5〜20ms・CT 6〜75ms、Larson 1987）⇒ 唇・舌の値で代用【C】
MUSCLE_FN = 6.1
MUSCLE_ZETA = 0.69
MUSCLE_DELAY_MS = 15.0
READ_TH = 0.20            # ⚠️【C】力（撃ち続けたときの発火率に寄る）がこれを越えた体を「撃った」と読む

# 連続値の範囲と段数（版44）
#   高さ      ln 120〜600Hz を対数で等間隔 30段【み】の耳（30段と140段が聞き分けられない §12-66）
#             下 120：話者102 は 134Hz 未満が同じに聞こえる（§12-67）／ 上 600：08-02 に崩れないと聞いた端
#   母音の長さ 10〜350ms を等間隔 33段 ／ 子音の長さ 1〜150ms を等間隔 15段
#             網の耳は 1コマ（10.67ms）ごとにしか長さを聞き分けない【測】measure_ear_res（§12-65）
#   大きさ    0.05〜2.0倍を対数で等間隔 16段（1段 約2.1dB）【み】09-30
#   ⟲ 版43 までは どれも 100段（運動単位の動員の数）、範囲の中を等間隔。高さは ln 0.001〜6.0（1〜400Hz）
RANGE = {"consonant_length": (0.001, 0.150),
         "vowel_length": (0.010, 0.350),
         "pitch": (float(np.log(120.0)), float(np.log(600.0)))}
VOLUME_RANGE = (0.05, 2.0)
CONT_KEYS = ("consonant_length", "vowel_length", "pitch")
POOL = {"consonant_length": 15, "vowel_length": 33, "pitch": 30}
VOL_POOL = 16
N_CMD = MAX_MORAE * N_CODE + MAX_MORAE * sum(POOL.values()) + VOL_POOL   # 840


def grid(key):
    """その値の段の並び（高さは ln Hz なので等間隔 ＝ 対数で等間隔。大きさは対数で等間隔）"""
    if key == "volume":
        lo, hi = VOLUME_RANGE
        return np.exp(np.linspace(np.log(lo), np.log(hi), VOL_POOL))
    lo, hi = RANGE[key]
    return np.linspace(lo, hi, POOL[key])


def to_step(key, x):
    """値 → 段（0 から）。一番近い段に丸める"""
    return int(np.argmin(np.abs(grid(key) - x)))


def muscle_force(spikes, dt_ms=10.67):
    """版49：発火の列 [歩, 体] → 最後の歩の力 [体]。2次の系（直流の利得 1：撃ち続けると発火率に寄る）"""
    spikes = np.asarray(spikes, np.float64)
    w = 2.0 * np.pi * MUSCLE_FN
    d = int(round(MUSCLE_DELAY_MS / dt_ms))
    sub = 20
    h = dt_ms / 1000.0 / sub
    y = np.zeros(spikes.shape[1]); v = np.zeros_like(y)
    for t in range(len(spikes)):
        u = spikes[t - d] if t >= d else 0.0
        for _ in range(sub):
            v += (w * w * (u - y) - 2.0 * MUSCLE_ZETA * w * v) * h
            y += v * h
    return y


class Mouth:
    def __init__(self, n_code=N_CODE, seed=0):
        self.n_code = n_code
        rng = np.random.default_rng(seed)
        k = max(1, int(round(n_code * CODE_SPARSE)))
        # モーラごとの型（疎な符号）。位置をまたいで同じ型を使う ＝ 位置間で構造を共有
        self.code = np.zeros((N_INVENTORY, n_code), np.float32)
        for m in range(N_INVENTORY):
            self.code[m, rng.choice(n_code, k, replace=False)] = 1.0
        self.k = k
        self.n_cmd = MAX_MORAE * n_code + MAX_MORAE * sum(POOL.values()) + VOL_POOL

    def decode(self, spikes):
        """運動の領野の発火 → (morae, volume)。型がどのモーラにも近くなければ沈黙"""
        s = spikes[0]; off = 0
        picks = []
        for i in range(MAX_MORAE):
            seg = s[off:off + self.n_code]; off += self.n_code
            if seg.sum() == 0:
                picks.append(None); continue
            # 型の一致：そのモーラの型のうち何割が撃っているか（撃ちすぎも合わないと見なす）
            hit = self.code @ seg / self.k
            extra = seg.sum() / max(self.k, 1)
            m = int(np.argmax(hit))
            picks.append(m if hit[m] >= MATCH_MIN and extra <= 2.0 else None)
        # 連続値：撃った体の数 n → 段 n − 1（段 k は k ＋ 1 体で表す。1体も撃たなければ一番下の段）
        cont = np.zeros((MAX_MORAE, 3), np.float64)
        for i in range(MAX_MORAE):
            for j, kk in enumerate(CONT_KEYS):
                P = POOL[kk]
                n = int(s[off:off + P].sum()); off += P
                cont[i, j] = grid(kk)[min(max(n - 1, 0), P - 1)]
        n = int(s[off:off + VOL_POOL].sum())
        vol = float(grid("volume")[min(max(n - 1, 0), VOL_POOL - 1)])
        return [(m, *cont[i]) for i, m in enumerate(picks) if m is not None], vol

    def encode(self, morae, volume):
        """指令 → 随伴放電に渡す写し（運動の領野が出したものと同じ並び）。値は一番近い段に丸める"""
        v = np.zeros(self.n_cmd, np.float32); off = 0
        for i in range(MAX_MORAE):
            if i < len(morae):
                v[off:off + self.n_code] = self.code[morae[i][0]]
            off += self.n_code
        for i in range(MAX_MORAE):
            for j, kk in enumerate(CONT_KEYS):
                if i < len(morae):
                    v[off:off + to_step(kk, morae[i][1 + j]) + 1] = 1.0
                off += POOL[kk]
        v[off:off + to_step("volume", volume) + 1] = 1.0
        return v[None, :]

    def audio_query(self, morae, volume):
        """AudioQuery のパラメータ（arkhe で voicevox.py に渡す形）"""
        return {"morae": [{"mora_id": int(m), "consonant_length": float(c),
                           "vowel_length": float(vv), "pitch": float(p)}
                          for m, c, vv, p in morae],
                "volumeScale": float(volume), "speedScale": 1.0,
                "pitchScale": 0.0, "intonationScale": 1.0,
                "prePhonemeLength": 0.0, "postPhonemeLength": 0.0}
