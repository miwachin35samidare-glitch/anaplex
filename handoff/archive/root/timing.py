"""歩と ms の対応 —— 人の値は ms で書き、1歩の長さから歩数を出す

1歩を変えても壊れないように、歩の数を直接書かない。
⚠️ 人から出た値と、Claude が置いた値を分けて書く。
"""
import numpy as np

DT_MS = 10.67          # 1歩（耳の1フレーム。24kHz で256サンプル）


def steps(ms):
    return max(1, int(round(ms / DT_MS)))


# ── 人から出ている値 ──
SYN_DELAY_MS = 1.102          # ヒト局所シナプス潜時 1.102 ± 0.408ms
MEMBRANE_MS = (16.0, 24.0)    # ヒト錐体細胞の膜時定数
CA3_MEMBRANE_MS = 55.9        # ヒト CA3
AREA_TAU_MS = {"一次": 100.0, "連合": 190.0, "高次": 320.0}   # 段の時定数（ヒト脳波×10／マカク実測）
CORTEX_HZ = 0.3               # ヒト皮質の発火率
CA3_HZ = 9.6                  # ヒト CA3
SPARSE = 0.025                # ヒト海馬で1つの項目に反応する割合（2.5%未満）

# 1歩の中で何回まわるか（整理券）。1周 ＝ シナプス1本ぶんの遅れ
RECUR = max(1, int(round(DT_MS / SYN_DELAY_MS)))          # ≒ 10

# 時間細胞（ヒト：Umbach ほか 2020）
#   受容野は覆う期間に合わせて伸び縮みする。幅は一定ではなく、遅れが大きいほど広い（対数圧縮）
#   人の中央値は 2.5〜3.4秒だが、それは数十秒の課題に対する値。比として使う
TIMEFIELD_FRAC = 3.0 / 30.0   # 覆う期間に対する受容野の幅の比（人の 2.5〜3.4秒 ÷ 課題の長さ）


def time_fields(span_ms, n_cell):
    """覆う期間 span_ms を n_cell 体で覆う。遅れが大きいほど広い（対数圧縮）
    返り値: 好みの遅れ（歩）と、その幅（歩）"""
    span = steps(span_ms)
    # 対数で等間隔に置く ⇒ 隣どうしの間隔が遅れに比例して広がる
    pref = np.exp(np.linspace(np.log(1.0), np.log(span), n_cell))
    width = np.maximum(TIMEFIELD_FRAC * pref, 1.0)        # 幅も遅れに比例（下限1歩）
    return pref.astype(np.float32), width.astype(np.float32)


# ── Claude が置いた値（人から出ていない。⚠️ 引き直す候補） ──
CHAIN_SPAN_MS = 341.0         # 時間細胞が覆う期間（音節1つぶん、として置いた）
RAMP_SPAN_MS = 2732.0         # 傾きを持つ細胞の長さ
REWIRE_MS = 1067.0            # 入れ替えの間隔（下限だけ人から出る：恒常性より十分長く）
LATENCY_MS = 53.0             # 口から耳に届くまで（人の随伴放電は発話の107.5ms前）
