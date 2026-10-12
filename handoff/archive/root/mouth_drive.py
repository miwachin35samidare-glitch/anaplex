"""補助輪を外す —— 網が口を駆動する

これまで  いつ言うかは乱数（補助輪）。運動の領野は、乱数が決めた指令を読み取っていただけ
これから  運動の領野の表現のニューロンの自発活動が、そのまま口の指令になる
          （前の作りの決め：喃語のもとは運動野の自発活動。外す条件は「網が口を駆動するようになったら」）

運動の領野の入口（作り物の声の版）
  0..N_SYL-1              音節ごと。撃ったらその音節を言い出す
  N_SYL..2*N_SYL-1        母音の始まり（声の側が決める。指令の写しには乗る）
  2*N_SYL..               強さの段
撃った体が無ければ黙る（0モーラ＝沈黙。人の発話にも必ず含まれる）
"""
import numpy as np
import test_supp as ts
import test_supp_cmd as tc


def speak(motor_spikes, inv, rng):
    """運動の領野が撃った体 [1, N_CMD] → この歩に出す音（メル）と、指令の写し
    返り値: (mel [歩, N_MEL] または None, cmd [歩, N_CMD])"""
    syl = np.nonzero(motor_spikes[0, :ts.N_SYL])[0]
    if len(syl) == 0:
        return None, motor_spikes                      # 黙る
    s = int(syl[0])                                    # 同時に撃ったら若い方（口は1つ）
    cons, vow = inv[s]
    nc = max(2, int(rng.normal(7, 3)))
    nv = max(4, int(rng.normal(16, 5)))
    vol = max(0.3, rng.normal(1.0, 0.15))
    segs = [cons * vol] * nc + [vow * vol] * nv
    n = len(segs) + ts.LATENCY
    mel = np.clip(rng.normal(0.05, 0.03, (n, ts.N_MEL)), 0, 1)
    mel[ts.LATENCY:] = np.clip(np.array(segs) + mel[ts.LATENCY:], 0, 1)
    cmd = np.zeros((n, tc.N_CMD), np.float32)
    cmd[0, s] = 1                                      # 子音の始まり
    cmd[nc, ts.N_SYL + s] = 1                          # 母音の始まり
    cmd[0, 2 * ts.N_SYL + int(np.searchsorted(tc.VOL_EDGES, vol))] = 1   # 強さの段
    return mel.astype(np.float32), cmd
