"""黒質1体の試し台（網は使わない。何も書かない。CPU で数秒）
  python3 bench_sn.py
【み】10-15「それ、跳ねるの？」「いいよ」（作り込む前に、撃ち方の形 A・B で人の跳ねになるかを見る）
人の的：ふだん 4.56Hz【ヒト】Ramayya 2014 ／ 跳ね 約2.5〜3.6倍・幅 約150ms【ヒト】Ramayya 2014・Zaghloul 2009
置いたもの
  1歩 10.67ms（今の中脳と同じ刻み）
  興奮の入力 2,520本（8,000本 × 31.5%【ラット】Henny 2012）。ふだんの率 RATE_IN（⚠️【C】送り手の率は模型では領野ごとに違う）
  AMPA：届いた歩だけ（減り 6.0ms ＜ 1歩）／ NMDA：減り 75.5ms【マウス】Pearlstein 2015 ⇒ 1歩で ×0.868
  AMPA と NMDA の電荷の比 1:1（⚠️【C】人の値なし）
  A：撃つ確率 ＝ ふだんの確率 × exp(G ×（届いた分 − ふだん届く分））（今の midbrain.py の形）
  B：自分で撃ち続けるリズム（膜が毎歩 I0 ずつ上がり、閾値で撃って 0 に戻る。膜の減り τm）＋ 届いた分を膜に足す
     τm 20ms（⚠️【C】）。I0 は入力なしで 4.56Hz になる値。入力のばらつきでリズムが揺れる
出来事：入力のうち FRAC の線が、D 歩のあいだ RATE_EV で撃つ。強さは「山がふだんの約3倍」になるよう FRAC を探す
見るもの：山の倍率 ／ 2倍以上の幅 ／ 山の半分以上の幅（ふだんからの増え分で）／ 山の時刻
"""
import math, numpy as np

DT = 10.67
P0 = 4.56 * DT / 1000.0
N_IN = 2520
RATE_IN = 5.0                          # ⚠️【C】
RATE_EV = 40.0                         # ⚠️【C】出来事のときの送り手の率
DEC_N = math.exp(-DT / 75.5)
CHARGE_N = 1.0 / (1.0 / (1.0 - DEC_N))  # NMDA の1発の電荷を AMPA と同じに（足し合わせ ＝ 1）
TRIALS = 4000
WARM, PRE, POST = 40, 30, 40
rng = np.random.default_rng(0)


def inputs(frac, dur):
    """[歩, 試行] の届いた本数（AMPA の分・NMDA の分）"""
    T = WARM + PRE + POST
    lam = np.full((T, 1), RATE_IN * DT / 1000.0 * N_IN)
    n_ev = int(round(frac * N_IN))
    lam = np.repeat(lam, TRIALS, 1)
    lam[WARM + PRE:WARM + PRE + dur] += n_ev * (RATE_EV - RATE_IN) * DT / 1000.0
    k = rng.poisson(lam)
    nm = np.zeros_like(lam)
    acc = np.full(TRIALS, RATE_IN * DT / 1000.0 * N_IN * CHARGE_N / (1.0 - DEC_N))   # ふだんの定常から
    for t in range(T):
        acc = acc * DEC_N + k[t] * CHARGE_N
        nm[t] = acc
    return k[WARM:].astype(float), nm[WARM:]


def run_a(frac, dur, G):
    a, n = inputs(frac, dur)
    d = a + n
    base = RATE_IN * DT / 1000.0 * N_IN * 2.0          # ふだん届く分（AMPA ＋ NMDA の定常）
    p = np.clip(P0 * np.exp(G * (d - base)), 0, 1)
    return (rng.random(p.shape) < p).mean(1)


def run_b(frac, dur, wsyn, tau_m=20.0):
    a, n = inputs(frac, dur)
    alpha = math.exp(-DT / tau_m)
    base = RATE_IN * DT / 1000.0 * N_IN * 2.0
    steps = 1000.0 / 4.56 / DT                          # ふだんの間隔（歩）
    i0 = (1 - alpha) / (1 - alpha ** steps)              # 入力なしで steps 歩で閾値 1 に届く
    v = rng.random(TRIALS)
    out = np.zeros(a.shape[0])
    for t in range(a.shape[0]):
        v = v * alpha + i0 + wsyn * (a[t] + n[t] - base)
        f = v >= 1.0
        out[t] = f.mean()
        v[f] = 0.0
    return out


def shape(r):
    b = r[:PRE].mean()
    x = r[PRE:] / max(b, 1e-9)
    pk = x.max()
    over2 = (x >= 2.0).sum() * DT
    half = 1 + (pk - 1) / 2
    fw = (x >= half).sum() * DT
    return b * 1000 / DT, pk, over2, fw, (x.argmax() + 0.5) * DT


def tune(fn, dur, target=3.0, **kw):
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        pk = shape(fn(mid, dur, **kw))[1]
        lo, hi = (mid, hi) if pk < target else (lo, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    print(f"入力 {N_IN}本・ふだん {RATE_IN}Hz・出来事 {RATE_EV}Hz ／ NMDA 1歩で ×{DEC_N:.3f} ／ 試行 {TRIALS}", flush=True)
    G = math.log(25.0 / 4.0) / 8                         # 今の midbrain.py の G
    for name, fn, kw in (("A 今の形（G そのまま）", run_a, dict(G=G)),
                         ("A G を 1/10", run_a, dict(G=G / 10)),
                         ("B リズム＋足し合わせ", run_b, dict(wsyn=0.002)),
                         ("B 重さ 1/5", run_b, dict(wsyn=0.0004))):
        print(f"\n{name}", flush=True)
        print("  出来事の長さ   入力のうち撃つ割合   ふだん      山      2倍以上の幅   半分以上の幅   山の時刻", flush=True)
        for dur in (1, 5, 10, 15):
            fr = tune(fn, dur, **kw)
            hz, pk, o2, fw, tp = shape(fn(fr, dur, **kw))
            print(f"  {dur * DT:6.0f}ms      {fr * 100:8.2f}%        {hz:5.2f}Hz  ×{pk:4.2f}   {o2:7.0f}ms     {fw:7.0f}ms     {tp:5.0f}ms", flush=True)
