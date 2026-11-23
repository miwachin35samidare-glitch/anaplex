"""黒質1体の試し台3：B の跳ねの中の間隔が霊長類（約28〜33ms）より長い訳を、候補ごとに確かめる（網は使わない。何も書かない。CPU で数分）
  python3 bench_sn3.py
【み】10-15「それ確認できないの？」
的【マカク】Bayer・Lau・Glimcher 2007：ふだん 5.3Hz・CV 0.61・撃ったあと半分戻るまで 約106ms ／ 跳ね ＝ 間隔が 約33ms より短い（大きい正の誤差で平均 約28ms）
  ⚠️ 論文は「跳ねの間隔」を、33ms より短い間隔の平均として出している ⇒ ここも同じ物差しで：出来事のあいだの間隔のうち 33ms 未満の割合と、その平均
B（bench_sn2 で霊長類に合わせた形）：τm 40ms・自分のリズム ＋ 届いた分
候補
  0 そのまま（全部の体に同じ強さで入る）
  1 入力の偏り：出来事の入力が体ごとに違う（入力の割合を体ごとに指数の分布から。平均は同じ）⚠️ 分布の形は【C】
  4〜6 一部の体だけに出来事の入力がまとめて入る（平均は同じ）⚠️【C】
  2 NMDA の電圧の効き（Mg の栓）：NMDA の分 × 1／(1 ＋ [Mg]／3.57 × exp(−0.062 V))（Jahr・Stevens 1990【ラット】培養の海馬）[Mg] 1mM
     膜の値 0 → −60mV・閾値 1 → −40mV ⚠️【C】
  3 1 と 2 の両方
それぞれ、ふだんを率 4.56Hz・CV 0.61・半分戻るまで 106ms に合わせ直してから、山が約3倍の強さの出来事（107ms・160ms）を入れる
"""
import math, numpy as np

DT = 10.67
N_IN = 2520
RATE_IN = 5.0
RATE_EV = 40.0
DEC_N = math.exp(-DT / 75.5)
CHARGE_N = 1.0 - DEC_N
LAM0 = RATE_IN * DT / 1000.0 * N_IN
TM = 40.0
ALPHA = math.exp(-DT / TM)
rng = np.random.default_rng(0)


def drive(T, trials, frac=0.0, t0=0, dur=0, hetero=False):
    """AMPA の分・NMDA の分（ふだんの量を引いたもの）[歩, 試行]"""
    lam = np.full((T, trials), LAM0)
    if frac > 0:
        if hetero is True:
            f = rng.exponential(frac, trials)
        elif hetero:                                   # 一部の体（割合 hetero）だけに、その分まとめて入る
            f = np.where(rng.random(trials) < hetero, frac / hetero, 0.0)
        else:
            f = np.full(trials, frac)
        f = np.minimum(f, 1.0)
        lam[t0:t0 + dur] += (f * N_IN) * (RATE_EV - RATE_IN) * DT / 1000.0
    k = rng.poisson(lam).astype(float)
    acc = np.full(trials, LAM0)
    nm = np.empty_like(k)
    for t in range(T):
        acc = acc * DEC_N + k[t] * CHARGE_N
        nm[t] = acc
    return k - LAM0, nm - LAM0


def mg(v):
    V = -60.0 + 20.0 * v
    return 1.0 / (1.0 + math.exp(0) * np.exp(-0.062 * V) / 3.57)


MG_REF = float(mg(np.array(0.5)))     # 膜の真ん中での効きで割って、ふだんの強さをそろえる


def sim(a, n, i0, w, use_mg):
    v = rng.random(a.shape[1]) * 0.5
    s = np.zeros_like(a, dtype=bool)
    for t in range(a.shape[0]):
        nn = n[t] * (mg(v) / MG_REF if use_mg else 1.0)
        v = v * ALPHA + i0 + w * (a[t] + nn)
        f = v >= 1.0
        s[t] = f
        v[f] = 0.0
    return s


def base_stats(s):
    rate = s.mean() * 1000 / DT
    isis = [np.diff(np.nonzero(s[:, j])[0]) for j in range(s.shape[1])]
    isis = [x for x in isis if len(x)]
    if not isis:
        return 0.0, float("nan"), float("nan")
    isi = np.concatenate(isis) * DT
    cv = isi.std() / isi.mean()
    p = s.mean()
    cond = np.array([(s[:-k] & s[k:]).mean() / max(s[:-k].mean(), 1e-12) / max(p, 1e-12) for k in range(1, 40)])
    half = (np.argmax(cond >= 0.5) + 1) * DT if (cond >= 0.5).any() else float("nan")
    return rate, cv, half


def fit(use_mg):
    a, n = drive(3000, 300)
    best = None
    lo = 1 - ALPHA
    for i0 in np.linspace(lo * 0.9, lo * 1.1, 21):
        for w in (0.0007, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005):
            r, cv, h = base_stats(sim(a, n, i0, w, use_mg))
            if r < 0.5 or not np.isfinite(h):
                continue
            err = abs(r - 4.56) / 4.56 + abs(cv - 0.61) / 0.61 + abs(h - 106) / 106
            if best is None or err < best[0]:
                best = (err, i0, w, r, cv, h)
    return best


def event(i0, w, use_mg, hetero, dur, frac, pre=30, post=40, trials=3000):
    a, n = drive(pre + post, trials, frac, pre, dur, hetero)
    s = sim(a, n, i0, w, use_mg)
    r = s.mean(1)
    x = r[pre:] / max(r[:pre].mean(), 1e-12)
    win = s[pre:pre + dur + 3]
    isis = [np.diff(np.nonzero(win[:, j])[0]) * DT for j in range(trials)]
    isi = np.concatenate([q for q in isis if len(q)]) if any(len(q) for q in isis) else np.array([])
    bi = isi[isi < 33.0]
    base_isi = [np.diff(np.nonzero(s[:pre, j])[0]) * DT for j in range(trials)]
    b = np.concatenate([q for q in base_isi if len(q)])
    return (x.max(), (x >= 2).sum() * DT, (len(bi) / max(len(isi), 1)), (bi.mean() if len(bi) else float("nan")),
            (b < 33.0).mean())


def tune(i0, w, use_mg, hetero, dur):
    lo, hi = 0.0, 0.3 if hetero in (False, True) else 0.3 * hetero
    for _ in range(14):
        mid = (lo + hi) / 2
        pk = event(i0, w, use_mg, hetero, dur, mid, trials=1500)[0]
        lo, hi = (mid, hi) if pk < 3.0 else (lo, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    print("的【マカク】：跳ねの間隔 33ms 未満（大きい正の誤差で平均 約28ms）／ ふだん CV 0.61・半分戻るまで 106ms", flush=True)
    for name, use_mg, hetero in (("0 そのまま", False, False), ("1 入力の偏り（指数の分布）", False, True),
                                  ("2 NMDA の電圧の効き", True, False), ("3 偏り ＋ 電圧の効き", True, True),
                                  ("4 一部の体だけ強く（30%）", False, 0.3), ("5 一部の体だけ強く（10%）", False, 0.1),
                                  ("6 一部の体だけ強く（10%）＋ 電圧の効き", True, 0.1)):
        b = fit(use_mg)
        print(f"\n{name}：ふだん 率 {b[3]:.2f}Hz・CV {b[4]:.2f}・半分戻るまで {b[5]:.0f}ms（i0 {b[1]:.4f}・w {b[2]:.4f}）", flush=True)
        print("  出来事の長さ   山     2倍以上の幅   出来事の間隔のうち 33ms 未満   その平均   （ふだんの 33ms 未満）", flush=True)
        for dur in (10, 15):
            fr = tune(b[1], b[2], use_mg, hetero, dur)
            pk, o2, share, mb, base_share = event(b[1], b[2], use_mg, hetero, dur, fr)
            print(f"  {dur * DT:6.0f}ms     ×{pk:4.2f}   {o2:6.0f}ms          {share * 100:5.1f}%              {mb:5.1f}ms    （{base_share * 100:4.1f}%）", flush=True)
