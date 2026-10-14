"""黒質1体の試し台2：霊長類の撃ち方（ばらつき）に合わせてから、跳ねるかを見る（網は使わない。何も書かない。CPU で1〜2分）
  python3 bench_sn2.py
【み】10-15「リズムを調べて」「やってみて」
合わせる的（ふだん）
  率 4.56Hz【ヒト】Ramayya 2014 ／ 間隔の CV 0.61・撃ったあと半分戻るまで 約106ms【マカク】Bayer・Lau・Glimcher 2007
跳ねの的：約3倍・幅 約150ms【ヒト】Ramayya 2014・Zaghloul 2009 ／ 跳ねの間隔 約28〜33ms【マカク】Bayer 2007
置いたもの（bench_sn.py と同じ）：1歩 10.67ms ／ 興奮の入力 2,520本・ふだん 5Hz ／ AMPA は届いた歩だけ・NMDA 減り 75.5ms ／ 電荷の比 1:1
  B：自分で撃ち続ける膜（毎歩 i0 ずつ上がり、閾値 1 で撃って 0 に戻る。減り τm 20ms ⚠️【C】）＋ 届いた分 × w
     入力のばらつきだけで膜が揺れる。i0（率）と w（ばらつき）を、率 4.56Hz・CV 0.61 に合わせる
  A2：毎歩のくじ（A）＋ 撃ったあと撃ちにくい（撃ってからの歩 s で 1 − exp(−s／τr)）。p0 と τr を率・CV に合わせる。G は本数で割り直す
"""
import math, numpy as np

DT = 10.67
N_IN = 2520
RATE_IN = 5.0
RATE_EV = 40.0
DEC_N = math.exp(-DT / 75.5)
CHARGE_N = 1.0 - DEC_N
LAM0 = RATE_IN * DT / 1000.0 * N_IN
BASE = LAM0 * 2.0
rng = np.random.default_rng(0)


def drive(T, trials, frac=0.0, t0=0, dur=0):
    """[歩, 試行] の（届いた分 − ふだん）"""
    lam = np.full((T, trials), LAM0)
    lam[t0:t0 + dur] += int(round(frac * N_IN)) * (RATE_EV - RATE_IN) * DT / 1000.0
    k = rng.poisson(lam).astype(float)
    acc = np.full(trials, LAM0)
    out = np.empty_like(k)
    for t in range(T):
        acc = acc * DEC_N + k[t] * CHARGE_N
        out[t] = k[t] + acc - BASE
    return out


def sim_b(d, i0, w, tau_m=20.0):
    alpha = math.exp(-DT / tau_m)
    v = rng.random(d.shape[1]) * 0.5
    s = np.zeros_like(d, dtype=bool)
    for t in range(d.shape[0]):
        v = v * alpha + i0 + w * d[t]
        f = v >= 1.0
        s[t] = f
        v[f] = 0.0
    return s


def sim_a2(d, p0, tau_r, G):
    since = np.full(d.shape[1], 50.0)
    s = np.zeros_like(d, dtype=bool)
    for t in range(d.shape[0]):
        h = 1.0 - np.exp(-since / tau_r)
        p = np.clip(p0 * np.exp(G * d[t]) * h, 0, 1)
        f = rng.random(d.shape[1]) < p
        s[t] = f
        since = np.where(f, 0.0, since + 1.0)
    return s


def stats(s):
    """率・間隔の CV・撃ったあと半分戻るまで（ms）"""
    rate = s.mean() * 1000 / DT
    isis = []
    for j in range(s.shape[1]):
        idx = np.nonzero(s[:, j])[0]
        if len(idx) > 2:
            isis.append(np.diff(idx))
    isi = np.concatenate(isis) * DT if isis else np.array([1.0])
    cv = isi.std() / isi.mean()
    p = s.mean()
    cond = []
    for k in range(1, 40):
        cond.append((s[:-k] & s[k:]).mean() / max(s[:-k].mean(), 1e-12) / max(p, 1e-12))
    cond = np.array(cond)
    half = (np.argmax(cond >= 0.5) + 1) * DT if (cond >= 0.5).any() else float("nan")
    return rate, cv, half


HALF = True     # 撃ったあと半分戻るまで（106ms）も合わせる


def fit(sim, a_range, b_range, T=3000, trials=300):
    """率 4.56Hz・CV 0.61 に一番近い組"""
    d = drive(T, trials)
    best = None
    for a in a_range:
        for b in b_range:
            r, cv, h = stats(sim(d, a, b))
            err = abs(r - 4.56) / 4.56 + abs(cv - 0.61) / 0.61 + (abs(h - 106.0) / 106.0 if HALF else 0.0)
            if not np.isfinite(err) or r < 0.5:
                continue
            if best is None or err < best[0]:
                best = (err, a, b, r, cv, h)
    return best


def event(sim, a, b, dur, frac, pre=30, post=40, trials=3000):
    d = drive(pre + post, trials, frac, pre, dur)
    s = sim(d, a, b)
    r = s.mean(1)
    base = r[:pre].mean()
    x = r[pre:] / max(base, 1e-12)
    # 跳ねの中の間隔（出来事のあいだに2回以上撃った試行の、その間の間隔）
    sub = s[pre:pre + max(dur, 3)]
    isi = []
    for j in range(trials):
        idx = np.nonzero(sub[:, j])[0]
        if len(idx) >= 2:
            isi.append(np.diff(idx).mean() * DT)
    return x.max(), (x >= 2).sum() * DT, (x.argmax() + 0.5) * DT, (np.mean(isi) if isi else float("nan"))


def tune_ev(sim, a, b, dur):
    lo, hi = 0.0, 0.3
    for _ in range(14):
        mid = (lo + hi) / 2
        pk = event(sim, a, b, dur, mid, trials=1500)[0]
        lo, hi = (mid, hi) if pk < 3.0 else (lo, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    print("ふだんに合わせる（率 4.56Hz・CV 0.61・撃ったあと半分戻るまで 約106ms【マカク】）", flush=True)
    best = None
    for tm in (20.0, 40.0, 80.0, 160.0):         # 膜の減り τm ⚠️【C】も一緒に探す
        sb = lambda d, i0, w, tm=tm: sim_b(d, i0, w, tm)
        al = math.exp(-DT / tm); lo_i0 = 1 - al
        r = fit(sb, np.linspace(lo_i0 * 0.90, lo_i0 * 1.10, 21), [0.0002, 0.0004, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005])
        print(f"  B（τm {tm:.0f}ms）：i0 {r[1]:.4f}・w {r[2]:.4f} ⇒ 率 {r[3]:.2f}Hz・CV {r[4]:.2f}・半分戻るまで {r[5]:.0f}ms", flush=True)
        if best is None or r[0] < best[0]:
            best = r + (tm,)
    b = best
    sim_bt = lambda d, i0, w: sim_b(d, i0, w, b[6])
    print(f"  ⇒ B は τm {b[6]:.0f}ms", flush=True)
    Gs = math.log(25.0 / 4.0) / 8 / 10
    a2 = lambda d, p0, tr: sim_a2(d, p0, tr, Gs)
    a = fit(a2, np.linspace(0.04, 0.30, 14), [4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0])
    print(f"  A2：p0 {a[1]:.3f}・τr {a[2]:.1f}歩 ⇒ 率 {a[3]:.2f}Hz・CV {a[4]:.2f}・半分戻るまで {a[5]:.0f}ms", flush=True)
    for name, sim, x, y in (("B", sim_bt, b[1], b[2]), ("A2", a2, a[1], a[2])):
        print(f"\n{name}：山がふだんの約3倍になる強さで", flush=True)
        print("  出来事の長さ  入力のうち撃つ割合   山     2倍以上の幅   山の時刻   跳ねの中の間隔", flush=True)
        for dur in (1, 5, 10, 15):
            fr = tune_ev(sim, x, y, dur)
            pk, o2, tp, isi = event(sim, x, y, dur, fr)
            print(f"  {dur * DT:6.0f}ms      {fr * 100:7.2f}%     ×{pk:4.2f}   {o2:7.0f}ms     {tp:5.0f}ms     {isi:5.0f}ms", flush=True)
