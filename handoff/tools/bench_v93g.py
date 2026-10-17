"""版93 の試し台 その7：黒質の「撃ったあと戻る深さ」v_r を振り、ふだんの撃ち方と抑制1発の応えが両方合う形があるかを見る
  python3 bench_v93g.py
  （網は使わない。何も書き換えない。CPU）
【み】10-18 1:47「a」（調べる）⇒ 人・霊長類の仕組みの値は無い。霊長類で測れているのは外から見た形だけ：
  ふだん 4.56Hz【ヒト】・CV 0.61・撃ったあと半分戻るまで 約106ms【マカク】Bayer 2007 ／ 抑制1発で 10〜40ms 約6割減・50〜60ms で戻る【マカク】Matsumoto 2007
撃ち方 B（τm 40ms）で撃ったあと v を 0 ではなく v_r（≤ 0）へ戻す。v_r ＝ 0 が今の形
  v_r ＝ 0 では 4.56Hz を出すのに v の行き着く先が閾値すれすれ ⇒ 少しの抑制で長く止まる（bench_v93f 版4）
  v_r が深いと、行き着く先を閾値より上に置いたまま、撃ったあとの静かな時間を作れる
入力（bench_v93e と同じ）：興奮 PPN 512・視床下核 64・前頭前野 8（ポアソン）／ 抑制 吻内側被蓋核 64（17.8Hz）
抑制1発：吻内側被蓋核の 64本が1周にそろって撃つ（受け手に g_i × 1）。外側手綱核の1発で吻内側被蓋核がそろって撃つ形の近似
"""
import math, numpy as np, torch

DT = 1.067; TAU = 40.0; TN = 75.5
A, AN = math.exp(-DT / TAU), math.exp(-DT / TN)
EXC = [(512, 23.2), (64, 19.3), (8, 3.30)]; NE = 584
torch.manual_seed(0)


def ex(sh):
    return sum(torch.distributions.Binomial(n, torch.full(sh, hz * DT / 1000)).sample() for n, hz in EXC) / NE


def inh(sh):
    return torch.distributions.Binomial(64, torch.full(sh, 17.8 * DT / 1000)).sample() / 64


def run(ge, gi, i0, vr, nstep, keep=False, volley=None):
    sh = ge.shape
    v = torch.rand(sh); S = torch.zeros(sh)
    cnt = torch.zeros(sh); last = torch.full(sh, -1.0); s1 = torch.zeros(sh); s2 = torch.zeros(sh); ni = torch.zeros(sh)
    rec = []
    for t in range(nstep):
        e = ex(sh); i = inh(sh)
        if volley is not None and t in volley:
            i = torch.ones(sh)
        S = S * AN + e * (1 - AN)
        v = (v * A + i0 + ge * (0.5 * e + 0.5 * S) - gi * i).clamp(min=-5.0)
        f = v >= 1.0
        ff = f & (last >= 0)
        d = (t - last)[ff]; s1[ff] += d; s2[ff] += d ** 2; ni[ff] += 1
        last[f] = t; cnt += f.float()
        v = torch.where(f, vr, v)
        if keep:
            rec.append(f)
    hz = cnt / (nstep * DT / 1000)
    m = s1 / ni.clamp(min=1); cv = (s2 / ni.clamp(min=1) - m ** 2).clamp(min=0).sqrt() / m.clamp(min=1e-9)
    return hz, cv, (torch.stack(rec, -1) if keep else None)


def half_recovery(sp):
    """sp [..., 時刻] ⇒ 撃ったあと、撃つ確からしさがふだんの半分に戻るまで（ms）"""
    x = sp.reshape(-1, sp.shape[-1]).float(); p = x.mean()
    for k in range(1, 400):
        c = (x[:, :-k] * x[:, k:]).mean() / x[:, :-k].mean().clamp(min=1e-12) / p
        if c >= 0.5:
            return k * DT
    return float("nan")


def main():
    VR = [0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    GE = np.geomspace(0.02, 2.0, 7); GI = np.r_[0.0, np.geomspace(0.05, 5.0, 7)]
    print("1 ふだんに合わせる（率 4.56Hz・CV 0.61・半分戻るまで 106ms）", flush=True)
    finals = []
    for vr in VR:
        ge, gi = np.meshgrid(GE, GI, indexing="ij"); ge, gi = ge.ravel(), gi.ravel(); G = len(ge)
        lo0 = (1 - A) * 0.5; I0 = np.geomspace(lo0, 0.3, 24); NC = 4
        T3 = lambda x: torch.tensor(x).view(G, 1, 1).expand(G, len(I0), NC).float()
        z = torch.tensor(I0).view(1, -1, 1).expand(G, len(I0), NC).float()
        curve = run(T3(ge), T3(gi), z, torch.tensor(vr), int(4000 / DT))[0].mean(2).numpy()
        i0 = np.full(G, np.nan)
        for k in range(G):
            c = np.maximum.accumulate(curve[k])
            if c[0] <= 4.56 <= c[-1]:
                i0[k] = np.exp(np.interp(4.56, c, np.log(I0)))
        ok = ~np.isnan(i0); g_e, g_i, i_0 = ge[ok], gi[ok], i0[ok]; G2 = len(g_e)
        T2 = lambda x: torch.tensor(x).view(G2, 1).expand(G2, 16).float()
        hz, cv, _ = run(T2(g_e), T2(g_i), T2(i_0), torch.tensor(vr), int(15000 / DT))
        hz, cv = hz.mean(1).numpy(), cv.mean(1).numpy()
        err = np.abs(hz - 4.56) / 4.56 + np.abs(cv - 0.61) / 0.61
        order = np.argsort(err)[:4]
        for k in order:
            _, _, sp = run(torch.full((32,), g_e[k]), torch.full((32,), g_i[k]), torch.full((32,), i_0[k]), torch.tensor(vr), int(20000 / DT), keep=True)
            h = half_recovery(sp)
            vinf = (i_0[k] + g_e[k] * sum(n * hz_ for n, hz_ in EXC) / NE * DT / 1000 - g_i[k] * 17.8 * DT / 1000) / (1 - A)
            finals.append((vr, g_e[k], g_i[k], i_0[k], hz[k], cv[k], h, vinf))
            print(f"  v_r {vr:5.1f}：g_e {g_e[k]:.3f}・g_i {g_i[k]:.3f}・i0 {i_0[k]:.4f} ⇒ 率 {hz[k]:.2f}Hz・CV {cv[k]:.2f}・半分戻るまで {h:.0f}ms ／ 行き着く先 {vinf:.2f}", flush=True)

    print("\n2 抑制1発（吻内側被蓋核 64本がそろって撃つ）⇒ 黒質（Matsumoto 2007：10〜40ms 約0.4倍・50〜60ms で戻る）", flush=True)
    t0 = int(3000 / DT)
    for vr, ge, gi, i0, hz, cv, h, vinf in finals:
        if gi == 0:
            continue
        _, _, sp = run(torch.full((4096,), ge), torch.full((4096,), gi), torch.full((4096,), i0), torch.tensor(vr), t0 + int(300 / DT), keep=True, volley={t0})
        r = sp.float().mean(0); base = float(r[int(1000 / DT): t0].mean())
        w = lambda a, b: float(r[t0 + int(a / DT): t0 + int(b / DT)].mean()) / base
        print(f"  v_r {vr:5.1f}・g_i {gi:.3f}（CV {cv:.2f}・半分 {h:.0f}ms）：0〜10 {w(0, 10):.2f}・10〜40 {w(10, 40):.2f}・40〜50 {w(40, 50):.2f}・50〜60 {w(50, 60):.2f}・60〜100 {w(60, 100):.2f}・100〜200 {w(100, 200):.2f}", flush=True)


if __name__ == "__main__":
    main()
