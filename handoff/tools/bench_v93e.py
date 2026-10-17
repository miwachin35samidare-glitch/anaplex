"""版93 の試し台 その5：黒質の入力（版93 で模型にある送り手だけ）で、率 4.56Hz・CV 0.61 になる強さを探す
  python3 bench_v93e.py [秒=20]
  （網は使わない。何も書き換えない。CPU）
【み】10-17 22:47「a」：版93 は今ある送り手で率と CV だけ合わせる。どの線も1本の強さは同じ。Charara の比は版95 で外節・網様部・線条体が入ってから
撃ち方 B（τm 40ms【決め 10-15】）：v ← v·α ＋ i0 ＋ g_e·(AMPA ＋ NMDA) − g_i·I、閾値 1 で 0 へ
  AMPA 1周 ／ NMDA 75.5ms【マウス】Pearlstein 2015（黒質で決めた形）。AMPA と NMDA は半々 ⚠️（黒質の割合の値は無い）
送り手（受け手1体あたりの線）
  興奮：PPN 512本（23.2Hz【ヒト】）・視床下核の辺縁 64本（19.3Hz【ヒト】）・前頭前野 8本（3.30Hz【ヒト】、まばら＝ノブ【み】10-14）・上丘の誤差（ふだんほぼ撃たない ⇒ 0）
  抑制：吻内側被蓋核 64本（17.8Hz【マカク】）
  ⚠️ 線の本数は切片の送り手の全員（PPN・視床下核・吻内側被蓋核）。割合の値は無い
的：率 4.56Hz【ヒト】Ramayya 2014 ／ CV 0.61・撃ったあと半分戻るまで 約106ms【マカク】Bayer 2007
求め方：g_e・g_i を格子で振り、i0 は率が的に合うよう補間 ⇒ CV が的に近い組
"""
import sys, math, numpy as np, torch

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
DT = 1.067; TAU = 40.0; TN = 75.5
EXC = [("PPN", 512, 23.2), ("視床下核", 64, 19.3), ("前頭前野", 8, 3.30)]
INH = [("吻内側被蓋核", 64, 17.8)]
NE = sum(n for _, n, _ in EXC); NI = sum(n for _, n, _ in INH)
torch.manual_seed(0)


def run(ge, gi, i0, nstep):
    a = math.exp(-DT / TAU); aN = math.exp(-DT / TN)
    sh = ge.shape
    v = torch.rand(sh); S = torch.zeros(sh); cnt = torch.zeros(sh); last = torch.full(sh, -1.0)
    s1 = torch.zeros(sh); s2 = torch.zeros(sh); ni = torch.zeros(sh)
    for t in range(nstep):
        e = sum(torch.distributions.Binomial(n, torch.full(sh, hz * DT / 1000)).sample() for _, n, hz in EXC) / NE
        i = sum(torch.distributions.Binomial(n, torch.full(sh, hz * DT / 1000)).sample() for _, n, hz in INH) / NI
        S = S * aN + e * (1 - aN)
        v = (v * a + i0 + ge * (0.5 * e + 0.5 * S) - gi * i).clamp(min=-5.0)
        f = v >= 1.0
        ff = f & (last >= 0)
        if ff.any():
            d = (t - last)[ff]; s1[ff] += d; s2[ff] += d ** 2; ni[ff] += 1
        last[f] = t; cnt += f.float(); v = torch.where(f, torch.zeros_like(v), v)
    hz = cnt / (nstep * DT / 1000)
    m = s1 / ni.clamp(min=1); cv = (s2 / ni.clamp(min=1) - m ** 2).clamp(min=0).sqrt() / m.clamp(min=1e-9)
    return hz, cv


def main():
    GE = np.geomspace(0.05, 200, 14); GI = np.r_[0, np.geomspace(0.05, 200, 10)]; I0 = np.geomspace(1e-4, 0.1, 26)
    ge, gi = np.meshgrid(GE, GI, indexing="ij"); ge, gi = ge.ravel(), gi.ravel(); G = len(ge); NC = 4
    T3 = lambda x: torch.tensor(x).view(G, 1, 1).expand(G, len(I0), NC).float()
    z0 = torch.tensor(I0).view(1, -1, 1).expand(G, len(I0), NC).float()
    nfit = int(4000 / DT)
    curve = run(T3(ge), T3(gi), z0, nfit)[0].mean(2).numpy()
    i0 = np.full(G, np.nan)
    for k in range(G):
        c = np.maximum.accumulate(curve[k])
        if c[0] <= 4.56 <= c[-1]:
            i0[k] = np.exp(np.interp(4.56, c, np.log(I0)))
    ok = ~np.isnan(i0); ge, gi, i0 = ge[ok], gi[ok], i0[ok]; G = len(ge)
    hz, cv = run(torch.tensor(ge).view(G, 1).expand(G, 16).float(), torch.tensor(gi).view(G, 1).expand(G, 16).float(),
                 torch.tensor(i0).view(G, 1).expand(G, 16).float(), int(SEC * 1000 / DT))
    hz, cv = hz.mean(1).numpy(), cv.mean(1).numpy()
    good = (abs(hz - 4.56) < 0.5)
    print(f"試し台その5：黒質（{SEC:.0f}秒）率が 4.56±0.5Hz に入った組 {good.sum()} ／ {G}", flush=True)
    order = np.argsort(abs(cv - 0.61) + 10 * (~good))
    print("CV が 0.61 に近い組（上位 12）", flush=True)
    for k in order[:12]:
        drive = ge[k] * (sum(n * hz_ for _, n, hz_ in EXC) / NE) * DT / 1000
        print(f"  g_e {ge[k]:8.3f}・g_i {gi[k]:8.3f}・i0 {i0[k]:.5f} ⇒ 率 {hz[k]:.2f}Hz・CV {cv[k]:.2f} ／ 入力の興奮の平均 {drive:.4f}・i0 との比 {drive / i0[k]:.1f}", flush=True)
    print(f"CV の幅（率が合った組）：{cv[good].min():.2f}〜{cv[good].max():.2f}", flush=True)


if __name__ == "__main__":
    main()
