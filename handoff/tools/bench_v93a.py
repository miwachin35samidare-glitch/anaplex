"""版93 の試し台 その1：核1つずつ、外からの入力を決まった率で入れて（開いた輪）、τm と入力の強さを霊長類の値に合わせる
  python3 bench_v93a.py [秒=10]
  （網は使わない。何も書き換えない。CPU で回る。コンテナで回した結果を Claude が報告する）
【み】10-17 0:51「まぁそれでやってみて」（核どうしの線 A）／ 0:54「Aでいいよ」（内節・網様部にも i0）
撃ち方 B（周回 1.067ms）：v ← v·α ＋ i0 ＋ g_e·E − g_i·I、閾値 1 で撃って 0 へ。α ＝ exp(−dt/τm)
  E・I ＝ 送り手ごとに（終末の割合 ÷ 線の本数）×（その周に撃った送り手の数）の和。送り手は決まった率で撃つ（ポアソン）
合わせる的（核ごと）
  外節【マカク】率 69.7Hz（Starr 2005）・CV 1.15（2022 Nat Commun）
      グルタミン酸（NBQX）を止める −56.7%・GABA-A（gabazine）を止める +115.8%（Kita 2004）／ 両方止める +10%（Kita 2005：62.2→68.4）
  視床下核【ヒト】率 19.3Hz・CV 1.29（Steigerwald 2008）
      【マカク】グルタミン酸（NBQX＋CPP）−16%・gabazine +26%・両方 ≒ gabazine だけ（Polyakova 2020）
  内節・網様部：止める実験の数が無い ⇒ その2 で（率と CV だけ）
線（受け手1体あたり）
  外節 ← 線条体 100本（【マカク】Percheron）抑制・終末 0.80 ／ ← 外節 64本（核の中の枝。切片の上限）抑制・0.05 ／ ← 視床下核 53本（64 × 82.7%、Sato 2000b）興奮
  視床下核 ← 外節 54本（64 × 84.2%、Sato 2000a）抑制 ／ ← 前頭前野 946本（切片の上限、⚠️ 霊長類の割合は未調査）興奮 3.30Hz（【ヒト】Verma 2024）
  ⚠️ 終末の割合は樹状突起の値（数で多い方、Shink & Smith 1995【リスザル】）。細胞体の違い（内節で外節 48%）はまだ入れていない
  ⚠️ 興奮は AMPA の形（1周で効く）だけ。NMDA の長い尾はまだ
求め方：τm ごとに g_e・g_i を格子で振り、i0 は率が的に合うよう補間で決める ⇒ 止めたときの率の比が的に一番近い g_e・g_i ⇒ そのときの CV
"""
import sys, math, numpy as np, torch

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
DT = 1.067                       # ms（周回）
NSTEP = int(SEC * 1000 / DT)
torch.manual_seed(0)

NUC = {
    "外節": dict(R=69.7, CV=1.15, glu=1 - 0.567, gaba=1 + 1.158, both=68.4 / 62.2,
               lines=[("線条体", 100, 2.1, -1, 0.80), ("外節", 64, 69.7, -1, 0.05), ("視床下核", 53, 19.3, +1, 1.0)]),
    "視床下核": dict(R=19.3, CV=1.29, glu=1 - 0.16, gaba=1 + 0.26, both=1 + 0.26,
                 lines=[("外節", 54, 69.7, -1, 1.0), ("前頭前野", 946, 3.30, +1, 1.0)]),
}
TAUS = [5.0, 10.0, 20.0, 40.0, 80.0]
GE = np.geomspace(0.02, 50.0, 14)
GI = np.geomspace(0.02, 50.0, 14)
I0 = np.geomspace(1e-4, 0.6, 28)
NC = 6                          # 1組あたりの細胞数


def drive(lines, shape, sign):
    """送り手を決まった率で撃たせ、その周の入力（終末の割合 ÷ 本数 × 撃った数）を返す"""
    tot = None
    sh = sum(s for _, _, _, sg, s in lines if sg == sign)
    for name, n, hz, sg, s in lines:
        if sg != sign:
            continue
        p = hz * DT / 1000.0
        k = torch.distributions.Binomial(n, torch.full(shape, p)).sample()
        x = (s / sh) / n * k
        tot = x if tot is None else tot + x
    return tot if tot is not None else torch.zeros(shape)


def run(lines, tau, ge, gi, i0, nstep=NSTEP):
    """ge・gi・i0 は同じ形の配列。返り値：率 Hz・CV（細胞ごと）"""
    a = math.exp(-DT / tau)
    shape = ge.shape
    v = torch.rand(shape)
    cnt = torch.zeros(shape); last = torch.full(shape, -1.0)
    s1 = torch.zeros(shape); s2 = torch.zeros(shape); ni = torch.zeros(shape)
    # 入力の平均は1周あたり（割合の和 1）× 率 × dt。g は「その平均が閾値の何倍か」の目盛りにしない（そのまま）
    for t in range(nstep):
        e = drive(lines, shape, +1); i = drive(lines, shape, -1)
        v = v * a + i0 + ge * e - gi * i
        v = v.clamp(min=-5.0)
        f = v >= 1.0
        if f.any():
            isi = (t - last)[f & (last >= 0)]
            ff = f & (last >= 0)
            s1[ff] += (t - last)[ff]; s2[ff] += ((t - last)[ff]) ** 2; ni[ff] += 1
            last[f] = t; cnt[f] += 1
        v = torch.where(f, torch.zeros_like(v), v)
    hz = cnt / (nstep * DT / 1000.0)
    m = s1 / ni.clamp(min=1); var = s2 / ni.clamp(min=1) - m ** 2
    cv = var.clamp(min=0).sqrt() / m.clamp(min=1e-9)
    return hz, cv


def fit_i0(lines, tau, ge, gi, R, nstep):
    """格子の i0 ごとに率を出し、的 R に合う i0 を補間（ge・gi は [G] の配列）"""
    G = len(ge)
    g_e = torch.tensor(ge).view(G, 1, 1).expand(G, len(I0), NC).float()
    g_i = torch.tensor(gi).view(G, 1, 1).expand(G, len(I0), NC).float()
    i0 = torch.tensor(I0).view(1, -1, 1).expand(G, len(I0), NC).float()
    hz, _ = run(lines, tau, g_e, g_i, i0, nstep)
    curve = hz.mean(2).numpy()                       # [G, len(I0)]
    out = np.full(G, np.nan)
    for k in range(G):
        c = np.maximum.accumulate(curve[k])         # 単調に（揺らぎを均す）
        if c[0] <= R <= c[-1]:
            out[k] = np.exp(np.interp(R, c, np.log(I0)))
    return out


def main():
    nfit = max(2000, NSTEP // 3)
    print(f"試し台その1：{SEC:.0f}秒（周回 {NSTEP:,}）・1組 {NC} 体", flush=True)
    for nm, d in NUC.items():
        print(f"\n== {nm}：率 {d['R']}Hz・CV {d['CV']} ／ 止めたときの比 グルタミン酸 {d['glu']:.2f}・GABA {d['gaba']:.2f}・両方 {d['both']:.2f}", flush=True)
        for tau in TAUS:
            ge, gi = np.meshgrid(GE, GI, indexing="ij"); ge, gi = ge.ravel(), gi.ravel()
            i0 = fit_i0(d["lines"], tau, ge, gi, d["R"], nfit)
            ok = ~np.isnan(i0)
            if not ok.any():
                print(f"  τm {tau:4.0f}ms：率に届く組なし", flush=True)
                continue
            ge, gi, i0 = ge[ok], gi[ok], i0[ok]
            G = len(ge)
            T = lambda x: torch.tensor(x).view(G, 1).expand(G, NC).float()
            z = torch.zeros(G, NC)
            base, cvb = run(d["lines"], tau, T(ge), T(gi), T(i0), nfit)
            glu, _ = run(d["lines"], tau, z, T(gi), T(i0), nfit)
            gab, _ = run(d["lines"], tau, T(ge), z, T(i0), nfit)
            b = np.maximum(base.mean(1).numpy(), 1e-9)
            rg, rb = glu.mean(1).numpy() / b, gab.mean(1).numpy() / b
            err = np.log(np.maximum(rg, 1e-3) / d["glu"]) ** 2 + np.log(np.maximum(rb, 1e-3) / d["gaba"]) ** 2 \
                + np.log(np.maximum(b, 1e-3) / d["R"]) ** 2
            k = int(np.argmin(err))
            # 選んだ組を長く回して確かめる（両方止める も）
            one = lambda x: torch.full((1, 64), float(x))
            hz, cv = run(d["lines"], tau, one(ge[k]), one(gi[k]), one(i0[k]))
            hzg, _ = run(d["lines"], tau, one(0), one(gi[k]), one(i0[k]))
            hzb, _ = run(d["lines"], tau, one(ge[k]), one(0), one(i0[k]))
            hz0, _ = run(d["lines"], tau, one(0), one(0), one(i0[k]))
            r = max(float(hz.mean()), 1e-9)
            print(f"  τm {tau:4.0f}ms：g_e {ge[k]:.3f}・g_i {gi[k]:.3f}・i0 {i0[k]:.4f} ⇒ 率 {r:5.1f}Hz・CV {float(cv.mean()):.2f}"
                  f" ／ 止める グルタミン酸 {float(hzg.mean()) / r:.2f}・GABA {float(hzb.mean()) / r:.2f}・両方 {float(hz0.mean()) / r:.2f}", flush=True)


if __name__ == "__main__":
    main()
