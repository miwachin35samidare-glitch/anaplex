"""版93 の試し台 その6：中脳の鎖（境 → 外側手綱核 → 吻内側被蓋核 → 黒質）の強さを霊長類の応えに合わせ、動く最低数を測り直す
  python3 bench_v93f.py
  （網は使わない。何も書き換えない。CPU）
【み】10-18 0:17「調べて」→ 0:20・0:22 の PDF で値（設計案_基底核と小脳 10-18 0:20〜）
置いたもの（撃ち方 B、周回 1.067ms、τm 40ms）
  境の細胞：送り手として決まった率で撃つ（ふだん 33Hz【マカク】Hong & Hikosaka 2008）。合図で +40%（150ms）
  外側手綱核：i0 あり【マウス】Fedorov 2026 ／ ふだん 30Hz【マカク】／ 境から興奮（受け手1体は境の全員から）
       強さ g_L：境 +40% ⇒ 外側手綱核 +60%（Matsumoto 2007 の 26 → 42Hz・Hong & Hikosaka 2008 図6）に合わせる
  吻内側被蓋核：i0 ＝ 0【み】10-18 0:15 ／ 外側手綱核から興奮のみ ⇒ 強さ g_R は率 17.8Hz【マカク】Hong 2011 で決まる
  黒質：bench_v93e の組（g_e 0.065・g_i 2.0・i0 は率 4.56Hz に合わせ直し）、PPN 512・視床下核 64・前頭前野 8 はポアソン、抑制は吻内側被蓋核の全員
  AMPA と NMDA（75.5ms）半々は外側手綱核・吻内側被蓋核・黒質で同じ ⚠️
確かめ（合わせない）
  1 外側手綱核をそろって1発 ⇒ ドーパミン 10〜40ms で 約6割減・50〜60ms で戻る（Matsumoto 2007）
  2 境 +40%（150ms）⇒ ドーパミンが下がるか（大きさ・遅れ）
動く最低数（設計案_脳幹 §12-42 と同じ見方：合図のあとのドーパミンの下がりが、ふだんの揺れの何倍か t）
  各核の体数 N ＝ 16・32・64・128・256 で
"""
import math, numpy as np, torch

DT = 1.067; TAU = 40.0; TN = 75.5
A, AN = math.exp(-DT / TAU), math.exp(-DT / TN)
torch.manual_seed(0)


class Pop:
    def __init__(self, n, i0, shape=()):
        self.v = torch.rand(*shape, n); self.S = torch.zeros(*shape, n); self.i0 = i0

    def step(self, e, i, ge, gi):
        self.S = self.S * AN + e * (1 - AN)
        self.v = (self.v * A + self.i0 + ge * (0.5 * e + 0.5 * self.S) - gi * i).clamp(min=-5.0)
        f = self.v >= 1.0
        self.v = torch.where(f, torch.zeros_like(self.v), self.v)
        return f.float()


def chain(N, gL, gR, i0L, i0S, nstep, events=(), volley=(), B=1):
    """B 回を並べて回す。返り値：各核の発火の平均（時刻ごと）"""
    L = Pop(N, i0L, (B,)); R = Pop(N, 0.0, (B,)); S = Pop(N, i0S, (B,))
    out = {"境": [], "外側手綱核": [], "吻内側被蓋核": [], "黒質": []}
    fL = torch.zeros(B, N); fR = torch.zeros(B, N)
    p_g = 33.0 * DT / 1000
    for t in range(nstep):
        boost = any(a <= t < b for a, b in events)
        g = (torch.rand(B, N) < p_g * (1.4 if boost else 1.0)).float()
        eL = g.mean(1, keepdim=True).expand(B, N)                 # 境の全員から
        nL = L.step(eL, torch.zeros_like(eL), gL, 0.0)
        if t in volley:
            nL = torch.ones_like(nL)
        eR = fL.mean(1, keepdim=True).expand(B, N)                 # 1周遅れで外側手綱核の全員から
        nR = R.step(eR, torch.zeros_like(eR), gR, 0.0)
        ex = (torch.distributions.Binomial(512, torch.full((B, N), 23.2 * DT / 1000)).sample()
              + torch.distributions.Binomial(64, torch.full((B, N), 19.3 * DT / 1000)).sample()
              + torch.distributions.Binomial(8, torch.full((B, N), 3.30 * DT / 1000)).sample()) / (512 + 64 + 8)
        iS = fR.mean(1, keepdim=True).expand(B, N)
        nS = S.step(ex, iS, 0.065, 2.0)
        fL, fR = nL, nR
        for k, v in (("境", g), ("外側手綱核", nL), ("吻内側被蓋核", nR), ("黒質", nS)):
            out[k].append(v.mean(1))
    return {k: torch.stack(v, 1) / (DT / 1000) for k, v in out.items()}     # [B, t] Hz


def fit_i0(N, gL, gR, i0L, i0S, it=8):
    for _ in range(it):
        o = chain(N, gL, gR, i0L, i0S, int(3000 / DT), B=4)
        rL, rS = float(o["外側手綱核"].mean()), float(o["黒質"].mean())
        i0L *= float(np.clip((30.0 / max(rL, 0.5)) ** 0.5, 0.6, 1.6))
        i0S *= float(np.clip((4.56 / max(rS, 0.2)) ** 0.5, 0.6, 1.6))
    return i0L, i0S, o


def main():
    N = 64
    print("1 強さを合わせる（N 64）", flush=True)
    best = None
    for gL in (0.5, 1.0, 2.0, 4.0, 8.0):
        i0L, i0S = 0.1, 0.058
        # 吻内側被蓋核の強さ：率 17.8Hz
        lo, hi = 0.1, 50.0
        for _ in range(12):
            gR = math.sqrt(lo * hi)
            i0L, i0S, o = fit_i0(N, gL, gR, i0L, i0S, it=3)
            rR = float(o["吻内側被蓋核"].mean())
            lo, hi = (gR, hi) if rR < 17.8 else (lo, gR)
        i0L, i0S, o = fit_i0(N, gL, gR, i0L, i0S, it=6)
        T = int(2000 / DT); ev = (int(1000 / DT), int(1150 / DT))
        o = chain(N, gL, gR, i0L, i0S, T, events=[ev], B=16)
        base = o["外側手綱核"][:, : ev[0]].mean(); up = o["外側手綱核"][:, ev[0]:ev[1]].mean()
        resp = float(up / base - 1)
        print(f"  g_L {gL:4.1f}・g_R {gR:6.2f}・i0（外側手綱核）{i0L:.4f}・i0（黒質）{i0S:.4f} ⇒ 率 外側手綱核 {float(base):.1f}・吻内側被蓋核 {float(o['吻内側被蓋核'].mean()):.1f}・黒質 {float(o['黒質'][:, :ev[0]].mean()):.2f}Hz ／ 境 +40% で外側手綱核 {resp * 100:+.0f}%", flush=True)
        if best is None or abs(resp - 0.6) < abs(best[0] - 0.6):
            best = (resp, gL, gR, i0L, i0S)
    _, gL, gR, i0L, i0S = best
    print(f"  ⇒ 使う：g_L {gL}・g_R {gR:.2f}", flush=True)

    print("\n2 確かめ：外側手綱核をそろって1発（Matsumoto 2007：10〜40ms で 約6割減、50〜60ms で戻る）", flush=True)
    t0 = int(1000 / DT)
    o = chain(N, gL, gR, i0L, i0S, int(1300 / DT), volley={t0}, B=64)
    s = o["黒質"].mean(0)
    base = float(s[: t0].mean())
    for a, b in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 120)):
        m = float(s[t0 + int(a / DT): t0 + int(b / DT)].mean())
        print(f"  {a:3d}〜{b:3d}ms：ドーパミン {m:.2f}Hz（ふだん {base:.2f}）{(m / base - 1) * 100:+.0f}%", flush=True)

    print("\n3 境 +40%（150ms）⇒ ドーパミン と 動く最低数（t ＝ 下がり ÷ ふだんの揺れ、§12-42）", flush=True)
    for N2 in (16, 32, 64, 128, 256):
        i0L2, i0S2, _ = fit_i0(N2, gL, gR, i0L, i0S, it=5)
        ev = (int(1000 / DT), int(1150 / DT))
        o = chain(N2, gL, gR, i0L2, i0S2, int(1500 / DT), events=[ev], B=40)
        s = o["黒質"]
        win = slice(ev[0] + int(20 / DT), ev[1] + int(50 / DT))
        pre = s[:, : ev[0]]
        d = s[:, win].mean(1) - pre.mean(1)                     # 1回ごとの下がり
        # ふだんの揺れ：同じ長さの窓を合図の前から取ったときの散らばり
        L = win.stop - win.start
        segs = torch.stack([pre[:, k: k + L].mean(1) for k in range(0, pre.shape[1] - L, L)], 1)
        sd = float((segs - pre.mean(1, keepdim=True)).std())
        t = float(-d.mean()) / max(sd, 1e-9)
        print(f"  N {N2:3d}：ドーパミン ふだん {float(pre.mean()):.2f}Hz・合図の間 {float(s[:, win].mean()):.2f}Hz（{float(d.mean() / pre.mean()) * 100:+.0f}%）・t {t:.1f}", flush=True)


if __name__ == "__main__":
    main()
