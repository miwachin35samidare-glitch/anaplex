"""版93 の試し台 その2：閉じた輪（外節・視床下核・内節・網様部を本当に繋ぐ）で CV がどこまで上がるか
  python3 bench_v93b.py [τm=20] [秒=10]
  （網は使わない。何も書き換えない。CPU で回る）
【み】10-17 9:41「すすめて」（その1：開いた輪では CV が出ない ⇒ 閉じた輪で測る。上がらなければ足す前に相談）
置いたもの（その1 と同じ形、核どうしの線 A）
  各核 64体 ／ 撃ち方 B（周回 1.067ms）：v ← v·α ＋ i0 ＋ g_e·E − g_i·I、閾値 1 で撃って 0 へ
  線を出す細胞（送り手の切片のうち、Sato 2000 の割合）・受け手はその全員から受ける
    視床下核 → 外節 82.7%・内節 69.3%・網様部 24.0% ／ 外節 → 視床下核 84.2%・内節 31.6%・網様部 65.8% ／ 外節 → 外節（核の中の枝）全員
  線条体 2,288体（【ヒト】2.1Hz、ポアソン）→ 外節・内節・網様部 受け手1体に 100本（【マカク】Percheron）
  前頭前野 946体（【ヒト】3.30Hz、ポアソン）→ 視床下核 全員（⚠️ 割合は未調査）
  終末の割合（樹状突起、Shink & Smith 1995）：抑制の中で 線条体 0.80・外節 0.05
  遅れ：線条体 → 2周 ／ 核どうし 1周（probe_tick_nuclei と同じ置き方）
  g_e・g_i：その1（3秒）で止める比に合わせた値。内節・網様部は止める実験の数が無いので外節と同じ（⚠️ 仮）
  i0：閉じた輪で率が的に入るよう回しながら合わせる
測るもの：率・CV・核全体のそろい方（5ms ごとの数のばらつき ÷ ポアソンなら出る分）・外節と視床下核で 8体だけ止めたときの比
"""
import sys, math, numpy as np, torch

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
DT = 1.067
torch.manual_seed(0); np.random.seed(0)
N = 64
NUC = ["外節", "視床下核", "内節", "網様部"]
R = {"外節": 69.7, "視床下核": 19.3, "内節": 82.5, "網様部": 60.5}
CVT = {"外節": 1.15, "視床下核": 1.29, "内節": 0.89, "網様部": None}
# その1 の結果（3秒、止める比に合わせた組）
FIT = {20.0: {"外節": (2.466, 15.004, 0.1497), "視床下核": (0.405, 0.037, 0.0576)},
       40.0: {"外節": (2.466, 15.004, 0.1360), "視床下核": (0.740, 0.067, 0.0387)},
       10.0: {"外節": (1.351, 15.004, 0.2025), "視床下核": (0.740, 0.405, 0.1303)}}
N_STR, N_CX = 2288, 946
HZ_STR, HZ_CX = 2.1, 3.30
off = {nm: i * N for i, nm in enumerate(NUC)}
NIN = len(NUC) * N
S_STR, S_CX = NIN, NIN + N_STR
NSRC = NIN + N_STR + N_CX
DMAX = 2


def proj(frac, gen):
    """送り手の切片のうち、線を出す細胞（割合 frac）"""
    k = int(round(N * frac))
    return torch.randperm(N, generator=gen)[:k]


def build():
    gen = torch.Generator().manual_seed(1)
    WE = torch.zeros(DMAX + 1, NIN, NSRC); WI = torch.zeros(DMAX + 1, NIN, NSRC)
    P = {("視床下核", "外節"): 0.827, ("視床下核", "内節"): 0.693, ("視床下核", "網様部"): 0.240,
         ("外節", "視床下核"): 0.842, ("外節", "内節"): 0.316, ("外節", "網様部"): 0.658, ("外節", "外節"): 1.0}
    senders = {k: proj(v, gen) for k, v in P.items()}
    for rc in NUC:
        r0 = off[rc]
        # 抑制：線条体 0.80・外節 0.05（樹状突起）。視床下核は抑制が外節だけ
        inh = []
        if rc != "視床下核":
            inh.append(("線条体", 0.80))
        if ("外節", rc) in senders:
            inh.append(("外節", 0.05 if rc != "視床下核" else 1.0))
        sh = sum(s for _, s in inh)
        for r in range(r0, r0 + N):
            for src, s in inh:
                if src == "線条体":
                    idx = S_STR + torch.randperm(N_STR, generator=gen)[:100]; d = 2
                else:
                    idx = off[src] + senders[(src, rc)]; idx = idx[idx != r]; d = 1
                WI[d, r, idx] += (s / sh) / len(idx)
            # 興奮：視床下核から（外節・内節・網様部）／ 前頭前野から（視床下核）
            if rc == "視床下核":
                idx = torch.arange(S_CX, S_CX + N_CX); d = 1
            else:
                idx = off["視床下核"] + senders[("視床下核", rc)]; d = 1
            WE[d, r, idx] += 1.0 / len(idx)
    return WE, WI


def run(WE, WI, ge, gi, i0, nstep, block=None, rec=False):
    a = math.exp(-DT / TAU)
    ring = torch.zeros(DMAX + 1, NSRC)
    v = torch.rand(NIN)
    cnt = torch.zeros(NIN); last = torch.full((NIN,), -1.0)
    s1 = torch.zeros(NIN); s2 = torch.zeros(NIN); ni = torch.zeros(NIN)
    pop = [] if rec else None
    p_str, p_cx = HZ_STR * DT / 1000, HZ_CX * DT / 1000
    gE, gI = ge.clone(), gi.clone()
    if block is not None:
        idx, kind = block
        if kind in ("glu", "both"):
            gE[idx] = 0
        if kind in ("gaba", "both"):
            gI[idx] = 0
    for t in range(nstep):
        E = sum(WE[d] @ ring[d] for d in range(1, DMAX + 1))
        I = sum(WI[d] @ ring[d] for d in range(1, DMAX + 1))
        v = (v * a + i0 + gE * E - gI * I).clamp(min=-5.0)
        f = v >= 1.0
        v = torch.where(f, torch.zeros_like(v), v)
        ring = torch.roll(ring, 1, 0)
        ring[0].zero_()
        ring[0, :NIN] = f.float()
        ring[0, S_STR:S_CX] = (torch.rand(N_STR) < p_str).float()
        ring[0, S_CX:] = (torch.rand(N_CX) < p_cx).float()
        ff = f & (last >= 0)
        if ff.any():
            dt_ = (t - last)[ff]; s1[ff] += dt_; s2[ff] += dt_ ** 2; ni[ff] += 1
        last[f] = t; cnt += f.float()
        if rec:
            pop.append(f.float())
    hz = cnt / (nstep * DT / 1000)
    m = s1 / ni.clamp(min=1); cv = (s2 / ni.clamp(min=1) - m ** 2).clamp(min=0).sqrt() / m.clamp(min=1e-9)
    return hz, cv, (torch.stack(pop) if rec else None)


def main():
    if TAU not in FIT:
        sys.exit(f"τm は {sorted(FIT)} のどれか")
    WE, WI = build()
    ge = torch.zeros(NIN); gi = torch.zeros(NIN); i0 = torch.zeros(NIN)
    for nm in NUC:
        src = "外節" if nm in ("外節", "内節", "網様部") else "視床下核"
        e_, g_, z_ = FIT[TAU][src]
        sl = slice(off[nm], off[nm] + N)
        ge[sl], gi[sl], i0[sl] = e_, g_, z_
    print(f"試し台その2：閉じた輪 τm {TAU:.0f}ms・{SEC:.0f}秒・各核 {N}体", flush=True)
    # i0 を閉じた輪で合わせる
    for it in range(10):
        hz, cv, _ = run(WE, WI, ge, gi, i0, int(1500 / DT))
        msg = []
        for nm in NUC:
            sl = slice(off[nm], off[nm] + N)
            r = float(hz[sl].mean())
            i0[sl] *= float(np.clip((R[nm] / max(r, 0.5)) ** 0.5, 0.5, 2.0))
            msg.append(f"{nm} {r:5.1f}")
        print(f"  合わせ {it + 1}：" + "・".join(msg) + "Hz", flush=True)
    nstep = int(SEC * 1000 / DT)
    hz, cv, pop = run(WE, WI, ge, gi, i0, nstep, rec=True)
    print("\n核        率(的)          CV(的)      そろい（1 ＝ ばらばら）  i0", flush=True)
    for nm in NUC:
        sl = slice(off[nm], off[nm] + N)
        bins = pop[:, sl].sum(1)
        nb = int(5 / DT)
        b = bins[: (len(bins) // nb) * nb].view(-1, nb).sum(1)
        fano = float(b.var() / b.mean().clamp(min=1e-9))
        ct = CVT[nm]
        print(f"  {nm:<6} {float(hz[sl].mean()):5.1f}Hz({R[nm]})  {float(cv[sl].mean()):.2f}({ct if ct else '-'})"
              f"   {fano:6.2f}                {float(i0[sl].mean()):.4f}", flush=True)
    # 8体だけ止める（局所に薬を入れた形）
    print("\n8体だけ止めたときの比（的：外節 0.43・2.16・1.10 ／ 視床下核 0.84・1.26・≒1.26）", flush=True)
    for nm in ("外節", "視床下核"):
        idx = torch.arange(off[nm], off[nm] + 8)
        base = float(hz[idx].mean())
        out = []
        for kind in ("glu", "gaba", "both"):
            h, _, _ = run(WE, WI, ge, gi, i0, int(min(SEC, 5) * 1000 / DT), block=(idx, kind))
            out.append(f"{float(h[idx].mean()) / max(base, 1e-9):.2f}")
        print(f"  {nm}：グルタミン酸 {out[0]}・GABA {out[1]}・両方 {out[2]}", flush=True)


if __name__ == "__main__":
    main()
