"""版93 の試し台 その3：NMDA の尾を足して CV を測る（尾は CV に合わせない）
  python3 bench_v93c.py [秒=6]
  （網は使わない。何も書き換えない。CPU で回る）
【み】10-17 12:20「まぁAでいいよ」
  人の淡蒼球の NMDA：NR2A と NR2D（【ヒト】Küppenbender 1998、割合は不明）
  尾の長さ：組み換えの受容体（室温）を体温へ（÷3.3 ＝ Q10 2.5 で 13℃ 分、Korinek 2010 の幅 1.7〜3.7 の中）
    NR2A 70.8ms → 21ms（Wyllie 1998 の速い方。遅い方 350ms は ⚠️ 入れていない）／ NR2D 4,408ms → 1,340ms
  割合：NR2A だけ ／ NR2D だけ ／ 半々 の3通り（CV で選ばない）
  NMDA 全体の強さ：外節で NBQX −56.7%・CPP −32.4%・gabazine +115.8%（【マカク】Kita 2004）に合わせる
    視床下核は NBQX＋CPP −16%・gabazine +26%（【マカク】Polyakova 2020）。NMDA の割合は外節と同じ（⚠️ 仮）
手順：1 開いた輪（その1 の形）で、外節・視床下核の g_e・g_i・i0・NMDA の割合 f を止める比に合わせる
      2 閉じた輪（その2 の形、各核 256体）で i0 を率に合わせ直し、CV・隣どうしの相関・8体だけ止めた比を測る
⚠️ 内節・網様部は止める実験の数が無いので外節と同じ g・f（その2 と同じ仮）
"""
import sys, math, os, numpy as np, torch

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
DT = 1.067
TAU = 40.0
Q = 3.3
TN = {"NR2A": 70.8 / Q, "NR2D": 4408.0 / Q}
MIX = {"NR2A だけ": {"NR2A": 1.0}, "NR2D だけ": {"NR2D": 1.0}, "半々": {"NR2A": 0.5, "NR2D": 0.5}}
KGS = float(os.environ.get("KGS", "0"))   # 版2：外節 → 視床下核の割合（0 ＝ 64体のときの数 54）
if os.environ.get("ONLYHALF"):
    MIX = {"半々": MIX["半々"]}
torch.manual_seed(0); np.random.seed(0)

# ---------- 1 開いた輪 ----------
OPEN = {
    "外節": dict(R=69.7, lines=[("線条体", 100, 2.1, -1, 0.80), ("外節", 64, 69.7, -1, 0.05), ("視床下核", 53, 19.3, +1, 1.0)],
               tgt={"ampa": 1 - 0.567, "nmda": 1 - 0.324, "gaba": 1 + 1.158}),
    "視床下核": dict(R=19.3, lines=[("外節", 54, 69.7, -1, 1.0), ("前頭前野", 946, 3.30, +1, 1.0)],
                 tgt={"glu": 1 - 0.16, "gaba": 1 + 0.26}),
}


def drive(lines, shape, sign):
    tot = torch.zeros(shape)
    sh = sum(s for _, _, _, sg, s in lines if sg == sign)
    for _, n, hz, sg, s in lines:
        if sg == sign:
            k = torch.distributions.Binomial(n, torch.full(shape, hz * DT / 1000)).sample()
            tot += (s / sh) / n * k
    return tot


def run_open(lines, ge, gi, i0, f, mix, nstep, cut=None):
    """cut：'ampa'・'nmda'・'glu'・'gaba' のどれかを止める"""
    a = math.exp(-DT / TAU); shape = ge.shape
    aN = {k: math.exp(-DT / TN[k]) for k in mix}
    S = {k: torch.zeros(shape) for k in mix}
    v = torch.rand(shape); cnt = torch.zeros(shape)
    wa = 0.0 if cut in ("ampa", "glu") else 1.0
    wn = 0.0 if cut in ("nmda", "glu") else 1.0
    wi = 0.0 if cut == "gaba" else 1.0
    for _ in range(nstep):
        e = drive(lines, shape, +1); i = drive(lines, shape, -1)
        en = 0.0
        for k, m in mix.items():
            S[k] = S[k] * aN[k] + e * (1 - aN[k])
            en = en + m * S[k]
        x = (1 - f) * e * wa + f * en * wn
        v = (v * a + i0 + ge * x - gi * i * wi).clamp(min=-5.0)
        fr = v >= 1.0; cnt += fr.float(); v = torch.where(fr, torch.zeros_like(v), v)
    return cnt / (nstep * DT / 1000)


def fit(nm, mix, nstep):
    d = OPEN[nm]
    GE = np.geomspace(0.05, 50, 12); GI = np.geomspace(0.02, 50, 12); I0 = np.geomspace(1e-4, 0.6, 24)
    FS = [0.2, 0.4, 0.6, 0.8] if nm == "外節" else [FIX_F[0]]
    best = None
    for f in FS:
        ge, gi = np.meshgrid(GE, GI, indexing="ij"); ge, gi = ge.ravel(), gi.ravel(); G = len(ge)
        g_e = torch.tensor(ge).view(G, 1, 1).expand(G, len(I0), 4).float()
        g_i = torch.tensor(gi).view(G, 1, 1).expand(G, len(I0), 4).float()
        z0 = torch.tensor(I0).view(1, -1, 1).expand(G, len(I0), 4).float()
        curve = run_open(d["lines"], g_e, g_i, z0, f, mix, nstep).mean(2).numpy()
        i0 = np.full(G, np.nan)
        for k in range(G):
            c = np.maximum.accumulate(curve[k])
            if c[0] <= d["R"] <= c[-1]:
                i0[k] = np.exp(np.interp(d["R"], c, np.log(I0)))
        ok = ~np.isnan(i0)
        if not ok.any():
            continue
        ge, gi, i0 = ge[ok], gi[ok], i0[ok]; G = len(ge)
        T = lambda x: torch.tensor(x).view(G, 1).expand(G, 8).float()
        base = run_open(d["lines"], T(ge), T(gi), T(i0), f, mix, nstep).mean(1).numpy().clip(1e-6)
        err = np.log(base / d["R"]) ** 2
        for cut, tv in d["tgt"].items():
            r = run_open(d["lines"], T(ge), T(gi), T(i0), f, mix, nstep, cut=cut).mean(1).numpy() / base
            err = err + np.log(np.maximum(r, 1e-3) / tv) ** 2
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), ge[k], gi[k], i0[k], f)
    return best


# ---------- 2 閉じた輪（その2 の形） ----------
N = 256
NUC = ["外節", "視床下核", "内節", "網様部"]
R = {"外節": 69.7, "視床下核": 19.3, "内節": 82.5, "網様部": 60.5}
CVT = {"外節": 1.15, "視床下核": 1.29, "内節": 0.89, "網様部": "-"}
N_STR, N_CX = 2288, 946
off = {nm: i * N for i, nm in enumerate(NUC)}
NIN = len(NUC) * N; S_STR, S_CX = NIN, NIN + N_STR; NSRC = NIN + N_STR + N_CX
P = {("視床下核", "外節"): 0.827, ("視床下核", "内節"): 0.693, ("視床下核", "網様部"): 0.240,
     ("外節", "視床下核"): 0.842, ("外節", "内節"): 0.316, ("外節", "網様部"): 0.658, ("外節", "外節"): 1.0}


def build():
    gen = torch.Generator().manual_seed(1)
    rows = {"E1": [], "I1": [], "I2": []}
    senders = {k: torch.randperm(N, generator=gen)[:int(round(N * v))] for k, v in P.items()}
    for rc in NUC:
        inh = ([("線条体", 0.80)] if rc != "視床下核" else []) + ([("外節", 0.05 if rc != "視床下核" else 1.0)] if ("外節", rc) in P else [])
        sh = sum(s for _, s in inh)
        for r in range(off[rc], off[rc] + N):
            for src, s in inh:
                if src == "線条体":
                    idx = S_STR + torch.randperm(N_STR, generator=gen)[:100]; key = "I2"
                else:
                    idx = off[src] + senders[(src, rc)]; idx = idx[idx != r]
                    k = int(round(64 * P[(src, rc)]))
                    if src == "外節" and rc == "視床下核" and KGS:   # 版2：外節 → 視床下核の割合を変える（A ラット 2% ／ C 振る）
                        k = max(1, int(round(N * KGS)))
                    idx = idx[torch.randperm(len(idx), generator=gen)[:k]]; key = "I1"
                rows[key] += [(r, int(j), (s / sh) / len(idx)) for j in idx]
            if rc == "視床下核":
                idx = torch.arange(S_CX, S_CX + N_CX)
            else:
                idx = off["視床下核"] + senders[("視床下核", rc)]
                idx = idx[torch.randperm(len(idx), generator=gen)[:int(round(64 * P[("視床下核", rc)]))]]
            rows["E1"] += [(r, int(j), 1.0 / len(idx)) for j in idx]
    M = {}
    for k, rr in rows.items():
        t = torch.tensor(rr, dtype=torch.float64)
        M[k] = torch.sparse_coo_tensor(t[:, :2].long().T, t[:, 2].float(), (NIN, NSRC)).coalesce().to_sparse_csr()
    return M


def run_closed(M, ge, gi, i0, f, mix, nstep, block=None, rec=False):
    a = math.exp(-DT / TAU)
    aN = {k: math.exp(-DT / TN[k]) for k in mix}
    S = {k: torch.zeros(NIN) for k in mix}
    r1 = torch.zeros(NSRC); r2 = torch.zeros(NSRC)
    v = torch.rand(NIN); cnt = torch.zeros(NIN); last = torch.full((NIN,), -1.0)
    s1 = torch.zeros(NIN); s2 = torch.zeros(NIN); ni = torch.zeros(NIN); pop = []
    wa = torch.ones(NIN); wn = torch.ones(NIN); wi = torch.ones(NIN)
    if block is not None:
        idx, kind = block
        if kind in ("ampa", "glu"):
            wa[idx] = 0
        if kind in ("nmda", "glu"):
            wn[idx] = 0
        if kind == "gaba":
            wi[idx] = 0
    for t in range(nstep):
        e = torch.mv(M["E1"], r1); inh = torch.mv(M["I1"], r1) + torch.mv(M["I2"], r2)
        en = 0.0
        for k, m in mix.items():
            S[k] = S[k] * aN[k] + e * (1 - aN[k]); en = en + m * S[k]
        x = (1 - f) * e * wa + f * en * wn
        v = (v * a + i0 + ge * x - gi * inh * wi).clamp(min=-5.0)
        fr = v >= 1.0; v = torch.where(fr, torch.zeros_like(v), v)
        r2 = r1.clone()
        r1 = torch.zeros(NSRC); r1[:NIN] = fr.float()
        r1[S_STR:S_CX] = (torch.rand(N_STR) < 2.1 * DT / 1000).float()
        r1[S_CX:] = (torch.rand(N_CX) < 3.30 * DT / 1000).float()
        ff = fr & (last >= 0)
        if ff.any():
            dd = (t - last)[ff]; s1[ff] += dd; s2[ff] += dd ** 2; ni[ff] += 1
        last[fr] = t; cnt += fr.float()
        if rec:
            pop.append(fr.float())
    hz = cnt / (nstep * DT / 1000)
    m = s1 / ni.clamp(min=1); cv = (s2 / ni.clamp(min=1) - m ** 2).clamp(min=0).sqrt() / m.clamp(min=1e-9)
    return hz, cv, (torch.stack(pop) if rec else None)


FIX_F = [None]


def main():
    nfit = int(1500 / DT)
    M = build()
    print(f"試し台その3：NMDA の尾（τm {TAU:.0f}ms、各核 {N}体、{SEC:.0f}秒）", flush=True)
    for name, mix in MIX.items():
        bg = fit("外節", mix, nfit)
        if bg is None:
            print(f"\n== {name}：外節が率に届かない", flush=True); continue
        FIX_F[0] = bg[4]
        bs = fit("視床下核", mix, nfit)
        f = bg[4]
        print(f"\n== {name}（尾 {', '.join(f'{k} {TN[k]:.0f}ms' for k in mix)}）NMDA の割合 f {f:.1f}", flush=True)
        print(f"  開いた輪で合わせた：外節 g_e {bg[1]:.2f}・g_i {bg[2]:.2f}・i0 {bg[3]:.4f}（ずれ {bg[0]:.3f}）"
              + (f" ／ 視床下核 g_e {bs[1]:.2f}・g_i {bs[2]:.2f}・i0 {bs[3]:.4f}（ずれ {bs[0]:.3f}）" if bs else " ／ 視床下核 届かない"), flush=True)
        if bs is None:
            continue
        ge = torch.zeros(NIN); gi = torch.zeros(NIN); i0 = torch.zeros(NIN)
        for nm in NUC:
            src = bs if nm == "視床下核" else bg
            sl = slice(off[nm], off[nm] + N); ge[sl], gi[sl], i0[sl] = src[1], src[2], src[3]
        for _ in range(8):
            hz, _, _ = run_closed(M, ge, gi, i0, f, mix, int(1200 / DT))
            for nm in NUC:
                sl = slice(off[nm], off[nm] + N)
                i0[sl] *= float(np.clip((R[nm] / max(float(hz[sl].mean()), 0.5)) ** 0.5, 0.5, 2.0))
        nstep = int(SEC * 1000 / DT)
        hz, cv, pop = run_closed(M, ge, gi, i0, f, mix, nstep, rec=True)
        nb = int(round(10 / DT))
        for nm in NUC:
            sl = slice(off[nm], off[nm] + N)
            x = pop[:, sl]; x = x[: (len(x) // nb) * nb].view(-1, nb, N).sum(1)
            x = (x - x.mean(0)) / x.std(0).clamp(min=1e-9)
            g = torch.Generator().manual_seed(3)
            i1 = torch.randint(0, N, (300,), generator=g); i2 = (i1 + 1 + torch.randint(0, N - 1, (300,), generator=g)) % N
            print(f"  {nm:<5} 率 {float(hz[sl].mean()):5.1f}Hz（{R[nm]}）・CV {float(cv[sl].mean()):.2f}（{CVT[nm]}）・相関 {float((x[:, i1] * x[:, i2]).mean()):.3f}", flush=True)
        idx = torch.arange(off["外節"], off["外節"] + 8); base = float(hz[idx].mean())
        out = []
        for kind in ("ampa", "nmda", "gaba"):
            h, _, _ = run_closed(M, ge, gi, i0, f, mix, int(3000 / DT), block=(idx, kind))
            out.append(f"{float(h[idx].mean()) / max(base, 1e-9):.2f}")
        print(f"  外節 8体だけ止める：NBQX {out[0]}（0.43）・CPP {out[1]}（0.68）・gabazine {out[2]}（2.16）", flush=True)


if __name__ == "__main__":
    main()
