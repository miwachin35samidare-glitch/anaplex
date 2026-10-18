"""版93 の試し台 その8：網と同じ入り方で、版93 の中脳の核（視床下核の辺縁・境・外側手綱核・吻内側被蓋核・黒質）を合わせ直す
  python3 bench_v93h.py
  （網は使わない。何も書き換えない。CPU）
網と同じにしたところ（試し台その6・7 との違い）
  ・領野と脳幹の送り手（前頭前野の表現・来なかった・側坐核の表現・誤差・PPN）は1歩（10周）に1回、周の頭で届く【決め 10-16】
    ⚠️ 送り手の撃ち方そのものは試し台の代わり（決まった率のポアソン）。網に入れたら網の発火になる
  ・核どうしは1周遅れ・受け手1体は送り手の別々の 64体から（送り手が 64 より少なければ全員）【決め 10-17】／ 前頭前野 → 黒質は 8本【み 10-14】
決めた形（10-18）
  視床下核の辺縁：i0・← 前頭前野の表現（3.30Hz【ヒト】）＋「来なかった」（1本あたり同じ）【み 13:23】。強さはグルタミン酸を止めて −16%【マカク】Polyakova 2020・率 19.3Hz【ヒト】
     NMDA は淡蒼球の尾（NR2A 21ms・NR2D 1,340ms 半々、割合 f は外節の合わせ）⚠️ 試し台その3 と同じ仮
  境（内節の一部）：i0・← 視床下核の辺縁（線を出すのは 69.3%【マカク】Sato 2000b）興奮 ／ ← 側坐核（表現＋誤差）抑制【み 13:36】
     強さ：外節で合わせた g_e・g_i・f を使う（試し台その2・3 の内節と同じ仮）・率 33Hz【マカク】
  外側手綱核：i0 ≥ 0・← 境・NMDA 尾2本（22ms 65% ・137ms 35%）【み 13:17】・率 30Hz【マカク】・境 +40% ⇒ +60% に近い g_L
  吻内側被蓋核：i0（入力なしで 6.7Hz【ラット】）・← 外側手綱核・NMDA 75.5ms（黒質の形 ⚠️）・率 17.8Hz【マカク】
  黒質：v_r 0・← PPN（23.2Hz【ヒト】）・視床下核の辺縁・前頭前野 8本 興奮 ／ ← 吻内側被蓋核 抑制・NMDA 75.5ms
     g_e：率 4.56Hz・CV 0.61・半分戻るまで 106ms に一番近いもの（試し台その7 と同じ選び方。CV は見張る項目）
     g_S：外側手綱核をそろって1発で 10〜40ms 約0.4倍【マカク】Matsumoto 2007
版2【み】10-18 14:56「合わせ直すなら今合わせる必要無くない？」⇒ 境の視床下核からの強さ ge_gph は仮に「視床下核を止めて 0.95 倍」【マカク】2022 Nat Commun（内節 変わらず）で。版95 で外節と一緒に合わせ直す
  0・1 の結果は1回目のものを使う（ENV FAST=1）
確かめ：1発の形 ／「来なかった」を 150ms 撃たせて境 +40% にしたときの外側手綱核・ドーパミン（サル 約5 → 約2Hz）
"""
import math, sys, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_v93c as C

DT = 1.067; NP = 10; TAU = 40.0; A = math.exp(-DT / TAU)
N = 256; K = 64
EXT = {"表現": (946, 3.30), "来なかった": (64, 0.0), "側坐核の表現": (146, 2.1), "側坐核の誤差": (146, 0.0), "PPN": (512, 23.2)}
# ⚠️ 側坐核の表現の率：線条体の棘細胞 2.1Hz【ヒト】Valsky 2020・Singh 2016 を代わりに（網の側坐核の率は網で決まる）
TAILS = {"淡蒼球": [(21.0, 0.5), (1340.0, 0.5)], "手綱": [(22.0, 0.65), (137.0, 0.35)], "黒質": [(75.5, 1.0)]}
torch.manual_seed(0)


class Pop:
    def __init__(self, B, n, i0, tails, f):
        self.v = torch.rand(B, n); self.i0 = i0; self.f = f
        self.tl = [(math.exp(-DT / t), m) for t, m in tails]
        self.S = [torch.zeros(B, n) for _ in tails]

    def step(self, e, i, ge, gi, cut_e=False):
        en = 0.0
        for k, (a, m) in enumerate(self.tl):
            self.S[k] = self.S[k] * a + e * (1 - a); en = en + m * self.S[k]
        x = 0.0 if cut_e else (1 - self.f) * e + self.f * en
        self.v = (self.v * A + self.i0 + ge * x - gi * i).clamp(min=-5.0)
        s = self.v >= 1.0
        self.v = torch.where(s, torch.zeros_like(self.v), self.v)
        return s.float()


def pick(n_src, n_tgt, k, gen, frac=1.0):
    """線を出す送り手（割合 frac）の中から、受け手ごとに別々の k 体"""
    pool = torch.randperm(n_src, generator=gen)[:max(1, int(round(n_src * frac)))]
    k = min(k, len(pool))
    return torch.stack([pool[torch.randperm(len(pool), generator=gen)[:k]] for _ in range(n_tgt)])


class Mid:
    def __init__(self, P, B, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.P, self.B = P, B
        self.ix = {
            "表現→視": pick(946 + 64, N, K, g),                     # 前頭前野（表現 946 ＋ 来なかった 64）を1つの送り手として
            "視→境": pick(N, N, K, g, 0.693),
            "側→境": pick(146 + 146, N, K, g),
            "境→手": pick(N, N, K, g),
            "手→吻": pick(N, N, K, g),
            "PPN→黒": pick(512, N, K, g),
            "視→黒": pick(N, N, K, g),
            "表現→黒": pick(946, N, 8, g),
            "吻→黒": pick(N, N, K, g),
        }
        fg = P["f"]
        self.stn = Pop(B, N, P["i0_stn"], TAILS["淡蒼球"], fg)
        self.gph = Pop(B, N, P["i0_gph"], TAILS["淡蒼球"], fg)
        self.lhb = Pop(B, N, P["i0_lhb"], TAILS["手綱"], 0.5)
        self.rmtg = Pop(B, N, P["i0_rmtg"], TAILS["黒質"], 0.5)
        self.sn = Pop(B, N, P["i0_sn"], TAILS["黒質"], 0.5)
        z = torch.zeros(B, N)
        self.f = {"視": z, "境": z, "手": z, "吻": z, "黒": z}

    def run(self, nstep, omit_hz=None, volley=None, cut=None, keep=False, jit=None):
        """nstep 歩（各 10周）。omit_hz：{歩: 来なかったの率}。volley：外側手綱核をそろって撃たせる歩（周の頭）。返り値：核ごとの率 [B, 歩]"""
        P, B = self.P, self.B
        out = {k: [] for k in ("視", "境", "手", "吻", "黒")}
        sp = []
        jit = jit if jit is not None else torch.zeros(B, dtype=torch.long)
        for t in range(nstep):
            rel = t - jit
            ext = {}
            for nm, (n, hz) in EXT.items():
                r = torch.full((B, n), hz * NP * DT / 1000)
                if nm == "来なかった" and omit_hz:
                    for (a, b), h in omit_hz.items():
                        on = ((rel >= a) & (rel < b)).float()[:, None]
                        r = r + on * h * NP * DT / 1000
                ext[nm] = (torch.rand(B, n) < r).float()
            pfc = torch.cat([ext["表現"], ext["来なかった"]], 1)
            nac = torch.cat([ext["側坐核の表現"], ext["側坐核の誤差"]], 1)
            acc = {k: torch.zeros(B, N) for k in out}
            for p in range(NP):
                f = self.f
                head = 1.0 if p == 0 else 0.0
                e_stn = head * pfc[:, self.ix["表現→視"]].mean(2)
                e_gph = f["視"][:, self.ix["視→境"]].mean(2); i_gph = head * nac[:, self.ix["側→境"]].mean(2)
                e_lhb = f["境"][:, self.ix["境→手"]].mean(2)
                e_rm = f["手"][:, self.ix["手→吻"]].mean(2)
                nE = K + K + 8
                e_sn = (head * ext["PPN"][:, self.ix["PPN→黒"]].sum(2) + f["視"][:, self.ix["視→黒"]].sum(2)
                        + head * ext["表現"][:, self.ix["表現→黒"]].sum(2)) / nE
                i_sn = f["吻"][:, self.ix["吻→黒"]].mean(2)
                z = torch.zeros(B, N)
                s_stn = self.stn.step(e_stn, z, P["ge_stn"], 0.0, cut_e=(cut == "視グル"))
                s_gph = self.gph.step(e_gph, i_gph, P["ge_gph"], P["gi_gp"], cut_e=(cut == "境視"))
                s_lhb = self.lhb.step(e_lhb, z, P["g_L"], 0.0)
                if volley is not None and p == 0:
                    vm = (rel == volley)[:, None]
                    s_lhb = torch.where(vm, torch.ones_like(s_lhb), s_lhb)
                    self.lhb.v = torch.where(vm, torch.zeros_like(self.lhb.v), self.lhb.v)
                s_rm = self.rmtg.step(e_rm, z, P["g_R"], 0.0)
                s_sn = self.sn.step(e_sn, i_sn, P["ge_sn"], P["g_S"])
                self.f = {"視": s_stn, "境": s_gph, "手": s_lhb, "吻": s_rm, "黒": s_sn}
                for k in out:
                    acc[k] += self.f[k]
                if keep:
                    sp.append(s_sn)
            for k in out:
                out[k].append(acc[k].mean(1))
        hz = {k: torch.stack(v, 1) / (NP * DT / 1000) for k, v in out.items()}
        if keep:
            hz["発火"] = torch.stack(sp, -1)
        return hz


def rate(P, key, nstep=300, B=4, cut=None, warm=200):
    m = Mid(P, B); m.run(warm)
    return float(m.run(nstep, cut=cut)[key].mean())


def bisect(P, name, key, tgt, lo=0.0, hi=0.3, it=12, **kw):
    for _ in range(it):
        P[name] = (lo + hi) / 2
        r = rate(P, key, **kw)
        lo, hi = (P[name], hi) if r < tgt else (lo, P[name])
    P[name] = (lo + hi) / 2
    return r


def cv_half(sp):
    x = sp.reshape(-1, sp.shape[-1]).float()
    cvs = []
    for row in x:
        t = torch.nonzero(row).flatten().float()
        if len(t) > 3:
            d = t[1:] - t[:-1]; cvs.append(float(d.std() / d.mean()))
    p = x.mean(); half = float("nan")
    for k in range(1, 400):
        c = (x[:, :-k] * x[:, k:]).mean() / x[:, :-k].mean().clamp(min=1e-12) / p
        if c >= 0.5:
            half = k * DT; break
    return (float(np.mean(cvs)) if cvs else float("nan")), half


def main():
    import os
    print("0 外節の g_e・g_i・NMDA の割合 f（試し台その3 の開いた輪、尾は半々）", flush=True)
    if os.environ.get("FAST"):
        ge_gp, gi_gp, f = 4.056, 12.055, 0.4                       # 1回目の結果
    else:
        _, ge_gp, gi_gp, _, f = C.fit("外節", C.MIX["半々"], int(3000 / DT))
    print(f"  g_e {ge_gp:.3f}・g_i {gi_gp:.3f}・f {f}", flush=True)
    P = dict(f=f, ge_gp=ge_gp, gi_gp=gi_gp, ge_gph=ge_gp, ge_stn=1.0, i0_stn=0.02, i0_gph=0.05, i0_lhb=0.0, g_L=1.0,
             i0_rmtg=0.027, g_R=0.24, i0_sn=0.007, ge_sn=0.9, g_S=0.15)

    print("\n1 視床下核の辺縁：グルタミン酸を止めて −16%・率 19.3Hz", flush=True)
    best = None
    if os.environ.get("FAST"):
        P["ge_stn"], P["i0_stn"] = 0.857, 0.0337
        print("  1回目の結果：g_e 0.857・i0 0.0337", flush=True)
    for ge in (() if os.environ.get("FAST") else (0.1, 0.3, 1.0, 3.0, 10.0)):
        P["ge_stn"] = ge
        r = bisect(P, "i0_stn", "視", 19.3)
        rc = rate(P, "視", cut="視グル")
        print(f"  g_e {ge:5.1f}・i0 {P['i0_stn']:.4f} ⇒ {r:.1f}Hz ／ 止める {rc / r:.2f}（的 0.84）", flush=True)
        if best is None or abs(rc / r - 0.84) < abs(best[0] - 0.84):
            best = (rc / r, ge, P["i0_stn"])
    if best:
        _, P["ge_stn"], P["i0_stn"] = best
        lo, hi = best[1] / 3, best[1] * 3
    for _ in (() if os.environ.get("FAST") else range(7)):
        P["ge_stn"] = math.sqrt(lo * hi); r = bisect(P, "i0_stn", "視", 19.3); rc = rate(P, "視", cut="視グル") / r
        lo, hi = (P["ge_stn"], hi) if rc > 0.84 else (lo, P["ge_stn"])
    if not os.environ.get("FAST"):
        print(f"  ⇒ g_e {P['ge_stn']:.3f}・i0 {P['i0_stn']:.4f}（止める {rc:.2f}）", flush=True)

    print("\n2 境：視床下核を止めて 0.95 倍（仮）・率 33Hz ／ 抑制は外節の g_i", flush=True)
    lo, hi = 0.01, ge_gp
    for _ in range(9):
        P["ge_gph"] = math.sqrt(lo * hi)
        r = bisect(P, "i0_gph", "境", 33.0)
        rc = rate(P, "境", cut="境視") / r
        print(f"  g_e {P['ge_gph']:.3f}・i0 {P['i0_gph']:.4f} ⇒ {r:.1f}Hz ／ 止める {rc:.2f}", flush=True)
        lo, hi = (P["ge_gph"], hi) if rc > 0.95 else (lo, P["ge_gph"])

    print("\n3 外側手綱核：率 30Hz（i0 ≥ 0）・境 +40% ⇒ +60% に近い g_L", flush=True)
    # 境 +40% を作る「来なかった」の率を先に探す（g_L に関係しない）
    lo, hi = 0.0, 200.0
    for _ in range(10):
        h = (lo + hi) / 2
        m = Mid(P, 8); m.run(200); o = m.run(60, omit_hz={(30, 44): h})
        up = float(o["境"][:, 30:44].mean() / o["境"][:, :30].mean()) - 1
        lo, hi = (h, hi) if up < 0.4 else (lo, h)
    h_ev = (lo + hi) / 2
    print(f"  「来なかった」{h_ev:.1f}Hz を 150ms ⇒ 境 {up * 100:+.0f}%（視床下核の辺縁 {float(o['視'][:, 30:44].mean()):.1f}Hz）", flush=True)
    best = None
    for gL in (0.5, 0.8, 1.0, 1.4, 2.0, 3.0):
        P["g_L"] = gL
        r = bisect(P, "i0_lhb", "手", 30.0)
        m = Mid(P, 8); m.run(200); o = m.run(60, omit_hz={(30, 44): h_ev})
        base = float(o["手"][:, :30].mean()); g = float(o["手"][:, 30:44].mean()) / base - 1
        print(f"  g_L {gL:.1f}・i0 {P['i0_lhb']:.4f} ⇒ {base:.1f}Hz ／ +{g * 100:.0f}%", flush=True)
        if abs(base - 30) < 3 and (best is None or abs(g - 0.6) < abs(best[0] - 0.6)):
            best = (g, gL, P["i0_lhb"])
    _, P["g_L"], P["i0_lhb"] = best
    print(f"  ⇒ g_L {P['g_L']}・i0 {P['i0_lhb']:.4f}", flush=True)

    print("\n4 吻内側被蓋核：入力なしで 6.7Hz・ふだん 17.8Hz", flush=True)
    g0 = P["g_R"]; P["g_R"] = 0.0
    bisect(P, "i0_rmtg", "吻", 6.7)
    P["g_R"] = g0
    lo, hi = 0.01, 10.0
    for _ in range(12):
        P["g_R"] = math.sqrt(lo * hi); r = rate(P, "吻")
        lo, hi = (P["g_R"], hi) if r < 17.8 else (lo, P["g_R"])
    print(f"  i0 {P['i0_rmtg']:.5f}・g_R {P['g_R']:.3f} ⇒ {r:.1f}Hz", flush=True)

    print("\n5 黒質：g_e を率 4.56Hz・CV 0.61・半分戻るまで 106ms で選ぶ（g_S 0.15 で仮に）", flush=True)
    best = None
    for ge in (0.3, 0.6, 0.9, 1.5, 2.5, 4.0):
        P["ge_sn"] = ge
        r = bisect(P, "i0_sn", "黒", 4.56, hi=0.1)
        m = Mid(P, 2); m.run(200); o = m.run(int(10000 / (NP * DT)), keep=True)
        cv, hf = cv_half(o["発火"])
        err = abs(cv - 0.61) / 0.61 + abs(hf - 106) / 106
        print(f"  g_e {ge:.1f}・i0 {P['i0_sn']:.5f} ⇒ {float(o['黒'].mean()):.2f}Hz・CV {cv:.2f}・半分 {hf:.0f}ms", flush=True)
        if best is None or err < best[0]:
            best = (err, ge, P["i0_sn"])
    _, P["ge_sn"], P["i0_sn"] = best

    print("\n6 黒質 ← 吻内側被蓋核 g_S：1発で 10〜40ms 約0.4倍", flush=True)
    def volley(B=64):
        m = Mid(P, B); m.run(200)
        jit = torch.randint(0, 30, (B,))
        o = m.run(110, volley=40, jit=jit)                         # ⟲ 80 だと jit の分だけ足りなかった
        s = torch.stack([o["黒"][b, jit[b]:jit[b] + 60] for b in range(B)])   # 歩の刻み（10.67ms）
        return s
    lo, hi = 0.01, 2.0
    for _ in range(9):
        P["g_S"] = math.sqrt(lo * hi)
        bisect(P, "i0_sn", "黒", 4.56, hi=0.1)
        s = volley().mean(0); base = float(s[:40].mean())
        r = float(s[41:44].mean()) / base                            # 1発の歩の次の歩から3歩 ＝ 約10〜40ms
        print(f"  g_S {P['g_S']:.3f}・i0 {P['i0_sn']:.5f} ⇒ 10〜40ms {r:.2f} 倍", flush=True)
        lo, hi = (P["g_S"], hi) if r > 0.4 else (lo, P["g_S"])
    P["g_S"] = math.sqrt(lo * hi); bisect(P, "i0_sn", "黒", 4.56, hi=0.1)

    print("\n7 確かめ", flush=True)
    print("  使う値：" + "・".join(f"{k} {v:.4g}" for k, v in P.items()), flush=True)
    m = Mid(P, 4); m.run(200); o = m.run(int(10000 / (NP * DT)), keep=True)
    cv, hf = cv_half(o["発火"])
    print("  ふだん：" + "・".join(f"{k} {float(o[k].mean()):.1f}Hz" for k in ("視", "境", "手", "吻", "黒")) + f" ／ 黒質 CV {cv:.2f}・半分 {hf:.0f}ms", flush=True)
    s = volley(128).mean(0); base = float(s[:40].mean())
    print("  1発（歩 10.67ms ごと）：" + " ".join(f"{float(x) / base:.2f}" for x in s[40:52]), flush=True)
    m = Mid(P, 32); m.run(200); o = m.run(60, omit_hz={(30, 44): h_ev})
    print("  「来なかった」150ms：" + "・".join(f"{k} {float(o[k][:, :30].mean()):.1f} → {float(o[k][:, 31:45].mean()):.1f}Hz" for k in ("視", "境", "手", "吻", "黒"))
          + "（サル：外側手綱核 26 → 42・ドーパミン 約5 → 約2）", flush=True)


if __name__ == "__main__":
    main()
