"""版95 の先生（凍結する核）を一つにまとめた試し台 —— 今の値のまま繋いで、どこが的から外れるかを測る
  python3 teacher_v95.py [秒=5]        （網は使わない。何も書き換えない。CPU）
【み】10-20 22:30「いいよ」（引き継ぎ §22〜24：先生は 率・止める比・1発の応え を調べた数値どおりに。CV は見張る項目【み選】「ア」）
まとめたもの（各 256体【決め 10-18】・撃ち方 B・周 1.067ms・τm 40ms）
  版93 の5核（midbrain_v93.py の LINES・NUC）：視床下核・境・外側手綱核・吻内側被蓋核・黒質
  試し台その3（bench_v93c.py、NMDA 半々）：外節・内節・網様部（g は bench_v93c の fit を同じ条件で回して出す）
  合わせた所（「同じなら一つに」【み】10-16）
    視床下核：版93 の辺縁（前頭前野から）＋ 試し台の視床下核（外節から）＝ 1つ。強さは試し台の fit（グルタミン酸・GABA の両方の止める比で合わせた方）
              前頭前野からは別々の 64体【決め 10-17】
    境：内節の一部（Parent 2001【マカク】、設計案 10-17 23:32）。群れは分けたまま（率 33Hz・行き先 外側手綱核）。入る線は版93 のまま
       ⚠️ 境に外節・線条体（背側）から入るかは版93 に無い ⇒ 足していない
  入力（学ぶ側・外から。人の率のポアソン、周ごと）
    線条体 2,288体 2.1Hz【ヒト】・側坐核 表現 146体 2.1Hz【ヒト】＋ 誤差 146体 0Hz・前頭前野 946体 3.30Hz【ヒト】・PPN 512体 23.2Hz【ヒト】・上丘の誤差 64体 0Hz（出来事の入力）
    ⚠️「跳ね・そろい」はまだ（7核が的に入ってから）
  届く時刻：核どうし・領野・PPN は次の周 ／ 線条体は2周（bench_v93c と同じ）
測るもの
  0 今の値のまま（i0 も今の値）の率
  1 i0 だけを率に合わせたあとの率・CV（CV は見張る項目、見るだけ）
  2 止める比（8体だけ）：外節 NBQX 0.43・CPP 0.68・gabazine 2.16【マカク】Kita 2004 ／ 視床下核 グルタミン酸 0.84・GABA 1.26【マカク】Polyakova 2020
  3 1発の応え：外側手綱核をそろって1発 ⇒ 黒質 10〜40ms 約0.4倍・50〜60ms で戻る【マカク】Matsumoto 2007
              境を +40%（150ms、i0 を足して起こす）⇒ 外側手綱核 +60〜70%【マカク】Hong & Hikosaka 2008・Matsumoto 2007
"""
import sys, os, math, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ONLYHALF", "1")
import bench_v93c as C
import midbrain_v93 as MB

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
DT = 1.067; TAU = 40.0; ALPHA = math.exp(-DT / TAU)
N = 256; K = 64
WARM = int(1500 / DT)          # 版2：慣らし 1.5秒
torch.manual_seed(0); np.random.seed(0)

NUCS = ["外節", "視床下核", "内節", "境", "網様部", "外側手綱核", "吻内側被蓋核", "黒質"]
HZ = {"外節": 69.7, "視床下核": 19.3, "内節": 82.5, "境": 33.0, "網様部": 60.5, "外側手綱核": 30.0, "吻内側被蓋核": 17.8, "黒質": 4.56}
CVT = {"外節": 1.15, "視床下核": 1.29, "内節": 0.89, "黒質": 0.61}
EXT = {"線条体": (2288, 2.1), "側坐核": (292, None), "前頭前野": (946, 3.30), "PPN": (512, 23.2), "上丘": (64, 0.0)}
SAT = {("視床下核", "外節"): 0.827, ("視床下核", "内節"): 0.693, ("視床下核", "網様部"): 0.240,
       ("外節", "視床下核"): 0.842, ("外節", "内節"): 0.316, ("外節", "網様部"): 0.658, ("外節", "外節"): 1.0}   # Sato 2000【マカク】


def lines(gp, st):
    """(送り手, 受け手, 符号, G, k, 線を出す割合, 遅れ周)。1本の重さ ＝ G ÷ k（k 本が全部同じ周に届くと G）"""
    ge, gi, _, f = gp[1], gp[2], gp[3], gp[4]
    L = []
    sh = 0.85                                                     # 線条体 0.80 ＋ 外節 0.05（Shink & Smith 1995）
    for t in ("外節", "内節", "網様部"):
        L.append(("線条体", t, -1, gi * 0.80 / sh, 100, 1.0, 2))
        L.append(("外節", t, -1, gi * 0.05 / sh, int(round(64 * SAT[("外節", t)])), SAT[("外節", t)], 1))
        L.append(("視床下核", t, +1, ge, int(round(64 * SAT[("視床下核", t)])), SAT[("視床下核", t)], 1))
    L.append(("外節", "視床下核", -1, st[2], int(round(64 * SAT[("外節", "視床下核")])), SAT[("外節", "視床下核")], 1))
    L.append(("前頭前野", "視床下核", +1, st[1], K, 1.0, 1))
    # 版93 の線（midbrain_v93.LINES）。1本の重さ g ÷ 割る数 を G ÷ k に写す
    div = {("境", +1): 64, ("境", -1): 64, ("外側手綱核", +1): 64, ("吻内側被蓋核", +1): 64, ("黒質", +1): 136, ("黒質", -1): 64}
    for s, t, sg, g, k, fr in MB.LINES:
        if t == "視床下核":
            continue                                              # 視床下核は上の試し台の形に合わせた
        L.append((s, t, sg, g * k / div[(t, sg)], k, fr, 1))
    return L


class Teacher:
    def __init__(self, gp, st, gen_seed=1):
        g = torch.Generator().manual_seed(gen_seed)
        self.off = {}; o = 0
        for k in NUCS:
            self.off[k] = o; o += N
        self.NC = o
        for s, (n, _) in EXT.items():
            self.off[s] = o; o += n
        self.NX = o
        size = lambda s: N if s in NUCS else EXT[s][0]
        T, S, W, SG, D = [], [], [], [], []
        for s, t, sg, G, k, fr, d in lines(gp, st):
            pool = torch.randperm(size(s), generator=g)[:max(1, int(round(size(s) * fr)))]
            k = min(k, len(pool))
            for r in range(N):
                idx = pool[torch.randperm(len(pool), generator=g)[:k]]
                if s == t:
                    idx = idx[idx != r]
                T.append(torch.full((len(idx),), self.off[t] + r)); S.append(idx + self.off[s])
                W.append(torch.full((len(idx),), G / k)); SG.append(torch.full((len(idx),), sg)); D.append(torch.full((len(idx),), d))
        self.t, self.s, self.w = torch.cat(T), torch.cat(S), torch.cat(W).float()
        sg, d = torch.cat(SG), torch.cat(D)
        self.sel = {(x, dd): torch.nonzero((sg == x) & (d == dd)).squeeze(1) for x in (1, -1) for dd in (1, 2)}
        # 細胞ごと：i0・NMDA（f・尾）
        f = gp[4]
        tl = {"外節": "淡蒼球", "視床下核": "淡蒼球", "内節": "淡蒼球", "網様部": "淡蒼球", "境": "淡蒼球",
              "外側手綱核": "手綱", "吻内側被蓋核": "黒質", "黒質": "黒質"}
        fN = {k: (f if k in ("外節", "視床下核", "内節", "網様部") else MB.NUC[k]["f"]) for k in NUCS}
        i0 = {k: (gp[3] if k in ("外節", "内節", "網様部") else (st[3] if k == "視床下核" else MB.NUC[k]["i0"])) for k in NUCS}
        self.i0 = torch.cat([torch.full((N,), float(i0[k])) for k in NUCS])
        self.f = torch.cat([torch.full((N,), float(fN[k])) for k in NUCS])
        cols = []
        for k in NUCS:
            (t1, m1), (t2, m2) = MB.TAILS[tl[k]]
            cols.append([math.exp(-DT / t1), m1, math.exp(-DT / t2), m2])
        c = torch.tensor(cols).repeat_interleave(N, 0)
        self.a1, self.m1, self.a2, self.m2 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        self.hz_ext = torch.zeros(self.NX - self.NC)
        for s, (n, hz) in EXT.items():
            a = self.off[s] - self.NC
            if s == "側坐核":
                self.hz_ext[a:a + 146] = 2.1                          # 表現 2.1Hz【ヒト】・誤差 0
            else:
                self.hz_ext[a:a + n] = hz

    def sl(self, k):
        return slice(self.off[k], self.off[k] + N)

    def run(self, nstep, cut=None, kick=None, rec=False, warm=WARM):
        """cut：(核, 体の数, 'ampa'|'nmda'|'glu'|'gaba')。kick：周 → {核: 足す i0 か 'fire'}（周は慣らしのあとから数える）
        warm：慣らしの周（数えない）。版2：冷えた状態から測っていた誤りを直した"""
        NC, NX = self.NC, self.NX
        kick = {t + warm: v_ for t, v_ in kick.items()} if kick else None
        nstep = nstep + warm
        r1 = torch.zeros(NX); r2 = torch.zeros(NX)
        v = torch.rand(NC); S1 = torch.zeros(NC); S2 = torch.zeros(NC)
        wa = torch.ones(NC); wn = torch.ones(NC); wi = torch.ones(NC)
        if cut is not None:
            k, m, kind = cut; a = self.off[k]
            if kind in ("ampa", "glu"): wa[a:a + m] = 0
            if kind in ("nmda", "glu"): wn[a:a + m] = 0
            if kind == "gaba": wi[a:a + m] = 0
        cnt = torch.zeros(NC); last = torch.full((NC,), -1.0); s1 = torch.zeros(NC); s2 = torch.zeros(NC); ni = torch.zeros(NC)
        pop = []
        pe = self.hz_ext * DT / 1000
        for t in range(nstep):
            E = torch.zeros(NC); I = torch.zeros(NC)
            for (sg, d), ix in self.sel.items():
                if len(ix) == 0: continue
                x = (r1 if d == 1 else r2)[self.s[ix]] * self.w[ix]
                (E if sg > 0 else I).index_add_(0, self.t[ix], x)
            S1 = S1 * self.a1 + E * (1 - self.a1); S2 = S2 * self.a2 + E * (1 - self.a2)
            x = (1 - self.f) * E * wa + self.f * (self.m1 * S1 + self.m2 * S2) * wn
            i0 = self.i0
            fire_now = None
            if kick and t in kick:
                i0 = i0.clone()
                for k, val in kick[t].items():
                    if val == "fire":
                        fire_now = self.sl(k)
                    else:
                        i0[self.sl(k)] += val
            v = (v * ALPHA + i0 + x - I * wi).clamp(min=-5.0)
            fr = v >= 1.0
            if fire_now is not None:
                fr[fire_now] = True
            v = torch.where(fr, torch.zeros_like(v), v)
            r2 = r1; r1 = torch.zeros(NX); r1[:NC] = fr.float()
            r1[NC:] = (torch.rand(NX - NC) < pe).float()
            if t < warm:
                continue
            ff = fr & (last >= 0)
            if ff.any():
                dd = (t - last)[ff]; s1[ff] += dd; s2[ff] += dd ** 2; ni[ff] += 1
            last[fr] = t; cnt += fr.float()
            if rec: pop.append(torch.stack([fr[self.sl(k)].float().mean() for k in NUCS]))
        hz = cnt / ((nstep - warm) * DT / 1000)
        m = s1 / ni.clamp(min=1); cv = (s2 / ni.clamp(min=1) - m ** 2).clamp(min=0).sqrt() / m.clamp(min=1e-9)
        return hz, cv, (torch.stack(pop) if rec else None)

    def table(self, hz, cv=None):
        out = []
        for k in NUCS:
            s = f"{k} {float(hz[self.sl(k)].mean()):.1f}({HZ[k]})"
            if cv is not None and k in CVT:
                s += f" CV {float(cv[self.sl(k)].mean()):.2f}({CVT[k]})"
            out.append(s)
        return "・".join(out)


def main():
    nfit = int(1500 / DT)
    mix = C.MIX["半々"]
    print("試し台の fit（bench_v93c と同じ、NMDA 半々）を回す", flush=True)
    gp = C.fit("外節", mix, nfit); C.FIX_F[0] = gp[4]; st = C.fit("視床下核", mix, nfit)
    print(f"  外節 g_e {gp[1]:.3f}・g_i {gp[2]:.3f}・i0 {gp[3]:.4f}・f {gp[4]:.1f} ／ 視床下核 g_e {st[1]:.3f}・g_i {st[2]:.3f}・i0 {st[3]:.4f}", flush=True)
    th = Teacher(gp, st)
    print(f"先生：核 {len(NUCS)}×{N}体・線 {len(th.t):,} 本", flush=True)
    hz, cv, _ = th.run(int(2000 / DT))
    print("0 今の値のまま：" + th.table(hz), flush=True)
    for it in range(10):
        hz, _, _ = th.run(int(1500 / DT))
        for k in NUCS:
            r = float(hz[th.sl(k)].mean())
            if k == "外側手綱核" and r > HZ[k] and float(th.i0[th.sl(k)].mean()) < 1e-4:
                continue
            th.i0[th.sl(k)] *= float(np.clip((HZ[k] / max(r, 0.3)) ** 0.5, 0.5, 2.0))
    nstep = int(SEC * 1000 / DT)
    hz, cv, _ = th.run(nstep)
    print("1 i0 を率に合わせたあと：" + th.table(hz, cv), flush=True)
    print("  i0：" + "・".join(f"{k} {float(th.i0[th.sl(k)].mean()):.4f}" for k in NUCS), flush=True)
    print("2 止める比（8体）", flush=True)
    for k, kinds, tg in (("外節", ("ampa", "nmda", "gaba"), (0.43, 0.68, 2.16)), ("視床下核", ("glu", "gaba"), (0.84, 1.26))):
        base = float(hz[th.off[k]:th.off[k] + 8].mean())
        out = []
        for kind, tv in zip(kinds, tg):
            h, _, _ = th.run(int(3000 / DT), cut=(k, 8, kind))
            out.append(f"{kind} {float(h[th.off[k]:th.off[k] + 8].mean()) / max(base, 1e-9):.2f}({tv})")
        print(f"  {k}：" + "・".join(out), flush=True)
    print("3 1発の応え（20回の平均）", flush=True)
    T0 = int(300 / DT); L = int(200 / DT)
    acc = None
    for rep in range(20):
        _, _, p = th.run(T0 + L, kick={T0: {"外側手綱核": "fire"}}, rec=True)
        acc = p if acc is None else acc + p
    p = acc / 20 * 1000 / DT
    sn = NUCS.index("黒質")
    base = float(p[T0 - int(200 / DT):T0, sn].mean())
    w = lambda a, b: float(p[T0 + int(a / DT):T0 + int(b / DT), sn].mean()) / max(base, 1e-9)
    print(f"  外側手綱核を1発 ⇒ 黒質 10〜40ms {w(10, 40):.2f} 倍（約0.4）・50〜60ms {w(50, 60):.2f} 倍（約1）・ふだん {base:.2f}Hz", flush=True)
    # 境 +40%：150ms だけ境に i0 を足す（足す量は二分法で +40% に）
    lo, hi = 0.0, 0.1
    ib, il = NUCS.index("境"), NUCS.index("外側手綱核")
    for _ in range(8):
        d = (lo + hi) / 2
        acc = None
        for rep in range(6):
            _, _, p = th.run(T0 + int(150 / DT), kick={t: {"境": d} for t in range(T0, T0 + int(150 / DT))}, rec=True)
            acc = p if acc is None else acc + p
        p = acc / 6
        up = float(p[T0:, ib].mean()) / float(p[:T0, ib].mean()) - 1
        lo, hi = (d, hi) if up < 0.40 else (lo, d)
    lh = float(p[T0:, il].mean()) / float(p[:T0, il].mean()) - 1
    print(f"  境 {up * 100:+.0f}%（150ms）⇒ 外側手綱核 {lh * 100:+.0f}%（+60〜70%）", flush=True)
    json.dump({"gp": list(map(float, gp[1:])), "st": list(map(float, st[1:])),
               "i0": {k: float(th.i0[th.sl(k)].mean()) for k in NUCS}}, open("teacher_v95_fit.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
