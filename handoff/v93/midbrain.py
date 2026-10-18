"""中脳と手綱・視床下核の辺縁・境の小さな核（版93）—— 撃ち方 B を周ごとに回す。入るシナプスは学ぶ
設計の経緯・出どころ：handoff/docs/設計案_基底核と小脳.md（10-16〜10-18）、bench_v93a〜h
⟲ 版43b〜92 の形（1歩ごとに撃つ確率 p ＝ ふだん × exp(G × ずれ)、学ばない）は消した

置くもの（各 256体【み】10-18 12:44。受け手1体は送り手の別々の 64体から【み】10-17）
  核              ふだんの率          i0（自分で撃つ）                    入る線
  視床下核の辺縁   19.3Hz【ヒト】      あり【マカク】Polyakova 2020          ← 前頭前野の表現（興奮）【み】10-18 13:23
  境（内節の一部） 33Hz【マカク】       あり【マカク】                       ← 視床下核の辺縁（線を出すのは 69.3%【マカク】Sato 2000b、興奮）
                                                                         ← 側坐核の表現＋誤差（抑制）【み】10-18 13:36
  外側手綱核       30Hz【マカク】       あり（i0 ≥ 0）【マウス】Fedorov 2026  ← 境（興奮）
  吻内側被蓋核     17.8Hz【マカク】     あり（入力なしで 6.7Hz【ラット】）    ← 外側手綱核（興奮）
  黒質            4.56Hz【ヒト】       あり（撃ったら 0 へ）                 ← PPN 64・視床下核の辺縁 64・前頭前野の表現 8・上丘の誤差 64（興奮）
                                                                         ← 吻内側被蓋核 64（抑制）
  ⟲ 版43b の「来なかった → 視床下核の辺縁」は外した【み】10-18 16:57：torch_brain が誤差の素子の中の値から計算した値で、
     ニューロンの発火ではない（⇒ 来ないことに撃つ細胞を本物の素子として作り直す項目、設計案 10-18 17:10）
撃ち方 B（周 1.067ms ごと・10周を1回の呼び出しで）【決め 10-16】
  v ← v·α ＋ i0 ＋ (1−f)·E ＋ f·Σ m·N − I、閾値 1 で撃って 0 へ。α ＝ exp(−周／40ms)【決め 10-15】
  E・I ＝ Σ 強さ g × 重さ w × 届いた発火 ÷ 線の数。N ＝ E を NMDA の時定数でならしたもの
  NMDA の尾：視床下核の辺縁・境 ＝ 淡蒼球の決め（NR2A 21ms・NR2D 1,340ms 半々、f 0.4）【ヒト】Küppenbender 1998【み】10-17
             外側手綱核 ＝ 22ms（65%）・137ms（35%）、f 0.5【ヒト】HPA 手綱核 2A:2B ≒ 1:2 ＋ Vicini 1998 表1 ÷3.3【み】10-18 13:17
             吻内側被蓋核・黒質 ＝ 75.5ms、f 0.5【マウス】Pearlstein 2015 ⚠️ 吻内側被蓋核は値なし
届く時刻
  核どうし：撃った周の次の周【決め 10-16】
  領野（前頭前野・側坐核・上丘）：領野が撃った周（Group2 の _last_fire）の次の周【み】10-18 16:18「撃った周にちゃんと入力が入るのが正しい」
  PPN（脳幹）：周ごとに撃つ（torch_brain）。撃った周の次の周【み】10-18 16:27
  1歩の最後の周に撃った分は、次の歩の周0 に届く
強さ（bench_v93h 版2・bench_v93f 版6 の値）⚠️ 領野の率は試し台の代わり（前頭前野 3.3Hz・側坐核 2.1Hz）で合わせた
  ⇒ 網の率が違えば核の率がずれる。check_v93.py で測る
学び（興奮の線だけ。重さは 0〜W_MAX、初め 1）
  黒質：Harnett 2009【ラット】の痕跡3本・決め B（跳ねの痕跡は入力の速い痕跡が小さいときの跳ねだけで作る）【み】10-17
        SN_LTD_ORDER ＝ False で A の値・形に戻る（「何かあったときAを辿れるように」）
  ほかの核：決め A（送り手の痕跡があるうちに受け手が撃つと跡 ⇒（門 − 1）で強まる・弱まる）【み】10-16 ⚠️ 量 ETA_A・長さはノブ
門（三つ目の因子）：黒質の率を 225ms でならして ÷ 4.56Hz（今のまま）【み】10-17 0:42。周の発火を1歩分たしてならす
"""
import math
import torch
from torch_area import DT_MS

DT = 1.067                       # 周（ms）
R_PASS = 10                      # 1歩の周（torch_area.RECUR と同じ）
TAU_M = 40.0
ALPHA = math.exp(-DT / TAU_M)
N_DEF = 256
K_DEF = 64

NAMES_MB = ("視床下核", "境", "外側手綱核", "吻内側被蓋核", "黒質")
HZ = {"黒質": 4.56, "視床下核": 19.3, "境": 33.0, "外側手綱核": 30.0, "吻内側被蓋核": 17.8}
TAILS = {"淡蒼球": ((21.0, 0.5), (1340.0, 0.5)), "手綱": ((22.0, 0.65), (137.0, 0.35)), "黒質": ((75.5, 1.0), (75.5, 0.0))}
NUC = {
    "視床下核":     dict(i0=0.0337,  tails="淡蒼球", f=0.4),
    "境":          dict(i0=0.0616,  tails="淡蒼球", f=0.4),
    "外側手綱核":   dict(i0=0.0117,  tails="手綱", f=0.5),
    "吻内側被蓋核": dict(i0=0.02699, tails="黒質", f=0.5),
    "黒質":        dict(i0=0.00917, tails="黒質", f=0.5),
}
# 線：(送り手, 受け手, 符号, 強さ g, 受け手1体の本数, 線を出す送り手の割合)
LINES = [
    ("前頭前野", "視床下核", +1, 0.857, K_DEF, 1.0),
    ("視床下核", "境", +1, 0.0761, K_DEF, 0.693),
    ("側坐核", "境", -1, 12.055, K_DEF, 1.0),           # 側坐核の表現＋誤差を1つの送り手として（1本あたり同じ）
    ("境", "外側手綱核", +1, 1.0, K_DEF, 1.0),
    ("外側手綱核", "吻内側被蓋核", +1, 0.2544, K_DEF, 1.0),
    ("PPN", "黒質", +1, 0.9, K_DEF, 1.0),
    ("視床下核", "黒質", +1, 0.9, K_DEF, 1.0),
    ("前頭前野", "黒質", +1, 0.9, 8, 1.0),
    ("上丘", "黒質", +1, 0.9, K_DEF, 1.0),
    ("吻内側被蓋核", "黒質", -1, 0.152, K_DEF, 1.0),
]
# 割る線の数（受け手・符号ごと）。黒質の興奮は 136（PPN 64 ＋ 視床下核 64 ＋ 前頭前野 8、bench_v93g・h と同じ）。上丘の 64 も同じ1本の重さ
DIV_FIX = {("黒質", +1): 136.0}

SN_LTD_ORDER = True               # 決め B（False で A）
HARNETT = {True: dict(ta=131.0, tb=10000.0, tc=284.0, Ap=2.99e-5 * 0.605, Am=2.96e-4, A0=9.34e-5),
           False: dict(ta=161.0, tb=1321.0, tc=300.0, Ap=3.85e-5 * 0.64, Am=5.47e-5, A0=1.42e-4)}
ORDER_A = 0.05
ETA_A = 1e-3                      # ⚠️ ノブ
TAU_PRE_A = 20.0                  # ⚠️ ノブ（送り手の痕跡、ms）
TAU_ELIG_A = 225.0                # ⚠️ ノブ（跡の長さ、ms。門のならしと同じ）
W_MAX = 4.0
DA_TAU = int(round(225.0 / DT_MS))   # 門のならし【ヒト】Zaghloul 2009


class Midbrain:
    def __init__(self, n=None, dev="cpu", gen=None):
        self.n = {k: N_DEF for k in NAMES_MB} if n is None else dict(n)
        self.dev, self.gen = dev, gen
        self.p0 = HZ["黒質"] * DT_MS / 1000.0
        self.da = torch.tensor(self.p0, device=dev)
        self.spk = {k: torch.zeros(self.n[k], device=dev) for k in NAMES_MB}   # この歩のどこかで撃ったか（測る道具用）
        self.rate_sum = torch.zeros(len(NAMES_MB), device=dev)
        self.steps = 0

    def _pick(self, n_src, n_tgt, k, frac):
        g = self.gen
        pool = torch.randperm(n_src, device=self.dev, generator=g)[:max(1, int(round(n_src * frac)))]
        k = min(k, len(pool))
        return torch.stack([pool[torch.randperm(len(pool), device=self.dev, generator=g)[:k]] for _ in range(n_tgt)])

    def wire(self, n_src):
        """n_src：外から来る送り手の数 {"前頭前野": 表現の体数, "側坐核": 表現＋誤差の体数, "上丘": 誤差の体数, "PPN": 体数}"""
        dev = self.dev
        self.n_src = dict(n_src)
        # 送り手の並び X ＝ [核の細胞（NAMES_MB の順）| 外の送り手（n_src の順）]
        self.off = {}
        o = 0
        for k in NAMES_MB:
            self.off[k] = o; o += self.n[k]
        self.NC = o
        for s, n in self.n_src.items():
            self.off[s] = o; o += n
        self.NX = o
        size = lambda s: self.n[s] if s in NAMES_MB else self.n_src[s]
        cnt = {}
        for s, t, sg, g, k, fr in LINES:
            cnt[(t, sg)] = cnt.get((t, sg), 0) + min(k, max(1, int(round(size(s) * fr))))
        div = {key: DIV_FIX.get(key, float(v)) for key, v in cnt.items()}
        tgt, src, geff, sign, kind = [], [], [], [], []
        for s, t, sg, g, k, fr in LINES:
            idx = self._pick(size(s), self.n[t], k, fr)
            r = torch.arange(self.n[t], device=dev).unsqueeze(1).expand_as(idx)
            tgt.append((r + self.off[t]).reshape(-1)); src.append((idx + self.off[s]).reshape(-1))
            m = idx.numel()
            geff.append(torch.full((m,), g / div[(t, sg)], device=dev))
            sign.append(torch.full((m,), sg, device=dev, dtype=torch.int8))
            kind.append(torch.full((m,), 0 if sg < 0 else (1 if t == "黒質" else 2), device=dev, dtype=torch.int8))
        self.tgt, self.src = torch.cat(tgt), torch.cat(src)
        self.geff, self.sign, self.kind = torch.cat(geff), torch.cat(sign), torch.cat(kind)
        self.w = torch.ones(len(self.tgt), device=dev)
        self.exc = self.sign > 0
        self.ie = torch.nonzero(self.exc).squeeze(1); self.ii = torch.nonzero(~self.exc).squeeze(1)
        self.tE, self.tI = self.tgt[self.ie], self.tgt[self.ii]
        self.iH = torch.nonzero(self.kind == 1).squeeze(1)       # 黒質の興奮の線（Harnett）
        self.iA = torch.nonzero(self.kind == 2).squeeze(1)       # ほかの核の興奮の線（決め A）
        # 細胞ごとの値
        i0, f, a1, m1, a2, m2 = [], [], [], [], [], []
        for k in NAMES_MB:
            c = NUC[k]; (t1, w1), (t2, w2) = TAILS[c["tails"]]
            n = self.n[k]
            for lst, v in ((i0, c["i0"]), (f, c["f"]), (a1, math.exp(-DT / t1)), (m1, w1), (a2, math.exp(-DT / t2)), (m2, w2)):
                lst.append(torch.full((n,), float(v), device=dev))
        self.i0, self.f, self.a1, self.m1, self.a2, self.m2 = (torch.cat(x) for x in (i0, f, a1, m1, a2, m2))
        self.v = torch.rand(self.NC, device=dev, generator=self.gen)
        self.S1 = torch.zeros(self.NC, device=dev); self.S2 = torch.zeros(self.NC, device=dev)
        self.prev = torch.zeros(self.NC, device=dev)
        self.carry = torch.zeros(self.NX - self.NC, device=dev)
        # 学びの痕跡（送り手ごと：X の長さ。線ごと：c・跡）
        h = HARNETT[SN_LTD_ORDER]
        self.h = dict(h, ea=math.exp(-DT / h["ta"]), eb=math.exp(-DT / h["tb"]), ec=math.exp(-DT / h["tc"]))
        self.e_pre = math.exp(-DT / TAU_PRE_A)
        self.e_elig = math.exp(-DT_MS / TAU_ELIG_A)
        self.ha = torch.zeros(self.NX, device=dev); self.hb = torch.zeros(self.NX, device=dev)   # 黒質の a・b
        self.pa = torch.zeros(self.NX, device=dev)                                            # 決め A の送り手の痕跡
        self.c = torch.zeros(len(self.iH), device=dev)
        self.elig = torch.zeros(len(self.iA), device=dev)
        self.tH, self.sH = self.tgt[self.iH], self.src[self.iH]
        self.tA, self.sA = self.tgt[self.iA], self.src[self.iA]
        self.sn = slice(self.off["黒質"], self.off["黒質"] + self.n["黒質"])
        self.seg = [(k, self.off[k], self.off[k] + self.n[k]) for k in NAMES_MB]

    def ext_passes(self, spk, ps):
        """外の送り手1つ：この歩に撃ったか [n] と撃った周 [n] ⇒ 周ごとの発火 [R, n]"""
        m = torch.zeros(R_PASS, len(spk), device=self.dev)
        m.scatter_(0, ps.clamp(0, R_PASS - 1).long().unsqueeze(0), spk.float().unsqueeze(0))
        return m

    @torch.no_grad()
    def step(self, ext, learn=True):
        """1歩（10周）。ext：{送り手: 周ごとの発火 [R, n]}（n_src と同じ順・同じ数）。返り値：門"""
        R = R_PASS
        m = torch.cat([ext[s].float() for s in self.n_src], 1)      # [R, 外の送り手]
        arr = torch.cat([self.carry.unsqueeze(0), m[:-1]])            # 周 p に届くのは周 p−1 に撃った分
        self.carry = m[-1]
        h = self.h
        acc = torch.zeros(self.NC, device=self.dev)
        for p in range(R):
            X = torch.cat([self.prev, arr[p]])
            if learn:                                                  # 痕跡の減り（届く前）
                self.ha.mul_(h["ea"]); self.hb.mul_(h["eb"]); self.pa.mul_(self.e_pre)
            d = self.geff * self.w * X[self.src]
            E = torch.zeros(self.NC, device=self.dev).index_add_(0, self.tE, d[self.ie])
            I = torch.zeros(self.NC, device=self.dev).index_add_(0, self.tI, d[self.ii])
            self.S1 = self.S1 * self.a1 + E * (1 - self.a1)
            self.S2 = self.S2 * self.a2 + E * (1 - self.a2)
            v = (self.v * ALPHA + self.i0 + (1 - self.f) * E + self.f * (self.m1 * self.S1 + self.m2 * self.S2) - I).clamp(min=-5.0)
            fire = v >= 1.0
            self.v = torch.where(fire, torch.zeros_like(v), v)
            new = fire.float()
            acc += new
            if learn:
                self._learn_pass(X, new)
                self.ha.add_(X); self.hb.add_(X); self.pa.add_(X)     # 届いた分を足す（bench_v93d と同じ順）
            self.prev = new
        self.rate_sum += torch.stack([acc[a:b].sum() / (b - a) for _, a, b in self.seg])
        self.steps += 1
        self.spk = {k: (acc[a:b] > 0).float() for k, a, b in self.seg}
        self.da = self.da + (acc[self.sn].mean() - self.da) / DA_TAU    # 周の発火を1歩分たして（1歩あたりの数）ならす
        gate = self.da / self.p0
        if learn and len(self.iA):
            self.w[self.iA] = (self.w[self.iA] + ETA_A * (gate - 1.0) * self.elig).clamp(0.0, W_MAX)
            self.elig.mul_(self.e_elig)
        return gate

    def _learn_pass(self, X, new):
        h = self.h
        if len(self.iH):                                               # 黒質（Harnett、bench_v93d と同じ順：減り → 跳ね → 入力）
            pre = X[self.sH]; post = new[self.tH]
            a, b = self.ha[self.sH], self.hb[self.sH]
            self.c.mul_(h["ec"])
            dw = h["Ap"] * a * b * post
            self.c += post * ((a < ORDER_A).float() if SN_LTD_ORDER else 1.0)
            dw = dw - (h["Am"] * self.c + h["A0"]) * pre
            self.w[self.iH] = (self.w[self.iH] + dw).clamp(0.0, W_MAX)
        if len(self.iA):                                               # 決め A：送り手の痕跡があるうちに受け手が撃つ ⇒ 跡
            self.elig += self.pa[self.sA] * new[self.tA]

    def rates_hz(self):
        t = max(self.steps, 1) * DT_MS / 1000.0
        return {k: float(self.rate_sum[i]) / t for i, k in enumerate(NAMES_MB)}

    def state_dict(self):
        return {"w": self.w, "da": self.da, "v": self.v, "S1": self.S1, "S2": self.S2}

    def load_state_dict(self, d):
        if d.get("w") is None or d["w"].shape != self.w.shape:
            print("⚠️ 中脳の核の重さは形が違うので読まない（初めの重さ 1 から）", flush=True)
            return
        for k in ("w", "da", "v", "S1", "S2"):
            setattr(self, k, d[k].to(self.dev))
