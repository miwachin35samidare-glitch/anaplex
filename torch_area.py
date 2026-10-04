"""arkhe 用（torch / ROCm）—— numpy 版と同じ機構を GPU に移したもの

⚠️ このコンテナに torch が無いので、一度も動かしていない。arkhe で回すのが初回。
   numpy 版（area4.py / group.py / hippocampus.py / thalamus.py / brain.py）と同じ形に書いてある。
   まず selftest() を通してから網を組むこと。

環境（40_環境 より）
  PyTorch 2.13.0+rocm7.2 / RX 9070 XT 16GB（gfx1201）。VRAM は 12GB まで使う
  🔴 毎歩 float() で GPU から値を取り出すと 1.4〜1.5倍 遅くなる。取り出しは1回だけにする
  🔴 小さい演算を何度も呼ぶと、呼ぶ回数の値段が支配する ⇒ 領野はまとめて1回で進める

持ち方（numpy 版と同じ）
  辺は送り手ごとに連ねる。撃った送り手の分だけを触る（発火の疎を計算に効かせる）
  contacts と e_src は持たない（抑制かどうかと、並び順から導く）
  短期可塑性は共有の表＋行の索引。触ったときに最後の歩からまとめて減らす
"""
import math
import numpy as np
import torch

# 抑制の1接続の強さ（学ばない。版45【み】10-01「学ばない形にしても」）
#   【ヒト】側頭葉 MTG の抑制性 → 興奮性 18組：g_syn / g_input 平均 0.02648（1接続。Campagnola 2022 ／ synapse_model_human.json）
#   接触 4.0 で割って1本の重さに（amp ＝ 4.0 × w）。短期可塑性も同じ表から（静止時の1発目 ＝ 1）
#   ⚠️ 速い籠細胞だけなら 0.69nS ≈ 漏れの 0.07倍（Szegedi 2017）。こちらは型を混ぜた平均
#   ⚠️ 人の抑制 → 錐体 のシナプスが学ぶかは測られていない（Mansvelder 2019）。学ばないのは【み】の判断
W_INH = 0.02648 / 4.0

# 版47：錐体 → 抑制の体（形A の直し）【み選】10-02 B ／【み】10-02「Aを直して」
#   版45 の撃ち方の単位で 1mV：静止の見込み −1.98（焼いた素子の「何も撃っていない」）を
#   【ヒト】速い抑制細胞の静止から閾値まで 20.9mV（Wilbers 2023）に当てる【C】
K_MV = 1.98 / 20.9
#   相手の数：興奮性 180（ノブ【C】。人の抑制細胞の入力の数は見つからない）
#   錐体が【ヒト】覚醒 3.30Hz で撃つとき抑制の体が【ヒト】覚醒 4.70Hz になる数（Verma 2024。EPSP・閾値は Wilbers 2023）
#   ⟲ 版46（戻した）は 1,080：錐体 0.45%/歩・抑制 2Hz（Peyrache 2012）で出した数。0.45%/歩 は人の値ではなかった
INH_N_EXC = 180
#   海馬・嗅内：【ヒト】Le Van Quyen 2008（てんかん 11人、深部微小電極）ふだん 錐体 3.1Hz ／ 抑制 5.1Hz ⇒ 194
INH_N_EXC_HIPPO = 194
#   錐体 → 抑制の体 の1本：対数正規 1.65 ± 1.59mV（【ヒト】Wilbers 2023）を K_MV で版45 の単位に
INH_W_SIG = math.sqrt(math.log(1.0 + 0.96 ** 2))
INH_W_MU = -INH_W_SIG ** 2 / 2
INH_W_MEAN = 1.65 * K_MV
#   弱まり：大きい繋がり（【ヒト】4.21mV 超、Molnár 2016）だけ、送り手が 10歩に5回以上撃った歩ごとに
#   0.52 の 40乗根（【ヒト】40Hz の連発 40回で 0.52倍）。弱い繋がりは弱まらない
INH_LTD_TH = 4.21 * K_MV
# 版48：抑制の体の撃つ閾値。【ヒト】速い抑制細胞の静止から閾値まで 20.9mV（Wilbers 2023）を版45 の単位に（＝ 1.98）
INH_TH = 20.9 * K_MV
INH_LTD_F = 0.52 ** (1.0 / 40.0)
INH_LTD_WIN, INH_LTD_N = 10, 5

DT_MS = 10.67
SYN_DELAY_MS = 1.102
RECUR = max(1, int(round(DT_MS / SYN_DELAY_MS)))      # 1歩の中の周回（整理券）≒10
TAUS = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32)
THETA = 0.5
HUMAN_INPUT_PER_STEP = 135.0   # 【ヒト】毎歩 受けている入力の本数（30,000シナプス × 0.45%/歩）
RATE_HUMAN = 0.0045             # 【ヒト】皮質の細胞が 1歩（10.67ms）に撃つ確率 0.45%（約 0.4Hz）
#   シナプススケーリングの目標。【ヒト】iPS 由来の皮質細胞で、活動電位を止めると興奮性シナプス全体が
#   強まって活動が戻る（Cordella 2022）⇒ 感じているのは細胞自身の活動、動かすのはシナプスの強さ全体
# ✗ G_FIRE 0.855 は使わない。コンダクタンスで動く素子の値（前の作り）で、このラインの素子
#   （重みはロジットに足される量）には持ち越せない。落-039（連続版の 0.53 を持ち込んで暴走）と同じ形


def steps(ms):
    return max(1, int(round(ms / DT_MS)))


class Prior(torch.nn.Module):
    """素子の中の「いつもの見込み」。焼いて凍結し、同じ型で共有する"""

    def __init__(self, hidden=32, dev="cuda"):
        super().__init__()
        n = len(TAUS)
        self.W1 = torch.nn.Parameter(torch.randn(n, hidden, device=dev) / n ** 0.5, False)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden, device=dev), False)
        self.W2 = torch.nn.Parameter(torch.randn(hidden, 1, device=dev) / hidden ** 0.5, False)
        self.b2 = torch.nn.Parameter(torch.zeros(1, device=dev), False)

    @torch.no_grad()
    def logit(self, f):
        return (torch.tanh(f @ self.W1 + self.b1) @ self.W2 + self.b2).squeeze(-1)

    @classmethod
    def load(cls, path, dev="cuda"):
        """numpy 版で焼いた prior_v1.npz をそのまま読む"""
        d = np.load(path)
        m = cls(hidden=d["W1"].shape[1], dev=dev)
        with torch.no_grad():
            for k in ("W1", "b1", "W2", "b2"):
                getattr(m, k).copy_(torch.tensor(d[k], device=dev))
        return m


class STP:
    """短期可塑性。値は172行の共有の表に置き、1本は行の索引だけ持つ（ヒト側頭葉の当てはめ）"""

    def __init__(self, n_edge, send_inh, recv_inh, path, dev="cuda", gen=None):
        import json
        d = json.load(open(path, encoding="utf-8"))
        keys = ("ex2ex", "ex2in", "in2ex", "in2in")
        tbl, off = [], {}
        for k in keys:
            off[k] = len(tbl)
            tbl += [list(map(float, r)) for r in d["行"][k]]
        self.tbl = torch.tensor(tbl, dtype=torch.float32, device=dev)
        self.dec_d = torch.exp(-DT_MS / 1000.0 / self.tbl[:, 2].clamp(min=1e-4))
        self.dec_f = torch.exp(-DT_MS / 1000.0 / self.tbl[:, 4].clamp(min=1e-4))
        self.N = self.tbl[:, 5].round().clamp(min=1)
        grp = (send_inh.long() * 2 + recv_inh.long())          # ex2ex/ex2in/in2ex/in2in
        row = torch.zeros(n_edge, dtype=torch.long, device=dev)
        for gi, k in enumerate(("ex2ex", "ex2in", "in2ex", "in2in")):
            m = grp == gi
            n = int(m.sum())
            if n:
                row[m] = off[k] + torch.randint(0, len(d["行"][k]), (n,), device=dev, generator=gen)
        self.row = row.to(torch.int16)       # 表は172行なので 16bit で足りる（版43a-m。使うときに広げる）
        self.deplete = self.tbl[row, 1] < 0
        self.state = torch.where(self.deplete, torch.ones(n_edge, device=dev),
                                 torch.zeros(n_edge, device=dev))
        self.fac = torch.zeros(n_edge, device=dev)
        self.last = torch.zeros(n_edge, dtype=torch.int32, device=dev)
        self.gen = gen

    @torch.no_grad()
    def release_at(self, e, t):
        """触った辺だけ、最後に触った歩からまとめて減らして放出する"""
        r = self.row[e].long()
        n = (t - self.last[e]).float()
        dep = self.deplete[e]
        self.state[e] = self.state[e] * torch.where(dep, torch.ones_like(n), self.dec_d[r] ** n)
        self.fac[e] = self.fac[e] * self.dec_f[r] ** n
        self.last[e] = t
        Pr0 = self.tbl[r, 0]
        P = (Pr0 + (1 - Pr0) * self.fac[e]).clamp(0, 1)
        N = self.N[r]
        avail = torch.where(dep, self.state[e], 1 - self.state[e])
        cnt = torch.binomial((N * avail).round().clamp(min=0), P, generator=self.gen)
        out = cnt / (N * Pr0).clamp(min=1e-6)
        k = torch.where(dep, P, self.tbl[r, 1])
        self.state[e] = torch.where(dep, self.state[e] * (1 - k),
                                    self.state[e] + (1 - self.state[e]) * k)
        self.fac[e] = self.fac[e] + (1 - self.fac[e]) * self.tbl[r, 3]
        return out

    def reorder(self, o):
        self.row, self.state, self.fac = self.row[o], self.state[o], self.fac[o]
        self.last, self.deplete = self.last[o], self.deplete[o]


class TimeCells:
    """時間に広がった受容野（聴覚）／ 時間細胞（海馬）／ 傾きを持つ細胞
    運動の領野は n_cell=0（集団の状態が動くことで時間を表すので、貯めない）"""

    def __init__(self, n_src, n_cell=128, n_ramp=8, span_ms=341.0, ramp_span_ms=2732.0,
                 M=16, dev="cuda", gen=None):
        self.n_src, self.M, self.n_cell, self.n_ramp, self.dev = n_src, M, n_cell, n_ramp, dev
        if n_cell:
            # 遅れが大きい体ほど広い（対数圧縮。【ヒト】Umbach ほか 2020）
            pref = torch.exp(torch.linspace(0.0, np.log(steps(span_ms)), n_cell, device=dev))
            self.pref, self.half = pref, (0.1 * pref).clamp(min=1.0) / 2
        else:
            self.pref = self.half = None
        self.ramp_len = steps(ramp_span_ms)
        self.age = torch.full((n_src, M), 10 ** 9, dtype=torch.long, device=dev)
        self.gen, self._cache = gen, None

    @property
    def n_cells(self):
        return self.n_src * (self.n_cell + self.n_ramp)

    @torch.no_grad()
    def spikes(self):
        if self._cache is not None:
            return self._cache
        parts = []
        if self.n_cell:
            a = self.age.unsqueeze(-1).float()
            hit = ((a - self.pref).abs() <= self.half) & (a <= self.pref.max())
            parts.append(hit.any(1).float())
        y = (self.age.min(1).values.float() / self.ramp_len).clamp(0, 1).unsqueeze(-1)
        h = self.n_ramp // 2
        up = torch.rand(self.n_src, h, device=self.dev, generator=self.gen) < y * 0.3
        dn = torch.rand(self.n_src, self.n_ramp - h, device=self.dev, generator=self.gen) < (1 - y) * 0.3
        parts.append(torch.cat([up, dn], 1).float())
        self._cache = torch.cat(parts, 1).reshape(-1)
        return self._cache

    @torch.no_grad()
    def step(self, fired):
        self._cache = None
        self.age = torch.where(self.age < 10 ** 8, self.age + 1, self.age)
        f = fired > 0
        if f.any():
            oldest = self.age.argmax(1)
            idx = torch.nonzero(f).squeeze(1)
            self.age[idx, oldest[idx]] = 0


class Area:
    """1つの領野。表現・誤差・抑制のニューロンと、疎なシナプス"""

    def __init__(self, n_ch, prior, n_ext=0, eta=0.03, n_partner=200, pool_mult=3,
                 rewire_every=None, homeo_tau=2000.0, n_cell=128, n_ramp=8,
                 inh_frac=0.16, stp_path=None, tau_scale=1.0, gate=True,
                 spont_hz=0.3, complete=False, motor=False, dev="cuda", seed=0,
                 w_int8=False, fire_th=1.0, x_gain=1.0, tgt_by_fire=False, contacts_exc=3.3,
                 inh_n_exc=None):
        g = torch.Generator(device=dev); g.manual_seed(seed)
        self.dev, self.gen, self.prior = dev, g, prior
        self.n_ch, self.n_ext, self.eta, self.homeo_tau = n_ch, n_ext, eta, homeo_tau
        self.tau_scale, self.gate, self.complete, self.motor = tau_scale, gate, complete, motor
        self.spont_p = spont_hz * DT_MS / 1000.0
        self.pred_boost, self.w_int8 = 0.9, w_int8
        # シナプススケーリングの目標：細胞自身の発火率を【ヒト】0.45%/歩 に
        #   ⟲ 版4「本数を135へ」は scale で本数を動かせない。版18「強さを 0.855 へ」は素子の単位が違う
        self.scaling_target = RATE_HUMAN
        # 表現のニューロンが撃つ閾値。【ヒト】L2/3 錐体細胞は同時に 134±28本 で 50% の確率
        #   （人が毎歩受けている入力 135本 とほぼ同じ ＝ 平均がちょうど閾値）
        #   こちらの重みは1本 1.0 ではないので、合計の値として置く（ノブ）
        self.fire_th = fire_th
        self.x_gain = x_gain        # 入口から直に来る駆動を何本ぶんと数えるか
        # 誤差の的を「下からの入力（本数 × 重さ）が閾値を越えるか」にする領野
        #   【ヒト】皮質の錐体細胞の出力は、集まった入力を閾値と比べた撃つ／撃たない
        self.tgt_by_fire = tgt_by_fire
        # 1つの繋がりの興奮性の接触の数。【ヒト】皮質 3.3（既定）
        self.contacts_exc = contacts_exc
        self.recur = RECUR
        self.D = 8 if motor else steps(SYN_DELAY_MS * 3)     # 運動は自分の発火が戻る道を広く持つ
        self.rewire_every = steps(1067.0) if rewire_every is None else rewire_every
        self.decay = torch.exp(-1.0 / (TAUS.to(dev) * tau_scale))
        self.n_inh = max(1, int(round(n_ch * inh_frac / (1 - inh_frac))))
        self.n_src = n_ch + n_ext + self.n_inh
        self.ch = TimeCells(self.n_src, n_cell=(0 if motor else n_cell), n_ramp=n_ramp,
                            dev=dev, gen=g)
        self.n_send = self.ch.n_cells + self.n_src * self.D
        self.n_recv = n_ch * 2 + self.n_inh
        n_partner = min(n_partner, self.n_send)
        self.n_partner = n_partner
        # 版47：抑制の体の相手は 興奮性 INH_N_EXC ＋ 抑制性（inh_frac を保つ）
        n_ie = INH_N_EXC if inh_n_exc is None else inh_n_exc
        n_pi = min(int(round(n_ie / (1 - inh_frac))), self.n_send)
        self.n_partner_inh = n_pi
        self.n_partner_vec = torch.cat([torch.full((n_ch * 2,), float(n_partner), device=dev),
                                        torch.full((self.n_inh,), float(n_pi), device=dev)])
        # 送り手が抑制性か（辺の相手を選ぶ前に要る）
        is_inh_send = torch.zeros(self.n_send, dtype=torch.bool, device=dev)
        inh_from = (n_ch + n_ext) * (self.ch.n_cell + self.ch.n_ramp)
        is_inh_send[inh_from:self.ch.n_cells] = True
        is_inh_send[self.ch.n_cells + (n_ch + n_ext) * self.D:] = True
        # 辺の相手：受け手1体あたり inh_frac を抑制性から、残りを興奮性から引く
        #   【ヒト】E:I はシナプス数で 84:16（L2/3 電子顕微鏡）。外から来る入力も全部この 84 の側に入る
        #   ⚠️ 前は送り元全体から一様に引いていた。抑制の割合が「抑制の体 ÷ 送り元」になり、
        #      外からの入力が多い領野ほど薄まっていた（聴覚 2.8% ／ 高次 4.4% ／ CA3 2.4%）
        inh_pool = torch.nonzero(is_inh_send).squeeze(1)
        exc_pool = torch.nonzero(~is_inh_send).squeeze(1)
        def pick(n_rows, k_all):
            k_inh = int(round(k_all * inh_frac))
            s = torch.empty((n_rows, k_all), dtype=torch.long, device=dev)
            s[:, :k_inh] = inh_pool[torch.randint(0, len(inh_pool), (n_rows, k_inh),
                                                  device=dev, generator=g)]
            s[:, k_inh:] = exc_pool[torch.randint(0, len(exc_pool), (n_rows, k_all - k_inh),
                                                  device=dev, generator=g)]
            return s.reshape(-1)
        # 表現・誤差は n_partner、抑制の体は n_pi（版47）
        src = torch.cat([pick(n_ch * 2, n_partner), pick(self.n_inh, n_pi)])
        recv = torch.cat([torch.arange(n_ch * 2, device=dev).repeat_interleave(n_partner),
                          (torch.arange(self.n_inh, device=dev) + n_ch * 2).repeat_interleave(n_pi)])
        o = torch.argsort(src)
        self.e_src_sorted = src[o]                 # 刈り込みのときだけ使う
        self.e_recv = recv[o].int()
        self.w = torch.zeros(len(o), device=dev)
        self.start = torch.zeros(self.n_send + 1, dtype=torch.long, device=dev)
        self.start.scatter_add_(0, self.e_src_sorted + 1,
                                torch.ones(len(o), dtype=torch.long, device=dev))
        self.start = self.start.cumsum(0)
        self.is_inh_send = is_inh_send
        self.e_inh = is_inh_send[self.e_src_sorted]
        self.w[self.e_inh] = W_INH                 # 版45：抑制は人の強さで始まり、学ばない
        recv_inh = torch.zeros(self.n_recv, dtype=torch.bool, device=dev)
        recv_inh[n_ch * 2:] = True
        self.stp = STP(len(o), self.e_inh, recv_inh[self.e_recv.long()], stp_path, dev, g) \
            if stp_path else None
        self.tb = torch.zeros(self.n_recv, len(TAUS), device=dev)
        self.buf = torch.zeros(self.n_src, self.D, device=dev)
        self.inh_spk = torch.zeros(self.n_inh, device=dev)
        self.scale = torch.ones(self.n_recv, device=dev)
        self.recv_ema = torch.zeros(self.n_recv, device=dev)
        self.drive = torch.zeros(self.n_recv, device=dev)
        self.budget = self.n_partner_vec.clone()      # 版47：受け手の種類ごとの相手の数
        self.t = 0
        self._e_cache = None

    @property
    def n_cells(self):
        return self.n_ch * 2 + self.n_inh + self.ch.n_cells

    @torch.no_grad()
    def _fired_edges(self):
        if self._e_cache is not None:
            return self._e_cache
        s = torch.cat([self.ch.spikes(), self.buf.reshape(-1)])
        fired = torch.nonzero(s).squeeze(1)
        if len(fired) == 0:
            self._e_cache = torch.empty(0, dtype=torch.long, device=self.dev)
            return self._e_cache
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        fired, lens = fired[keep], lens[keep]
        if len(fired) == 0:
            self._e_cache = torch.empty(0, dtype=torch.long, device=self.dev)
            return self._e_cache
        base = self.start[fired].repeat_interleave(lens)
        off = torch.arange(int(lens.sum()), device=self.dev) - \
            torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev),
                       lens.cumsum(0)[:-1]]).repeat_interleave(lens)
        self._e_cache = base + off
        return self._e_cache

    @torch.no_grad()
    def step(self, x, ext=None, learn=True):
        """x: 細胞体に入る（駆動する入力）  ext: 樹状突起に入る（調整する入力）"""
        for i in range(self.recur):
            out = self._pass(x, ext, learn and i == 0, i == self.recur - 1, i == 0)
        return out

    @torch.no_grad()
    def _pass(self, x, ext, learn, advance, first):
        e = self._fired_edges()
        exc = torch.zeros(self.n_recv, device=self.dev)
        ihn = torch.zeros(self.n_recv, device=self.dev)
        if len(e):
            amp = torch.where(self.e_inh[e], 4.0, 3.3) * self.w[e]
            if self.stp is not None:
                if first:
                    self._amp = torch.ones(len(self.w), device=self.dev)
                    self._amp[e] = self.stp.release_at(e, self.t)
                amp = amp * self._amp[e]
            r = self.e_recv[e].long()
            exc.scatter_add_(0, r, torch.where(self.e_inh[e], 0.0, amp))
            ihn.scatter_add_(0, r, torch.where(self.e_inh[e], amp, 0.0))
        exc = exc * self.scale; ihn = ihn * self.scale
        p = torch.sigmoid(self.prior.logit(self.tb) + exc / (1.0 + ihn))
        tgt = torch.cat([x, x, x.mean().expand(self.n_inh)])
        err_sig = tgt - p
        if learn and len(e):
            r = self.e_recv[e].long()
            gg = self.eta * err_sig[r]
            self.w[e] += torch.where(self.e_inh[e], torch.zeros_like(gg), gg)   # 版45：抑制は学ばない
            a = 1.0 / self.homeo_tau
            cnt = torch.zeros(self.n_recv, device=self.dev)
            cnt.scatter_add_(0, r, torch.ones(len(e), device=self.dev))
            self.recv_ema += a * (cnt - self.recv_ema)
            self.drive += a * (exc.abs() - self.drive)
            self.scale *= (1 - 0.01 * (self.drive / self.scaling_target - 1)).clamp(0.5, 1.5) ** a
        err = (err_sig[:self.n_ch] > THETA).float()
        rep = x.clone()
        if self.complete and x.sum() > 0:
            # 予測は撃たせない。入力に足して閾値に届きやすくするだけ（【ヒト】遠い樹状突起）
            rep = torch.maximum(rep, ((x + self.pred_boost * p[self.n_ch:self.n_ch * 2]) > 1.0).float())
        if self.spont_p > 0:
            rep = torch.maximum(rep, (torch.rand(self.n_ch, device=self.dev,
                                                 generator=self.gen) < self.spont_p).float())
        self.inh_spk = (p[self.n_ch * 2:] > 0.5).float()
        self.pred_rep = p[self.n_ch:self.n_ch * 2]
        if advance:
            self.tb = self.tb * self.decay + torch.cat(
                [rep, rep, self.inh_spk]).unsqueeze(1) * (1 - self.decay)
            src = rep if ext is None else torch.cat([rep, ext])
            src = torch.cat([src, self.inh_spk])
            self.ch.step(src)
            self.buf = torch.cat([src.unsqueeze(1), self.buf[:, :-1]], 1)
            self.t += 1
            self._e_cache = None
            if learn and self.rewire_every and self.t % self.rewire_every == 0:
                self.rewire()
        z = 0.0 if not self.gate else 1.0
        return rep * z, err * z, p[:self.n_ch]

    @torch.no_grad()
    def rewire(self):
        """資源（受け手ごとの総量）と順位（一番弱い1本）。予備からランダムに開通する"""
        target = self.n_partner * 0.5
        self.budget = (self.budget + 0.5 * torch.sign(target - self.recv_ema)).clamp(1, self.n_partner * 3)
        # 受け手ごとに最小の1本を切り、送り手を振り直して並べ直す
        order = torch.argsort(self.e_recv)
        w_ord, r_ord = self.w[order], self.e_recv[order].long()
        big = self.w.abs().max() + 1.0
        key = r_ord.float() * big + w_ord.abs()
        first = torch.zeros(self.n_recv, dtype=torch.long, device=self.dev)
        srt = torch.argsort(key)
        seen = torch.zeros(self.n_recv, dtype=torch.bool, device=self.dev)
        for i in srt.tolist():                      # 受け手ごとの最初＝最小
            rr = int(r_ord[i])
            if not seen[rr]:
                seen[rr] = True
                first[rr] = order[i]
        cuts = first[seen]
        self.e_src_sorted[cuts] = torch.randint(0, self.n_send, (len(cuts),),
                                                device=self.dev, generator=self.gen)
        self.w[cuts] = 0.0
        o = torch.argsort(self.e_src_sorted)
        self.e_src_sorted = self.e_src_sorted[o]
        self.e_recv, self.w = self.e_recv[o], self.w[o]
        self.e_inh = self.is_inh_send[self.e_src_sorted]
        self.w[self.e_inh] = W_INH                 # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        if self.stp is not None:
            self.stp.reorder(o)
        self.start = torch.zeros(self.n_send + 1, dtype=torch.long, device=self.dev)
        self.start.scatter_add_(0, self.e_src_sorted + 1,
                                torch.ones(len(self.w), dtype=torch.long, device=self.dev))
        self.start = self.start.cumsum(0)


def selftest(stp_path="synapse_model_human.json", dev="cuda"):
    """arkhe で最初に通すもの。numpy 版と同じ形で動くか"""
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    prior = Prior(dev=dev)
    a = Area(n_ch=40, prior=prior, n_ext=20, n_partner=500, stp_path=stp_path, dev=dev)
    x = (torch.rand(40, device=dev) < 0.05).float()
    e = torch.zeros(20, device=dev)
    import time
    for _ in range(3):
        a.step(x, e)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        a.step(x, e)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    print(f"体 {a.n_cells:,}  辺 {len(a.w):,}  1歩 {ms:.2f}ms  "
          f"（1歩 {DT_MS}ms に{'間に合う' if ms < DT_MS else f'{ms/DT_MS:.1f}倍おそい'}）")
    print(f"VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")
