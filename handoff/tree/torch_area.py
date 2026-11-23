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
import os

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
# 【ヒト】起きている皮質の錐体 3.30 ± 2.52Hz（Verma 2024、覚醒・リップルの外、Utah II/III層、3人・437体）⇒ 1歩に 3.52%
#   ⟲ 版52 までは 0.45%/歩。前の作りの 1.5Hz（【ラット・サル】）× 1歩 3ms が、1歩 10.67ms になっても残った数（約 0.42Hz）
#     版45b（10-02【み】「間違ってるなら直して」）の直しが arkhe に入らないまま版が進んでいた（10-08 grep）。版53 で入れた
RATE_HUMAN = 3.30 * DT_MS / 1000.0
HUMAN_INPUT_PER_STEP = 30000 * RATE_HUMAN   # 毎歩 受けている入力の本数（30,000シナプス × RATE_HUMAN）。どこからも使っていない
#   シナプススケーリングの目標。【ヒト】iPS 由来の皮質細胞で、活動電位を止めると興奮性シナプス全体が
#   強まって活動が戻る（Cordella 2022）⇒ 感じているのは細胞自身の活動、動かすのはシナプスの強さ全体
# ✗ G_FIRE 0.855 は使わない。コンダクタンスで動く素子の値（前の作り）で、このラインの素子
#   （重みはロジットに足される量）には持ち越せない。落-039（連続版の 0.53 を持ち込んで暴走）と同じ形


# 版103d：放出のくじ（二項分布）を、一様乱数1つで二項分布の累積を引く形に（GPU のとき）【み】11-01 16:43「乱数なんだから変わるでしょ」
#   分布は同じ（N は表の 1〜32 ⇒ 32回までの足し上げ）。出目は torch.binomial と違う。CPU と ANAPLEX_BINOM=torch では今まで通り
#   【測】measure_prof103（版103b）：aten::binomial 1歩 6.88ms（ROCm）
BINOM_NMAX = 32


def _binom_core(n, p, u):
    """二項分布 B(n, p) を一様乱数 u で（累積が u を越える所まで数える）"""
    flip = p > 0.5                                   # p > 0.5 は B(n, 1 − p) を引いて n − k（q ** n が float32 で 0 に落ちないように）
    ps = torch.where(flip, 1.0 - p, p).clamp(0.0, 0.5)
    q = 1.0 - ps
    r = ps / q
    pk = q ** n
    cdf = pk
    k = torch.zeros_like(n)
    for i in range(BINOM_NMAX):
        k = k + ((u > cdf) & (n > i)).to(n.dtype)
        pk = pk * (n - i) / (i + 1) * r
        cdf = cdf + pk
    return torch.where(flip, n - k, k)


def binom(cn, P, gen):
    """版103d：torch.binomial(cn, P, generator=gen) と同じ分布"""
    if cn.device.type != "cuda" or os.environ.get("ANAPLEX_BINOM") == "torch":
        return torch.binomial(cn, P, generator=gen)
    u = torch.rand(cn.shape, device=cn.device, generator=gen)
    from torch_group2 import _run
    return _run(_binom_core, cn, P, u)


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


def _rel_pre(e, row, last, deplete, state, fac, dec_d, dec_f, tbl, Nt, t):
    """版78：放出の前半（触った辺だけ、最後に触った歩からまとめて減らす）。式は版77 の release_at と同じ"""
    r = row[e].long()
    n = (t - last[e]).float()
    dep = deplete[e]
    st = state[e] * torch.where(dep, torch.ones_like(n), dec_d[r] ** n)
    fc = fac[e] * dec_f[r] ** n
    Pr0 = tbl[r, 0]
    P = (Pr0 + (1 - Pr0) * fc).clamp(0, 1)
    N = Nt[r]
    avail = torch.where(dep, st, 1 - st)
    return r, dep, st, fc, Pr0, P, N, (N * avail).round().clamp(min=0)


def _rel_post(cnt, r, dep, st, fc, Pr0, P, N, tbl):
    """版78：放出の後半（出た量・状態の更新）"""
    out = cnt / (N * Pr0).clamp(min=1e-6)
    k = torch.where(dep, P, tbl[r, 1])
    st2 = torch.where(dep, st * (1 - k), st + (1 - st) * k)
    fc2 = fc + (1 - fc) * tbl[r, 3]
    return out, st2, fc2


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
        self.grp_off = [off[k] for k in keys]          # 版53：生やした辺の行を引き直すため
        self.grp_len = [len(d["行"][k]) for k in keys]

    @torch.no_grad()
    def redraw(self, idx, send_inh, recv_inh):
        """版53：生やした辺の行を、新しい種類の組から引き直して状態を初めに戻す"""
        grp = send_inh.long() * 2 + recv_inh.long()
        for gi in range(4):
            m = grp == gi
            n = int(m.sum())
            if n:
                self.row[idx[m]] = (self.grp_off[gi] + torch.randint(0, self.grp_len[gi], (n,), device=idx.device,
                                                                     generator=self.gen)).to(self.row.dtype)
        dep = self.tbl[self.row[idx].long(), 1] < 0
        self.deplete[idx] = dep
        self.state[idx] = dep.float()
        self.fac[idx] = 0.0

    @torch.no_grad()
    def release_at(self, e, t):
        """触った辺だけ、最後に触った歩からまとめて減らして放出する"""
        # 版78：前半・後半を torch.compile で（くじ binomial は外で。式・くじの順は版77 と同じ）
        from torch_group2 import _run
        r, dep, st, fc, Pr0, P, N, cn = _run(_rel_pre, e, self.row, self.last, self.deplete, self.state, self.fac,
                                             self.dec_d, self.dec_f, self.tbl, self.N, t)
        self.last.index_fill_(0, e, t)     # 版54b
        cnt = binom(cn, P, self.gen)   # 版103d
        out, st2, fc2 = _run(_rel_post, cnt, r, dep, st, fc, Pr0, P, N, self.tbl)
        self.state[e] = st2
        self.fac[e] = fc2
        return out

    def reorder(self, o):
        # 版92：同じ置き場所へ1本ずつ写す（⟲ 版91 まで5本の新しい写しを全部作ってから入れ替え ⇒ 刈り込みの山）
        for t in (self.row, self.state, self.fac, self.last, self.deplete):
            t.copy_(t[o])


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
                 inh_n_exc=None, relay_exc=None, inh_exc_from_rep=False, inh_slice=None, n_om=0,
                 rr_k=0, rr_seed=0, rr_w=None):
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
        # 版98：inh_frac 0 ⇒ 抑制の体なし（プルキンエ・分子層：抑制は外の領野から【ヒト】の作り）
        self.n_inh = max(1, int(round(n_ch * inh_frac / (1 - inh_frac)))) if inh_frac > 0 else 0
        self.n_src = n_ch + n_ext + self.n_inh
        self.ch = TimeCells(self.n_src, n_cell=(0 if motor else n_cell), n_ramp=n_ramp,
                            dev=dev, gen=g)
        self.n_send = self.ch.n_cells + self.n_src * self.D
        # 版101：来ないことに撃つ体（組み表_版101）。行は抑制の体のうしろに n_om 体（帯ごとに1体。誤差の体と同じ相手の引き方）
        self.n_om = n_om
        self.n_recv = n_ch * 2 + self.n_inh + n_om
        n_partner = min(n_partner, self.n_send)
        # 版51：視床の中継の細胞が自分の領野の中から受ける興奮の本数（relay_exc。ノブ【C】）。抑制性の相手の数は変えない
        self.n_inh_partner_rows = int(round(n_partner * inh_frac))
        if relay_exc is not None:
            n_partner = self.n_inh_partner_rows + relay_exc
        self.no_lat_exc = relay_exc == 0     # 版61：中継の表現・誤差の行は自分の領野の中から興奮を受けない
        # 版64：表現・誤差の行の抑制の相手を、抑制の体の1歩前の列から重なりなしで（切片の見方）
        self.inh_slice = inh_slice is not None
        k_rep_inh = self.n_inh_partner_rows
        if self.inh_slice:
            k_rep_inh = max(1, int(round(self.n_inh * inh_slice)))
            n_partner = k_rep_inh + (n_partner - self.n_inh_partner_rows)
            self.n_inh_partner_rows = k_rep_inh
        self.n_partner = n_partner
        # 版47：抑制の体の相手は 興奮性 INH_N_EXC ＋ 抑制性（inh_frac を保つ）
        n_ie = INH_N_EXC if inh_n_exc is None else inh_n_exc
        n_pi = min(int(round(n_ie / (1 - inh_frac))), self.n_send)
        self.n_partner_inh = n_pi
        self.n_partner_vec = torch.cat([torch.full((n_ch * 2,), float(n_partner), device=dev),
                                        torch.full((self.n_inh,), float(n_pi), device=dev),
                                        torch.full((n_om,), float(n_partner), device=dev)])   # 版101
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
        # 版57：海馬の抑制の体は 194 の前提（相手は錐体）どおり、遅れの列の興奮の行（時間細胞ではない）から引く
        self.inh_exc_from_rep = inh_exc_from_rep
        exc_pool_rep = exc_pool[exc_pool >= self.ch.n_cells]
        def pick(n_rows, k_all, k_inh=None, pool=None):
            pool = exc_pool if pool is None else pool
            k_inh = int(round(k_all * inh_frac)) if k_inh is None else k_inh
            s = torch.empty((n_rows, k_all), dtype=torch.long, device=dev)
            if k_inh > 0:                          # 版98：抑制の体が無い領野（プルキンエ・分子層）は引かない
                s[:, :k_inh] = inh_pool[torch.randint(0, len(inh_pool), (n_rows, k_inh),
                                                      device=dev, generator=g)]
            if k_all - k_inh > 0:
                s[:, k_inh:] = pool[torch.randint(0, len(pool), (n_rows, k_all - k_inh),
                                                      device=dev, generator=g)]
            return s.reshape(-1)
        # 表現・誤差は n_partner、抑制の体は n_pi（版47）
        def pick_rows(n_rows):                      # 表現・誤差（版101：来ない体も）の行。版64 の抑制の引き直しも同じ
            sr = pick(n_rows, n_partner, self.n_inh_partner_rows if (relay_exc is not None or self.inh_slice) else None)
            if self.inh_slice and n_rows:          # 版64：抑制の分を、抑制の体の1歩前の列から重なりなしで引き直す
                col0 = self.ch.n_cells + (n_ch + n_ext + torch.arange(self.n_inh, device=dev)) * self.D
                sel = torch.rand((n_rows, self.n_inh), device=dev, generator=g).argsort(1)[:, :k_rep_inh]
                sr = sr.view(n_rows, n_partner)
                sr[:, :k_rep_inh] = col0[sel]
                sr = sr.reshape(-1)
            return sr
        src_rep = pick_rows(n_ch * 2)
        # 版113：皮質の 表現 → 表現（同じ領野の錐体どうし）の本数を人の値に（組み表_版113_案 §13【み】11-19 12:33）
        #   表現の行の興奮の相手のうち rr_k 本を、自分以外の表現の体（1歩前の列）から重なりなしで。残りは自分の表現の列を除いた興奮から
        #   rr_k ＝ round(0.158 ×（体数 − 1））【ヒト】Planert 2025（側頭葉 L2/3 の対）。乱数は別の生成器 ⇒ rr_k 0 なら版112 と同じ網
        self.rr_k = int(rr_k)
        if self.rr_k > 0:
            gr = torch.Generator(device=dev); gr.manual_seed(int(rr_seed))
            c0 = self.ch.n_cells
            own = c0 + torch.arange(n_ch, device=dev) * self.D
            is_own = torch.zeros(self.n_send, dtype=torch.bool, device=dev); is_own[own] = True
            pool_x = exc_pool[~is_own[exc_pool]]
            k_i = self.n_inh_partner_rows if (relay_exc is not None or self.inh_slice) else int(round(n_partner * inh_frac))
            n_ex = n_partner - k_i
            assert self.rr_k <= min(n_ch - 1, n_ex), (self.rr_k, n_ch, n_ex)
            rows_ = src_rep.view(n_ch * 2, n_partner)[n_ch:]          # 表現の行（誤差の行のうしろ）
            key = torch.rand((n_ch, n_ch), device=dev, generator=gr)
            key[torch.arange(n_ch, device=dev), torch.arange(n_ch, device=dev)] = 2.0   # 自分は引かない
            rows_[:, k_i:k_i + self.rr_k] = own[key.argsort(1)[:, :self.rr_k]]
            if n_ex - self.rr_k > 0:
                rows_[:, k_i + self.rr_k:] = pool_x[torch.randint(0, len(pool_x), (n_ch, n_ex - self.rr_k), device=dev, generator=gr)]
        src = torch.cat([src_rep,
                         pick(self.n_inh, n_pi, pool=(exc_pool_rep if inh_exc_from_rep else None))])
        recv = torch.cat([torch.arange(n_ch * 2, device=dev).repeat_interleave(n_partner),
                          (torch.arange(self.n_inh, device=dev) + n_ch * 2).repeat_interleave(n_pi)])
        if n_om:                                   # 版101：来ない体の行（抑制の体のうしろ）。乱数は上の行を引いたあと（版100 の行は同じ）
            src = torch.cat([src, pick_rows(n_om)])
            recv = torch.cat([recv, (torch.arange(n_om, device=dev) + n_ch * 2 + self.n_inh).repeat_interleave(n_partner)])
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
        if self.rr_k > 0:                          # 版113：表現 → 表現 の初めの重さは人の分布（下からの線と同じ引き方）÷ 接触
            es_ = self.e_src_sorted; er_ = self.e_recv.long()
            m_rr = (er_ >= n_ch) & (er_ < 2 * n_ch) & (es_ >= c0) & (es_ < c0 + n_ch * self.D) & ((es_ - c0) % self.D == 0)
            assert int(m_rr.sum()) == n_ch * self.rr_k
            self.w[m_rr] = rr_w(n_ch * self.rr_k, gr) / contacts_exc
        recv_inh = torch.zeros(self.n_recv, dtype=torch.bool, device=dev)
        recv_inh[n_ch * 2:n_ch * 2 + self.n_inh] = True   # 版101：来ない体の行は興奮
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
        return self.n_ch * 2 + self.n_inh + self.n_om + self.ch.n_cells

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
