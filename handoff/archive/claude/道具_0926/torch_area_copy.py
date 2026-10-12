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
import numpy as np
import torch

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


class Area:
    """1つの領野。表現・誤差・抑制のニューロンと、疎なシナプス
    ⚠️ 切り分け用の写し：Group2 に渡すための組み立て（__init__）だけを写した"""

    def __init__(self, n_ch, prior, n_ext=0, eta=0.03, n_partner=200, pool_mult=3,
                 rewire_every=None, homeo_tau=2000.0, n_cell=128, n_ramp=8,
                 inh_frac=0.16, stp_path=None, tau_scale=1.0, gate=True,
                 spont_hz=0.3, complete=False, motor=False, dev="cuda", seed=0,
                 w_int8=False, fire_th=1.0, x_gain=1.0, tgt_by_fire=False, contacts_exc=3.3):
        g = torch.Generator(device=dev); g.manual_seed(seed)
        self.dev, self.gen, self.prior = dev, g, prior
        self.n_ch, self.n_ext, self.eta, self.homeo_tau = n_ch, n_ext, eta, homeo_tau
        self.tau_scale, self.gate, self.complete, self.motor = tau_scale, gate, complete, motor
        self.spont_p = spont_hz * DT_MS / 1000.0
        self.pred_boost, self.w_int8 = 0.9, w_int8
        self.scaling_target = RATE_HUMAN
        self.fire_th = fire_th
        self.x_gain = x_gain
        self.tgt_by_fire = tgt_by_fire
        self.contacts_exc = contacts_exc
        self.recur = RECUR
        self.D = 8 if motor else steps(SYN_DELAY_MS * 3)
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
        is_inh_send = torch.zeros(self.n_send, dtype=torch.bool, device=dev)
        inh_from = (n_ch + n_ext) * (self.ch.n_cell + self.ch.n_ramp)
        is_inh_send[inh_from:self.ch.n_cells] = True
        is_inh_send[self.ch.n_cells + (n_ch + n_ext) * self.D:] = True
        inh_pool = torch.nonzero(is_inh_send).squeeze(1)
        exc_pool = torch.nonzero(~is_inh_send).squeeze(1)
        k_inh = int(round(n_partner * inh_frac))
        src = torch.empty((self.n_recv, n_partner), dtype=torch.long, device=dev)
        src[:, :k_inh] = inh_pool[torch.randint(0, len(inh_pool), (self.n_recv, k_inh),
                                                device=dev, generator=g)]
        src[:, k_inh:] = exc_pool[torch.randint(0, len(exc_pool), (self.n_recv, n_partner - k_inh),
                                                device=dev, generator=g)]
        src = src.reshape(-1)
        recv = torch.arange(self.n_recv, device=dev).repeat_interleave(n_partner)
        o = torch.argsort(src)
        self.e_src_sorted = src[o]
        self.e_recv = recv[o].int()
        self.w = torch.zeros(len(o), device=dev)
        self.start = torch.zeros(self.n_send + 1, dtype=torch.long, device=dev)
        self.start.scatter_add_(0, self.e_src_sorted + 1,
                                torch.ones(len(o), dtype=torch.long, device=dev))
        self.start = self.start.cumsum(0)
        self.is_inh_send = is_inh_send
        self.e_inh = is_inh_send[self.e_src_sorted]
        self.stp = None
        self.tb = torch.zeros(self.n_recv, len(TAUS), device=dev)
        self.buf = torch.zeros(self.n_src, self.D, device=dev)
        self.inh_spk = torch.zeros(self.n_inh, device=dev)
        self.scale = torch.ones(self.n_recv, device=dev)
        self.recv_ema = torch.zeros(self.n_recv, device=dev)
        self.drive = torch.zeros(self.n_recv, device=dev)
        self.budget = torch.full((self.n_recv,), float(n_partner), device=dev)
        self.t = 0
        self._e_cache = None

    @property
    def n_cells(self):
        return self.n_ch * 2 + self.n_inh + self.ch.n_cells
