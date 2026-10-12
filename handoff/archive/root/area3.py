"""領野 v3 —— 表現のニューロンも学ぶ ／ 抑制性を体にする ／ 傾きを持つ細胞（TimeCells の中）

表現のニューロン（人：予測符号化の表現の側）
  これまで  来た入力をそのまま撃つだけ。学ばない
  これから  自分も予測を持ち、自分のシナプスを自分の誤差で変える。
            撃つのは「来たもの」のまま（表現なので黙らない）。学ぶのは予測の側だけ
            ⇒ この予測が、下の段へ送る「上からの予測」になる

抑制性の体（人：抑制性ニューロンは自分で入力を受けて撃つ。札ではない）
  これまで  送り手に「抑制性」の札を付けていただけ
  これから  抑制性のニューロンを n_inh 体置く。領野の中の発火を受けて撃ち、
            その出力が誤差・表現のニューロンの分母に入る（短絡）
            割合は人：シナプス数の E:I ＝ 84:16
"""
import numpy as np
from element1 import Prior, TAUS, DECAY, THETA, sigmoid
from chains import TimeCells
from stp import STP


class Area3:
    def __init__(self, n_ch, prior, n_ext=0, B=1, L=32, M=16, D=3, eta=0.03,
                 n_partner=20, pool_mult=3, rewire_every=100, homeo_tau=2000.0,
                 seed=0, n_cell=32, width=3, inh_frac=0.16, stp=True, tau_scale=1.0, gate=True,
                 spont_hz=0.3, complete=False):
        self.n_ch, self.n_ext, self.B, self.D = n_ch, n_ext, B, D
        self.eta, self.rewire_every, self.homeo_tau = eta, rewire_every, homeo_tau
        self.tau_scale = tau_scale        # 段の時定数（一次 1.0 ／ 連合 1.9 ／ 高次 3.2〜4.8）
        self.decay = np.exp(-1.0 / (TAUS * tau_scale)).astype(np.float32)
        self.gate = gate                  # まぶた：False なら外に何も出さない（場所取りだけ）
        # 自発活動：入力が無くても撃つ。人では育つときも大人でも残る
        #   育ち：活動の多い体が残る ／ 大人：内側から予測を立てる側
        #   ヒト皮質の発火率 0.3Hz（前の作りの決-768）。1歩 10.67ms なら 0.32%/歩
        self.spont_p = spont_hz * 0.01067
        # パターン完成：細胞体に入力が来ないとき、表現のニューロンは自分の予測で撃つ
        #   （Urbanczik・Senn の2区画の形。入力があれば起きたこと、無ければ来るはずだったものを運ぶ）
        self.complete = complete
        self.pred_boost = 0.9        # 予測だけでは 1.0 に届かない（撃たせず、撃ちやすくするだけ）
        self.prior = prior
        self.n_inh = max(1, int(round(n_ch * inh_frac / (1 - inh_frac))))   # 抑制性の体
        self.n_src = n_ch + n_ext + self.n_inh
        self.ch = TimeCells(self.n_src, L=L, n_cell=n_cell, width=width, M=M, B=B, seed=seed)
        self.n_send = self.ch.n_cells + self.n_src * D
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.scaling_target = 0.855
        # 受け手は3種：誤差のニューロン ／ 表現のニューロン（予測の側）／ 抑制性のニューロン
        self.n_recv = n_ch * 2 + self.n_inh
        self.n_pool = int(n_partner * pool_mult)
        self.n_partner = n_partner
        self.idx = np.stack([np.stack([rng.choice(self.n_send, self.n_pool, replace=False)
                                       for _ in range(self.n_recv)]) for _ in range(B)])
        self.w = np.zeros((B, self.n_recv, self.n_pool), np.float32)
        self.live = np.zeros((B, self.n_recv, self.n_pool), bool)
        self.live[:, :, :n_partner] = True
        self.budget = np.full((B, self.n_recv), float(n_partner), np.float32)
        self.recv_ema = np.zeros((B, self.n_recv), np.float32)
        self.drive = np.zeros((B, self.n_recv), np.float32)
        self.scale = np.ones((B, self.n_recv), np.float32)
        self.tb = np.zeros((B, self.n_recv, len(TAUS)), np.float32)
        self.buf = np.zeros((B, self.n_src, D), np.float32)
        self.inh_spk = np.zeros((B, self.n_inh), np.float32)
        # 送り手が抑制性か（＝抑制性の体から出たものか）
        src_inh = np.zeros(self.n_src, bool)
        src_inh[n_ch + n_ext:] = True
        self.is_inh = np.concatenate([np.repeat(src_inh, self.ch.n_cell + self.ch.n_ramp),
                                      np.repeat(src_inh, D)])
        self.contacts = np.where(self.is_inh[self.idx], 4.0, 3.3).astype(np.float32)
        recv_inh = np.zeros(self.n_recv, bool); recv_inh[n_ch * 2:] = True
        self.stp = STP(self.w.shape, self.is_inh[self.idx],
                       np.broadcast_to(recv_inh[None, :, None], self.w.shape).copy(), rng) if stp else None
        self.t = 0

    @property
    def n_cells(self):
        return self.n_ch * 2 + self.n_inh + self.ch.n_cells

    def step(self, x, ext=None, learn=True):
        s = np.concatenate([self.ch.spikes(), self.buf.reshape(self.B, -1)], 1)
        got = s[np.arange(self.B)[:, None, None], self.idx] * self.live
        if self.stp is not None:
            got = got * self.stp.release(got)
        inh = self.is_inh[self.idx]
        lg, _ = self.prior.logit(self.tb)
        exc = (self.contacts * self.w * got * ~inh).sum(2) * self.scale
        ihn = (self.contacts * self.w * got * inh).sum(2) * self.scale
        p = sigmoid(lg + exc / (1.0 + ihn))
        # 的：誤差と表現は自分の入口 x、抑制性は領野の発火の混み具合
        tgt = np.concatenate([x, x, np.repeat(x.mean(1, keepdims=True), self.n_inh, 1)], 1)
        e = tgt - p
        if learn:
            d_e = self.eta * e[..., None] * got
            shunt = (exc / (1.0 + ihn) ** 2)[..., None]
            self.w += np.where(inh, -self.eta * e[..., None] * got * shunt, d_e)
            self.w[inh] = np.maximum(self.w[inh], 0.0)
            a = 1.0 / self.homeo_tau
            self.recv_ema += a * (got.sum(2) - self.recv_ema)
            self.drive += a * (np.abs(exc) - self.drive)
            self.scale *= np.clip(1 - 0.01 * (self.drive / self.scaling_target - 1), 0.5, 1.5) ** a
        err = (e[:, :self.n_ch] > THETA).astype(np.float32)
        self.last_x, self.last_err = x, err          # 段ごとの当て損ねを見るため        # 誤差のニューロン：外れたら撃つ
        rep = x.astype(np.float32)                                  # 表現のニューロン：来たもので撃つ
        if self.complete:
            # 人の形：樹状突起の予測は、それだけでは細胞体を撃たせない。撃ちやすさを変えるだけ。
            #   遠い樹状突起で検出されたパターンは発火を起こさず、少しだけ電位を上げる
            #   ⇒ 入力の合計が閾値に届かなくても、予測が高ければ届く。入力が全く無ければ撃たない
            drive = x.astype(np.float32) + self.pred_boost * p[:, self.n_ch:self.n_ch * 2]
            near = (x.sum(1, keepdims=True) > 0)          # 何かは来ている（全くの無入力では撃たない）
            rep = np.maximum(rep, ((drive > 1.0) & near).astype(np.float32))
        if self.spont_p > 0:                                        # 自発活動（入力が無くても撃つ）
            rep = np.maximum(rep, (self.rng.random(rep.shape) < self.spont_p).astype(np.float32))
        self.inh_spk = (p[:, self.n_ch * 2:] > 0.5).astype(np.float32)   # 抑制性：自分で受けて撃つ
        self.pred_rep = p[:, self.n_ch:self.n_ch * 2]               # 表現の側の予測（上から下へ送るもの）
        self.tb = self.tb * self.decay + np.concatenate(
            [rep, rep, self.inh_spk], 1)[..., None] * (1 - self.decay)
        src = [rep] + ([] if ext is None else [ext.astype(np.float32)]) + [self.inh_spk]
        src = np.concatenate(src, 1)
        self.ch.step(src)
        self.buf = np.concatenate([src[..., None], self.buf[..., :-1]], -1)
        self.t += 1
        if learn and self.rewire_every and self.t % self.rewire_every == 0:
            self.rewire()
        if not self.gate:                 # まぶたを閉じている間は外に出さない
            return rep * 0, err * 0, p[:, :self.n_ch]
        return rep, err, p[:, :self.n_ch]

    def rewire(self):
        target = self.n_partner * 0.5
        self.budget = np.clip(self.budget + 0.5 * np.sign(target - self.recv_ema), 1, self.n_pool)
        for b in range(self.B):
            for i in range(self.n_recv):
                live = np.nonzero(self.live[b, i])[0]
                want = int(round(self.budget[b, i]))
                n_cut = max(len(live) - want, 0) + (1 if len(live) >= want else 0)
                if n_cut:
                    weak = live[np.argsort(np.abs(self.w[b, i, live]))[:n_cut]]
                    self.live[b, i, weak] = False; self.w[b, i, weak] = 0.0
                rest = np.nonzero(~self.live[b, i])[0]
                n_grow = min(want - int(self.live[b, i].sum()), len(rest))
                if n_grow > 0:
                    pick = self.rng.choice(rest, n_grow, replace=False)
                    self.live[b, i, pick] = True
                    self.idx[b, i, pick] = self.rng.integers(0, self.n_send, n_grow)
