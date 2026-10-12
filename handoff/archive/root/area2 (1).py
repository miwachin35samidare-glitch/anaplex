"""領野 v2 —— 過去を運ぶ体（人の時間細胞の形）＋ シナプスの成長（資源・順位・予備・束）

ニューロン   表現 n_ch ／ 誤差 n_ch ／ 時間細胞 (n_ch + n_ext) × n_cell
シナプス     誤差のニューロンが、時間細胞と短い遅れから受ける。疎（相手を選んで持つ）
成長
  資源  受け手ごとに持てる総量 T を恒常性で決める（共通の固定値にしない）
        T は「自分がどれだけ受け取っているか」で動く：受け取りすぎなら減り、足りなければ増える
        持てる本数 ＝ T ÷ 1本あたりの取り分 ⇒ 受け手ごとに違う
  順位  その中で一番弱い1本（束）から消える。絶対閾値を持たない
  予備  休止中の候補プール（強さ0・総数の外）。刈った跡は予備からランダムに1本開通する
  束    1人の相手に接触 contacts 本。生えるのも刈られるのも束ごと
  順序  消えるのが先、生えるのが後
"""
import numpy as np
from element1 import Prior, TAUS, DECAY, THETA, sigmoid
from chains import TimeCells
from stp import STP


class Area2:
    def __init__(self, n_ch, prior, n_ext=0, B=1, L=32, M=16, D=3, eta=0.03,
                 n_partner=20, pool_mult=3, rewire_every=100,
                 homeo_tau=2000.0, seed=0, n_cell=32, width=3, inh_frac=0.16, stp=True):
        self.n_ch, self.n_ext, self.B, self.L, self.D = n_ch, n_ext, B, L, D
        self.eta, self.rewire_every = eta, rewire_every
        self.prior, self.homeo_tau = prior, homeo_tau
        self.scaling_target = 0.855      # 受け手の総コンダクタンス G_FIRE（人から。生-249）
        self.n_src = n_ch + n_ext
        self.ch = TimeCells(self.n_src, L=L, n_cell=n_cell, width=width, M=M, B=B)
        self.n_send = self.ch.n_cells + self.n_src * D
        self.n_partner = n_partner                      # 繋がっている相手（＝総数の中）
        self.n_pool = int(n_partner * pool_mult)        # 予備込みの枠。超えた分は休止中
        rng = np.random.default_rng(seed)
        self.idx = np.stack([np.stack([rng.choice(self.n_send, self.n_pool, replace=False)
                                       for _ in range(n_ch)]) for _ in range(B)])
        self.w = np.zeros((B, n_ch, self.n_pool), np.float32)
        self.live = np.zeros((B, n_ch, self.n_pool), bool)
        self.live[:, :, :n_partner] = True              # 先頭 n_partner 本だけが繋がっている
        self.budget = np.full((B, n_ch), float(n_partner), np.float32)   # 資源（持てる相手の数）
        self.recv = np.zeros((B, n_ch), np.float32)     # 受け取っている量（恒常性の観測）
        self.tb = np.zeros((B, n_ch, len(TAUS)), np.float32)
        self.buf = np.zeros((B, self.n_src, D), np.float32)
        self.is_inh = rng.random(self.n_send) < inh_frac   # 抑制性の送り手（人：シナプス数の E:I 84:16）
        # 短期可塑性：状態は接続ごと（束で1つ）。受け手（誤差のニューロン）は興奮性とする
        self.contacts = np.where(self.is_inh[self.idx], 4.0, 3.3).astype(np.float32)  # 人の実測
        self.stp = STP(self.w.shape, self.is_inh[self.idx],
                       np.zeros(self.w.shape, bool), rng) if stp else None
        self.scale = np.ones((B, n_ch), np.float32)      # シナプススケーリング（強さの恒常性）
        self.drive = np.zeros((B, n_ch), np.float32)
        self.rng, self.t = rng, 0

    @property
    def n_cells(self):
        return self.n_ch * 2 + self.ch.n_cells

    @property
    def n_syn(self):
        return int(self.live.sum() * self.contacts)

    def senders(self):
        return np.concatenate([self.ch.spikes(), self.buf.reshape(self.B, -1)], 1)

    def step(self, x, ext=None, learn=True):
        s = self.senders()
        got = s[np.arange(self.B)[:, None, None], self.idx] * self.live
        if self.stp is not None:
            got = got * self.stp.release(got)          # 枯れと増強（人の当てはめそのまま）
        inh = self.is_inh[self.idx]
        lg, _ = self.prior.logit(self.tb)
        exc_sum = (self.contacts * self.w * got * ~inh).sum(2) * self.scale
        inh_sum = (self.contacts * self.w * got * inh).sum(2) * self.scale
        p = sigmoid(lg + exc_sum / (1.0 + inh_sum))          # 抑制は分母に入る（短絡）。
        # 薄まるのはシナプスの入力だけで、素子自身の見込み（＝静止の側）は薄まらない
        e = x - p
        err = (e > THETA).astype(np.float32)
        if learn:
            # 興奮も抑制も同じ誤差から学ぶ（前の作りと同じ）。違うのは短絡の効き方の分だけ
            d_e = self.eta * e[..., None] * got
            shunt = (exc_sum / (1.0 + inh_sum) ** 2)[..., None]   # 抑制は「薄める効き」の分だけ動く
            d_i = -self.eta * e[..., None] * got * shunt
            self.w += np.where(inh, d_i, d_e)
            self.w[inh] = np.maximum(self.w[inh], 0.0)       # 符号の射影（抑制は 0 以上）
            a = 1.0 / self.homeo_tau
            self.recv += a * (got.sum(2) - self.recv)               # 受け取っている量の平均（本数の恒常性）
            self.drive += a * (np.abs(exc_sum) - self.drive)        # 受け取っている強さの平均
            # シナプススケーリング：受け手ごとに入辺を一括で乗算して総量を保つ（人にある機構）
            self.scale *= np.clip(1 - 0.01 * (self.drive / self.scaling_target - 1), 0.5, 1.5) ** a
        rep = x.astype(np.float32)
        self.tb = self.tb * DECAY + rep[..., None] * (1 - DECAY)
        src = rep if ext is None else np.concatenate([rep, ext.astype(np.float32)], 1)
        self.ch.step(src)
        self.buf = np.concatenate([src[..., None], self.buf[..., :-1]], -1)
        self.t += 1
        if learn and self.rewire_every and self.t % self.rewire_every == 0:
            self.rewire()
        return rep, err, p

    def rewire(self):
        """資源で総量を決め、順位で一番弱い束を切り、予備からランダムに1束生やす"""
        target = self.n_partner * 0.5                              # 受け取りの目標（人の値ではない。ノブ）
        self.budget = np.clip(self.budget + 0.5 * np.sign(target - self.recv), 1, self.n_pool)
        for b in range(self.B):
            for i in range(self.n_ch):
                live = np.nonzero(self.live[b, i])[0]
                want = int(round(self.budget[b, i]))
                # 消えるのが先：総量を超えている分＋最弱の1本
                n_cut = max(len(live) - want, 0) + (1 if len(live) >= want else 0)
                if n_cut:
                    weak = live[np.argsort(np.abs(self.w[b, i, live]))[:n_cut]]
                    self.live[b, i, weak] = False
                    self.w[b, i, weak] = 0.0
                # 生えるのが後：予備（休止中）からランダムに、総量に届くまで
                rest = np.nonzero(~self.live[b, i])[0]
                n_grow = min(want - int(self.live[b, i].sum()), len(rest))
                if n_grow > 0:
                    pick = self.rng.choice(rest, n_grow, replace=False)
                    self.live[b, i, pick] = True
                    self.idx[b, i, pick] = self.rng.integers(0, self.n_send, n_grow)
