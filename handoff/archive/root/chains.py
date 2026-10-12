"""過去を運ぶ体 —— 人の時間細胞の形（Umbach ほか 2020、ヒト海馬・嗅内皮質の単一ニューロン）

人で分かっていること
  ・出来事の始まりからの決まった時間の窓で撃つ。窓は細胞ごとに違い、互いに重なる
  ・集団として、その期間の全体を覆う
  ・候補 509体のうち 79体（15.5%）が時間細胞
  ⇒ 1歩に1体ではなく、同じ「k歩前」を複数の体が重なって表す

これまでの代役（16本の並びを順に使い回す）との違い
  同じ k が、どの並びに入ったかで別の体になっていた ⇒ 1体あたりの学ぶ回数が 1/16 に薄まった
  人の形では、体は「自分の窓に入る出来事があれば撃つ」ので、どの出来事でも同じ体が撃つ
"""
import numpy as np
from timing import time_fields, steps, CHAIN_SPAN_MS, RAMP_SPAN_MS


class TimeCells:
    def __init__(self, n_src, L=None, n_cell=32, width=None, M=16, B=1, n_ramp=8,
                 ramp_len=None, ramp_p=0.3, seed=0, span_ms=CHAIN_SPAN_MS):
        self.n_src, self.M, self.B = n_src, M, B
        self.n_cell = n_cell
        # 人（Umbach ほか 2020）：受容野は覆う期間に合わせて伸び縮みし、
        #   遅れが大きい体ほど広い（対数圧縮）。一定の幅ではない
        self.pref, w = time_fields(span_ms, n_cell) if n_cell else (np.zeros(0, np.float32), np.zeros(0, np.float32))
        self.L = int(np.ceil(self.pref.max())) if n_cell else 1
        self.half = w / 2.0
        self.n_ramp = n_ramp
        self.ramp_len = steps(RAMP_SPAN_MS) if ramp_len is None else ramp_len
        self.ramp_p = ramp_p
        self.rng = np.random.default_rng(seed)
        self.age = np.full((B, n_src, M), 10 ** 9, np.int64)
        self._cache = None

    @property
    def n_cells(self):
        return self.n_src * (self.n_cell + self.n_ramp)

    def spikes(self):
        """自分の窓に入る出来事が1つでもあれば撃つ ＋ 傾きを持つ細胞 [B, n_src × (n_cell + n_ramp)]

        傾きを持つ細胞（ramping）：人で時間細胞と並んで見つかる。決まった瞬間ではなく、
        時間とともに発火が上がっていく（または下がっていく）ことで粗い時間を運ぶ（Umbach ほか 2020）
        """
        if self._cache is not None:          # 1歩の中では中身が変わらないので作り直さない
            return self._cache
        a = self.age[..., None].astype(np.float32)                     # [B, src, M, 1]
        hit = (np.abs(a - self.pref) <= self.half) & (a <= self.L)
        time_cells = hit.any(2).astype(np.float32) if self.n_cell else \
            np.zeros(self.age.shape[:2] + (0,), np.float32)             # [B, src, n_cell]
        youngest = self.age.min(2).astype(np.float32)                   # 一番最近の出来事からの歩数
        r = np.clip(youngest[..., None] / self.ramp_len, 0, 1)          # 0 → 1 に上がる
        up = (self.rng.random(r.shape[:2] + (self.n_ramp // 2,)) < r[..., :self.n_ramp // 2] * self.ramp_p)
        dn = (self.rng.random(r.shape[:2] + (self.n_ramp - self.n_ramp // 2,))
              < (1 - r[..., :self.n_ramp - self.n_ramp // 2]) * self.ramp_p)
        ramp = np.concatenate([up, dn], -1).astype(np.float32)
        self._cache = np.concatenate([time_cells, ramp], -1).reshape(self.B, -1)
        return self._cache

    def step(self, fired):
        self._cache = None                   # 歩を進めたら作り直す
        self.age = np.where(self.age < 10 ** 8, self.age + 1, self.age)
        f = fired > 0
        if f.any():
            oldest = np.argmax(self.age, 2)
            b, s = np.nonzero(f)
            self.age[b, s, oldest[b, s]] = 0
