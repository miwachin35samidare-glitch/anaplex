"""デフォルメ版の網 —— 段を建てる（回す前に全部建てておく）

段（並べ方は脳を模す。一次・連合・高次）
  一次  聴覚（耳を受ける）／ 運動（口へ出す）／ 視覚（まぶたを閉じて場所取り）
  連合  聴覚の上に1段。繋ぐ先を持たない
  高次  聴覚と運動をまたぐ1つ。語・統語はここ
段の時定数（1歩 10.67ms）
  一次 9歩（100ms）／ 連合 18歩（190ms）／ 高次 30〜45歩（320〜480ms）
繋ぎ方
  下 → 上   誤差のニューロンの発火を上の段の入口へ（予測符号化で上へ送るのは誤差）
  上 → 下   上の段の表現のニューロンの予測を、下の段の誤差のニューロンの樹状突起へ
  横        運動 → 聴覚 の指令の写し（随伴放電）
視覚
  建てておいて、視覚の段が来るまで出力を止める（まぶた）。
  人でも、目が開く前から網膜の自発的な波が視覚系を組み上げている
"""
import numpy as np
from element1 import Prior
from area4 import Area4
from hippocampus import Hippocampus
from group import AreaGroup

TAU = {"一次": 1.0, "連合": 1.9, "高次": 3.2}     # 一次を1とした比（人 約5倍・マカク 3.3〜4.1倍から）


class Brain:
    def __init__(self, prior, n_ear=40, n_cmd=20, n_eye=40, n_assoc=40, n_high=40,
                 n_partner=500, n_cell=128, eta=0.3, stp=True, seed=0, eye_open=False,
                 hippo=True, hippo_scale=4.0):
        def mk(n_ch, n_ext, tau, gate=True, sd=0):
            return Area4(n_ch=n_ch, prior=prior, n_ext=n_ext, eta=eta, n_partner=n_partner,
                         n_cell=n_cell, stp=stp, tau_scale=tau, gate=gate, seed=seed + sd)
        self.聴覚 = mk(n_ear, n_cmd + n_assoc, TAU["一次"], sd=1)   # 耳 ＋ 指令の写し ＋ 連合からの予測
        self.運動 = mk(n_cmd, n_high, TAU["一次"], sd=2)            # 口 ＋ 高次からの予測
        self.視覚 = mk(n_eye, 0, TAU["一次"], gate=eye_open, sd=3)   # まぶた
        self.連合 = mk(n_assoc, n_high, TAU["連合"], sd=4)          # 聴覚の誤差 ＋ 高次からの予測
        self.高次 = mk(n_high, n_assoc, TAU["高次"], sd=5)          # 連合の誤差 ＋ 運動の誤差
        # 海馬：皮質（高次）の誤差を受け、CA1 の出力を皮質（連合）へ返す
        #   人：嗅内皮質を介して皮質と繋がる。速く覚えて、あとで皮質へ移す
        self.海馬 = Hippocampus(prior, n_in=n_high, scale=hippo_scale,
                              n_partner=n_partner, n_cell=n_cell, eta=eta,
                              seed=seed + 6) if hippo else None
        self.ca1_out = None
        # 皮質5領野をまとめて1回で進める（意味は変えない。呼ぶ回数だけ減らす）
        self.group = AreaGroup([self.聴覚, self.運動, self.視覚, self.連合, self.高次])
        self.n_ear, self.n_cmd, self.n_assoc, self.n_high = n_ear, n_cmd, n_assoc, n_high
        self.pred = {k: np.zeros((1, getattr(self, k).n_ch), np.float32)
                     for k in ("連合", "高次")}                      # 上からの予測（1歩前のもの）
        self.up = {"聴覚": np.zeros((1, n_ear), np.float32),
                   "連合": np.zeros((1, n_assoc), np.float32),
                   "運動": np.zeros((1, n_cmd), np.float32),
                   "高次": np.zeros((1, n_high), np.float32)}

    @property
    def n_cells(self):
        n = sum(getattr(self, k).n_cells for k in ("聴覚", "運動", "視覚", "連合", "高次"))
        return n + (self.海馬.n_cells if self.海馬 else 0)

    def step(self, ear, cmd, eye=None, learn=True):
        """ear [1, n_ear]  cmd [1, n_cmd]（口の指令の写し）  eye [1, n_eye] または None

        領野をまたぐのは1歩に1回。すべての領野が「1歩前の他の領野」を見て、同時に進む。
        （人でも領野をまたぐには伝導の遅れがある。順に呼んで同じ歩のうちに届けるのは人と違った）
        """
        to_bits = lambda x, n: ((x if x is not None else np.zeros((1, n), np.float32)) > 0.5
                                ).astype(np.float32)
        u = self.up                                     # 1歩前の、他の領野からの発火
        ins = [
            (self.聴覚, ear, np.concatenate([cmd, to_bits(self.pred["連合"], self.n_assoc)], 1)),
            (self.運動, cmd, to_bits(self.pred["高次"], self.n_high)),
            (self.視覚, np.zeros((1, self.視覚.n_ch), np.float32) if eye is None else eye, None),
            (self.連合, self._fit(u["聴覚"], self.n_assoc), to_bits(self.pred["高次"], self.n_high)),
            (self.高次, self._fit(u["連合"], self.n_high), self._fit(u["運動"], self.n_assoc)),
        ]
        out = self.group.step([x for _, x, _ in ins], [e for _, _, e in ins], learn)
        rep_a, err_a, _ = out[0]
        rep_m, err_m, _ = out[1]
        _, err_c, _ = out[3]
        _, err_h, _ = out[4]
        if self.海馬 is not None:
            self.ca1_out, _, _ = self.海馬.step(self.up.get("高次", err_h * 0) * 0 + err_h, learn)
        self.up = {"聴覚": err_a, "連合": err_c, "運動": err_m, "高次": err_h}
        self.pred["連合"] = self.連合.pred_rep
        self.pred["高次"] = self.高次.pred_rep
        return rep_a, err_a, rep_m, err_m

    @staticmethod
    def _fit(x, n):
        """段ごとに入口の数が違うので、足りなければ0で埋め、多ければ切る（体数はみわが決めるまでの仮）"""
        if x.shape[1] == n:
            return x
        out = np.zeros((x.shape[0], n), np.float32)
        m = min(n, x.shape[1])
        out[:, :m] = x[:, :m]
        return out
