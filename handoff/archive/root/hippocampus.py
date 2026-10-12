"""海馬 —— 歯状回（パターン分離）／ CA3（パターン完成）／ CA1（出口）

人から引いた値（ヒト）
  体数（片側）    歯状回 15M ／ 門 2M ／ CA3-2 2.7M ／ CA1 16M ／ 海馬台 4.5M
                （West & Gundersen 1990、ステレオロジー）
                ⇒ 比は 歯状回 15 : CA3 2.7 : CA1 16。デフォルメ版もこの比で置く
  疎さ          1つの項目に強く反応するのは 2.5% 未満（患者20名の単一ニューロン記録）
                概念の細胞は見せた絵の 1〜3% にしか反応しない。発火率 約0.5Hz
  CA3 の再帰     接続確率 1.27%（789組中10組）。距離によらず広く繋がる
                1体あたりの再帰入力 16,100本（マウス 13,200）
  CA3 の確実さ   シナプスの成功率 0.90（マウス 0.62。皮質の放出確率 0.33 と対比）
  CA3 の膜時定数  55.9ms（マウス 58.3ms）⇒ 皮質の一次（約100ms）より速い
  発火率        非硬化のヒト CA3 で 9.6Hz

デフォルメ
  学習の速さは値として置かない。シナプスの確実さ（0.90 対 0.33）と疎さの差から出させる
  素子は皮質と同じものを使う（型の違いは入れない）
"""
import numpy as np
from element1 import Prior, TAUS, THETA, sigmoid
from area4 import Area4
from group import AreaGroup

RATIO = {"歯状回": 15.0, "CA3": 2.7, "CA1": 16.0}     # ヒトの体数の比
SPARSE = 0.025                                        # 1つの項目に反応する割合（ヒト実測 2.5%未満）
CA3_RECUR_P = 0.0127                                  # CA3 どうしの接続確率（ヒト実測）
CA3_SUCCESS = 0.90                                    # CA3 のシナプスの成功率（ヒト実測）
TAU_CA3 = 55.9 / 10.67                                # 1歩 10.67ms に直した膜時定数（≈5.2歩）


class Hippocampus:
    def __init__(self, prior, n_in, scale=1.0, n_partner=200, n_cell=32, eta=0.3, seed=0,
                 recur_steps=None):
        """n_in: 入口（皮質から来る誤差の発火の数）。scale: ヒトの比に掛ける倍率"""
        n = {k: max(4, int(round(v * scale))) for k, v in RATIO.items()}
        self.n = n
        self.n_in = n_in
        mk = lambda n_ch, n_ext, sd, tau, comp=False: Area4(
            n_ch=n_ch, prior=prior, n_ext=n_ext, eta=eta, n_partner=n_partner,
            n_cell=n_cell, pool_mult=1, rewire_every=0, stp=False, spont_hz=0.5,
            tau_scale=tau, seed=seed + sd, complete=comp)
        # 歯状回：入口より体を多く置き、閾値を上げて疎にする（パターン分離）
        self.dg = mk(n["歯状回"], n_in, 1, TAU_CA3 / 9.4)
        # CA3：自分の出力が自分に戻る（再帰）。パターン完成
        self.ca3 = mk(n["CA3"], n["歯状回"] + n["CA3"], 2, TAU_CA3 / 9.4, comp=True)
        # CA1：CA3 の出力を受けて皮質へ返す
        self.ca1 = mk(n["CA1"], n["CA3"], 3, TAU_CA3 / 9.4, comp=True)
        self.ca3_prev = np.zeros((1, n["CA3"]), np.float32)
        # 苔状線維：歯状回 → CA3。1本が強く、少数で撃たせる（起爆シナプス）
        #   ⇒ 切り詰めではなく、CA3 の1体が歯状回の少数と繋がる形にする
        g = np.random.default_rng(seed + 9)
        self.mossy = np.stack([g.choice(n["歯状回"], min(4, n["歯状回"]), replace=False)
                               for _ in range(n["CA3"])])      # [CA3, 4]
        # 嗅内皮質 → 歯状回：1体が入口の約2%を拾う（ヒト・霊長類の疎な投射）
        k = max(2, int(round(n_in * 0.02)) if n_in >= 100 else max(2, n_in // 8))
        self.perforant = np.stack([g.choice(n_in, k, replace=False) for _ in range(n["歯状回"])])
        self.schaffer = np.stack([g.choice(n["CA3"], min(4, n["CA3"]), replace=False)
                                  for _ in range(n["CA1"])])   # CA3 → CA1
        self.rng = np.random.default_rng(seed)
        # CA3 の再帰は「引き込み」。1歩で終わらせず、同じ歩の中で何度か回して落ち着かせる
        #   （人：CA3 は脳で最大の自己連想の網。一部から覚えた全体へ引き込む）
        #   引き込みの回数は領野の中の周回に任せる（Area4 の recur ＝ 10回）
        self.recur_steps = recur_steps
        # 歯状回・CA3・CA1 をまとめて1回で進める（意味は変えない。呼ぶ回数だけ減らす）
        self.group = AreaGroup([self.dg, self.ca3, self.ca1])

    @property
    def n_cells(self):
        return sum(a.n_cells for a in (self.dg, self.ca3, self.ca1))

    def sparsify(self, p, frac=SPARSE):
        """上位 frac だけ撃つ（歯状回の勝者総取り。ヒトの 2.5% に合わせる）"""
        k = max(1, int(round(p.shape[1] * frac)))
        th = np.partition(p, -k, axis=1)[:, -k][:, None]
        return (p >= th).astype(np.float32)

    def step(self, ctx_err, learn=True):
        """ctx_err: 皮質から来る誤差の発火 [1, n_in] → CA1 の出力 [1, n_CA1]"""
        # 入口を歯状回へ広げる（切り詰めではなく投射）。拾った本数の多い体が勝つ ＝ 疎にする
        drive = ctx_err[0][self.perforant].sum(1)[None, :]
        dg = self.sparsify(drive)                                # 上位 2.5% だけ撃つ
        mf = (dg[0][self.mossy].max(1) > 0).astype(np.float32)[None, :]   # 起爆：どれか1本で撃つ
        # CA3 → CA1 は1歩前の CA3（領野をまたぐのは1歩に1回）
        sch = (self.ca3_prev[0][self.schaffer].max(1) > 0).astype(np.float32)[None, :]
        # 3つをまとめて1回で進める。引き込みは領野の中の周回が担う（二重に回さない）
        outs = self.group.step(
            [dg, mf, sch],
            [self._fit(ctx_err, self.n_in), np.concatenate([dg, self.ca3_prev], 1), self.ca3_prev],
            learn)
        # CA3 のシナプスは確実に伝わる（ヒト 0.90）。皮質の 0.33 との差が速く覚える理由
        ca3 = outs[1][0] * (self.rng.random(outs[1][0].shape) < CA3_SUCCESS)
        self.ca3_prev = ca3
        return outs[2][0], dg, ca3

    @staticmethod
    def _fit(x, n):
        if x.shape[1] == n:
            return x
        out = np.zeros((x.shape[0], n), np.float32)
        m = min(n, x.shape[1])
        out[:, :m] = x[:, :m]
        return out
