"""領野をまとめて1回で処理する —— 呼ぶ回数の値段を減らす

なぜ要るか
  1歩に80回の呼び出し（領野8つ × 周回10回）。1回 0.4ms のうち、中の計算は 0.02ms。
  20倍が呼び出しの費用。触る辺は1%なので、疎は効いている。効いていないのは回数だけ。

人と関係しないこと
  領野の境目   辺の番号をずらして並べるだけ。どの辺がどの領野かは決まったまま
  学習         シナプスは1本ずつ別の重み。更新も辺ごと。値を共有しない
  領野ごとの値  時定数・自発の率・まぶた・完成の有無は、受け手ごとの配列のまま残る
  時間細胞     領野ごとの塊のままにする（混ぜると聴覚の時間細胞が運動の入力で撃ってしまう）
"""
import numpy as np
from element1 import TAUS, THETA, sigmoid


class AreaGroup:
    """作り終えた Area4 を何本かまとめて、1回で進める"""

    def __init__(self, areas):
        self.areas = list(areas)
        n = len(self.areas)
        # 辺をひとつに繋ぐ（送り手・受け手の番号をずらす）
        self.off_send = np.cumsum([0] + [a.n_send for a in self.areas])
        self.off_recv = np.cumsum([0] + [a.n_recv for a in self.areas])
        self.e_src = np.concatenate([np.repeat(np.arange(a.n_send, dtype=np.int64), np.diff(a.start))
                                     + self.off_send[i] for i, a in enumerate(self.areas)])
        self.e_recv = np.concatenate([a.e_recv + self.off_recv[i] for i, a in enumerate(self.areas)])
        self.e_area = np.concatenate([np.full(len(a.w), i, np.int32) for i, a in enumerate(self.areas)])
        o = np.argsort(self.e_src, kind="stable")
        self.e_src, self.e_recv, self.e_area = self.e_src[o], self.e_recv[o], self.e_area[o]
        self.w = np.concatenate([a.w for a in self.areas])[o]
        self.e_inh = np.concatenate([a.e_inh for a in self.areas])[o]
        self.n_send_all = int(self.off_send[-1])
        self.n_recv_all = int(self.off_recv[-1])
        self.start = np.zeros(self.n_send_all + 1, np.int64)
        np.add.at(self.start, self.e_src + 1, 1)
        self.start = np.cumsum(self.start)
        # 受け手ごとの値（領野ごとに違うものも、ここでは1本の配列になる）
        self.scale = np.concatenate([a.scale for a in self.areas])
        self.drive = np.concatenate([a.drive for a in self.areas])
        self.recv_ema = np.concatenate([a.recv_ema for a in self.areas])
        self.decay = np.stack([np.repeat(a.decay[None, :], a.n_recv, 0) for a in self.areas]).reshape(-1, len(TAUS)) \
            if False else np.concatenate([np.repeat(a.decay[None, :], a.n_recv, 0) for a in self.areas])
        self.tb = np.concatenate([a.tb[0] for a in self.areas])
        self.prior = self.areas[0].prior
        self.eta = self.areas[0].eta
        self.homeo_tau = self.areas[0].homeo_tau
        self.scaling_target = self.areas[0].scaling_target
        self.recur = self.areas[0].recur
        self.t = 0
        self._e_cache = None
        # 受け手が「どの領野の、どの役か」
        self.r_area = np.concatenate([np.full(a.n_recv, i, np.int32) for i, a in enumerate(self.areas)])
        self.r_kind = np.concatenate([np.concatenate([np.zeros(a.n_ch, np.int8),        # 0 誤差
                                                      np.ones(a.n_ch, np.int8),          # 1 表現
                                                      np.full(a.n_inh, 2, np.int8)])     # 2 抑制
                                      for a in self.areas])

    def _fired_edges(self, fired):
        if len(fired) == 0:
            return np.empty(0, np.int64)
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        if not keep.any():
            return np.empty(0, np.int64)
        f, lens = fired[keep], lens[keep]
        return np.repeat(self.start[f], lens) + (np.arange(lens.sum()) - np.repeat(
            np.concatenate([[0], np.cumsum(lens)[:-1]]), lens))

    def step(self, xs, exts, learn=True):
        """xs / exts: 領野ごとの入力（1歩前のもの）。1歩ぶん進める"""
        for i in range(self.recur):
            out = self._pass(xs, exts, learn and i == 0, advance=(i == self.recur - 1))
        return out

    def _pass(self, xs, exts, learn, advance):
        # 送り手の発火は1歩の中で変わらないので、1回だけ作る
        if self._e_cache is None:
            sp = [np.concatenate([a.ch.spikes()[0], a.buf.reshape(-1)]) for a in self.areas]
            self._e_cache = self._fired_edges(np.nonzero(np.concatenate(sp))[0])
        e = self._e_cache
        exc = np.zeros(self.n_recv_all, np.float32)
        ihn = np.zeros(self.n_recv_all, np.float32)
        if len(e):
            amp = np.where(self.e_inh[e], 4.0, 3.3).astype(np.float32) * self.w[e]
            np.add.at(exc, self.e_recv[e], np.where(self.e_inh[e], 0.0, amp))
            np.add.at(ihn, self.e_recv[e], np.where(self.e_inh[e], amp, 0.0))
        exc *= self.scale; ihn *= self.scale
        lg, _ = self.prior.logit(self.tb[None, :, :])
        p = sigmoid(lg[0] + exc / (1.0 + ihn))
        # 的：誤差と表現は自分の入口、抑制は領野の混み具合
        tgt = np.zeros(self.n_recv_all, np.float32)
        for i, a in enumerate(self.areas):
            lo, hi = self.off_recv[i], self.off_recv[i + 1]
            x = xs[i][0]
            tgt[lo:lo + a.n_ch] = x
            tgt[lo + a.n_ch:lo + 2 * a.n_ch] = x
            tgt[lo + 2 * a.n_ch:hi] = x.mean()
        err_sig = tgt - p
        if learn and len(e):
            g = self.eta * err_sig[self.e_recv[e]]
            shunt = (exc / (1.0 + ihn) ** 2)[self.e_recv[e]]
            self.w[e] += np.where(self.e_inh[e], -g * shunt, g)
            self.w[e] = np.where(self.e_inh[e], np.maximum(self.w[e], 0.0), self.w[e])
            a_ = 1.0 / self.homeo_tau
            cnt = np.zeros(self.n_recv_all, np.float32); np.add.at(cnt, self.e_recv[e], 1.0)
            self.recv_ema += a_ * (cnt - self.recv_ema)
            self.drive += a_ * (np.abs(exc) - self.drive)
            self.scale *= np.clip(1 - 0.01 * (self.drive / self.scaling_target - 1), 0.5, 1.5) ** a_
        # 領野ごとに出力を作る
        outs = []
        for i, a in enumerate(self.areas):
            lo = self.off_recv[i]
            pe = p[lo:lo + a.n_ch]; pr = p[lo + a.n_ch:lo + 2 * a.n_ch]
            pi = p[lo + 2 * a.n_ch:self.off_recv[i + 1]]
            x = xs[i]
            err = (err_sig[lo:lo + a.n_ch] > THETA).astype(np.float32)[None, :]
            rep = x.astype(np.float32)
            if a.complete and x.sum() > 0:
                rep = np.maximum(rep, ((x[0] + a.pred_boost * pr) > 1.0).astype(np.float32)[None, :])
            if a.spont_p > 0:
                rep = np.maximum(rep, (a.rng.random(rep.shape) < a.spont_p).astype(np.float32))
            inh_spk = (pi > 0.5).astype(np.float32)[None, :]
            a.pred_rep = pr[None, :]
            if advance:
                lo2, hi2 = self.off_recv[i], self.off_recv[i + 1]
                self.tb[lo2:hi2] = self.tb[lo2:hi2] * self.decay[lo2:hi2] + np.concatenate(
                    [rep[0], rep[0], inh_spk[0]])[:, None] * (1 - self.decay[lo2:hi2])
                src = rep[0] if exts[i] is None else np.concatenate([rep[0], exts[i][0].astype(np.float32)])
                src = np.concatenate([src, inh_spk[0]])
                a.ch.step(src[None, :])
                a.buf = np.concatenate([src[:, None], a.buf[:, :-1]], 1)
            outs.append((rep * (1 if a.gate else 0), err * (1 if a.gate else 0), pe[None, :]))
        if advance:
            self.t += 1
            self._e_cache = None      # 歩を進めたら作り直す
        return outs
