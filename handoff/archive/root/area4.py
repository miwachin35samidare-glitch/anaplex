"""領野 v4 —— 発火の疎を計算にも効かせる

v3 との違いは持ち方だけ。機構（抑制の短絡・符号の射影・短期可塑性・放出・スケーリング・
資源と順位の成長・パターン完成・自発活動・段の時定数・まぶた）はそのまま。

持ち方
  v3  受け手ごとに「自分の相手」を並べ、毎歩ぜんぶ読む
      ⇒ 撃った数が0でも全部でも同じ時間。発火の疎が計算に効かない
  v4  送り手ごとに「自分の届け先」を連ねて持つ（辺の配列を送り手順に並べる）
      撃った送り手の分だけを触る ⇒ 触る量が発火率に比例し、しかも続きの領域になる
  状態（短期可塑性・適格度）も、最後に触った歩からまとめて減らす（毎歩ぜんぶ触らない）
"""
import numpy as np
from element1 import Prior, TAUS, THETA, sigmoid
from chains import TimeCells
from stp import STP, DT_S
from timing import steps, RECUR, SYN_DELAY_MS, REWIRE_MS


class Area4:
    def __init__(self, n_ch, prior, n_ext=0, L=None, D=None, eta=0.03,
                 n_partner=200, pool_mult=3, rewire_every=None, homeo_tau=2000.0,
                 seed=0, n_cell=32, width=3, inh_frac=0.16, stp=True,
                 tau_scale=1.0, gate=True, spont_hz=0.3, complete=False, recur=None,
                 w_int8=False, stp_f16=False, timecells=True, n_ramp=8, motor=False):
        # 短い遅れ（人は 1.1ms。1歩が下限）。運動の領野は時間細胞を持たない代わりに、
        # 自分の発火が戻る道をここで持つ（回転する動きは領野の中の再帰が支える）
        D = (steps(SYN_DELAY_MS * 3) if D is None else D) if not motor else (D or 8)
        rewire_every = steps(REWIRE_MS) if rewire_every is None else rewire_every
        self.n_ch, self.n_ext, self.D, self.eta = n_ch, n_ext, D, eta
        self.rewire_every, self.homeo_tau, self.prior = rewire_every, homeo_tau, prior
        self.tau_scale, self.gate, self.complete = tau_scale, gate, complete
        self.spont_p = spont_hz * DT_S
        self.pred_boost = 0.9
        # 1歩の中で何回まわるか（整理券）。1周 ＝ シナプス1本ぶんの遅れ 1.1ms（ヒト実測）
        #   1歩 10.67ms ÷ 1.1ms ≒ 10回。1歩の中で起きたことの前後が、この回数でつく
        self.recur = RECUR if recur is None else recur
        # 持ち方の旗（GPU に移してから、速さと精度を両方測って決める）
        #   w_int8   重みを受け手ごとの尺度 ＋ 256段で持つ（19 → 16B）
        #   stp_f16  短期可塑性の状態を粗い小数で持つ（16 → 12B）
        self.w_int8, self.stp_f16 = w_int8, stp_f16
        self.scaling_target = 0.855
        self.decay = np.exp(-1.0 / (TAUS * tau_scale)).astype(np.float32)
        self.n_inh = max(1, int(round(n_ch * inh_frac / (1 - inh_frac))))
        self.n_src = n_ch + n_ext + self.n_inh
        # 時間の持ち方は領野で違う（すべて【ヒト】の実測から）
        #   聴覚  時間に広がった受容野（何ms前の入力に反応するか）
        #   海馬  時間細胞（出来事から何秒後）
        #   運動  持たない。集団の状態が動くことで時間を表す
        #        （BrainGate、四肢麻痺の患者2名の発話中の単一ニューロン。
        #         合図のあと、どの語でも同じ大きな『蹴り』→ そのあと回転する動き）
        self.motor = motor
        self.ch = TimeCells(self.n_src, n_cell=(0 if motor else (n_cell if timecells else 0)),
                            B=1, seed=seed, n_ramp=n_ramp)
        self.n_send = self.ch.n_cells + self.n_src * D
        self.n_recv = n_ch * 2 + self.n_inh
        rng = np.random.default_rng(seed)
        self.rng = rng
        n_partner = min(n_partner, self.n_send)
        self.n_partner, self.n_pool = n_partner, min(int(n_partner * pool_mult), self.n_send)

        # ── 辺を送り手ごとに連ねて持つ ──
        idx = np.stack([rng.choice(self.n_send, self.n_pool, replace=False)
                        for _ in range(self.n_recv)])
        live = np.zeros((self.n_recv, self.n_pool), bool)
        live[:, :n_partner] = True
        r, c = np.nonzero(live)
        src = idx[r, c]
        o = np.argsort(src, kind="stable")
        self.e_src = src[o].astype(np.int32)          # 送り手
        self.e_recv = r[o].astype(np.int32)           # 届け先
        self.w = np.zeros(len(o), np.float32)         # 重み（辺ごと）
        self.start = np.zeros(self.n_send + 1, np.int64)
        np.add.at(self.start, self.e_src + 1, 1)
        self.start = np.cumsum(self.start)
        # 予備（休止中・強さ0・総数の外）は、受け手ごとに候補の番号だけ持つ
        self.spare = [idx[i, n_partner:].copy() for i in range(self.n_recv)]

        src_inh = np.zeros(self.n_src, bool); src_inh[n_ch + n_ext:] = True
        is_inh_send = np.concatenate([np.repeat(src_inh, self.ch.n_cell + self.ch.n_ramp),
                                      np.repeat(src_inh, D)])
        self.e_inh = is_inh_send[self.e_src]
        self.is_inh_send = is_inh_send
        # contacts は e_inh から決まる（抑制4.0／興奮3.3。ヒト実測）ので持たない
        # e_src も、送り手ごとに並べてあるので位置から分かる。刈り込みのときだけ作り直す
        del self.e_src
        recv_inh = np.zeros(self.n_recv, bool); recv_inh[n_ch * 2:] = True
        self.stp = STP((len(o),), self.e_inh, recv_inh[self.e_recv], rng) if stp else None

        self.tb = np.zeros((1, self.n_recv, len(TAUS)), np.float32)
        self.buf = np.zeros((self.n_src, D), np.float32)
        self.inh_spk = np.zeros((1, self.n_inh), np.float32)
        self.scale = np.ones(self.n_recv, np.float32)
        self.recv_ema = np.zeros(self.n_recv, np.float32)
        self.drive = np.zeros(self.n_recv, np.float32)
        self.budget = np.full(self.n_recv, float(n_partner), np.float32)
        self.t = 0

    @property
    def n_cells(self):
        return self.n_ch * 2 + self.n_inh + self.ch.n_cells

    @property
    def n_syn(self):
        return int(len(self.w) * 3.3)

    def stp_amp_cache(self, e):
        """同じ歩の2周目以降は、1周目に放出した量をそのまま使う（小胞を二重に消費しない）"""
        return self._amp_cache.get(len(e), 1.0) if False else self._last_amp[e] \
            if hasattr(self, '_last_amp') else 1.0

    def _fired_edges(self, fired):
        """撃った送り手 → 触る辺の位置（送り手順に連なっているので続きの領域になる）"""
        if len(fired) == 0:
            return np.empty(0, np.int64)
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        if not keep.any():
            return np.empty(0, np.int64)
        f, lens = fired[keep], lens[keep]
        return np.repeat(self.start[f], lens) + (np.arange(lens.sum()) - np.repeat(
            np.concatenate([[0], np.cumsum(lens)[:-1]]), lens))

    def step(self, x, ext=None, learn=True):
        """1歩の中で recur 回まわる。学ぶのは1回目だけ（1歩に1回の学習）"""
        out = None
        for i in range(self.recur):
            out = self._pass(x, ext, learn and i == 0, advance=(i == self.recur - 1), first_pass=(i == 0))
        return out

    def _pass(self, x, ext=None, learn=True, advance=True, first_pass=True):
        fired = np.nonzero(self.ch.spikes()[0])[0]
        bf = np.nonzero(self.buf.reshape(-1))[0]
        if len(bf):
            fired = np.concatenate([fired, bf + self.ch.n_cells])
        e = self._fired_edges(np.sort(fired))                    # 触る辺はここだけ
        exc = np.zeros(self.n_recv, np.float32)
        ihn = np.zeros(self.n_recv, np.float32)
        if len(e):
            amp = np.where(self.e_inh[e], 4.0, 3.3).astype(np.float32) * self.w[e]
            if self.stp is not None:
                # 放出は1歩に1回（1歩の中の周回では、小胞は減らない）
                if first_pass:
                    rel = self.stp.release_at(e, self.t)
                    if not hasattr(self, '_last_amp'):
                        self._last_amp = np.ones(len(self.w), np.float32)
                    self._last_amp[e] = rel
                else:
                    rel = self._last_amp[e]
                amp = amp * rel
            np.add.at(exc, self.e_recv[e], np.where(self.e_inh[e], 0.0, amp))
            np.add.at(ihn, self.e_recv[e], np.where(self.e_inh[e], amp, 0.0))
        exc *= self.scale; ihn *= self.scale
        lg, _ = self.prior.logit(self.tb)
        p = sigmoid(lg[0] + exc / (1.0 + ihn))
        tgt = np.concatenate([x[0], x[0], np.repeat(x.mean(), self.n_inh)])
        err_sig = tgt - p
        if learn and len(e):
            g = self.eta * err_sig[self.e_recv[e]]
            shunt = (exc / (1.0 + ihn) ** 2)[self.e_recv[e]]
            self.w[e] += np.where(self.e_inh[e], -g * shunt, g)
            self.w[e] = np.where(self.e_inh[e], np.maximum(self.w[e], 0.0), self.w[e])
            a = 1.0 / self.homeo_tau
            cnt = np.zeros(self.n_recv, np.float32); np.add.at(cnt, self.e_recv[e], 1.0)
            self.recv_ema += a * (cnt - self.recv_ema)
            self.drive += a * (np.abs(exc) - self.drive)
            self.scale *= np.clip(1 - 0.01 * (self.drive / self.scaling_target - 1), 0.5, 1.5) ** a
        err = ((err_sig[:self.n_ch] > THETA)).astype(np.float32)[None, :]
        rep = x.astype(np.float32)
        if self.complete:
            drive = x[0] + self.pred_boost * p[self.n_ch:self.n_ch * 2]
            if x.sum() > 0:
                rep = np.maximum(rep, (drive > 1.0).astype(np.float32)[None, :])
        if self.spont_p > 0:
            rep = np.maximum(rep, (self.rng.random(rep.shape) < self.spont_p).astype(np.float32))
        self.inh_spk = (p[self.n_ch * 2:] > 0.5).astype(np.float32)[None, :]
        self.pred_rep = p[self.n_ch:self.n_ch * 2][None, :]
        if advance:                       # 歩を進めるのは、まわり終えたときだけ
            self.tb = self.tb * self.decay + np.concatenate(
                [rep[0], rep[0], self.inh_spk[0]])[None, :, None] * (1 - self.decay)
            src = rep[0] if ext is None else np.concatenate([rep[0], ext[0].astype(np.float32)])
            src = np.concatenate([src, self.inh_spk[0]])
            self.ch.step(src[None, :])
            self.buf = np.concatenate([src[:, None], self.buf[:, :-1]], 1)
            self.t += 1
            if self.rewire_every and self.t % self.rewire_every == 0:
                self.rewire()
        if not self.gate:
            return rep * 0, err * 0, p[:self.n_ch][None, :]
        return rep, err, p[:self.n_ch][None, :]

    def rewire(self):
        """資源（受け手ごとの総量）と順位（一番弱い1本）。予備からランダムに開通する
        辺の並びは送り手順に保つ（弱い辺の送り手を差し替えて、並べ直す）"""
        target = self.n_partner * 0.5
        self.budget = np.clip(self.budget + 0.5 * np.sign(target - self.recv_ema), 1, self.n_pool)
        e_src = np.repeat(np.arange(self.n_send, dtype=np.int32), np.diff(self.start))
        order = np.argsort(self.e_recv, kind="stable")
        cuts = []
        for i in range(self.n_recv):
            mine = order[np.searchsorted(self.e_recv[order], i):
                         np.searchsorted(self.e_recv[order], i, side="right")]
            if len(mine) == 0:
                continue
            cuts.append(mine[np.argmin(np.abs(self.w[mine]))])
        if not cuts:
            return
        cuts = np.asarray(cuts)
        e_src[cuts] = self.rng.integers(0, self.n_send, len(cuts)).astype(np.int32)
        self.w[cuts] = 0.0
        o = np.argsort(e_src, kind="stable")             # 送り手順に並べ直す
        e_src = e_src[o]
        self.e_recv, self.w = self.e_recv[o], self.w[o]
        self.e_inh = self.is_inh_send[e_src]
        if self.stp is not None:
            self.stp.reorder(o)
        self.start = np.zeros(self.n_send + 1, np.int64)
        np.add.at(self.start, e_src + 1, 1)
        self.start = np.cumsum(self.start)
