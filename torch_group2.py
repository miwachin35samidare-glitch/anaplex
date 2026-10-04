"""領野をまとめて1回で進める v2（arkhe 用）—— 領野ごとのループを全部消す

v1 との違いは、書く場所だけ。仕組みは1つも増えていない。
  v1  周回の中に「的を作るループ」と「出力を作るループ」、歩を進めるところに時間細胞のループ
      ⇒ 8領野 × 周回10 で、小さな呼び出しが何度も走る
  v2  入口・出力・時間細胞を、それぞれ1本の配列／行列にまとめて一度に扱う
      領野の境目は番号で残る（辺と同じやり方）

⚠️ 一致することを確かめてから使う（check(areas) が v1 と同じ結果になるか見る）

arkhe の実測（2026-09-11）
  別々 41.95ms → v1 23.90ms（周回1回 1.94ms ＋ 歩の固定費 5.23ms）
"""
import math
import torch
from torch_area import TAUS, THETA, DT_MS, W_INH
from torch_area import INH_W_SIG, INH_W_MU, INH_W_MEAN, INH_LTD_TH, INH_LTD_F, INH_LTD_WIN, INH_LTD_N   # 版47
from torch_area import INH_TH   # 版48

# 傾きの細胞の発火（版39）：【ヒト】皮質の単一細胞の発火率 0.3Hz を1歩に（0.32%）
#   ⟲ 版38 までは 30%/歩（chains.py の既定値。出どころの記録なし【C】。人の 30〜90倍）
RAMP_P = 0.3 * DT_MS / 1000.0
# 学んだ跡の長さ（版40）【ラット】大人の聴覚野・生体：修飾の信号が 1秒前・同時なら効き、
#   3秒前後では効かない（Brain Stimulation 2025）⇒ 時定数 1秒。【ヒト】【霊長類】は見つからない【み】09-27
ELIG_TAU = 1000.0 / DT_MS


class TimeCellsGroup:
    """8領野ぶんの時間細胞を、ひとつの行列で持つ（領野をまたいで混ざらない）"""

    def __init__(self, areas):
        dev = areas[0].dev
        self.dev = dev
        self.areas = areas
        self.M = areas[0].ch.M
        self.n_src = [a.ch.n_src for a in areas]
        self.off = torch.cumsum(torch.tensor([0] + self.n_src, device=dev), 0)
        self.age = torch.cat([a.ch.age for a in areas])          # [送り元の合計, M]
        # 領野ごとに時間細胞の数が違う（運動は 0）ので、出す幅も領野ごとに持つ
        self.n_cell = [a.ch.n_cell for a in areas]
        self.n_ramp = [a.ch.n_ramp for a in areas]
        self.pref = [a.ch.pref for a in areas]
        self.half = [a.ch.half for a in areas]
        self.ramp_len = [a.ch.ramp_len for a in areas]
        self.gen = areas[0].ch.gen
        self._cache = None

    @torch.no_grad()
    def spikes(self):
        """[送り手の口の合計] を1本で返す（領野ごとの塊が順に並ぶ）"""
        if self._cache is not None:
            return self._cache
        out = []
        for i, a in enumerate(self.areas):
            lo, hi = int(self.off[i]), int(self.off[i + 1])
            age = self.age[lo:hi]
            parts = []
            if self.n_cell[i]:
                f = age.unsqueeze(-1).float()
                hit = ((f - self.pref[i]).abs() <= self.half[i]) & (f <= self.pref[i].max())
                parts.append(hit.any(1).float())
            y = (age.min(1).values.float() / self.ramp_len[i]).clamp(0, 1).unsqueeze(-1)
            h = self.n_ramp[i] // 2
            up = torch.rand(hi - lo, h, device=self.dev, generator=self.gen) < y * RAMP_P
            dn = torch.rand(hi - lo, self.n_ramp[i] - h, device=self.dev,
                            generator=self.gen) < (1 - y) * RAMP_P
            parts.append(torch.cat([up, dn], 1).float())
            out.append(torch.cat(parts, 1).reshape(-1))
        self._cache = torch.cat(out)
        return self._cache

    @torch.no_grad()
    def step(self, fired_all):
        """fired_all: [送り元の合計]（領野ごとの塊が順に並ぶ）。1回で全部進める"""
        self._cache = None
        self.age = torch.where(self.age < 10 ** 8, self.age + 1, self.age)
        idx = torch.nonzero(fired_all > 0).squeeze(1)
        if len(idx):
            oldest = self.age.argmax(1)
            self.age[idx, oldest[idx]] = 0


class Group2:
    def __init__(self, areas):
        self.areas = list(areas)
        dev = self.areas[0].dev
        self.dev, self.recur = dev, self.areas[0].recur
        self.prior = self.areas[0].prior
        self.eta = self.areas[0].eta
        self.homeo_tau = self.areas[0].homeo_tau
        self.scaling_target = self.areas[0].scaling_target

        cs = lambda xs: torch.cumsum(torch.tensor([0] + xs, device=dev), 0)
        self.off_recv = cs([a.n_recv for a in self.areas])
        self.off_ch = cs([a.n_ch for a in self.areas])
        self.off_src = cs([a.n_src for a in self.areas])
        self.n_recv_all = int(self.off_recv[-1])
        self.n_ch_all = int(self.off_ch[-1])
        self.n_src_all = int(self.off_src[-1])

        # 送り手の口の並び：まとめると「時間細胞ぜんぶ → 各領野の buf」になる。
        # Area 側は「自分の時間細胞 → 自分の buf」なので、番号を写し直す
        ch_off = torch.cumsum(torch.tensor([0] + [a.ch.n_cells for a in self.areas], device=dev), 0)
        buf_base = int(ch_off[-1])
        buf_off = torch.cumsum(torch.tensor([0] + [a.n_src * a.D for a in self.areas], device=dev), 0)
        remap = []
        for i, a in enumerate(self.areas):
            m = torch.empty(a.n_send, dtype=torch.long, device=dev)
            m[:a.ch.n_cells] = torch.arange(a.ch.n_cells, device=dev) + ch_off[i]
            m[a.ch.n_cells:] = torch.arange(a.n_src * a.D, device=dev) + buf_base + buf_off[i]
            remap.append(m)
        self.off_send = torch.cat([ch_off[:-1], torch.tensor([buf_base], device=dev)])  # 参考用
        src = torch.cat([remap[i][a.e_src_sorted] for i, a in enumerate(self.areas)])
        recv = torch.cat([a.e_recv.long() + self.off_recv[i] for i, a in enumerate(self.areas)])
        w = torch.cat([a.w for a in self.areas])
        inh = torch.cat([a.e_inh for a in self.areas])
        self.n_send_all = buf_base + int(buf_off[-1])
        # 送り手が抑制性か（刈り込みで辺を並べ直したときに作り直すのに要る）
        self.is_inh_all = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)
        for i, a in enumerate(self.areas):
            self.is_inh_all[remap[i]] = a.is_inh_send
        # 刈り込みで生やす先の候補（領野ごと。番号は全体の通し）
        #   Area.rewire は自分の領野の送り元から引いていた。まとめたときに全領野に広がっていた
        self.pool_send = remap
        o = torch.argsort(src)
        # 辺の番号は 32bit で持つ（版43a-m【み】09-29 許可。番号は1つもずれない。64bit の半分）
        self.e_src, self.e_recv, self.w, self.e_inh = src[o].int(), recv[o].int(), w[o], inh[o]
        del src, recv, w, inh
        self.start = self._starts()

        self.stp = None
        if self.areas[0].stp is not None:
            import copy
            g = copy.copy(self.areas[0].stp)
            for k in ("row", "state", "fac", "last", "deplete"):
                setattr(g, k, torch.cat([getattr(a.stp, k) for a in self.areas])[o])
            self.stp = g

        self.scale = torch.cat([a.scale for a in self.areas])
        self.drive = torch.cat([a.drive for a in self.areas])
        self.recv_ema = torch.cat([a.recv_ema for a in self.areas])
        # 細胞自身の発火率（恒常性が感じるもの）。目標から始める
        self.rate_ema = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)
        # 恒常性の目標を受け手ごとに（版39b。torch_brain が種類ごとの【ヒト】の値を入れる）
        self.target_recv = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)
        self.scale_skip = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)   # 人の値が無い受け手は下げない
        # 学びの門（版40）：学んだ跡を辺ごとに持ち、脳幹が配る第三の信号 third の分だけ重みを変える
        self.elig = torch.zeros_like(self.w)
        self.elig_decay = math.exp(-1.0 / ELIG_TAU)
        self.third = torch.ones((), device=dev)
        self.th_gain = torch.ones((), device=dev)   # 覚醒で表現の閾値に掛ける倍率（版41。脳幹が決める）
        self.decay = torch.cat([a.decay.unsqueeze(0).expand(a.n_recv, -1) for a in self.areas])
        self.tb = torch.cat([a.tb for a in self.areas])
        # 短い遅れ D は領野ごとに違う（運動は自分の発火が戻る道を広く持つので 8、他は 1）
        # 揃えると中身が変わるので、buf は領野ごとに別々の幅のまま持ち、口の番号だけ繋ぐ
        self.bufs = [a.buf for a in self.areas]
        self.Ds = [a.D for a in self.areas]
        self.ch = TimeCellsGroup(self.areas)

        # 受け手が「どの領野の、どの役か」を印で持つ（ループの代わり）
        kind, area_of, ch_of = [], [], []
        for i, a in enumerate(self.areas):
            kind += [0] * a.n_ch + [1] * a.n_ch + [2] * a.n_inh
            area_of += [i] * a.n_recv
            ch_of += list(range(a.n_ch)) + list(range(a.n_ch)) + [-1] * a.n_inh
        self.kind = torch.tensor(kind, device=dev)
        self.area_of = torch.tensor(area_of, device=dev)
        self.ch_of = torch.tensor(ch_of, device=dev)
        self.is_err = self.kind == 0
        self.is_rep = self.kind == 1
        self.is_inh_r = self.kind == 2
        # 表現・誤差の受け手が、まとめた入口のどこに当たるか
        self.x_index = torch.zeros(self.n_recv_all, dtype=torch.long, device=dev)
        for i, a in enumerate(self.areas):
            lo = int(self.off_recv[i]); c = int(self.off_ch[i])
            self.x_index[lo:lo + a.n_ch] = torch.arange(a.n_ch, device=dev) + c
            self.x_index[lo + a.n_ch:lo + 2 * a.n_ch] = torch.arange(a.n_ch, device=dev) + c
        # 抑制の受け手には、その領野の入口の平均を渡す
        self.inh_area = self.area_of[self.is_inh_r]
        self.mean_map = torch.zeros(self.n_ch_all, len(self.areas), device=dev)
        for i, a in enumerate(self.areas):
            c = int(self.off_ch[i])
            self.mean_map[c:c + a.n_ch, i] = 1.0 / a.n_ch
        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])
        self.complete = torch.cat([torch.full((a.n_ch,), float(a.complete), device=dev)
                                   for a in self.areas])
        self.pred_boost = torch.cat([torch.full((a.n_ch,), a.pred_boost, device=dev)
                                     for a in self.areas])
        self.gate = torch.cat([torch.full((a.n_ch,), 1.0 if a.gate else 0.0, device=dev)
                               for a in self.areas])
        self.gen = self.areas[0].gen
        # 閾値：シナプスの合計が「同時に 134本」に当たるところ（人の値）
        #   x_gain は、入口から直に来る駆動を「何本ぶん」と数えるか
        self.fire_th = torch.cat([torch.full((a.n_ch,), a.fire_th, device=dev) for a in self.areas])
        self.x_gain = torch.cat([torch.full((a.n_ch,), a.x_gain, device=dev) for a in self.areas])
        # 興奮性の接触の数（受け手ごと）。既定は【ヒト】皮質 3.3
        self.c_exc = torch.cat([torch.full((a.n_recv,), float(getattr(a, "contacts_exc", 3.3)), device=dev)
                                for a in self.areas])
        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
        # 版47：錐体 → 抑制の体 の重みは人の分布で始める（予測誤差では学ばない。形A の直し）
        ei = self.is_inh_r[self.e_recv.long()] & ~self.e_inh
        self.w[ei] = self._inh_w(self.e_recv[ei].long())
        # 送り手ごとの直近 INH_LTD_WIN 歩の発火（弱まりの条件を見るため）
        self.hist = torch.zeros((INH_LTD_WIN, self.n_send_all), dtype=torch.uint8, device=dev)
        self.hist_n = torch.zeros(self.n_send_all, dtype=torch.int16, device=dev)
        self.hist_i = 0
        self.t = 0
        self._e_cache = None
        self._within = None
        self._base = None
        self.scaling_on = True          # 脳幹（brainstem.py）が状態で切り替える（版38）
        self.mod_send = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 調整する側の送り手（版39。torch_brain が印を付ける）
        self.send_gain = torch.ones(self.n_send_all, device=dev)   # 送り手ごとの重さ。眠りで上からの戻りを落とす（版38）
        self.eta_gain = 1.0             # 学ぶ速さの掛け算。起きているあいだの驚き（版38）
        self.exc_gain = None            # リップルのときだけ、興奮の入力を底上げする
        self.sleep_mode = False
        self.is_hippo = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)
        self.rewire_every = self.areas[0].rewire_every
        # 相手の数は受け手ごと（【ヒト】1体 30,000シナプス・相手 9,100人）。
        #   ⚠️ 先頭の領野の値を全体に当てていた。視床は時間細胞を持たないので口が小さく、
        #      その値が全部の領野に効いていた（視床の相手が少ないこと自体は人の形）
        self.n_partner = torch.cat([a.n_partner_vec for a in self.areas])   # 版47：抑制の体は相手が少ない
        self.budget = torch.cat([a.budget for a in self.areas])
        # 周回の中で撃った発火を、短い遅れの「1歩前」の枠に重ねるための番号
        self.within_index = torch.cat(
            [torch.arange(a.n_ch, device=dev) * a.D + buf_base + int(buf_off[i])
             for i, a in enumerate(self.areas)])
        # ここまでで、領野ごとの辺・短期可塑性・時間細胞の中身はすべて group 側へ写してある。
        # 元を抱えたままだと、同じものを2つ持つことになる
        self._release_areas()

    def _inh_w(self, r):
        """版47：錐体 → 抑制の体 の1本（繋がり1つ。平均 1.65mV を版45 の単位にした INH_W_MEAN）を接触の数で割って w に"""
        z = torch.randn(len(r), device=self.dev, generator=self.gen)
        return torch.exp(INH_W_MU + INH_W_SIG * z) * INH_W_MEAN / self.c_exc[r]

    def _starts(self):
        """送り手ごとの辺の始まり（e_src は並んでいる）。数えるだけなので 64bit の写しを作らない（版43a-m）"""
        cnt = torch.bincount(self.e_src, minlength=self.n_send_all)
        return torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev), cnt.cumsum(0)])

    def _release_areas(self):
        """まとめ終わった領野ごとの配列を手放す（中身は group に移っている）

        ⚠️ 手放したあと Area 単体では回せない。Group2 に入れたら group で回す
        ⚠️ buf は残す（self.bufs が同じものを指している）
        """
        for a in self.areas:
            for k in ("w", "e_src_sorted", "e_recv", "e_inh", "start", "is_inh_send"):
                if getattr(a, k, None) is not None:
                    setattr(a, k, None)
            if getattr(a, "stp", None) is not None:
                for k in ("row", "state", "fac", "last", "deplete"):
                    setattr(a.stp, k, None)
            if getattr(a.ch, "age", None) is not None:
                a.ch.age = None
        if str(self.dev) != "cpu":
            torch.cuda.empty_cache()

    @property
    def n_cells(self):
        return sum(a.n_cells for a in self.areas)

    @torch.no_grad()
    def _fired_edges(self):
        if self._e_cache is not None:
            return self._e_cache
        if self._base is None:                    # 時間細胞と buf は1歩に1回でいい
            self._base = torch.cat([self.ch.spikes()] + [b.reshape(-1) for b in self.bufs])
        s = self._base
        if self._within is not None:
            s = torch.maximum(s, self._within)    # 周回の中で撃った分も送り手になる
        fired = torch.nonzero(s).squeeze(1)
        empty = torch.empty(0, dtype=torch.long, device=self.dev)
        if len(fired) == 0:
            self._e_cache = empty
            return empty
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        fired, lens = fired[keep], lens[keep]
        if len(fired) == 0:
            self._e_cache = empty
            return empty
        base = self.start[fired].repeat_interleave(lens)
        off = torch.arange(int(lens.sum()), device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens)
        self._e_cache = base + off
        return self._e_cache

    @torch.no_grad()
    def rewire(self):
        """資源（受け手ごとの総量）と順位（一番弱い1本）。予備からランダムに開通する
        【ヒト】シナプス密度は成人期を通じて一定 ／ 消えるスパインは小さい（強さと相関）
        消えるのが先、生えるのが後。生やす先はランダム"""
        target = self.n_partner * 0.5
        self.budget = torch.minimum(
            (self.budget + 0.5 * torch.sign(target - self.recv_ema)).clamp(min=1.0),
            self.n_partner * 3)
        self.n_rewire = getattr(self, "n_rewire", 0) + 1        # 何回まわしたか
        big = self.w.abs().max() + 1.0
        key = self.e_recv.float() * big + self.w.abs()
        order = torch.argsort(key)
        r_sorted = self.e_recv[order]
        first = torch.ones_like(r_sorted, dtype=torch.bool)
        first[1:] = r_sorted[1:] != r_sorted[:-1]      # 受け手ごとの先頭 ＝ 一番弱い1本
        cuts = order[first]
        if len(cuts) == 0:
            return
        self.n_cut = getattr(self, "n_cut", 0) + len(cuts)      # 何本 切ったか
        # 生やす先：受け手と同じ領野の送り元から一様に（種類は送り元の割合のまま）
        ar = self.area_of[self.e_recv[cuts].long()]
        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)
        for i in range(len(self.areas)):
            m = ar == i
            k = int(m.sum())
            if k:
                pool = self.pool_send[i]
                new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev,
                                            generator=self.gen)]
        self.e_src[cuts] = new.to(self.e_src.dtype)
        o = torch.argsort(self.e_src)
        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]
        self.elig = self.elig[o]
        self.e_inh = torch.index_select(self.is_inh_all, 0, self.e_src)
        # ⟲ 版40 で直した：並べ直したあとに、並べ直す前の位置 cuts で 0 にしていた。
        #    開通した辺は切った辺の重みを持ち越し、代わりに無関係の辺の重みが 100歩ごとに受け手の数だけ消えていた
        pos = torch.empty_like(o)
        pos[o] = torch.arange(len(o), device=self.dev)
        cuts = pos[cuts]
        self.w[cuts] = 0.0             # 開通の値（決まったこと 09-11：開通は w0、学びの跡は 0）
        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        ei = cuts[self.is_inh_r[self.e_recv[cuts].long()] & ~self.e_inh[cuts]]
        if len(ei):                    # 版47：開通した 錐体 → 抑制の体 も人の分布から
            self.w[ei] = self._inh_w(self.e_recv[ei].long())
        self.elig[cuts] = 0.0
        if self.stp is not None:
            self.stp.reorder(o)
        self.start = self._starts()

    @torch.no_grad()
    def step(self, x_all, ext_all, learn=True):
        """x_all: まとめた入口 [n_ch_all]  ext_all: まとめた ext [送り元の合計 − n_ch_all − 抑制]"""
        self._within = None                  # 周回の中の発火（1歩の中だけ。歩はまたがない）
        for i in range(self.recur):
            out = self._pass(x_all, ext_all, learn and i == 0, i == self.recur - 1, i == 0)
        # 刈り込みは、周をまわし終えてから
        #   ⚠️ 学習するのは1周目、歩を進めるのは最後の周。両方を条件にすると永久に呼ばれない
        if learn and self.rewire_every and self.t % self.rewire_every == 0:
            self.rewire()
        return out

    @torch.no_grad()
    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
        if first and self._base is not None:   # 版47：送り手ごとの直近 INH_LTD_WIN 歩の発火を数える
            now = (self._base > 0).to(torch.uint8)
            self.hist_n += now.to(torch.int16) - self.hist[self.hist_i].to(torch.int16)
            self.hist[self.hist_i] = now
            self.hist_i = (self.hist_i + 1) % INH_LTD_WIN
        exc = torch.zeros(self.n_recv_all, device=self.dev)
        ihn = torch.zeros(self.n_recv_all, device=self.dev)
        exc_mod = torch.zeros(self.n_recv_all, device=self.dev)   # 調整する側から来た興奮（版39）
        if len(e):
            # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0
            re_ = self.e_recv[e].long()
            se_ = self.e_src[e].long()
            amp = torch.where(self.e_inh[e], 4.0, self.c_exc[re_]) * self.w[e]
            if self.stp is not None:
                if first:
                    self._amp = torch.ones(len(self.w), device=self.dev)
                    self._amp[e] = self.stp.release_at(e, self.t)
                amp = amp * self._amp[e]
            amp = amp * self.send_gain[se_]   # 脳幹：眠りで上からの戻りを落とす（版38）
            # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）
            #   興奮 ＝ (和 ＋ |和|) / 2 にはできないので、抑制の総量を別に持つ必要がある
            #   ⇒ 受け手の数だけの配列を2本用意して、1回の scatter_add にまとめる
            r = re_
            both = torch.zeros(self.n_recv_all * 2, device=self.dev)
            both.scatter_add_(0, r + self.n_recv_all * self.e_inh[e].long(), amp)
            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]
            mm = self.mod_send[se_] & ~self.e_inh[e]
            exc_mod = exc_mod.index_add(0, r[mm], amp[mm])
        exc = exc * self.scale; ihn = ihn * self.scale
        exc_mod = exc_mod * self.scale
        # リップルのとき、興奮の入力だけ底上げする（【ヒト】錐体細胞 5.2倍）
        #   抑制は外さない。抑制性の体は、増えた発火を受けて自分で強まる（人では 3.8倍）
        if getattr(self, "exc_gain", None) is not None:
            exc = exc * self.exc_gain
            exc_mod = exc_mod * self.exc_gain
        p = torch.sigmoid(self.prior.logit(self.tb) + exc / (1.0 + ihn))

        # 表現の発火は学びの前に出す（版36。的に使うため）。学びはこの周の exc・ihn を変えないので値は同じ
        pr = p[self.is_rep]
        # 表現のニューロンも、シナプスの合計が閾値を越えたら撃つ
        #   【ヒト】L2/3 錐体細胞は、同時に活動した興奮性シナプス 134±28本 で 50% の確率で撃つ
        #   （人が毎歩受けている入力 135本＝30,000シナプス×0.45%/歩 と、ほぼ同じ）
        #   ⇒ 平均的な入力がちょうど閾値。だから疎で、入力の揺らぎに敏感になる
        #   抑制（短絡）はこの合計に掛かるので、ここで初めて表現の発火に効く
        # 調整する側（上からの戻り・指令の写し）は予測 p には入るが、表現を撃たせない（版39）
        drive = ((exc - exc_mod)[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])
        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）
        if self.complete.any():           # 入力が無いところは予測が足しになる（撃たせはしない）
            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th * self.th_gain) & (x_all > 0)
            rep = torch.where(fill, torch.ones_like(rep), rep)
        rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,
                                             generator=self.gen) < self.spont_p).float())

        # 的：誤差と表現は自分の入口、抑制はその領野の入口の平均（ループなしで一度に）
        # 誤差の的：下からの入力でこの細胞が撃つか（0か1）。予測はシグモイドなので的も 0か1 の目盛り
        # 的は「下からの入力（本数 × 重さ）がこの細胞を撃たせるか」。学んでいる重み（抑制）は入れない
        #   ⟲ 版31 で表現と同じ抑制を入れたら、学びが抑制を育てて的そのものを消し、皮質が全部止まった
        #      （的が 0 ⇒ e ＝ −p ⇒ 抑制が強まる ⇒ 的がさらに立たない、の輪）。的は学ぶ量に依存させない
        x_t = torch.where(self.tgt_by_fire, (x_all * self.x_gain > self.fire_th).float(), x_all)
        self.last_xt = x_t                         # 測る道具用
        tgt = x_t[self.x_index]
        tgt = torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], tgt)
        # 表現の的は下からの入力（版39。版36 の「自分の発火」を外した）
        #   ⟲ 自分の発火を的にすると、学んだシナプスがそのまま駆動に入るので「いつも撃つ」が誤差0の解になり、
        #      全領野が撃ちっぱなしになった（版36〜38、表現 50〜64%）
        #   ⚠️ 何を的にするかは人からは決まらない【C】。区画（予測する網と駆動する網）を作る段で見直す
        err_sig = tgt - p

        if learn and len(e):
            r = self.e_recv[e].long()
            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain：驚きで上がる（青斑核。版42 で戻した）
            # 学びの門（版40）：その歩の学びは跡にためる。重みを変えるのは脳幹の第三の信号の分だけ
            #   【ヒト】皮質の古典的な対ではドーパミンを入れて初めて強まる（前の作り 生-217）
            self.elig.mul_(self.elig_decay)
            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            # 版47：錐体 → 抑制の体 も予測誤差では学ばない（下の弱まりだけ。⟲ 版45 は的 ＝ 入口の平均で学び上下した：形A）
            to_inh = self.is_inh_r[r]
            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg)
            # 版47：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）
            se = self.e_src[e].long()
            ltd = to_inh & ~self.e_inh[e] & (self.hist_n[se] >= INH_LTD_N) & \
                (self.w[e] * self.c_exc[r] > INH_LTD_TH)
            if bool(ltd.any()):
                self.w[e[ltd]] *= INH_LTD_F
            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）
            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))
            # 興奮も抑制も 0 以上（版36。【ヒト】繋がりの符号は送り手の型で決まる。前は興奮が負になれた）
            self.w.clamp_(min=0.0)
            a_ = 1.0 / self.homeo_tau
            cnt = torch.zeros(self.n_recv_all, device=self.dev)
            cnt.scatter_add_(0, r, torch.ones(len(e), device=self.dev))
            self.recv_ema += a_ * (cnt - self.recv_ema)
            self.drive += a_ * (exc.abs() - self.drive)
            # シナプススケーリング：受け手ごとに入辺を一括で乗算して、自分の発火率を目標に保つ
            #   目標は【ヒト】0.45%/歩。撃ちすぎなら下げ、足りなければ上げる
            #   【ヒト】iPS 由来の皮質細胞：活動を止めると興奮性シナプス全体が強まって活動が戻る
            #   ⚠️ 起きているあいだは止める。【ヒト】起きているあいだは強くなる方へ、
            #      眠っているあいだに戻す（TMS の応答が起きている時間で増え、眠ると元に戻る）
            #   ⚠️ 下げるのは皮質だけ。海馬は眠っているあいだ強くなるという報告がある【げっ歯類】
            if getattr(self, "scaling_on", True):
                f = (1 - 0.01 * (self.rate_ema / self.target_recv - 1)).clamp(0.5, 1.5) ** a_
                if getattr(self, "sleep_mode", False):
                    f = torch.where(self.is_hippo, torch.ones_like(f), f)
                # ⟲ 下限 0.05・上限 20 は torch 版だけにあった（numpy 原本は1歩の変化を 0.5〜1.5 に収めるだけ）
                #    版23 で7領野が下限に当たって止まり、発火率が人の 10〜50倍のままだった
                f = torch.where(self.scale_skip, torch.ones_like(f), f)   # 版39b：人の値が無い受け手
                self.scale = self.scale * f

        # 出力もループなしで（誤差・表現・抑制を印で切り出す）
        err = (err_sig[self.is_err] > THETA).float()
        # 版48：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ。焼いた見込み・p は使わない
        #   ⟲ 版47 まで p ＞ 0.5【C】。撃つと見込みが上がって撃ち続けた（形B）
        #   【ヒト】速い抑制細胞は 1歩の刻みでは履歴にほぼ左右されない（Lee・Dalley 2023）
        d_inh = (exc - exc_mod)[self.is_inh_r] / (1.0 + ihn[self.is_inh_r])
        inh_spk = (d_inh > INH_TH).float()
        if learn:                                  # 細胞自身がこの歩に撃ったか（恒常性の観測）
            fired = torch.zeros(self.n_recv_all, device=self.dev)
            fired[self.is_err], fired[self.is_rep], fired[self.is_inh_r] = err, rep, inh_spk
            self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)
        self.pred_rep = pr
        for i, a in enumerate(self.areas):        # 領野ごとに切り出して持たせる
            a.pred_rep = pr[int(self.off_ch[i]):int(self.off_ch[i]) + a.n_ch]

        # 周回の中の発火は、次の周で他の体に届く（1周 ＝ シナプス1本ぶんの遅れ 1.1ms）
        w_new = torch.zeros(self.n_send_all, device=self.dev)
        w_new[self.within_index] = rep
        self._within = w_new if self._within is None else torch.maximum(self._within, w_new)
        self._e_cache = None                      # 送り手が変わったので作り直す

        if advance:
            # 版50：行（領野ごとに 誤差・表現・抑制）と同じ並びで足す
            #   ⟲ 版49 まで cat([全領野の rep, 全領野の rep, 全領野の inh])（種類ごとの並び）。
            #      先頭の領野の誤差の行のほかは、別の体の発火を履歴にしていた【測】chk_tb（10-04）
            now = torch.zeros(self.n_recv_all, device=self.dev)
            now[self.is_err] = rep[self.x_index[self.is_err]]
            now[self.is_rep] = rep[self.x_index[self.is_rep]]
            now[self.is_inh_r] = inh_spk
            self.tb = self.tb * self.decay + now.unsqueeze(1) * (1 - self.decay)
            src = torch.empty(self.n_src_all, device=self.dev)
            eo = io = 0
            for i, a in enumerate(self.areas):        # 入口の並べ直しだけは領野の形に依る
                lo = int(self.off_src[i]); c = int(self.off_ch[i])
                src[lo:lo + a.n_ch] = rep[c:c + a.n_ch]
                if a.n_ext:
                    src[lo + a.n_ch:lo + a.n_ch + a.n_ext] = ext_all[eo:eo + a.n_ext]
                    eo += a.n_ext
                src[lo + a.n_ch + a.n_ext:int(self.off_src[i + 1])] = inh_spk[io:io + a.n_inh]
                io += a.n_inh
            self.ch.step(src)
            for i, a in enumerate(self.areas):
                lo, hi = int(self.off_src[i]), int(self.off_src[i + 1])
                self.bufs[i] = torch.cat([src[lo:hi].unsqueeze(1), self.bufs[i][:, :-1]], 1)
            self.t += 1
            self._e_cache = None
            self._base = None        # 時間細胞と buf が進んだので作り直す
            self._within = None      # 歩をまたいだら、周回の中の発火は消える
        return rep * self.gate, err * self.gate, p[self.is_err]
