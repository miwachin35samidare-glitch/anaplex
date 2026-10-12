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
# 版53：辺1本の寿命（【ヒト】の神経の棘 約2週。マウスに移植した人の神経、移植後 3〜6か月。Linaro 2019）
LIFE_STEPS = 14 * 24 * 3600 * 1000.0 / DT_MS
LAZY_EPOCH = 1024                   # 版54：学びをまとめる区切り（刻み）。減り^(−1024) ≈ 5.6万で float32 に収まる
REWIRE_STEPS = 1000                 # 刈り込みを見る間隔（約10.7秒）。寿命は確率で合わせる【C の置き方】
# 版53：静か・撃ちすぎを見る発火の長い平均（1時間【C】）。起きているあいだだけ進める
LONG_TAU = 3600 * 1000.0 / DT_MS
# 版59：不応期（周の数）。【ヒト】起きている人の内側側頭葉の単一細胞で 3ms 以内の間隔は 0.10%（Ison 2011）⇒ 3ms ÷ 1.102ms ≒ 3周
REFRACT_PASSES = 3
# 版63：周回の中で撃った抑制の体の発火も次の周で届ける【ヒト】Szegedi 2017：籠細胞の抑制は錐体の発火の 4.1ms 後【み】10-11
#   ANAPLEX_WITHIN_INH=0 で版62 と同じ（比べる用）
import os as _os
WITHIN_INH = _os.environ.get("ANAPLEX_WITHIN_INH", "1") != "0"
# ⟲ 版60 の f-I は外した（10-11【み】「電極刺して生体にありえない電流を流して測った数字ってことでしょ？」）
# 版53：起きているあいだ上げる恒常性の速さ（【ラット】Hengen 2016：率が約60%落ちたのが 3.5日ほどで戻る）
#   足りない割合 0.6 が 3.5日続くと 2.5倍 ⇒ 1歩あたり ln(2.5) ÷ (0.6 × 3.5日の歩数)【C の当てはめ】
K_WAKE_UP = math.log(2.5) / (0.6 * 3.5 * 24 * 3600 * 1000.0 / DT_MS)


class TimeCellsGroup:
    """8領野ぶんの時間細胞を、ひとつの行列で持つ（領野をまたいで混ざらない）"""

    def __init__(self, areas):
        dev = areas[0].dev
        self.dev = dev
        self.areas = areas
        self.M = areas[0].ch.M
        self.n_src = [a.ch.n_src for a in areas]
        self.off = torch.cumsum(torch.tensor([0] + self.n_src, device=dev), 0)
        self.off_cpu = [sum(self.n_src[:k]) for k in range(len(self.n_src) + 1)]   # 版54b：境目は CPU の数で
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
            lo, hi = self.off_cpu[i], self.off_cpu[i + 1]
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
        # 版54b：撃った送り手だけ一番古い枠を 0 に（撃っていない所は同じ値を書き戻す。nonzero で待たない）
        oldest = self.age.argmax(1)
        rows = torch.arange(len(fired_all), device=self.dev)
        self.age[rows, oldest] = torch.where(fired_all > 0, torch.zeros_like(oldest), self.age[rows, oldest])


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
        self.rate_long = self.target_recv.clone()   # 版53：発火の長い平均（1時間。静か・撃ちすぎの判定）
        self.scale_up = torch.ones(self.n_recv_all, device=dev)   # 版53：起きているあいだ上げる恒常性（興奮だけに掛ける）
        # 学びの門（版40）：学んだ跡を辺ごとに持ち、脳幹が配る第三の信号 third の分だけ重みを変える
        self.elig = torch.zeros_like(self.w)
        self.elig_decay = math.exp(-1.0 / ELIG_TAU)
        # 版54：学びを触ったときにまとめる。elig は「跡 × 減り^(−k)」（区切りの頭からの刻み k）、k_start は辺がまだ足していない最初の刻み
        self.k_start = torch.ones(len(self.w), dtype=torch.int16, device=dev)
        self.A = torch.zeros(LAZY_EPOCH + 1, dtype=torch.float64, device=dev)
        self.kcur = 0
        self._prior_l = None        # 版54：焼いた網の見込み（1歩に1回）
        self._amp_buf = None        # 版54：短期可塑性の値を持ち続ける配列
        self._amp_prev = None
        self._new_e = None
        self._both_base = None
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
        # 版54b：決まった印は番号の列でも持つ（印で選ぶと、数を読むために GPU を待つ）
        self.err_i, self.rep_i, self.inh_i = [torch.nonzero(m).squeeze(1) for m in (self.is_err, self.is_rep, self.is_inh_r)]
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
        # 版58：自発のくじは周ごと（1周 ＝ 【ヒト】シナプス1本の遅れ 1.1ms）。1回あたりは1周ぶんの確率にする
        #   ⟲ 版57 まで1歩ぶんの確率で周ごと10回引き、相手に届く自発が置いた率の約10倍だった【測】measure_v57b
        self.spont_p_pass = 1.0 - (1.0 - self.spont_p) ** (1.0 / self.recur)
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
        self.mod_drive_inh = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 版66：調整する側のうち抑制の体は撃たせる送り手
        self.send_gain = torch.ones(self.n_send_all, device=dev)   # 送り手ごとの重さ。眠りで上からの戻りを落とす（版38）
        self.eta_gain = 1.0             # 学ぶ速さの掛け算。起きているあいだの驚き（版38）
        self.exc_gain = None            # リップルのときだけ、興奮の入力を底上げする
        self.sleep_mode = False
        self.is_hippo = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)
        self.rewire_every = REWIRE_STEPS      # 版53（⟲ 版52 まで 100歩ごとに受け手ごと1本）
        # 相手の数は受け手ごと（【ヒト】1体 30,000シナプス・相手 9,100人）。
        #   ⚠️ 先頭の領野の値を全体に当てていた。視床は時間細胞を持たないので口が小さく、
        #      その値が全部の領野に効いていた（視床の相手が少ないこと自体は人の形）
        self.n_partner = torch.cat([a.n_partner_vec for a in self.areas])   # 版47：抑制の体は相手が少ない
        self.budget = torch.cat([a.budget for a in self.areas])
        # 周回の中で撃った発火を、短い遅れの「1歩前」の枠に重ねるための番号
        self.within_index = torch.cat(
            [torch.arange(a.n_ch, device=dev) * a.D + buf_base + int(buf_off[i])
             for i, a in enumerate(self.areas)])
        # 版63：抑制の体の行（送り元の並び 表現・ext・抑制 の最後）。並びは inh_spk と同じ（領野ごとに n_inh）
        self.within_inh_index = torch.cat(
            [(a.n_ch + a.n_ext + torch.arange(a.n_inh, device=dev)) * a.D + buf_base + int(buf_off[i])
             for i, a in enumerate(self.areas)])
        # ここまでで、領野ごとの辺・短期可塑性・時間細胞の中身はすべて group 側へ写してある。
        # 元を抱えたままだと、同じものを2つ持つことになる
        self._release_areas()
        # 版52 A・B：領野の境目と complete の有無は変わらないので CPU の数で持つ
        #   ⟲ 版51 まで GPU の配列。周回ごとに int() で読むたび GPU を待っていた（1歩 約360回）
        self.off_recv, self.off_ch, self.off_src = (self.off_recv.cpu().numpy(), self.off_ch.cpu().numpy(),
                                                    self.off_src.cpu().numpy())
        self._complete_any = bool(self.complete.any())
        # 版55：焼いた網は型ごと。受け手の並びで同じ網が続く塊にまとめる（全部同じなら1つ）
        segs = []
        for i, a in enumerate(self.areas):
            lo, hi = int(self.off_recv[i]), int(self.off_recv[i + 1])
            if segs and segs[-1][2] is a.prior and segs[-1][1] == lo:
                segs[-1][1] = hi
            else:
                segs.append([lo, hi, a.prior])
        self._prior_segs = segs
        self._base_of = None

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
    def _expand(self, fired):
        """撃った送り手 → その辺の番号（送り手ごとに連ねて持っているので、start から切り出す）"""
        empty = torch.empty(0, dtype=torch.long, device=self.dev)
        if len(fired) == 0:
            return empty
        lens = self.start[fired + 1] - self.start[fired]
        # 版54b：総数を1回だけ読む（長さ 0 の送り手は repeat_interleave で何も出さないので、外さなくても同じ）
        tot = int(lens.sum())
        if tot == 0:
            return empty
        base = self.start[fired].repeat_interleave(lens, output_size=tot)
        off = torch.arange(tot, device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens, output_size=tot)
        return base + off

    def _fired_edges(self):
        # 版52 C：時間細胞と buf の辺は1歩に1回だけ作り、周回の中で新しく撃った送り手の辺だけ足す
        #   ⟲ 版51 まで周回ごとに全部作り直していた。足す辺の組は同じ（並びだけ違う）【測】measure_speed 10-07
        if self._e_cache is not None:
            return self._e_cache
        if self._base is None or self._base_of is not self._base:
            if self._base is None:                # 時間細胞と buf は1歩に1回でいい
                self._base = torch.cat([self.ch.spikes()] + [b.reshape(-1) for b in self.bufs])
            self._base_of = self._base
            self._base_on = self._base > 0
            self._base_e = self._expand(torch.nonzero(self._base_on).squeeze(1))
        if self._within is None:
            self._e_cache = self._base_e
            self._new_e = None
        else:                                     # 周回の中で撃った分も送り手になる
            new = torch.nonzero((self._within > 0) & ~self._base_on).squeeze(1)
            self._new_e = self._expand(new) if len(new) else None     # 版54：周回の中の新しい分（別に足す）
            self._e_cache = torch.cat([self._base_e, self._new_e]) if self._new_e is not None else self._base_e
        return self._e_cache

    @torch.no_grad()
    def _flush(self, idx):
        """版54：idx の辺に、まだ足していない学びの分（k_start から今の刻みまで）を足して 0 で止める"""
        if idx is None or len(idx) == 0:
            return
        ks = self.k_start[idx].long()
        pend = self.elig[idx] * (self.A[self.kcur] - self.A[ks - 1]).float()
        self.w[idx] = (self.w[idx] + pend).clamp(min=0.0)
        self.k_start.index_fill_(0, idx, self.kcur + 1)     # 版54b

    @torch.no_grad()
    def flush_all(self):
        """版54：全部の辺へまとめて足す（刈り込み・保存・区切りの前）。メモリを抑えるため塊ごとに"""
        n = len(self.w); step = 1 << 24
        for a in range(0, n, step):
            s = slice(a, min(n, a + step))
            pend = self.elig[s] * (self.A[self.kcur] - self.A[self.k_start[s].long() - 1]).float()
            self.w[s] = (self.w[s] + pend).clamp(min=0.0)
        self.k_start.fill_(self.kcur + 1)

    @torch.no_grad()
    def _new_epoch(self):
        """版54：区切り。全部の辺へ足し、跡を新しい区切りの頭（今の刻み）に合わせ直す"""
        self.flush_all()
        self.elig.mul_(self.elig_decay ** self.kcur)
        self.k_start.fill_(1)
        self.A.zero_()
        self.kcur = 0

    def _first_per_recv(self, key):
        """受け手ごとに key が一番小さい辺（key が inf の辺しか無い受け手は −1）"""
        o1 = torch.argsort(key)
        o2 = torch.argsort(self.e_recv[o1].long(), stable=True)
        order = o1[o2]
        r_sorted = self.e_recv[order].long()
        first = torch.ones_like(r_sorted, dtype=torch.bool)
        first[1:] = r_sorted[1:] != r_sorted[:-1]
        pick = torch.full((self.n_recv_all,), -1, dtype=torch.long, device=self.dev)
        ok = first & torch.isfinite(key[order])
        pick[r_sorted[ok]] = order[ok]
        return pick

    @torch.no_grad()
    def rewire(self):
        """版53：細胞の撃ち具合で刈り方を決める（受け手ごと。寿命 LIFE_STEPS の確率で）
        ふだん          一番弱い興奮の辺を切り、興奮で生やす   【ヒト】刈り込みで消えるのは小さい棘
        静か            抑制の辺を切り、興奮で生やす           【ネズミ】入力が減ると抑制が外れる ／【ヒト iPS】止めると興奮性が増える
        撃ちすぎ        一番弱い興奮の辺を切り、抑制で生やす   【ネズミ】学ぶと抑制が増える
        ⇒ 人の率ちょうどで釣り合い、興奮と抑制の割合が保たれる（【ヒト】割合は領野・個人でほぼ一定）
        ⟲ 版52 まで：種類を見ずに一番弱い1本を切り、送り元の割合で生やした。抑制は固定の強さ・興奮は 0 で生えて 0 へ痩せるので
           切られるのはいつも興奮 ⇒ 興奮 → 抑制 の片道（【測】measure_v52a 視床の興奮 64 → 11本）"""
        self.flush_all()          # 版54：一番弱い辺を選ぶ前に、学びの分を全部足しておく
        self.n_rewire = getattr(self, "n_rewire", 0) + 1        # 何回まわしたか
        n = self.n_recv_all
        r_all = self.e_recv.long()
        inh = self.e_inh
        n_e = torch.bincount(r_all[~inh], minlength=n).float()
        n_i = torch.bincount(r_all[inh], minlength=n).float()
        R = float(self.rewire_every)
        do_e = (torch.rand(n, device=self.dev, generator=self.gen) < R * n_e / LIFE_STEPS) & (n_e > 0)
        do_i = (torch.rand(n, device=self.dev, generator=self.gen) < R * n_i / LIFE_STEPS) & (n_i > 0)
        judge = ~self.scale_skip
        quiet = judge & (self.rate_long < self.target_recv)
        over = judge & (self.rate_long > self.target_recv)
        inf = torch.full_like(self.w, float("inf"))
        weak_e = self._first_per_recv(torch.where(inh, inf, self.w))                       # 一番弱い興奮
        rand_i = self._first_per_recv(torch.where(inh, torch.rand(len(self.w), device=self.dev,
                                                                    generator=self.gen), inf))   # でたらめな抑制1本
        cut_e = (do_e | (over & do_i)) & (weak_e >= 0)
        cut_i = do_i & ~over & (rand_i >= 0)
        cuts = torch.cat([weak_e[cut_e], rand_i[cut_i]])
        if len(cuts) == 0:
            return
        to_inh = torch.cat([(over & do_i)[cut_e], ~quiet[cut_i]])   # 生やす種類（True ＝ 抑制）
        # 版61：中継（no_lat_exc の領野）の表現・誤差の行には興奮を生やさない（【マカク】中継どうしの興奮はほぼ無い）
        if getattr(self, "_no_lat_exc_r", None) is None:
            fl = torch.tensor([bool(getattr(a, "no_lat_exc", False)) for a in self.areas], device=self.dev)
            self._no_lat_exc_r = fl[self.area_of] & ~self.is_inh_r
        to_inh = to_inh | self._no_lat_exc_r[self.e_recv[cuts].long()]
        self.n_cut = getattr(self, "n_cut", 0) + len(cuts)      # 何本 切ったか
        self.n_e2i = getattr(self, "n_e2i", 0) + int((over & do_i & cut_e).sum())   # 興奮 → 抑制（撃ちすぎ）
        self.n_i2e = getattr(self, "n_i2e", 0) + int((quiet & cut_i).sum())          # 抑制 → 興奮（静か）
        # 生やす先：受け手と同じ領野の、決めた種類の送り元から一様に
        if getattr(self, "_pool_ei", None) is None:
            self._pool_ei = [(p[~self.is_inh_all[p]], p[self.is_inh_all[p]]) for p in self.pool_send]
            # 版57：海馬の抑制の体へ生やす興奮は、遅れの列の興奮の行（時間細胞ではない）から
            nb = self.n_send_all - sum(a.n_src * a.D for a in self.areas)      # 遅れの列の始まり（通し番号）
            self._pool_e_rep = [pe[pe >= nb] for pe, _ in self._pool_ei]
            self._inh_rep_area = [bool(getattr(a, "inh_exc_from_rep", False)) for a in self.areas]
            # 版64：inh_slice の領野の表現・誤差の行へ生やす抑制は、抑制の体の1歩前の列から
            self._inh_slice_area = [bool(getattr(a, "inh_slice", False)) for a in self.areas]
            oi = 0
            self._pool_i_col0 = []
            for a in self.areas:
                self._pool_i_col0.append(self.within_inh_index[oi:oi + a.n_inh])
                oi += a.n_inh
        ar = self.area_of[self.e_recv[cuts].long()]
        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)
        for i in range(len(self.areas)):
            for kind in (0, 1):
                m = (ar == i) & (to_inh == bool(kind))
                k = int(m.sum())
                if k:
                    pool = self._pool_ei[i][kind]
                    new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev, generator=self.gen)]
                    if kind == 1 and self._inh_slice_area[i]:   # 版64：表現・誤差の受け手は抑制の体の1歩前の列から
                        mr = m & ~self.is_inh_r[self.e_recv[cuts].long()]
                        kr = int(mr.sum())
                        if kr:
                            pc = self._pool_i_col0[i]
                            new[mr] = pc[torch.randint(0, len(pc), (kr,), device=self.dev, generator=self.gen)]
                    if kind == 0 and self._inh_rep_area[i]:     # 版57：抑制の体の受け手だけ引き直す
                        mi = m & self.is_inh_r[self.e_recv[cuts].long()]
                        ki = int(mi.sum())
                        if ki:
                            pr_ = self._pool_e_rep[i]
                            new[mi] = pr_[torch.randint(0, len(pr_), (ki,), device=self.dev, generator=self.gen)]
        self.e_src[cuts] = new.to(self.e_src.dtype)
        o = torch.argsort(self.e_src)
        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]
        self.elig = self.elig[o]
        self.k_start = self.k_start[o]     # 版54（flush_all のあとなので全部同じ値）
        self._amp_buf = None               # 版54：辺の並びが変わったので作り直す
        self._amp_prev = None
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
            # 版53：生やした辺の短期可塑性は新しい種類の組から（⟲ 版52 まで切った辺の行を持ち越していた）
            self.stp.redraw(cuts, self.e_inh[cuts], self.is_inh_r[self.e_recv[cuts].long()])
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
    def _sum(self, e, first):
        """版54：辺 e の分を受け手へ足す（版53 の _pass の中身と同じ式）。返り値は [興奮 ｜ 抑制]・調整する側の興奮"""
        both = torch.zeros(self.n_recv_all * 2, device=self.dev)
        mod = torch.zeros(self.n_recv_all, device=self.dev)
        if e is None or len(e) == 0:
            return both, mod
        # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0
        re_ = self.e_recv[e].long()
        se_ = self.e_src[e].long()
        amp = torch.where(self.e_inh[e], 4.0, self.c_exc[re_]) * self.w[e]
        if self.stp is not None:
            if first:      # 放出は1周目だけ。値の配列は持ち続け、前の歩に入れた所だけ 1 に戻す（版53 は毎歩作り直し）
                if self._amp_buf is None or len(self._amp_buf) != len(self.w):
                    self._amp_buf = torch.ones(len(self.w), device=self.dev)
                elif self._amp_prev is not None:
                    self._amp_buf.index_fill_(0, self._amp_prev, 1.0)     # 版54b
                self._amp_buf[e] = self.stp.release_at(e, self.t)
                self._amp_prev = e
                self._amp = self._amp_buf
            amp = amp * self._amp[e]
        amp = amp * self.send_gain[se_]   # 脳幹：眠りで上からの戻りを落とす（版38）
        # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）
        both.scatter_add_(0, re_ + self.n_recv_all * self.e_inh[e].long(), amp)
        mm = self.mod_send[se_] & ~self.e_inh[e]
        mm = mm & ~(self.mod_drive_inh[se_] & self.is_inh_r[re_])   # 版66：印の送り手 → 抑制の体 は撃たせる側
        mod = mod.index_add(0, re_, amp * mm)     # 版54b：選ばない辺は 0 を足す（選んで足すのと同じ値）
        return both, mod

    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
        if first and self._base is not None:   # 版47：送り手ごとの直近 INH_LTD_WIN 歩の発火を数える
            now = (self._base > 0).to(torch.uint8)
            self.hist_n += now.to(torch.int16) - self.hist[self.hist_i].to(torch.int16)
            self.hist[self.hist_i] = now
            self.hist_i = (self.hist_i + 1) % INH_LTD_WIN
        # 版54：1周目は撃った辺（時間細胞と buf）をそのまま。2周目は学んだあとの重みで同じ辺の分を作って残し、
        #   3〜10周目はそれを使い回す。どの周も、周回の中で新しく撃った送り手の分だけ足す（足す辺の組は版53 と同じ）
        if first:
            self._flush(e)
            both, exc_mod = self._sum(e, True)
            self._both_base = None
        else:
            if self._both_base is None:
                self._flush(self._base_e)
                self._both_base, self._mod_base = self._sum(self._base_e, False)
            if self._new_e is not None:
                self._flush(self._new_e)
                b2, m2 = self._sum(self._new_e, False)
                both, exc_mod = self._both_base + b2, self._mod_base + m2
            else:
                both, exc_mod = self._both_base, self._mod_base
        exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]
        exc = exc * (self.scale * self.scale_up); ihn = ihn * self.scale   # 版53：起きて上げた分は興奮だけに
        exc_mod = exc_mod * (self.scale * self.scale_up)
        # リップルのとき、興奮の入力だけ底上げする（【ヒト】錐体細胞 5.2倍）
        #   抑制は外さない。抑制性の体は、増えた発火を受けて自分で強まる（人では 3.8倍）
        if getattr(self, "exc_gain", None) is not None:
            exc = exc * self.exc_gain
            exc_mod = exc_mod * self.exc_gain
        if self._prior_l is None:        # 版54：tb は歩の終わりにしか変わらない ⇒ 1歩に1回
            self._prior_l = (self.prior.logit(self.tb) if len(self._prior_segs) == 1 else          # 版55
                             torch.cat([pr.logit(self.tb[lo:hi]) for lo, hi, pr in self._prior_segs]))
        p = torch.sigmoid(self._prior_l + exc / (1.0 + ihn))

        # 表現の発火は学びの前に出す（版36。的に使うため）。学びはこの周の exc・ihn を変えないので値は同じ
        pr = p[self.rep_i]
        # 表現のニューロンも、シナプスの合計が閾値を越えたら撃つ
        #   【ヒト】L2/3 錐体細胞は、同時に活動した興奮性シナプス 134±28本 で 50% の確率で撃つ
        #   （人が毎歩受けている入力 135本＝30,000シナプス×0.45%/歩 と、ほぼ同じ）
        #   ⇒ 平均的な入力がちょうど閾値。だから疎で、入力の揺らぎに敏感になる
        #   抑制（短絡）はこの合計に掛かるので、ここで初めて表現の発火に効く
        # 調整する側（上からの戻り・指令の写し）は予測 p には入るが、表現を撃たせない（版39）
        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])
        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）
        if self._complete_any:           # 入力が無いところは予測が足しになる（撃たせはしない）。版52 B
            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th * self.th_gain) & (x_all > 0)
            rep = torch.where(fill, torch.ones_like(rep), rep)
        rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,
                                             generator=self.gen) < self.spont_p_pass).float())   # 版58

        # 的：誤差と表現は自分の入口、抑制はその領野の入口の平均（ループなしで一度に）
        # 誤差の的：下からの入力でこの細胞が撃つか（0か1）。予測はシグモイドなので的も 0か1 の目盛り
        # 的は「下からの入力（本数 × 重さ）がこの細胞を撃たせるか」。学んでいる重み（抑制）は入れない
        #   ⟲ 版31 で表現と同じ抑制を入れたら、学びが抑制を育てて的そのものを消し、皮質が全部止まった
        #      （的が 0 ⇒ e ＝ −p ⇒ 抑制が強まる ⇒ 的がさらに立たない、の輪）。的は学ぶ量に依存させない
        if first:      # 版54b：入力 x は1歩のあいだ同じ ⇒ 的は1周目に出して使い回す
            x_t = torch.where(self.tgt_by_fire, (x_all * self.x_gain > self.fire_th).float(), x_all)
            self.last_xt = x_t                         # 測る道具用
            tgt = x_t[self.x_index]
            self._tgt = torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], tgt)
        tgt = self._tgt
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
            # 版54：全部の辺の跡を減らす代わりに、刻みを1つ進める（跡は 減り^(−k) を掛けて持つ）
            if self.kcur >= LAZY_EPOCH:
                self._new_epoch()
            k = self.kcur + 1
            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            # 版47：錐体 → 抑制の体 も予測誤差では学ばない（下の弱まりだけ。⟲ 版45 は的 ＝ 入口の平均で学び上下した：形A）
            to_inh = self.is_inh_r[r]
            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg) * (self.elig_decay ** (-k))
            # 版47：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）
            se = self.e_src[e].long()
            ltd = to_inh & ~self.e_inh[e] & (self.hist_n[se] >= INH_LTD_N) & \
                (self.w[e] * self.c_exc[r] > INH_LTD_TH)
            self.w[e] = torch.where(ltd, self.w[e] * INH_LTD_F, self.w[e])     # 版54b：判定で待たない
            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）
            # 版54：全部の辺へ足す代わりに A を1つ進める（辺は読む前にまとめて足す）
            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(()) * (self.elig_decay ** k)
            self.kcur = k
            # 興奮も抑制も 0 以上（版36。【ヒト】繋がりの符号は送り手の型で決まる。前は興奮が負になれた）
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
                # 版53：眠りは下げるだけ（⟲ 版52 まで両向き。上げる向きは人の根拠なしで【み】保留 10-02）
                f = (1 - 0.01 * (self.rate_ema / self.target_recv - 1)).clamp(0.5, 1.0) ** a_
                if getattr(self, "sleep_mode", False):
                    f = torch.where(self.is_hippo, torch.ones_like(f), f)
                # ⟲ 下限 0.05・上限 20 は torch 版だけにあった（numpy 原本は1歩の変化を 0.5〜1.5 に収めるだけ）
                #    版23 で7領野が下限に当たって止まり、発火率が人の 10〜50倍のままだった
                f = torch.where(self.scale_skip, torch.ones_like(f), f)   # 版39b：人の値が無い受け手
                self.scale = self.scale * f
            else:
                # 版53：起きているあいだは上げるだけ（【ラット】Hengen 2016。【み】10-08）。静かな細胞の興奮を強める
                up = torch.exp(K_WAKE_UP * (1.0 - self.rate_long / self.target_recv).clamp(min=0.0))
                self.scale_up = self.scale_up * torch.where(self.scale_skip, torch.ones_like(up), up)

        # 出力もループなしで（誤差・表現・抑制を印で切り出す）
        err = (err_sig[self.err_i] > THETA).float()
        # 版48：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ。焼いた見込み・p は使わない
        #   ⟲ 版47 まで p ＞ 0.5【C】。撃つと見込みが上がって撃ち続けた（形B）
        #   【ヒト】速い抑制細胞は 1歩の刻みでは履歴にほぼ左右されない（Lee・Dalley 2023）
        d_inh = (exc - exc_mod)[self.inh_i] / (1.0 + ihn[self.inh_i])
        inh_spk = (d_inh > INH_TH).float()
        # 版59：不応期。撃った周から REFRACT_PASSES 周たつまで撃たない（歩をまたいでも数える）
        if first:
            self._learn_step = learn
            if getattr(self, "_last_fire", None) is None:
                self._pc = 0
                self._last_fire = [torch.full(v.shape, -10 ** 6, dtype=torch.long, device=self.dev)
                                   for v in (rep, err, inh_spk)]
        self._pc += 1
        rep, err, inh_spk = [v * ((self._pc - lf) >= REFRACT_PASSES).float()
                             for v, lf in zip((rep, err, inh_spk), self._last_fire)]
        self._last_fire = [torch.where(v > 0, torch.full_like(lf, self._pc), lf)
                           for v, lf in zip((rep, err, inh_spk), self._last_fire)]
        # 版59：この歩のどこかで撃ったか（周の発火の重ね）
        self._any = [rep, err, inh_spk] if first else [torch.maximum(a, v) for a, v in zip(self._any, (rep, err, inh_spk))]
        if advance and getattr(self, "_learn_step", False):   # 細胞自身がこの歩に撃ったか（恒常性の観測。版59：歩のどこか）
            fired = torch.zeros(self.n_recv_all, device=self.dev)
            fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any
            self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)
            if not getattr(self, "scaling_on", True):   # 版53：長い平均は起きているあいだだけ
                self.rate_long += (1.0 / LONG_TAU) * (fired - self.rate_long)
        self.pred_rep = pr
        for i, a in enumerate(self.areas):        # 領野ごとに切り出して持たせる
            a.pred_rep = pr[int(self.off_ch[i]):int(self.off_ch[i]) + a.n_ch]

        # 周回の中の発火は、次の周で他の体に届く（1周 ＝ シナプス1本ぶんの遅れ 1.1ms）
        w_new = torch.zeros(self.n_send_all, device=self.dev)
        w_new[self.within_index] = rep
        if WITHIN_INH:                            # 版63：抑制の体の発火も次の周で届く
            w_new[self.within_inh_index] = inh_spk
        self._within = w_new if self._within is None else torch.maximum(self._within, w_new)
        self._e_cache = None                      # 送り手が変わったので作り直す

        if advance:
            rep, err, inh_spk = self._any     # 版59：最後の周ではなく、この歩のどこかで撃ったか
            # 版50：行（領野ごとに 誤差・表現・抑制）と同じ並びで足す
            #   ⟲ 版49 まで cat([全領野の rep, 全領野の rep, 全領野の inh])（種類ごとの並び）。
            #      先頭の領野の誤差の行のほかは、別の体の発火を履歴にしていた【測】chk_tb（10-04）
            now = torch.zeros(self.n_recv_all, device=self.dev)
            now[self.err_i] = rep[self.x_index[self.err_i]]
            now[self.rep_i] = rep[self.x_index[self.rep_i]]
            now[self.inh_i] = inh_spk
            self.tb = self.tb * self.decay + now.unsqueeze(1) * (1 - self.decay)
            self._prior_l = None      # 版54
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
        return rep * self.gate, err * self.gate, p[self.err_i]
