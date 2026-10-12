"""版53：刈り方を細胞の撃ち具合で決める ／ 恒常性の目標を人の起きているときの率に ／ 起きているあいだは上げる恒常性
（コンテナの写し用。編集は claude/道具_0926/apply_v53.py と同じ。頭書きは短くした）
"""
import ast, pathlib, sys

def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s

# ── torch_area.py ──
A = [("""HUMAN_INPUT_PER_STEP = 135.0   # 【ヒト】毎歩 受けている入力の本数（30,000シナプス × 0.45%/歩）
RATE_HUMAN = 0.0045             # 【ヒト】皮質の細胞が 1歩（10.67ms）に撃つ確率 0.45%（約 0.4Hz）
""", """# 【ヒト】起きている皮質の錐体 3.30 ± 2.52Hz（Verma 2024、覚醒・リップルの外、Utah II/III層、3人・437体）⇒ 1歩に 3.52%
#   ⟲ 版52 までは 0.45%/歩。前の作りの 1.5Hz（【ラット・サル】）× 1歩 3ms が、1歩 10.67ms になっても残った数（約 0.42Hz）
#     版45b（10-02【み】「間違ってるなら直して」）の直しが arkhe に入らないまま版が進んでいた（10-08 grep）。版53 で入れた
RATE_HUMAN = 3.30 * DT_MS / 1000.0
HUMAN_INPUT_PER_STEP = 30000 * RATE_HUMAN   # 毎歩 受けている入力の本数（30,000シナプス × RATE_HUMAN）。どこからも使っていない
"""),
     ("""        self.gen = gen

    @torch.no_grad()
    def release_at(self, e, t):""", """        self.gen = gen
        self.grp_off = [off[k] for k in keys]          # 版53：生やした辺の行を引き直すため
        self.grp_len = [len(d["行"][k]) for k in keys]

    @torch.no_grad()
    def redraw(self, idx, send_inh, recv_inh):
        \"\"\"版53：生やした辺の行を、新しい種類の組から引き直して状態を初めに戻す\"\"\"
        grp = send_inh.long() * 2 + recv_inh.long()
        for gi in range(4):
            m = grp == gi
            n = int(m.sum())
            if n:
                self.row[idx[m]] = (self.grp_off[gi] + torch.randint(0, self.grp_len[gi], (n,), device=idx.device,
                                                                     generator=self.gen)).to(self.row.dtype)
        dep = self.tbl[self.row[idx].long(), 1] < 0
        self.deplete[idx] = dep
        self.state[idx] = dep.float()
        self.fac[idx] = 0.0

    @torch.no_grad()
    def release_at(self, e, t):""")]

# ── torch_group2.py ──
OLD_REWIRE_HEAD = '''    @torch.no_grad()
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
'''
NEW_REWIRE_HEAD = '''    def _first_per_recv(self, key):
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
        self.n_cut = getattr(self, "n_cut", 0) + len(cuts)      # 何本 切ったか
        self.n_e2i = getattr(self, "n_e2i", 0) + int((over & do_i & cut_e).sum())   # 興奮 → 抑制（撃ちすぎ）
        self.n_i2e = getattr(self, "n_i2e", 0) + int((quiet & cut_i).sum())          # 抑制 → 興奮（静か）
        # 生やす先：受け手と同じ領野の、決めた種類の送り元から一様に
        if getattr(self, "_pool_ei", None) is None:
            self._pool_ei = [(p[~self.is_inh_all[p]], p[self.is_inh_all[p]]) for p in self.pool_send]
        ar = self.area_of[self.e_recv[cuts].long()]
        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)
        for i in range(len(self.areas)):
            for kind in (0, 1):
                m = (ar == i) & (to_inh == bool(kind))
                k = int(m.sum())
                if k:
                    pool = self._pool_ei[i][kind]
                    new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev, generator=self.gen)]
        self.e_src[cuts] = new.to(self.e_src.dtype)
'''
G2 = [("ELIG_TAU = 1000.0 / DT_MS\n", """ELIG_TAU = 1000.0 / DT_MS
# 版53：辺1本の寿命（【ヒト】の神経の棘 約2週。マウスに移植した人の神経、移植後 3〜6か月。Linaro 2019）
LIFE_STEPS = 14 * 24 * 3600 * 1000.0 / DT_MS
REWIRE_STEPS = 1000                 # 刈り込みを見る間隔（約10.7秒）。寿命は確率で合わせる【C の置き方】
# 版53：静か・撃ちすぎを見る発火の長い平均（1時間【C】）。起きているあいだだけ進める
LONG_TAU = 3600 * 1000.0 / DT_MS
# 版53：起きているあいだ上げる恒常性の速さ（【ラット】Hengen 2016：率が約60%落ちたのが 3.5日ほどで戻る）
#   足りない割合 0.6 が 3.5日続くと 2.5倍 ⇒ 1歩あたり ln(2.5) ÷ (0.6 × 3.5日の歩数)【C の当てはめ】
K_WAKE_UP = math.log(2.5) / (0.6 * 3.5 * 24 * 3600 * 1000.0 / DT_MS)
"""),
      ("        self.rewire_every = self.areas[0].rewire_every\n",
       "        self.rewire_every = REWIRE_STEPS      # 版53（⟲ 版52 まで 100歩ごとに受け手ごと1本）\n"),
      ("        self.scale_skip = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)   # 人の値が無い受け手は下げない\n",
       "        self.scale_skip = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)   # 人の値が無い受け手は下げない\n"
       "        self.rate_long = self.target_recv.clone()   # 版53：発火の長い平均（1時間。静か・撃ちすぎの判定）\n"
       "        self.scale_up = torch.ones(self.n_recv_all, device=dev)   # 版53：起きているあいだ上げる恒常性（興奮だけに掛ける）\n"),
      (OLD_REWIRE_HEAD, NEW_REWIRE_HEAD),
      ("""        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        ei = cuts[self.is_inh_r[self.e_recv[cuts].long()] & ~self.e_inh[cuts]]
        if len(ei):                    # 版47：開通した 錐体 → 抑制の体 も人の分布から
            self.w[ei] = self._inh_w(self.e_recv[ei].long())
        self.elig[cuts] = 0.0
        if self.stp is not None:
            self.stp.reorder(o)
""", """        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        ei = cuts[self.is_inh_r[self.e_recv[cuts].long()] & ~self.e_inh[cuts]]
        if len(ei):                    # 版47：開通した 錐体 → 抑制の体 も人の分布から
            self.w[ei] = self._inh_w(self.e_recv[ei].long())
        self.elig[cuts] = 0.0
        if self.stp is not None:
            self.stp.reorder(o)
            # 版53：生やした辺の短期可塑性は新しい種類の組から（⟲ 版52 まで切った辺の行を持ち越していた）
            self.stp.redraw(cuts, self.e_inh[cuts], self.is_inh_r[self.e_recv[cuts].long()])
"""),
      ("""        exc = exc * self.scale; ihn = ihn * self.scale
        exc_mod = exc_mod * self.scale
""", """        exc = exc * (self.scale * self.scale_up); ihn = ihn * self.scale   # 版53：起きて上げた分は興奮だけに
        exc_mod = exc_mod * (self.scale * self.scale_up)
"""),
      ("""            if getattr(self, "scaling_on", True):
                f = (1 - 0.01 * (self.rate_ema / self.target_recv - 1)).clamp(0.5, 1.5) ** a_
""", """            if getattr(self, "scaling_on", True):
                # 版53：眠りは下げるだけ（⟲ 版52 まで両向き。上げる向きは人の根拠なしで【み】保留 10-02）
                f = (1 - 0.01 * (self.rate_ema / self.target_recv - 1)).clamp(0.5, 1.0) ** a_
"""),
      ("""                f = torch.where(self.scale_skip, torch.ones_like(f), f)   # 版39b：人の値が無い受け手
                self.scale = self.scale * f
""", """                f = torch.where(self.scale_skip, torch.ones_like(f), f)   # 版39b：人の値が無い受け手
                self.scale = self.scale * f
            else:
                # 版53：起きているあいだは上げるだけ（【ラット】Hengen 2016。【み】10-08）。静かな細胞の興奮を強める
                up = torch.exp(K_WAKE_UP * (1.0 - self.rate_long / self.target_recv).clamp(min=0.0))
                self.scale_up = self.scale_up * torch.where(self.scale_skip, torch.ones_like(up), up)
"""),
      ("""            self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)
""", """            self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)
            if not getattr(self, "scaling_on", True):   # 版53：長い平均は起きているあいだだけ
                self.rate_long += (1.0 / LONG_TAU) * (fired - self.rate_long)
""")]

# ── torch_brain.py ──
BR = [("""            elif nm in ("聴覚", "視覚", "連合", "高次", "前頭前野", "嗅内浅", "嗅内深"):
                gg.target_recv[ih] = hz(2.0)
""", """            elif nm in ("聴覚", "視覚", "連合", "高次", "前頭前野"):
                gg.target_recv[ih] = hz(4.70)   # 版53：【ヒト】Verma 2024 覚醒の抑制（⟲ 2Hz は Peyrache 2012 の夜通しの値）
            elif nm in ("嗅内浅", "嗅内深", "CA3", "CA1"):
                # 版53：【ヒト】Le Van Quyen 2008 海馬 錐体 3.1Hz・抑制 5.1Hz（嗅内も同じ当てはめ。健診 13 と同じ【C】）
                gg.target_recv[lo:lo + 2 * nc] = hz(3.1)
                gg.target_recv[ih] = hz(5.1)
            elif nm == "歯状回":                # 版53：人の率の値が無い（項目に応じるのは 2.5% 未満、は率の目標ではない）
                gg.scale_skip[lo:hi] = True
"""),
      ("        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n",
       "        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n"
       "        gg.rate_long = gg.target_recv.clone()    # 版53\n")]

# ── babble.py ──
H_OLD = "#      脳の中の乱数を走らせるたびに OS の乱れから引き直す（種はログ。--seed N で再生）【み】10-07「脳ってそうだったか？」\n"
BB = [("VERSION = 52          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51,)      # 網の作りが同じで、state.pt をそのまま読める版（版52 は速さと乱数の種だけ）",
       "VERSION = 53          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52)   # 網の作りが同じで、state.pt をそのまま読める版（版52 は速さと乱数の種、版53 は刈り方と恒常性）"),
      (H_OLD, H_OLD +
       "#  53  刈り方を細胞の撃ち具合で（ふだん 弱い興奮→興奮 ／ 静か 抑制→興奮 ／ 撃ちすぎ 弱い興奮→抑制）。辺の寿命 14日【ヒト神経】／\n"
       "#      恒常性の目標を人の起きているときの率に（皮質 3.30・抑制 4.70Hz、海馬・嗅内 3.1・5.1Hz）／ 起きているあいだは上げるだけ【ラット】・眠りは下げるだけ\n"
       "#      【測】measure_v52a・v52b（刈り込みの片道が下がりの大元）【み】10-08・10-09\n"),
      ('"rate_ema": g.rate_ema, "tb": g.tb,', '"rate_ema": g.rate_ema, "rate_long": g.rate_long, "scale_up": g.scale_up, "tb": g.tb,'),
      ("""    if "rate_ema" in d:                    # 版20 から（それより前の state には無い）
        g.rate_ema = d["rate_ema"]
""", """    if "rate_ema" in d:                    # 版20 から（それより前の state には無い）
        g.rate_ema = d["rate_ema"]
    if "rate_long" in d:                   # 版53 から（それより前は目標から始める）
        g.rate_long, g.scale_up = d["rate_long"], d["scale_up"]
""")]

out = [edit("torch_area.py", A), edit("torch_group2.py", G2), edit("torch_brain.py", BR), edit("babble.py", BB)]
for p, s in out:
    p.write_text(s)
print("✅ 版53 を当てた（torch_area・torch_group2・torch_brain・babble）。網の作りは同じなので版51・52 の state.pt も読める")
