"""版54：1歩を縮める（意味は変えない）（コンテナの写し用。編集は claude/道具_0926/apply_v54.py と同じ）"""
import ast, pathlib, sys


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


G2 = []
G2.append(("LIFE_STEPS = 14 * 24 * 3600 * 1000.0 / DT_MS\n",
           "LIFE_STEPS = 14 * 24 * 3600 * 1000.0 / DT_MS\n"
           "LAZY_EPOCH = 1024                   # 版54：学びをまとめる区切り（刻み）。減り^(−1024) ≈ 5.6万で float32 に収まる\n"))
G2.append(("""        self.elig = torch.zeros_like(self.w)
        self.elig_decay = math.exp(-1.0 / ELIG_TAU)
""", """        self.elig = torch.zeros_like(self.w)
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
"""))
G2.append(("""    def _first_per_recv(self, key):
""", """    @torch.no_grad()
    def _flush(self, idx):
        \"\"\"版54：idx の辺に、まだ足していない学びの分（k_start から今の刻みまで）を足して 0 で止める\"\"\"
        if idx is None or len(idx) == 0:
            return
        ks = self.k_start[idx].long()
        pend = self.elig[idx] * (self.A[self.kcur] - self.A[ks - 1]).float()
        self.w[idx] = (self.w[idx] + pend).clamp(min=0.0)
        self.k_start[idx] = self.kcur + 1

    @torch.no_grad()
    def flush_all(self):
        \"\"\"版54：全部の辺へまとめて足す（刈り込み・保存・区切りの前）。メモリを抑えるため塊ごとに\"\"\"
        n = len(self.w); step = 1 << 24
        for a in range(0, n, step):
            s = slice(a, min(n, a + step))
            pend = self.elig[s] * (self.A[self.kcur] - self.A[self.k_start[s].long() - 1]).float()
            self.w[s] = (self.w[s] + pend).clamp(min=0.0)
        self.k_start.fill_(self.kcur + 1)

    @torch.no_grad()
    def _new_epoch(self):
        \"\"\"版54：区切り。全部の辺へ足し、跡を新しい区切りの頭（今の刻み）に合わせ直す\"\"\"
        self.flush_all()
        self.elig.mul_(self.elig_decay ** self.kcur)
        self.k_start.fill_(1)
        self.A.zero_()
        self.kcur = 0

    def _first_per_recv(self, key):
"""))
G2.append(("""        self.n_rewire = getattr(self, "n_rewire", 0) + 1        # 何回まわしたか
""", """        self.flush_all()          # 版54：一番弱い辺を選ぶ前に、学びの分を全部足しておく
        self.n_rewire = getattr(self, "n_rewire", 0) + 1        # 何回まわしたか
"""))
G2.append(("""        self.elig = self.elig[o]
        self.e_inh = torch.index_select(self.is_inh_all, 0, self.e_src)
""", """        self.elig = self.elig[o]
        self.k_start = self.k_start[o]     # 版54（flush_all のあとなので全部同じ値）
        self._amp_buf = None               # 版54：辺の並びが変わったので作り直す
        self._amp_prev = None
        self.e_inh = torch.index_select(self.is_inh_all, 0, self.e_src)
"""))
G2.append(("""        if self._within is None:
            self._e_cache = self._base_e
        else:                                     # 周回の中で撃った分も送り手になる
            new = torch.nonzero((self._within > 0) & ~self._base_on).squeeze(1)
            self._e_cache = torch.cat([self._base_e, self._expand(new)]) if len(new) else self._base_e
        return self._e_cache
""", """        if self._within is None:
            self._e_cache = self._base_e
            self._new_e = None
        else:                                     # 周回の中で撃った分も送り手になる
            new = torch.nonzero((self._within > 0) & ~self._base_on).squeeze(1)
            self._new_e = self._expand(new) if len(new) else None     # 版54：周回の中の新しい分（別に足す）
            self._e_cache = torch.cat([self._base_e, self._new_e]) if self._new_e is not None else self._base_e
        return self._e_cache
"""))
OLD_SUM = """        exc = torch.zeros(self.n_recv_all, device=self.dev)
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
"""
NEW_SUM = """        # 版54：1周目は撃った辺（時間細胞と buf）をそのまま。2周目は学んだあとの重みで同じ辺の分を作って残し、
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
"""
G2.append((OLD_SUM, NEW_SUM))
G2.append(("""    def _pass(self, x_all, ext_all, learn, advance, first):
""", """    def _sum(self, e, first):
        \"\"\"版54：辺 e の分を受け手へ足す（版53 の _pass の中身と同じ式）。返り値は [興奮 ｜ 抑制]・調整する側の興奮\"\"\"
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
                    self._amp_buf[self._amp_prev] = 1.0
                self._amp_buf[e] = self.stp.release_at(e, self.t)
                self._amp_prev = e
                self._amp = self._amp_buf
            amp = amp * self._amp[e]
        amp = amp * self.send_gain[se_]   # 脳幹：眠りで上からの戻りを落とす（版38）
        # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）
        both.scatter_add_(0, re_ + self.n_recv_all * self.e_inh[e].long(), amp)
        mm = self.mod_send[se_] & ~self.e_inh[e]
        mod = mod.index_add(0, re_[mm], amp[mm])
        return both, mod

    def _pass(self, x_all, ext_all, learn, advance, first):
"""))
G2.append(("        p = torch.sigmoid(self.prior.logit(self.tb) + exc / (1.0 + ihn))\n",
           "        if self._prior_l is None:        # 版54：tb は歩の終わりにしか変わらない ⇒ 1歩に1回\n"
           "            self._prior_l = self.prior.logit(self.tb)\n"
           "        p = torch.sigmoid(self._prior_l + exc / (1.0 + ihn))\n"))
G2.append(("""            self.elig.mul_(self.elig_decay)
""", """            # 版54：全部の辺の跡を減らす代わりに、刻みを1つ進める（跡は 減り^(−k) を掛けて持つ）
            if self.kcur >= LAZY_EPOCH:
                self._new_epoch()
            k = self.kcur + 1
"""))
G2.append(("""            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg)
""", """            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg) * (self.elig_decay ** (-k))
"""))
G2.append(("""            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))
""", """            # 版54：全部の辺へ足す代わりに A を1つ進める（辺は読む前にまとめて足す）
            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(()) * (self.elig_decay ** k)
            self.kcur = k
"""))
G2.append(("""            self.w.clamp_(min=0.0)
            a_ = 1.0 / self.homeo_tau
""", """            a_ = 1.0 / self.homeo_tau
"""))
G2.append(("""            self.tb = self.tb * self.decay + now.unsqueeze(1) * (1 - self.decay)
""", """            self.tb = self.tb * self.decay + now.unsqueeze(1) * (1 - self.decay)
            self._prior_l = None      # 版54
"""))

H_OLD = "#      【測】measure_v52a・v52b（刈り込みの片道が下がりの大元）【み】10-08・10-09\n"
BB = [("VERSION = 53          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52)   # 網の作りが同じで、state.pt をそのまま読める版（版52 は速さと乱数の種、版53 は刈り方と恒常性）",
       "VERSION = 54          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52, 53)   # 網の作りが同じで、state.pt をそのまま読める版（版52 速さと乱数の種、版53 刈り方と恒常性、版54 速さ）"),
      (H_OLD, H_OLD + "#  54  意味を変えずに1歩を縮める：周回の使い回し ／ 学びを触ったときにまとめる ／ 焼いた網を1歩1回（【み】10-10）\n"),
      ("""def save(b, done, path=STATE, log=None):
    g = b.g
""", """def save(b, done, path=STATE, log=None):
    g = b.g
    g.flush_all()                          # 版54：学びの分を全部の辺へ足してから書く
"""),
      ("""    g.bufs, g.ch.age, g.t = d["bufs"], d["age"], d["t"]
""", """    g.bufs, g.ch.age, g.t = d["bufs"], d["age"], d["t"]
    g._prior_l, g._amp_buf, g._amp_prev = None, None, None   # 版54：読み込んだ tb・辺で作り直す
""")]

if __name__ == "__main__":
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版54 を当てた（torch_group2.py・babble.py）")
