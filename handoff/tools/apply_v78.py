"""版78：周回の中の体ごとの計算を torch.compile でまとめる（速さだけ。式・くじの順は同じ）
（~/anaplex で。版77 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v78.py
【み】10-14「常時回ってるからこそ遅延が致命的なわけか」「それは実時間を下回る必要があるね」／ 設計案_速さ_版78.md の順（版78 compile → 版79 HIP グラフ）
【測】版77 1歩 27.1ms ／ GPU 約20.8ms ／ 呼び出し 約2,000回/歩
【測】コンテナ（CPU、1歩の命令を場所ごとに数えた。相手200）1歩 2,321命令のうち
  周回（_pass）の残り 924（1周 92）／ 撃った辺を集める 493 ／ _sum 299 ／ _flush 121 ／ 下からの表 142 ／ 放出 108 ／ 中脳 96 ／ 時間細胞 約300
  多い行：不応期の2行 210 ／ 補完（fill）80 ／ 駆動の式 70 ／ 抑制の体の駆動 50 ／ p 40 ／ CL の駆動 40 ／ 興奮の倍率 30 …
直すもの
  _drive_core：届いた入力 → 倍率 → 見込み p → 表現の駆動 → 撃つか・補完・自発（くじは外で、今と同じ順に引いて渡す）
  _out_core：誤差の出力 → 抑制の体の駆動 → 撃つか → 不応期 → この歩のどこかで撃ったか
    不応期の周の番号（_pc）は GPU の箱にも持つ（Python の数を渡すと、周ごとに組み直しになるため）。_pc も今まで通り進める
  _rel_pre・_rel_post：短期の増減の放出の前半・後半（くじ binomial は外で）。網・CL・下からの道の放出が全部これを通る
  _tc_core：時間細胞の撃つか（領野ごとのくじは外で、今と同じ順に引く）
【測】コンテナ（CPU・相手200）1歩の命令 4,164 → 1,886（まとめた核の呼び出し 96）／ 組み直し 0
⚠️ GPU では足す順・丸めが少し変わりうる（版77 と同じ「丸めの範囲」）
"""
import ast, pathlib, re, sys

G2 = [
    ("    cnt.scatter_add_(0, r, torch.ones(e.shape[0], device=w.device, dtype=w.dtype))\n"
     "    return cnt\n",
     "    cnt.scatter_add_(0, r, torch.ones(e.shape[0], device=w.device, dtype=w.dtype))\n"
     "    return cnt\n\n\n"
     "def _drive_core(both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry,\n"
     "                fire_th, th_gain, complete, pred_boost, spont_u, spont_p_pass, n_recv: int):\n"
     "    \"\"\"版78：周回の中の『届いた入力 → 見込み → 表現が撃つか』（式は版77 の _pass と同じ。くじ spont_u は外で引く）\"\"\"\n"
     "    exc, ihn = both[:n_recv], both[n_recv:]\n"
     "    exc = exc * (scale * scale_up); ihn = ihn * scale\n"
     "    exc_mod = exc_mod * (scale * scale_up)\n"
     "    if exc_gain is not None:\n"
     "        exc = exc * exc_gain\n"
     "        exc_mod = exc_mod * exc_gain\n"
     "    p = torch.sigmoid(prior_l + exc / (1.0 + ihn))\n"
     "    pr = p[rep_i]\n"
     "    drive = ((exc - exc_mod)[rep_i] + x_all * x_gain) / (1.0 + ihn[rep_i])\n"
     "    if cl_drive is not None:\n"
     "        drive = drive + cl_drive / (1.0 + ihn[rep_i])\n"
     "    if v_carry is not None:\n"
     "        drive = drive + v_carry\n"
     "    rep = (drive > fire_th * th_gain).float()\n"
     "    if complete is not None:\n"
     "        fill = (complete > 0) & ((drive + pred_boost * pr) > fire_th * th_gain) & (x_all > 0)\n"
     "        rep = torch.where(fill, torch.ones_like(rep), rep)\n"
     "    rep = torch.maximum(rep, (spont_u < spont_p_pass).float())\n"
     "    return exc, ihn, exc_mod, p, pr, drive, rep\n\n\n"
     "def _out_core(err_sig, exc, exc_mod, ihn, err_i, inh_i, v_carry_inh, rep, lf_r, lf_e, lf_i, pc, any_r, any_e, any_i):\n"
     "    \"\"\"版78：周回の中の『誤差 → 抑制の体 → 不応期 → この歩のどこかで撃ったか』（式は版77 と同じ。pc は周の番号の箱）\"\"\"\n"
     "    err = (err_sig[err_i] > THETA).float()\n"
     "    d_inh = (exc - exc_mod)[inh_i] / (1.0 + ihn[inh_i])\n"
     "    if v_carry_inh is not None:\n"
     "        d_inh = d_inh + v_carry_inh\n"
     "    inh_spk = (d_inh > INH_TH).float()\n"
     "    outs, lfs = [], []\n"
     "    for v, lf in ((rep, lf_r), (err, lf_e), (inh_spk, lf_i)):\n"
     "        v = v * ((pc - lf) >= REFRACT_PASSES).float()\n"
     "        outs.append(v)\n"
     "        lfs.append(torch.where(v > 0, pc, lf))\n"
     "    if any_r is None:\n"
     "        anys = list(outs)\n"
     "    else:\n"
     "        anys = [torch.maximum(a, v) for a, v in zip((any_r, any_e, any_i), outs)]\n"
     "    return d_inh, outs[0], outs[1], outs[2], lfs[0], lfs[1], lfs[2], anys[0], anys[1], anys[2]\n"),
    # 駆動の所
    ("        exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n"
     "        exc = exc * (self.scale * self.scale_up); ihn = ihn * self.scale   # 版53：起きて上げた分は興奮だけに\n"
     "        exc_mod = exc_mod * (self.scale * self.scale_up)\n",
     "        # 版53：起きて上げた分は興奮だけに（版78：_drive_core の中）\n"),
    ("        if getattr(self, \"exc_gain\", None) is not None:\n"
     "            exc = exc * self.exc_gain\n"
     "            exc_mod = exc_mod * self.exc_gain\n"
     "        if self._prior_l is None:",
     "        #   （版78：_drive_core の中）\n"
     "        if self._prior_l is None:"),
    ("        p = torch.sigmoid(self._prior_l + exc / (1.0 + ihn))\n",
     "        # 版78：自発のくじは今と同じ順（見込みのあと・学びの前）に引いて、_drive_core に渡す\n"
     "        spont_u = torch.rand(self.n_ch_all, device=self.dev, generator=self.gen)\n"
     "        exc, ihn, exc_mod, p, pr, drive, rep = _run(\n"
     "            _drive_core, both, exc_mod, self.scale, self.scale_up, getattr(self, \"exc_gain\", None), self._prior_l,\n"
     "            self.rep_i, x_all, self.x_gain, getattr(self, \"cl_drive\", None), getattr(self, \"v_carry\", None),\n"
     "            self.fire_th, self.th_gain, self.complete if self._complete_any else None, self.pred_boost,\n"
     "            spont_u, self.spont_p_pass, self.n_recv_all)\n"),
    ("        pr = p[self.rep_i]\n"
     "        # 表現のニューロンも、シナプスの合計が閾値を越えたら撃つ\n",
     "        # 表現のニューロンも、シナプスの合計が閾値を越えたら撃つ\n"),
    ("        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])\n"
     "        if getattr(self, \"cl_drive\", None) is not None:      # 版67：CL からの駆動（シナプス。誤差の的には入れない）\n"
     "            drive = drive + self.cl_drive / (1.0 + ihn[self.rep_i])\n"
     "        if getattr(self, \"v_carry\", None) is not None:       # 版72：前の歩から持ち越した膜の残り（時間の足し合わせ）\n"
     "            drive = drive + self.v_carry\n"
     "        self._drive_last = drive\n"
     "        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n"
     "        if self._complete_any:           # 入力が無いところは予測が足しになる（撃たせはしない）。版52 B\n"
     "            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th * self.th_gain) & (x_all > 0)\n"
     "            rep = torch.where(fill, torch.ones_like(rep), rep)\n"
     "        rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,\n"
     "                                             generator=self.gen) < self.spont_p_pass).float())   # 版58\n",
     "        # 版67：CL からの駆動（誤差の的には入れない）／ 版72：持ち越した膜の残り ／ 版41：覚醒で閾値が下がる\n"
     "        # 版52 B：入力が無いところは予測が足しになる ／ 版58：自発（以上 版78 で _drive_core の中）\n"
     "        self._drive_last = drive\n"),
    # 出力の所
    ("        err = (err_sig[self.err_i] > THETA).float()\n", ""),
    ("        d_inh = (exc - exc_mod)[self.inh_i] / (1.0 + ihn[self.inh_i])\n"
     "        if getattr(self, \"v_carry\", None) is not None:       # 版72：抑制の体の足し合わせ\n"
     "            if self.v_carry_inh is None:\n"
     "                self.v_carry_inh = torch.zeros_like(d_inh)\n"
     "            d_inh = d_inh + self.v_carry_inh\n"
     "        self._dinh_last = d_inh\n"
     "        inh_spk = (d_inh > INH_TH).float()\n",
     "        if getattr(self, \"v_carry\", None) is not None and self.v_carry_inh is None:   # 版72：抑制の体の足し合わせ\n"
     "            self.v_carry_inh = torch.zeros(len(self.inh_i), device=self.dev)\n"),
    ("        if first:\n"
     "            self._learn_step = learn\n"
     "            if getattr(self, \"_last_fire\", None) is None:\n"
     "                self._pc = 0\n"
     "                self._last_fire = [torch.full(v.shape, -10 ** 6, dtype=torch.long, device=self.dev)\n"
     "                                   for v in (rep, err, inh_spk)]\n"
     "        self._pc += 1\n"
     "        rep, err, inh_spk = [v * ((self._pc - lf) >= REFRACT_PASSES).float()\n"
     "                             for v, lf in zip((rep, err, inh_spk), self._last_fire)]\n"
     "        self._last_fire = [torch.where(v > 0, torch.full_like(lf, self._pc), lf)\n"
     "                           for v, lf in zip((rep, err, inh_spk), self._last_fire)]\n"
     "        # 版59：この歩のどこかで撃ったか（周の発火の重ね）\n"
     "        self._any = [rep, err, inh_spk] if first else [torch.maximum(a, v) for a, v in zip(self._any, (rep, err, inh_spk))]\n",
     "        if first:\n"
     "            self._learn_step = learn\n"
     "            if getattr(self, \"_last_fire\", None) is None:\n"
     "                self._pc = 0\n"
     "                self._pc_t = torch.zeros((), dtype=torch.long, device=self.dev)   # 版78：周の番号の箱（GPU）\n"
     "                self._last_fire = [torch.full((n,), -10 ** 6, dtype=torch.long, device=self.dev)\n"
     "                                   for n in (len(rep), len(self.err_i), len(self.inh_i))]\n"
     "        self._pc += 1\n"
     "        if getattr(self, \"_pc_t\", None) is None:          # 版77 の state から続けるとき\n"
     "            self._pc_t = torch.full((), self._pc, dtype=torch.long, device=self.dev)\n"
     "        else:\n"
     "            self._pc_t.add_(1)\n"
     "        # 版48：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ ／ 版59：不応期・この歩のどこかで撃ったか\n"
     "        # 版78：以上を _out_core（torch.compile）で1つに\n"
     "        an = (None, None, None) if first else tuple(self._any)\n"
     "        (d_inh, rep, err, inh_spk, lf0, lf1, lf2, a0, a1, a2) = _run(\n"
     "            _out_core, err_sig, exc, exc_mod, ihn, self.err_i, self.inh_i,\n"
     "            self.v_carry_inh if getattr(self, \"v_carry\", None) is not None else None,\n"
     "            rep, *self._last_fire, self._pc_t, *an)\n"
     "        self._dinh_last = d_inh\n"
     "        self._last_fire = [lf0, lf1, lf2]\n"
     "        self._any = [a0, a1, a2]\n"),
]
G2 = G2 + [('class TimeCellsGroup:\n',
  'def _tc_core(age, prefs, halfs, pmax, ups, dns, offs, n_cell, n_ramp, ramp_len):\n'
  '    """版78：時間細胞の撃つか（領野ごとの塊を順に）。くじ ups・dns は外で今の順に引く。式は版77 と同じ"""\n'
  '    out = []\n'
  '    for i in range(len(n_ramp)):\n'
  '        a = age[offs[i]:offs[i + 1]]\n'
  '        parts = []\n'
  '        if n_cell[i]:\n'
  '            f = a.unsqueeze(-1).float()\n'
  '            hit = ((f - prefs[i]).abs() <= halfs[i]) & (f <= pmax[i])\n'
  '            parts.append(hit.any(1).float())\n'
  '        y = (a.min(1).values.float() / ramp_len[i]).clamp(0, 1).unsqueeze(-1)\n'
  '        parts.append(torch.cat([ups[i] < y * RAMP_P, dns[i] < (1 - y) * RAMP_P], 1).float())\n'
  '        out.append(torch.cat(parts, 1).reshape(-1))\n'
  '    return torch.cat(out)\n'
  '\n'
  '\n'
  'class TimeCellsGroup:\n'),
 ('        out = []\n'
  '        for i, a in enumerate(self.areas):\n'
  '            lo, hi = self.off_cpu[i], self.off_cpu[i + 1]\n'
  '            age = self.age[lo:hi]\n'
  '            parts = []\n'
  '            if self.n_cell[i]:\n'
  '                f = age.unsqueeze(-1).float()\n'
  '                hit = ((f - self.pref[i]).abs() <= self.half[i]) & (f <= self.pref[i].max())\n'
  '                parts.append(hit.any(1).float())\n'
  '            y = (age.min(1).values.float() / self.ramp_len[i]).clamp(0, 1).unsqueeze(-1)\n'
  '            h = self.n_ramp[i] // 2\n'
  '            up = torch.rand(hi - lo, h, device=self.dev, generator=self.gen) < y * RAMP_P\n'
  '            dn = torch.rand(hi - lo, self.n_ramp[i] - h, device=self.dev,\n'
  '                            generator=self.gen) < (1 - y) * RAMP_P\n'
  '            parts.append(torch.cat([up, dn], 1).float())\n'
  '            out.append(torch.cat(parts, 1).reshape(-1))\n'
  '        self._cache = torch.cat(out)\n'
  '        return self._cache\n',
  '        # 版78：くじは領野の順・上り下りの順に外で引き、撃つかは _tc_core（torch.compile）で1回に\n'
  '        ups, dns = [], []\n'
  '        for i in range(len(self.areas)):\n'
  '            lo, hi = self.off_cpu[i], self.off_cpu[i + 1]\n'
  '            h = self.n_ramp[i] // 2\n'
  '            ups.append(torch.rand(hi - lo, h, device=self.dev, generator=self.gen))\n'
  '            dns.append(torch.rand(hi - lo, self.n_ramp[i] - h, device=self.dev, generator=self.gen))\n'
  '        if getattr(self, "_pmax", None) is None:\n'
  '            self._pmax = [(p.max() if p is not None else None) for p in self.pref]\n'
  '        self._cache = _run(_tc_core, self.age, self.pref, self.half, self._pmax, ups, dns, tuple(self.off_cpu),\n'
  '                           tuple(self.n_cell), tuple(self.n_ramp), tuple(float(x) for x in self.ramp_len))\n'
  '        return self._cache\n')]
AREA = [('class STP:\n',
  'def _rel_pre(e, row, last, deplete, state, fac, dec_d, dec_f, tbl, Nt, t):\n'
  '    """版78：放出の前半（触った辺だけ、最後に触った歩からまとめて減らす）。式は版77 の release_at と同じ"""\n'
  '    r = row[e].long()\n'
  '    n = (t - last[e]).float()\n'
  '    dep = deplete[e]\n'
  '    st = state[e] * torch.where(dep, torch.ones_like(n), dec_d[r] ** n)\n'
  '    fc = fac[e] * dec_f[r] ** n\n'
  '    Pr0 = tbl[r, 0]\n'
  '    P = (Pr0 + (1 - Pr0) * fc).clamp(0, 1)\n'
  '    N = Nt[r]\n'
  '    avail = torch.where(dep, st, 1 - st)\n'
  '    return r, dep, st, fc, Pr0, P, N, (N * avail).round().clamp(min=0)\n'
  '\n'
  '\n'
  'def _rel_post(cnt, r, dep, st, fc, Pr0, P, N, tbl):\n'
  '    """版78：放出の後半（出た量・状態の更新）"""\n'
  '    out = cnt / (N * Pr0).clamp(min=1e-6)\n'
  '    k = torch.where(dep, P, tbl[r, 1])\n'
  '    st2 = torch.where(dep, st * (1 - k), st + (1 - st) * k)\n'
  '    fc2 = fc + (1 - fc) * tbl[r, 3]\n'
  '    return out, st2, fc2\n'
  '\n'
  '\n'
  'class STP:\n'),
 ('        r = self.row[e].long()\n'
  '        n = (t - self.last[e]).float()\n'
  '        dep = self.deplete[e]\n'
  '        self.state[e] = self.state[e] * torch.where(dep, torch.ones_like(n), self.dec_d[r] ** n)\n'
  '        self.fac[e] = self.fac[e] * self.dec_f[r] ** n\n'
  '        self.last.index_fill_(0, e, t)     # 版54b\n'
  '        Pr0 = self.tbl[r, 0]\n'
  '        P = (Pr0 + (1 - Pr0) * self.fac[e]).clamp(0, 1)\n'
  '        N = self.N[r]\n'
  '        avail = torch.where(dep, self.state[e], 1 - self.state[e])\n'
  '        cnt = torch.binomial((N * avail).round().clamp(min=0), P, generator=self.gen)\n'
  '        out = cnt / (N * Pr0).clamp(min=1e-6)\n'
  '        k = torch.where(dep, P, self.tbl[r, 1])\n'
  '        self.state[e] = torch.where(dep, self.state[e] * (1 - k),\n'
  '                                    self.state[e] + (1 - self.state[e]) * k)\n'
  '        self.fac[e] = self.fac[e] + (1 - self.fac[e]) * self.tbl[r, 3]\n'
  '        return out\n',
  '        # 版78：前半・後半を torch.compile で（くじ binomial は外で。式・くじの順は版77 と同じ）\n'
  '        from torch_group2 import _run\n'
  '        r, dep, st, fc, Pr0, P, N, cn = _run(_rel_pre, e, self.row, self.last, self.deplete, self.state, self.fac,\n'
  '                                             self.dec_d, self.dec_f, self.tbl, self.N, t)\n'
  '        self.last.index_fill_(0, e, t)     # 版54b\n'
  '        cnt = torch.binomial(cn, P, generator=self.gen)\n'
  '        out, st2, fc2 = _run(_rel_post, cnt, r, dep, st, fc, Pr0, P, N, self.tbl)\n'
  '        self.state[e] = st2\n'
  '        self.fac[e] = fc2\n'
  '        return out\n')]
BB = [("VERSION = 77          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76)",
       "VERSION = 78          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77)"),
      ("#  77  torch.compile を戻した：核の中の番号の assert を外す【測】probe_compile3 で9つとも動く ／ 学びの値は fill_ で（CPU が待たない）\n",
       "#  77  torch.compile を戻した：核の中の番号の assert を外す【測】probe_compile3 で9つとも動く ／ 学びの値は fill_ で（CPU が待たない）\n"
       "#  78  周回の中の体ごとの計算（駆動・見込み・撃つか／誤差・抑制の体・不応期）・放出・時間細胞を torch.compile で（式・くじの順は同じ）\n"
       "#      【測】版77 1歩 27.1ms ／ コンテナで数えた命令 1歩 2,321（周回の残り 924）【み】10-14「実時間を下回る必要があるね」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py", "measure_speed75.py")


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 77:
        sys.exit(f"⚠️ babble.py が版77 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("torch_area.py", AREA), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版78 を当てた（torch_group2・torch_area・babble）。周回の中の体ごとの計算・放出・時間細胞を torch.compile で")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "78" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['78'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版78 でも回るようにした")
