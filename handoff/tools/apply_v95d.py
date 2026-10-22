"""版95d：速さの3つ（撃ち方・学び方の式は変えない）【み】10-22 22:12「その3つまではいいけど」
（~/anaplex で。版95c の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v95d.py
【測】arkhe measure_speed95（版95c、グラフの形のまま）：1歩 21.1ms ／ 下からの表 5.9ms（線条体への 885万本）／ NucNet 約3.5ms ／ 網の外 3.47ms
直すもの
  1 下からの表：大きい組（100万本以上）は、撃った送り手の線だけを計算する（放出のくじ・短期の増減・足し込み）
     撃ったかの印は全部の線で出す（学びが使う）。撃った線を数えるので CPU が GPU を1回待つ
     ⚠️ くじの分布は同じ。乱数の並びは変わる（同じ種でも同じ列にはならない。版52 から走らせるたびに種を変えている）
  2 NucNet：線ごとに受け手1体の本数が決まっている（64・100）⇒ 線を（符号・遅れ・本数）の組に並べ、受け手1体ぶんずつ並べて合計
     自分の核の中の線（外節 → 外節・棘細胞どうし）の「自分から自分」は、線を抜く代わりに重さ 0（足す値は同じ）
     float32 で足す（版95b の float64 の累積和をやめる。値は丸めの桁で変わる）
  3 核 → 領野の抑制の4本：線ごとに撃った線を数えていた（CPU が GPU を4回待つ）⇒ 下からの表と同じ形（数えない）で1回に
"""
import ast, pathlib, sys

TB = [
    # 1 下からの表：大きい組は撃った送り手の線だけ
    ('        x = torch.cat(vs)\n'
     '        te = self.t[self.pid[a:b]]\n',
     '        x = torch.cat(vs)\n'
     '        if b - a >= BU_COMPACT_MIN:                         # 版95d：大きい組は撃った送り手の線だけ\n'
     '            return self._drive_compact(gi, x, a, b, ta, tb, i0, i1)\n'
     '        te = self.t[self.pid[a:b]]\n'),
    ('    def _seg_ends(self, gi, n_out):\n',
     '    @torch.no_grad()\n'
     '    def _drive_compact(self, gi, x, a, b, ta, tb, i0, i1):\n'
     '        """版95d：drive と同じ式を、撃った送り手の線だけで（_bu_pre・_bu_post_v と同じ式。くじは同じ分布）"""\n'
     '        if gi not in getattr(self, "_tgt_rel", {}):\n'
     '            self._tgt_rel = getattr(self, "_tgt_rel", {})\n'
     '            self._tgt_rel[gi] = self.tgt[a:b] - ta\n'
     '        fired = x[self._gsrc[gi]] > 0\n'
     '        self.pre_now[a:b] = fired                            # 学びは全部の線の印を使う（今まで通り）\n'
     '        e = torch.nonzero(fired).squeeze(1)                  # 撃った線（ここで1回だけ CPU が GPU を待つ）\n'
     '        ea = e + a\n'
     '        te = self.t[self.pid[ea]]\n'
     '        r = self.row[ea].long()\n'
     '        dep = self.deplete[ea]\n'
     '        n = (te - self.last[ea]).float()\n'
     '        st = self.state[ea] * torch.where(dep, torch.ones_like(n), self.dec_d[r] ** n)\n'
     '        fc = self.fac[ea] * self.dec_f[r] ** n\n'
     '        Pr0 = self.tbl[r, 0]\n'
     '        P = (Pr0 + (1 - Pr0) * fc).clamp(0, 1)\n'
     '        N = self.N[r]\n'
     '        cn = (N * torch.where(dep, st, 1 - st)).round().clamp(min=0)\n'
     '        cnt = torch.binomial(cn, P, generator=self.gen)\n'
     '        rel = cnt / (N * Pr0).clamp(min=1e-6)\n'
     '        k = torch.where(dep, P, self.tbl[r, 1])\n'
     '        self.state[ea] = torch.where(dep, st * (1 - k), st + (1 - st) * k)\n'
     '        self.fac[ea] = fc + (1 - fc) * self.tbl[r, 3]\n'
     '        self.last[ea] = te.to(self.last.dtype)\n'
     '        tg = self._tgt_rel[gi][e]                            # 線は受け手の順 ⇒ 撃った線も受け手の順\n'
     '        out = _segsum(self.w[ea] * rel, torch.bincount(tg, minlength=tb - ta).cumsum(0))\n'
     '        self.t[i0:i1] += 1\n'
     '        return list(torch.split(out, self.nt[i0:i1]))\n'
     '\n'
     '    def _seg_ends(self, gi, n_out):\n'),
    ('class _BUTable:\n',
     'BU_COMPACT_MIN = 1_000_000   # 版95d：この本数より大きい組は、撃った送り手の線だけを計算する\n'
     '\n'
     '\n'
     'class _BUTable:\n'),
    # 3 核 → 領野の抑制：表にまとめて1回
    ('            self.g.ext_ihn = torch.zeros(self.g.n_recv_all, device=dev)\n'
     '            self._nucnet_build(g)                                    # 版95：核どうしの線（周の遅れ）\n',
     '            self.g.ext_ihn = torch.zeros(self.g.n_recv_all, device=dev)\n'
     '            self.but_inh = _BUTable({k_: s_ for k_, (s_, _r) in self.bu_inh.items()}, (list(self.bu_inh),))   # 版95d\n'
     '            self._nucnet_build(g)                                    # 版95：核どうしの線（周の遅れ）\n'),
    ('            for key, (sn_, _t) in INH_TO_AREA.items():          # 版95：核 → 領野（1歩前の表現の発火）\n'
     '                v = self.nuc[sn_]\n'
     '                s, rows = self.bu_inh[key]\n'
     '                d = s.drive(v)\n'
     '                self.g.ext_ihn.index_add_(0, rows, torch.cat([d, d]))\n',
     '            ks_ = self.but_inh.groups[0]                         # 版95d：表で1回（撃った線を数えない。式は _BUSyn.drive と同じ）\n'
     '            for key, d in zip(ks_, self.but_inh.drive(0, [self.nuc[INH_TO_AREA[k_][0]] for k_ in ks_])):\n'
     '                self.g.ext_ihn.index_add_(0, self.bu_inh[key][1], torch.cat([d, d]))\n'),
]

NUC_BUILD_NEW = '''    def _nucnet_build(self, g):
        """版95：核どうしの線を網の NucNet に（周の遅れ）。値は NUC_LINES・INH_SHARE（設計案_版95の組み表）
        版95d：線ごとに渡す（受け手1体ぶんずつ並べる）。自分の核の中の「自分から自分」は重さ 0（⟲ 版95 は線を抜いた）"""
        import math as _m
        from torch_group2 import NucNet
        dev, nn, gg = self.dev, self.n, self.g
        n_ch = gg.n_ch_all
        ch0 = lambda nm: int(self.off_ch[NAMES.index(nm)])
        sig = _m.sqrt(2.0 * _m.log(CL_W_MEAN / CL_W_MED))
        lines = []
        for srcs, tgt, sg, k, frac, dl, kind in NUC_LINES:
            pool = []
            for s in srcs:                                     # 線を出す送り手（体ごと）・送る体
                cells = torch.randperm(nn[s], device=dev, generator=g)[:max(1, int(round(nn[s] * frac)))] + ch0(s)
                pool.append(cells)
                if kind == "表誤":
                    pool.append(cells + n_ch)                  # 誤差の体（送り手の並びは [表現 | 誤差]）
            pool = torch.cat(pool)
            nt = nn[tgt]
            kk = min(k, len(pool))
            src = torch.stack([pool[torch.randperm(len(pool), device=dev, generator=g)[:kk]] for _ in range(nt)]).reshape(-1)
            tgt_ch = torch.arange(nt, device=dev) + ch0(tgt)
            if sg > 0:
                w = torch.exp(_m.log(CL_W_MED) + sig * torch.randn(len(src), device=dev, generator=g))
            else:
                sh = INH_SHARE.get(tgt, {}).get(srcs, None)
                tot = sum(INH_SHARE[tgt].values()) if sh is not None else None
                w = torch.full((len(src),), W_INH_BU * K_NUC * ((sh / tot) if sh is not None else 1.0) / kk, device=dev)
            if srcs == (tgt,):                                 # 自分の核の中：自分から自分は重さ 0
                w = torch.where((src % n_ch) == tgt_ch.repeat_interleave(kk), torch.zeros_like(w), w)
            lines.append(dict(src=src, w=w, sign=sg, delay=dl, k=kk, tgt_ch=tgt_ch))
        # 興奮の NMDA の尾（周ごと。版93 の式）：受け手の核の体ごと
        tl = {k_: torch.zeros(n_ch, device=dev) for k_ in ("a1", "m1", "a2", "m2", "f")}
        for k_ in NUC:
            (t1, w1), (t2, w2), f_ = NUC_TAIL[k_]
            o, n_ = ch0(k_), nn[k_]
            for kk_, v in (("a1", _m.exp(-1.067 / t1)), ("m1", w1), ("a2", _m.exp(-1.067 / t2)), ("m2", w2), ("f", f_)):
                tl[kk_][o:o + n_] = float(v)
        gg.nucnet = NucNet(lines, n_ch, gg.n_recv_all, gg.rep_i, gg.err_i, tl, dev)
'''

NUC_CLASS_NEW = '''

class NucNet:
    """版95：核どうしの線（周 1.1ms ごとに届く。核どうし 1周・線条体から 2周）【み】10-16「核は周回ごと。核どうしの遅れは周の数」
    送り手 ＝ 網の表現・誤差の体（[表現 n_ch | 誤差 n_ch] の並び）。興奮は受け手の表現の駆動へ（x と同じ入り方・誤差の的には入れない）、
    抑制は受け手の表現・誤差の行の抑制へ（÷(1＋抑制)）。興奮には周ごとの NMDA の尾（版93 の式）
    学び（決め A【み】10-15 23:51）：送り手の強い入力（直近 NUC_A_WIN 歩のうち NUC_A_N 歩以上撃った）のあとに強まる × 門。
      受け手の発火は条件にしない。⚠️ 量 NUC_A_ETA・強い入力の数・上限 NUC_A_WMAX はノブ【C】（弱まる側は人の値なし）
    版95d：線を（符号・遅れ・受け手1体の本数）の組に並べ、受け手1体ぶんずつ並べて合計（index_add の渋滞も累積和もなし）"""
    NUC_A_WIN, NUC_A_N, NUC_A_ETA, NUC_A_WMAX = 10, 8, 1e-3, 4.0

    def __init__(self, lines, n_ch, n_recv, rep_i, err_i, tail, dev):
        self.n_ch, self.n_recv, self.dev, self.tail = n_ch, n_recv, dev, tail
        grp = {}
        for L in lines:
            grp.setdefault((L["sign"], L["delay"], L["k"]), []).append(L)
        S_, W_, self.groups, self._tch = [], [], [], []
        o = 0
        for (sg, dl, k), Ls in sorted(grp.items()):
            src = torch.cat([L["src"] for L in Ls]).long()
            w = torch.cat([L["w"] for L in Ls]).float()
            tch = torch.cat([L["tgt_ch"] for L in Ls]).long()        # 受け手1体ぶん（k 本）ごとに1つ
            if sg > 0:
                tgt = tch
            else:                                                    # 抑制は表現の行と誤差の行の両方へ
                tgt = torch.cat([rep_i[tch], err_i[tch]]).long()
            self.groups.append((sg, dl, k, o, o + len(src), tgt))
            self._tch.append(tch)
            S_.append(src); W_.append(w); o += len(src)
        self.src = torch.cat(S_)
        self.w = torch.cat(W_)
        self.w0 = self.w.clone()
        self.hist = torch.zeros(self.NUC_A_WIN, n_ch * 2, dtype=torch.uint8, device=dev)
        self.cnt = torch.zeros(n_ch * 2, dtype=torch.int16, device=dev)
        self.hi = 0

    def flat(self):
        """測る道具用：線ごとの (送り手, 受け手の体（n_ch の並び）, 符号, 遅れ)"""
        D, Sg, Dl = [], [], []
        for (sg, dl, k, a, b, tgt), tch in zip(self.groups, self._tch):
            D.append(tch.repeat_interleave(k))
            Sg.append(torch.full((b - a,), sg, dtype=torch.long, device=self.dev))
            Dl.append(torch.full((b - a,), dl, dtype=torch.long, device=self.dev))
        return self.src, torch.cat(D), torch.cat(Sg), torch.cat(Dl)

    def state(self, n_ch, n_recv):
        z = lambda n: torch.zeros(n, device=self.dev)
        return {"nuc_x": z(n_ch), "nuc_x2": z(n_ch), "nuc_ihn": z(n_recv), "nuc_ihn2": z(n_recv),
                "nuc_s1": z(n_ch), "nuc_s2": z(n_ch)}

    def after(self, S, rep, err):
        """この周の発火 → 次の周（遅れ1）・その次の周（遅れ2）に届く分。置き場所は S（グラフの中で書き換える）"""
        v = torch.cat([rep, err])
        xs = {1: torch.zeros(self.n_ch, device=self.dev), 2: torch.zeros(self.n_ch, device=self.dev)}
        hs = {1: torch.zeros(self.n_recv, device=self.dev), 2: torch.zeros(self.n_recv, device=self.dev)}
        for sg, dl, k, a, b, tgt in self.groups:
            s = (self.w[a:b] * v[self.src[a:b]]).view(-1, k).sum(1)    # 受け手1体ぶんの合計
            if sg > 0:
                xs[dl].index_add_(0, tgt, s)
            else:
                hs[dl].index_add_(0, tgt, torch.cat([s, s]))
        raw = S["nuc_x2"] + xs[1]                                # 次の周に届く興奮
        T = self.tail
        s1 = S["nuc_s1"] * T["a1"] + raw * (1 - T["a1"])
        s2 = S["nuc_s2"] * T["a2"] + raw * (1 - T["a2"])
        S["nuc_s1"].copy_(s1); S["nuc_s2"].copy_(s2)
        S["nuc_x"].copy_((1 - T["f"]) * raw + T["f"] * (T["m1"] * s1 + T["m2"] * s2))
        S["nuc_x2"].copy_(xs[2])
        S["nuc_ihn"].copy_(S["nuc_ihn2"] + hs[1])
        S["nuc_ihn2"].copy_(hs[2])

    @torch.no_grad()
    def learn(self, any_src, d):
        """1歩に1回。any_src：送り手の体がこの歩のどこかで撃ったか（[表現 | 誤差]）／ d：学びの門（0〜1）"""
        now = (any_src > 0).to(torch.uint8)
        self.cnt += now.to(torch.int16) - self.hist[self.hi].to(torch.int16)
        self.hist[self.hi] = now
        self.hi = (self.hi + 1) % self.NUC_A_WIN
        g_on = (2.0 * torch.as_tensor(d, dtype=self.w.dtype, device=self.dev) - 1.0).clamp(0.0, 1.0)
        strong = (self.cnt[self.src] >= self.NUC_A_N).float()
        self.w.mul_(torch.exp(self.NUC_A_ETA * g_on * strong))
        torch.minimum(self.w, self.w0 * self.NUC_A_WMAX, out=self.w)

    def state_dict(self):
        return {"w": self.w, "cnt": self.cnt, "hist": self.hist, "hi": self.hi}

    def load_state_dict(self, d):
        if d.get("w") is None or d["w"].shape != self.w.shape:
            print("⚠️ 核どうしの線の重さは形が違うので読まない（初めの重さから）", flush=True)
            return
        self.w.copy_(d["w"]); self.cnt.copy_(d["cnt"]); self.hist.copy_(d["hist"]); self.hi = int(d["hi"])
'''
TG_PINS = [('        return () if nn_ is None else (nn_.w, nn_.src, nn_.dst)\n',
            '        return () if nn_ is None else (nn_.w, nn_.src) + tuple(g_[5] for g_ in nn_.groups)   # 版95d\n')]
BB = [("#  95c CL 4,096 → 64",
       "#  95d 速さ：下からの表は撃った送り手の線だけ・NucNet を並べた形で・核 → 領野の抑制を表で1回【み】10-22 22:12\n#  95c CL 4,096 → 64")]


def edit(s, pairs, path):
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:70]!r}。何も書いていない")
        s = s.replace(a, b)
    return s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    bt = pathlib.Path("babble.py").read_text()
    if babble.VERSION != 95 or "95c" not in bt or "95d" in bt:
        sys.exit("⚠️ 版95c ではない（apply_v95c.py のあとに1回だけ当てる）")
    s_tb = edit(pathlib.Path("torch_brain.py").read_text(), TB, "torch_brain.py")
    i = s_tb.find("    def _nucnet_build(self, g):"); j = s_tb.find("\n    def _send_rows(self, area, first, n):")
    if i < 0 or j < 0 or j < i:
        sys.exit("⚠️ torch_brain.py の _nucnet_build が見つからない。何も書いていない")
    s_tb = s_tb[:i] + NUC_BUILD_NEW + s_tb[j:]
    s_tg = edit(pathlib.Path("torch_group2.py").read_text(), TG_PINS, "torch_group2.py")
    i = s_tg.find("\n\nclass NucNet:"); j = s_tg.find("\n\ndef _segsum_nuc(v, ends):")
    if i < 0 or j < 0 or j < i:
        sys.exit("⚠️ torch_group2.py の NucNet が見つからない。何も書いていない")
    s_tg = s_tg[:i] + NUC_CLASS_NEW + s_tg[j:]
    s_bb = edit(bt, BB, "babble.py")
    for s in (s_tb, s_tg, s_bb):
        ast.parse(s)
    pathlib.Path("torch_brain.py").write_text(s_tb)
    pathlib.Path("torch_group2.py").write_text(s_tg)
    pathlib.Path("babble.py").write_text(s_bb)
    print("✅ 版95d を当てた（下からの表は撃った送り手の線だけ・NucNet を並べた形で・核 → 領野の抑制を表で1回）", flush=True)
