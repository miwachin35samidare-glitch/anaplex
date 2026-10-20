"""版94c：前のスレッドで決めた核の値を、素子の形へ戻す（版94b の上に当てる。印が合わなければ何も書かずに止まる）
  python3 apply_v94c.py
【み】10-20 18:17「値は前のスレッドで散々合わせたはずだけど何で無いの？」⇒ 18:21「ア」（素子に「自分で撃つ分」と「長い尾」を足す）
戻すもの（出どころは設計案_基底核と小脳 §4〜・引き継ぎ_版93 §4）
  1 線の本数：受け手1体は送り手の別々の 64体から【み】10-17 ／ 前頭前野 → 黒質 8本 ／ 視床下核 → 境 は線を出す送り手 69.3%【マカク】Sato 2000b
     側坐核 → 境 は表現＋誤差を1つの送り手として【み】10-18 13:36「a」（⟲ 版94 の「表現だけ」は Claude の置き方だった）
  2 膜の時定数 τm 40ms（核の表現の体の時間の足し合わせ。版93 で黒質のふだんの率・CV・戻りに合わせた値を5核に）
  3 自分で撃つ分 i0（根拠のある核だけ【み】10-16 23:51）：視床下核【マカク】Polyakova 2020 ／ 境 ／ 外側手綱核【マウス】Fedorov 2026【み】10-18 0:15 ／
     吻内側被蓋核【み】10-18 11:20 ／ 黒質（撃ったら 0）
     置き方：版93 の撃ち方 B での「入力なしで行き着く先 ÷ 閾値」（i0 ÷ (1 − α)）を、素子の閾値と 1歩の減りに写す
       ⟲ 版94 で「決まったこと『入力が無ければ素子は休む』と食い違う」と外したのは、Claude が【み】の決めを覆したもの
     抑制で割る（÷(1＋抑制)。CL からの駆動と同じ入り方）
  4 NMDA の長い尾：興奮の入力のうち f を、NMDA の時定数でならして届ける（版93 と同じ式。ならした値の行き着く先は入力と同じ）
     視床下核・境 ＝ 淡蒼球の決め（21ms・1,340ms 半々、f 0.4）【ヒト】Küppenbender 1998【み】10-17 ／
     外側手綱核 ＝ 22ms 65%・137ms 35%、f 0.5【ヒト】HPA ＋ Vicini 1998【み】10-18 13:17 ／ 吻内側被蓋核・黒質 ＝ 75.5ms、f 0.5【マウス】Pearlstein 2015
     ⚠️ 1歩 10.67ms で回す（版93 は周 1.067ms ごと）。21・22ms の速い尾は 1歩で 6割ほど減る
⚠️ 数値は合わせない（10-18 22:08【み】）。版93 の i0 は版93 の入力で合わせた値の写し ⇒ 全部組んだあと合わせ直す
"""
import ast, pathlib, sys

TB = [
    ('        # 版94：中脳の核へ入る線（受け手1体が K_NEW 本。⚠️ ノブ）。出どころは版93 と同じ線（設計案_基底核と小脳 10-16〜18）\n'
     '        #   ⟲ 版93 の midbrain.py（受け手1体が 64本・前頭前野 → 黒質 8本・側坐核は表現＋誤差）\n'
     '        self.p_pfc2stn = rp(nn["前頭前野"], nn["視床下核"])\n'
     '        self.p_stn2gpb = rp(nn["視床下核"], nn["境"])\n'
     '        self.p_gpb2lhb = rp(nn["境"], nn["外側手綱核"])\n'
     '        self.p_lhb2rmtg = rp(nn["外側手綱核"], nn["吻内側被蓋核"])\n'
     '        self.p_stn2sn = rp(nn["視床下核"], nn["黒質"])\n'
     '        self.p_pfc2sn = rp(nn["前頭前野"], nn["黒質"])\n'
     '        self.p_sc2sn = rp(nn["上丘"], nn["黒質"])\n'
     '        self.p_ppn2sn = rp(N_PPN, nn["黒質"])\n'
     '        self.p_nac2gpb = rp(nn["側坐核"], nn["境"])          # 抑制\n'
     '        self.p_rmtg2sn = rp(nn["吻内側被蓋核"], nn["黒質"])  # 抑制\n',
     '        # 版94c：中脳の核へ入る線。受け手1体は送り手の別々の 64体から【み】10-17 ／ 前頭前野 → 黒質 8本 ／\n'
     '        #   視床下核 → 境 は線を出す送り手 69.3%【マカク】Sato 2000b ／ 側坐核 → 境 は表現＋誤差【み】10-18 13:36\n'
     '        #   ⟲ 版94 は全部 K_NEW 8本・側坐核は表現だけ（Claude が前の決めを読まずに置いた）\n'
     '        def rpf(ns, nt, k, frac=1.0):\n'
     '            pool = torch.randperm(ns, device=dev, generator=g)[:max(1, int(round(ns * frac)))]\n'
     '            k = min(k, len(pool))\n'
     '            src = torch.stack([pool[torch.randperm(len(pool), device=dev, generator=g)[:k]] for _ in range(nt)])\n'
     '            return (src.reshape(-1), torch.arange(nt, device=dev).repeat_interleave(k), torch.full((nt,), float(k), device=dev))\n'
     '        self.p_pfc2stn = rpf(nn["前頭前野"], nn["視床下核"], K_NUC)\n'
     '        self.p_stn2gpb = rpf(nn["視床下核"], nn["境"], K_NUC, 0.693)\n'
     '        self.p_gpb2lhb = rpf(nn["境"], nn["外側手綱核"], K_NUC)\n'
     '        self.p_lhb2rmtg = rpf(nn["外側手綱核"], nn["吻内側被蓋核"], K_NUC)\n'
     '        self.p_stn2sn = rpf(nn["視床下核"], nn["黒質"], K_NUC)\n'
     '        self.p_pfc2sn = rpf(nn["前頭前野"], nn["黒質"], 8)\n'
     '        self.p_sc2sn = rpf(nn["上丘"], nn["黒質"], K_NUC)\n'
     '        self.p_ppn2sn = rpf(N_PPN, nn["黒質"], K_NUC)\n'
     '        self.p_nac2gpb = rpf(2 * nn["側坐核"], nn["境"], K_NUC)    # 抑制（側坐核の表現＋誤差）\n'
     '        self.p_rmtg2sn = rpf(nn["吻内側被蓋核"], nn["黒質"], K_NUC)  # 抑制\n'
     '        # 版94c：NMDA の長い尾（核ごと）と、自分で撃つ分 i0\n'
     '        import math as _m_nuc\n'
     '        tl = {"a1": [], "m1": [], "a2": [], "m2": [], "f": []}\n'
     '        for k_ in NUC:\n'
     '            (t1, w1), (t2, w2), f_ = NUC_TAIL[k_]\n'
     '            for kk, v in (("a1", _m_nuc.exp(-_DT_MS / t1)), ("m1", w1), ("a2", _m_nuc.exp(-_DT_MS / t2)), ("m2", w2), ("f", f_)):\n'
     '                tl[kk].append(torch.full((nn[k_],), float(v), device=dev))\n'
     '        self.nuc_tail = {kk: torch.cat(v) for kk, v in tl.items()}\n'
     '        n_nuc = sum(nn[k_] for k_ in NUC)\n'
     '        self.nuc_s1 = torch.zeros(n_nuc, device=dev); self.nuc_s2 = torch.zeros(n_nuc, device=dev)\n'
     '        dkm = _m_nuc.exp(-_DT_MS / TAU_M_NUC)\n'
     '        self.nuc_i0 = torch.cat([torch.full((nn[k_],), NUC_VINF[k_] * HUMAN_TH * (1.0 - dkm), device=dev) for k_ in NUC])\n'),
    ('                   "上丘→黒質": self.up["上丘"], "PPN→黒質": self._ppn}\n',
     '                   "上丘→黒質": self.up["上丘"], "PPN→黒質": self._ppn}\n'),
    ('            for key, v in (("側坐核→境", self.nac_rep), ("吻内側被蓋核→黒質", self.nuc["吻内側被蓋核"])):\n',
     '            for key, v in (("側坐核→境", self.nac_out), ("吻内側被蓋核→黒質", self.nuc["吻内側被蓋核"])):\n'),
    ('        self.nac_rep = torch.zeros(nn["側坐核"], device=dev)\n',
     '        self.nac_out = torch.zeros(2 * nn["側坐核"], device=dev)   # 版94c：側坐核の表現＋誤差\n'),
    ('        self.nac_rep = cut("側坐核", rep)\n',
     '        self.nac_out = torch.cat([cut("側坐核", rep), cut("側坐核", err)])   # 版94c\n'),
    ('            for nm_h in ("歯状回", "CA3", "CA1"):\n'
     '                o_h = int(self.off_ch[NAMES.index(nm_h)]); vd[o_h:o_h + self.n[nm_h]] = dk(SUM_TAU_HIPPO_MS)\n',
     '            for nm_h in ("歯状回", "CA3", "CA1"):\n'
     '                o_h = int(self.off_ch[NAMES.index(nm_h)]); vd[o_h:o_h + self.n[nm_h]] = dk(SUM_TAU_HIPPO_MS)\n'
     '            for nm_h in NUC:                                       # 版94c：核の τm 40ms（版93 の撃ち方 B と同じ値）\n'
     '                o_h = int(self.off_ch[NAMES.index(nm_h)]); vd[o_h:o_h + self.n[nm_h]] = dk(TAU_M_NUC)\n'),
    ('            self.cl_t += 1\n'
     '        outs = self.g.step(x, e, learn)\n',
     '            self.cl_t += 1\n'
     '            self.g.cl_drive[self.nuc_ch] = self.nuc_i0             # 版94c：核の自分で撃つ分（÷(1＋抑制)。CL と同じ入り方）\n'
     '        outs = self.g.step(x, e, learn)\n'),
    ('        if self.kick_left > 0:                                 # 運動視床の蹴り\n',
     '        # 版94c：核への興奮の入力のうち f を NMDA の尾でならして届ける（版93 の式。行き着く先は入力と同じ）\n'
     '        xn = x[self.nuc_ch]\n'
     '        T_ = self.nuc_tail\n'
     '        self.nuc_s1 = self.nuc_s1 * T_["a1"] + xn * (1 - T_["a1"])\n'
     '        self.nuc_s2 = self.nuc_s2 * T_["a2"] + xn * (1 - T_["a2"])\n'
     '        x[self.nuc_ch] = (1 - T_["f"]) * xn + T_["f"] * (T_["m1"] * self.nuc_s1 + T_["m2"] * self.nuc_s2)\n'
     '        if self.kick_left > 0:                                 # 運動視床の蹴り\n'),
    ('        self.mb = _MBView(nn, dev)                             # 測る道具のための見かけ\n',
     '        self.mb = _MBView(nn, dev)                             # 測る道具のための見かけ\n'),
    ('W_INH_BU = W_INH * 4.0             # 版94：核への抑制の線 1本（領野の中の抑制 1本 ＝ 4.0 × W_INH と同じ）。学ばない\n',
     'W_INH_BU = W_INH * 4.0             # 版94：核への抑制の線 1本（領野の中の抑制 1本 ＝ 4.0 × W_INH と同じ）。学ばない\n'
     '# 版94c：前のスレッドの決め（設計案_基底核と小脳 §4〜）\n'
     'K_NUC = 64                         # 受け手1体は送り手の別々の 64体から【み】10-17\n'
     'TAU_M_NUC = 40.0                   # 核の表現の体の時間の足し合わせ（版93 の τm）\n'
     '# 入力なしで行き着く先 ÷ 閾値（版93 の撃ち方 B の i0 ÷ (1 − α)、α ＝ exp(−1.067／40)）。1 を越えると自分で撃つ\n'
     'NUC_VINF = {k: v / (1.0 - __import__("math").exp(-1.067 / 40.0)) for k, v in\n'
     '            {"視床下核": 0.0337, "境": 0.0616, "外側手綱核": 0.0117, "吻内側被蓋核": 0.02699, "黒質": 0.00917}.items()}\n'
     '# NMDA の尾：((τ1, 割合), (τ2, 割合), f)\n'
     'NUC_TAIL = {"視床下核": ((21.0, 0.5), (1340.0, 0.5), 0.4), "境": ((21.0, 0.5), (1340.0, 0.5), 0.4),\n'
     '            "外側手綱核": ((22.0, 0.65), (137.0, 0.35), 0.5),\n'
     '            "吻内側被蓋核": ((75.5, 1.0), (75.5, 0.0), 0.5), "黒質": ((75.5, 1.0), (75.5, 0.0), 0.5)}\n'),
]
TB_CH = ('        self.cut_len = [self.n[nm] for nm in NAMES]\n',
         '        self.cut_len = [self.n[nm] for nm in NAMES]\n'
         '        self.nuc_ch = torch.cat([torch.arange(int(self.off_ch[NAMES.index(k)]), int(self.off_ch[NAMES.index(k)]) + self.n[k],\n'
         '                                              device=dev) for k in NUC])   # 版94c：核の表現の体（n_ch_all の並び）\n')
BB = [
    ("#  94b 核ごとの素子の型",
     "#  94c 前のスレッドの核の値を戻す（64本・69.3%・側坐核の表現＋誤差・τm 40ms・i0・NMDA の尾）【み】10-20「ア」\n#  94b 核ごとの素子の型"),
]


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
    if babble.VERSION != 94 or "94b" not in pathlib.Path("babble.py").read_text():
        sys.exit("⚠️ babble.py が版94b ではない（apply_v94.py → apply_v94b.py のあとに当てる）")
    out = [edit("torch_brain.py", TB + [TB_CH]), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版94c を当てた（核の線 64本・69.3%・側坐核の表現＋誤差・τm 40ms・自分で撃つ分・NMDA の尾）", flush=True)
