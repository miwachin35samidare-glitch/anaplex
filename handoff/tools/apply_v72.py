"""版72：時間の足し合わせ（膜が前の歩の入力を持ち越す）・学びの弱まる窓・K_PM 45・視床の閾値の出し直し（版71 の上）
（~/anaplex で。版71 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v72.py
【み】10-12「アだけど2発重なると撃つも要るんじゃないの」「本当に全部がそうなんだよね？」「ウ」「一旦アでやって」
1 時間の足し合わせ（表現の体・抑制の体）
  撃たなかった体は、この歩の駆動の残りを次の歩へ持ち越す（1歩 10.67ms で exp(−10.67／時定数) 倍）。撃ったら 0 に戻す
    皮質などの表現の体  15ms → 0.49【ヒト】L2/3 錐体の膜の時定数 12〜17ms（Eyal 2016 ほか）
    海馬（歯状回・CA3・CA1）の表現の体  30ms → 0.70【ヒト】海馬の錐体の EPSP の減り 25〜38ms（Testa-Silva 2010）⚠️ 膜の時定数ではない
    視床・背内側視床・側坐核・上丘  15ms【C】（人・霊長類の値なし。【み】「一旦アでやって」）
    抑制の体  6.8ms → 0.21【ヒト】速い籠細胞 中央 6.76ms（PMC9934405、8体）
  誤差の体は的と予測で撃つ規則のまま（足し合わせを入れない）
2 学びの弱まる窓（版69〜71 の下からのシナプス）
  受け手が 8〜12歩前（80〜130ms 前）に撃っていて、この歩に送り手が撃つと弱まる：40〜50組で −55% ⇒ 1組 ×0.45^(1/45)
    【ヒト】Testa-Silva 2010（人の海馬、0.14Hz で 40〜50組）：−80〜−130ms で傾き −55%（錐体 3体）
    ⚠️ 人の海馬の値。皮質の弱まる窓の人の値は無い ⇒ 皮質の道にも当てた【C】
    ⚠️ CL → 皮質（版67）の学びには入れていない（CL の体の率が低く、強まり続けは見ていない）
3 K_PM（シナプスのとき）53 → 45【測】measure_v67b 20（版71）：45 で指令の体 38.3%（35.9Hz）
4 視床の閾値（シナプスのとき）0.07 → 0.09：足し合わせ（0.49）を入れて、続く音で中継の効き 50%【マカク】になる値【計算】
ANAPLEX_BU=0 は版68 のまま（足し合わせも入らない）／ ANAPLEX_SUM=0 で足し合わせだけ外す
SAME_NET (66, 67, 68, 69, 70, 71)（Group2 の配線は同じ）
"""
import ast, pathlib, re, sys

BRAIN = [
    ("K_PM_SYN = 53            # 指令の体 35Hz：放出 0.51（35Hz の定常）⇒ 27 ÷ 0.51【計算】⚠️ measure_v67b で確かめる\n",
     "K_PM_SYN = 45            # 指令の体 35Hz【測】measure_v67b 20（版71）：45 で 38.3%（35.9Hz）。⟲ 版71 は 53（27 ÷ 放出 0.51 の見積もり）\n"),
    ("FIRE_TH_THAL_SYN = 0.07  # 視床：続く音で中継の効き 50%【マカク】Sincich 2007・Lee 1985 ⇒ 0.07【計算】ノブ（人の値なし）\n",
     "FIRE_TH_THAL_SYN = 0.09  # 視床：続く音で中継の効き 50%【マカク】Sincich 2007・Lee 1985 ⇒ 足し合わせ込みで 0.09【計算】ノブ（⟲ 版71 0.07）\n"
     "# 版72：時間の足し合わせ（1歩 10.67ms のあとに残る割合 ＝ exp(−10.67／時定数)）\n"
     "SUM_TAU_MS = 15.0        # 表現の体【ヒト】L2/3 錐体 12〜17ms（Eyal 2016 ほか）。視床・背内側視床・側坐核・上丘にも当てた【C】\n"
     "SUM_TAU_HIPPO_MS = 30.0  # 歯状回・CA3・CA1【ヒト】海馬の錐体の EPSP の減り 25〜38ms（Testa-Silva 2010）\n"
     "SUM_TAU_INH_MS = 6.8     # 抑制の体【ヒト】速い籠細胞 6.76ms（PMC9934405）\n"
     "LTD_LAG = (8, 12)        # 版72：受け手が 8〜12歩前（80〜130ms 前）に撃っていて送り手が撃つと弱まる【ヒト】Testa-Silva 2010\n"
     "LTD_F = 0.45             # 40〜50組で −55%【ヒト】Testa-Silva 2010（人の海馬）。皮質にも当てた【C】\n"
     "LTD_PAIRS = 45.0\n"),
    # _BUSyn：受け手の過去 12歩
    ("        self.post_prev = torch.zeros(self.n_tgt, device=dev)\n",
     "        self.post_prev = torch.zeros(self.n_tgt, device=dev)\n"
     "        self.post_hist = torch.zeros(LTD_LAG[1], self.n_tgt, device=dev)   # 版72：受け手が 1〜12歩前に撃ったか\n"),
    ("        self.w = self.w * torch.exp(ln_pp * (pre_e & post_e).float() + _CL_LN_POST / CL_PAIRS * pf.float())\n",
     "        ltd = (self.post_hist[LTD_LAG[0] - 1:LTD_LAG[1]].amax(0) > 0)[self.tgt] & pre_now   # 版72：弱まる窓\n"
     "        self.w = self.w * torch.exp(ln_pp * (pre_e & post_e).float() + _CL_LN_POST / CL_PAIRS * pf.float()\n"
     "                                    + _LN_LTD / LTD_PAIRS * ltd.float())\n"),
    ("        self.post_prev = post.clone()\n",
     "        self.post_prev = post.clone()\n"
     "        self.post_hist = torch.cat([(post > 0).float().unsqueeze(0), self.post_hist[:-1]], 0)\n"),
    ("class _BUSyn:\n",
     "_LN_LTD = __import__(\"math\").log(LTD_F)   # 版72\n\n\n"
     "class _BUSyn:\n"),
    # 足し合わせの準備（シナプスのとき）
    ("            self.g.fire_th = ft\n",
     "            self.g.fire_th = ft\n"
     "            if _os_cl.environ.get(\"ANAPLEX_SUM\", \"1\") != \"0\":       # 版72：時間の足し合わせ\n"
     "                import math as _m_sum\n"
     "                dk = lambda tau: _m_sum.exp(-10.67 / tau)\n"
     "                vd = torch.full((self.g.n_ch_all,), dk(SUM_TAU_MS), device=dev)\n"
     "                for nm_h in (\"歯状回\", \"CA3\", \"CA1\"):\n"
     "                    o_h = int(self.off_ch[NAMES.index(nm_h)]); vd[o_h:o_h + self.n[nm_h]] = dk(SUM_TAU_HIPPO_MS)\n"
     "                self.g.v_decay = vd\n"
     "                self.g.v_carry = torch.zeros(self.g.n_ch_all, device=dev)\n"
     "                self.g.v_decay_inh = dk(SUM_TAU_INH_MS)\n"
     "                self.g.v_carry_inh = None                              # 1周目に大きさを合わせて作る\n"),
]
G2 = [
    ("        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n",
     "        if getattr(self, \"v_carry\", None) is not None:       # 版72：前の歩から持ち越した膜の残り（時間の足し合わせ）\n"
     "            drive = drive + self.v_carry\n"
     "        self._drive_last = drive\n"
     "        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n"),
    ("        inh_spk = (d_inh > INH_TH).float()\n",
     "        if getattr(self, \"v_carry\", None) is not None:       # 版72：抑制の体の足し合わせ\n"
     "            if self.v_carry_inh is None:\n"
     "                self.v_carry_inh = torch.zeros_like(d_inh)\n"
     "            d_inh = d_inh + self.v_carry_inh\n"
     "        self._dinh_last = d_inh\n"
     "        inh_spk = (d_inh > INH_TH).float()\n"),
    ("            rep, err, inh_spk = self._any     # 版59：最後の周ではなく、この歩のどこかで撃ったか\n",
     "            rep, err, inh_spk = self._any     # 版59：最後の周ではなく、この歩のどこかで撃ったか\n"
     "            if getattr(self, \"v_carry\", None) is not None:   # 版72：撃たなかった体だけ残りを次の歩へ（撃ったら 0）\n"
     "                self.v_carry = self.v_decay * self._drive_last * (1.0 - rep)\n"
     "                self.v_carry_inh = self.v_decay_inh * self._dinh_last * (1.0 - inh_spk)\n"),
]
BB = [("VERSION = 71          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70)",
       "VERSION = 72          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71)"),
      ("#      視床の閾値 0.3 → 0.07（中継の効き 50%【マカク】Sincich 2007・Lee 1985）【み】10-12「ウで駄目ならイ」「もっと探してみて」\n",
       "#      視床の閾値 0.3 → 0.07（中継の効き 50%【マカク】Sincich 2007・Lee 1985）【み】10-12「ウで駄目ならイ」「もっと探してみて」\n"
       "#  72  時間の足し合わせ（表現 15ms・海馬 30ms・抑制 6.8ms【ヒト】、視床ほか 15ms【C】）・学びの弱まる窓（−80〜−130ms −55%【ヒト】海馬）・\n"
       "#      K_PM 45【測】・視床の閾値 0.09【計算】【み】10-12「2発重なると撃つも要るんじゃないの」「一旦アでやって」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py")


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
    if babble.VERSION != 71:
        sys.exit(f"⚠️ babble.py が版71 ではない（版 {babble.VERSION}）。先に apply_v71.py")
    out = [edit("torch_brain.py", BRAIN), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版72 を当てた（torch_brain・torch_group2・babble）。時間の足し合わせ・学びの弱まる窓・K_PM 45・視床の閾値 0.09")
    print("   ANAPLEX_BU=0 は版68 のまま ／ ANAPLEX_SUM=0 で足し合わせだけ外す")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "72" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['72'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版72 でも回るようにした")
