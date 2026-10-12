"""版73：下からのシナプスの学びを人の組の窓に直す（皮質 Verhoog 2013・海馬 Testa-Silva 2010）。組は一番近い発火どうし（版72 の上）
（~/anaplex で。版72 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v73.py
【み】10-12「どうして根幹に関わるシナプスの学びを外そうとする？」⇒ 学びは外さない。人の窓に直す
⟲ 版72 の誤り：皮質の道にも「受け手が先 −80〜−130ms で弱まる」（Testa-Silva の海馬の値）を当てた。人の新皮質は −100ms まで強まる（Verhoog 2013）
  しかも同じ論文の強まる窓を入れず、弱まりだけが毎歩かかって重さが潰れた（版72 健診：言えた 3/43）
窓（1歩 ＝ 10.67ms。k ＝ 何歩前か）
  皮質の道（受け手が皮質・視床・側坐核・上丘ほか）【ヒト】Verhoog 2013（人の側頭の新皮質、31人、0.14Hz で 50組）
    受け手が先：k 1〜4（−10〜−40ms）×1.26^(1/50)（+26%）／ k 5〜8（−40〜−80ms）×1.16^(1/50)（+13〜19% の中ほど）／ それより前は変えない
    入力が先：今までどおり（入力が同じ歩か1歩前、門 d で ×0.787〜1.218 を 75組）【ヒト】Frontiers Cell Neurosci 2021・Verhoog（+8ms より後は弱まる）
  海馬の道（受け手が歯状回・CA3・CA1：貫通路・苔状線維・シャッファー）【ヒト】Testa-Silva 2010（人の海馬、0.14Hz で 40〜50組）
    受け手が先：k 1〜8（−10〜−80ms）×1.58^(1/45)（+58%）／ k 9〜12（−80〜−130ms）×0.45^(1/45)（−55%）
    入力が先（同じ歩か1歩前）：×1.72^(1/45)（+10ms で +72%）
    ⚠️ 歯状回の顆粒細胞は錐体ではない（人の値なし）⇒ 錐体の値を当てた【C】
組の数え方：一番近い発火どうし（入力が撃った歩に、受け手の一番近い前の発火と組む ／ 受け手が撃った歩に、入力の同じ歩か1歩前と組む）
  【C】人の値は「間を空けた1組ずつ」だけ。続けて撃つときの足し合わさり方は人で決まっていない（10-12 夜 調べた）
  一番近い発火にした理由：人の強まる窓は、受け手の発火の後の脱分極（ADP、人 62ms）に入力が乗ることで決まる（Verhoog 2013）。
    受け手が次に撃てば脱分極は新しく始まる ⇒ 入力が見るのは一番近い前の受け手の発火【C の読み】
⚠️ CL → 皮質（版67）の学びは前の形のまま（受け手が先 1歩・×1.21^(1/75)）
SAME_NET (66, 67, 68, 69, 70, 71, 72)
"""
import ast, pathlib, re, sys

BRAIN = [
    ("LTD_LAG = (8, 12)        # 版72：受け手が 8〜12歩前（80〜130ms 前）に撃っていて送り手が撃つと弱まる【ヒト】Testa-Silva 2010\n"
     "LTD_F = 0.45             # 40〜50組で −55%【ヒト】Testa-Silva 2010（人の海馬）。皮質にも当てた【C】\n"
     "LTD_PAIRS = 45.0\n",
     "# 版73：下からのシナプスの組の窓（k ＝ 受け手が何歩前に撃ったか）。⟲ 版72 の LTD_LAG（皮質にも海馬の弱まりを当てた）は外した\n"
     "import math as _m_win\n"
     "WIN_CTX = {k: _m_win.log(1.26) / 50 for k in range(1, 5)}            # 【ヒト】Verhoog 2013 −10〜−40ms +26%（50組）\n"
     "WIN_CTX.update({k: _m_win.log(1.16) / 50 for k in range(5, 9)})      # −40〜−80ms +13〜19%\n"
     "WIN_HIP = {k: _m_win.log(1.58) / 45 for k in range(1, 9)}            # 【ヒト】Testa-Silva 2010 −10〜−80ms +58%（40〜50組）\n"
     "WIN_HIP.update({k: _m_win.log(0.45) / 45 for k in range(9, 13)})     # −80〜−130ms −55%\n"
     "LN_PRE_HIP = _m_win.log(1.72) / 45                                    # 海馬 入力が先 +10ms +72%\n"
     "WIN_MAXK = 12\n"),
    ("class _BUSyn:\n", "_LN_LTD = 0.0   # 版72 の名残（版73 で使わない）\n\n\nclass _BUSyn:\n"),
    ("_LN_LTD = __import__(\"math\").log(LTD_F)   # 版72\n\n\n", ""),
    ("        self.post_hist = torch.zeros(LTD_LAG[1], self.n_tgt, device=dev)   # 版72：受け手が 1〜12歩前に撃ったか\n",
     "        self.post_age = torch.full((self.n_tgt,), 10 ** 6, dtype=torch.long, device=dev)   # 版73：受け手が何歩前に撃ったか\n"
     "        self.hippo = False\n"
     "        tab = torch.zeros(WIN_MAXK + 2, device=dev)\n"
     "        self._tab = {False: tab.clone(), True: tab.clone()}\n"
     "        for k, v in WIN_CTX.items():\n"
     "            self._tab[False][k] = v\n"
     "        for k, v in WIN_HIP.items():\n"
     "            self._tab[True][k] = v\n"),
    ("        pre_now = self.pre[self.src] > 0\n"
     "        pre_e = pre_now | (self.pre_prev[self.src] > 0)\n"
     "        post_e = post[self.tgt] > 0\n"
     "        ln_pp = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS\n"
     "        pf = (self.post_prev[self.tgt] > 0) & ~post_e & pre_now\n"
     "        ltd = (self.post_hist[LTD_LAG[0] - 1:LTD_LAG[1]].amax(0) > 0)[self.tgt] & pre_now   # 版72：弱まる窓\n"
     "        self.w = self.w * torch.exp(ln_pp * (pre_e & post_e).float() + _CL_LN_POST / CL_PAIRS * pf.float()\n"
     "                                    + _LN_LTD / LTD_PAIRS * ltd.float())\n"
     "        self.pre_prev = self.pre.clone()\n"
     "        self.post_prev = post.clone()\n"
     "        self.post_hist = torch.cat([(post > 0).float().unsqueeze(0), self.post_hist[:-1]], 0)\n",
     "        # 版73：組は一番近い発火どうし\n"
     "        pre_now = self.pre[self.src] > 0\n"
     "        pre_e = pre_now | (self.pre_prev[self.src] > 0)            # 入力が同じ歩か1歩前\n"
     "        post_e = post[self.tgt] > 0\n"
     "        if self.hippo:                                              # 入力が先【ヒト】Testa-Silva\n"
     "            ln_pp = torch.full_like(d, LN_PRE_HIP) if torch.is_tensor(d) else LN_PRE_HIP\n"
     "        else:                                                       # 入力が先【ヒト】Frontiers 2021（門 d）\n"
     "            ln_pp = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS\n"
     "        k = self.post_age.clamp(max=WIN_MAXK + 1)[self.tgt]         # 受け手の一番近い前の発火が何歩前か\n"
     "        pf = self._tab[self.hippo][k] * (pre_now & ~post_e).float()   # 受け手が先\n"
     "        self.w = self.w * torch.exp(ln_pp * (pre_e & post_e).float() + pf)\n"
     "        self.pre_prev = self.pre.clone()\n"
     "        self.post_prev = post.clone()\n"
     "        self.post_age = torch.where(post > 0, torch.ones_like(self.post_age), self.post_age + 1)\n"),
    ("                self.bu[key] = _BUSyn(s_, t_, n_, self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n",
     "                self.bu[key] = _BUSyn(s_, t_, n_, self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n"
     "                self.bu[key].hippo = nm in (\"歯状回\", \"CA3\", \"CA1\")       # 版73：海馬の窓\n"),
]
BB = [("VERSION = 72          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71)",
       "VERSION = 73          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72)"),
      ("#      K_PM 45【測】・視床の閾値 0.09【計算】【み】10-12「2発重なると撃つも要るんじゃないの」「一旦アでやって」\n",
       "#      K_PM 45【測】・視床の閾値 0.09【計算】【み】10-12「2発重なると撃つも要るんじゃないの」「一旦アでやって」\n"
       "#  73  下からのシナプスの学びを人の組の窓に（皮質 Verhoog 2013・海馬 Testa-Silva 2010）。組は一番近い発火どうし【C】。\n"
       "#      ⟲ 版72 の皮質の弱まる窓（海馬の値を当てた誤り）【み】10-12「どうして根幹に関わるシナプスの学びを外そうとする？」\n")]
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
    if babble.VERSION != 72:
        sys.exit(f"⚠️ babble.py が版72 ではない（版 {babble.VERSION}）。先に apply_v72.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版73 を当てた（torch_brain・babble）。下からのシナプスの学びを人の組の窓に（皮質 Verhoog・海馬 Testa-Silva）、組は一番近い発火どうし")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "73" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['73'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版73 でも回るようにした")
