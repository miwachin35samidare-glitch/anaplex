"""版64：運動の領野の表現・誤差の体の抑制の相手を「抑制の体そのもの（1歩前の列）を 23% の割合で1回ずつ」にする（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v64.py      （版63 の上に当てる）
【み】10-11 22時44分「Aで様子を見よう」（A：切片の見方。抑制の相手は 抑制の体の数 × 人の繋がる割合）
【測】measure_v62b・版63（10-11）：運動の指令の体の抑制は 10周ほぼ平ら。抑制の体1発が 8歩（遅れの列 D＝8）同じ強さで写される
      運動の表現1体の抑制の相手：抑制の体の遅れの写し 50%（8列に均等）・抑制の行の傾きの細胞 50%（どちらも同じ体を何度も数える）
人
  【ヒト】Jiang 2012 PLoS Biol（てんかんの手術の新皮質、36℃、51組）：速い発火の細胞 → 錐体 遅れ 1.00ms・減りの時定数 8.5ms・繋がる割合 79/343（23%）
  【ヒト】Szegedi 2017・Wang 2023・Campagnola 2022：錐体 → 籠 → 錐体 の抑制は 3.5〜4.1ms
  ⇒ 1発の抑制は約1ms で届き、1歩（10.67ms）の終わりに約3割・2歩目の終わりに1割弱【計算】
写し方【C】：抑制の体の発火そのものを、歩の中（版63）と1歩前の列だけで届ける（周ごとに減る形は模型に無い）
  相手：抑制の体ごとに1回だけ、領野の抑制の体の数 × 23%（運動 157体 → 36本）。重さは W_INH のまま（【ヒト】Szegedi 2017）
変えるもの（運動だけ。INH_SLICE）
  torch_brain.py   INH_SLICE ＝ ("運動",)・P_INH_CONN ＝ 0.23。Area に inh_slice を渡す
  torch_area.py    inh_slice の領野は、表現・誤差の行の抑制の相手を 抑制の体の1歩前の列から重なりなしで k ＝ round(抑制の体 × 0.23) 本
                   興奮の相手の数は変えない（相手の合計は k ＋ 興奮）
  torch_group2.py  刈り込みで抑制を生やすとき、inh_slice の領野の表現・誤差の行は 抑制の体の1歩前の列から（⟲ 版61 と同じく、日をかけて戻らないように）
                   ⚠️ 生やすときは重なりを見ない（同じ体が2本になることはある）
変えないもの：抑制の体の受ける相手（興奮 180・抑制）・抑制の重さ W_INH・遅れの列 D＝8（興奮の写しは 8列のまま）・ほかの領野
⚠️ 視床は入れない：視床の介在ニューロン → 中継 の繋がる割合の人の値が無い（Jiang 2012 は皮質の値）
⚠️ 配線が変わる ⇒ 前の state.pt は読めない（SAME_NET を空に）。健診は0から
"""
import ast, pathlib, re, sys

BRAIN = [("RELAY_EXC = 0\n",
          "RELAY_EXC = 0\n"
          "# 版64：抑制の相手を切片の見方で（抑制の体そのものを1回ずつ、繋がる割合 × 体の数）にする領野【み】10-11「Aで様子を見よう」\n"
          "#   【ヒト】Jiang 2012：速い発火の細胞 → 錐体 の繋がる割合 79/343（23%）・減りの時定数 8.5ms（てんかんの組織）\n"
          "#   ⚠️ 視床は入れない（介在ニューロン → 中継 の割合の人の値なし）\n"
          "INH_SLICE = (\"運動\",)\n"
          "P_INH_CONN = 0.23\n"),
         ("            inh_exc_from_rep=(nm in TC_MOD),                 # 版57：抑制の体の興奮の相手は錐体（遅れの列）から。版62 で上丘も\n",
          "            inh_exc_from_rep=(nm in TC_MOD),                 # 版57：抑制の体の興奮の相手は錐体（遅れの列）から。版62 で上丘も\n"
          "            inh_slice=(P_INH_CONN if nm in INH_SLICE else None),   # 版64\n")]
AREA = [("                 inh_n_exc=None, relay_exc=None, inh_exc_from_rep=False):\n",
         "                 inh_n_exc=None, relay_exc=None, inh_exc_from_rep=False, inh_slice=None):\n"),
        ("        self.no_lat_exc = relay_exc == 0     # 版61：中継の表現・誤差の行は自分の領野の中から興奮を受けない\n",
         "        self.no_lat_exc = relay_exc == 0     # 版61：中継の表現・誤差の行は自分の領野の中から興奮を受けない\n"
         "        # 版64：表現・誤差の行の抑制の相手を、抑制の体の1歩前の列から重なりなしで（切片の見方）\n"
         "        self.inh_slice = inh_slice is not None\n"
         "        k_rep_inh = self.n_inh_partner_rows\n"
         "        if self.inh_slice:\n"
         "            k_rep_inh = max(1, int(round(self.n_inh * inh_slice)))\n"
         "            n_partner = k_rep_inh + (n_partner - self.n_inh_partner_rows)\n"
         "            self.n_inh_partner_rows = k_rep_inh\n"),
        ("        src = torch.cat([pick(n_ch * 2, n_partner, self.n_inh_partner_rows if relay_exc is not None else None),\n",
         "        src_rep = pick(n_ch * 2, n_partner, self.n_inh_partner_rows if (relay_exc is not None or self.inh_slice) else None)\n"
         "        if self.inh_slice:                         # 版64：抑制の分を、抑制の体の1歩前の列から重なりなしで引き直す\n"
         "            col0 = self.ch.n_cells + (n_ch + n_ext + torch.arange(self.n_inh, device=dev)) * self.D\n"
         "            sel = torch.rand((n_ch * 2, self.n_inh), device=dev, generator=g).argsort(1)[:, :k_rep_inh]\n"
         "            src_rep = src_rep.view(n_ch * 2, n_partner)\n"
         "            src_rep[:, :k_rep_inh] = col0[sel]\n"
         "            src_rep = src_rep.reshape(-1)\n"
         "        src = torch.cat([src_rep,\n")]
G2 = [("            self._inh_rep_area = [bool(getattr(a, \"inh_exc_from_rep\", False)) for a in self.areas]\n",
       "            self._inh_rep_area = [bool(getattr(a, \"inh_exc_from_rep\", False)) for a in self.areas]\n"
       "            # 版64：inh_slice の領野の表現・誤差の行へ生やす抑制は、抑制の体の1歩前の列から\n"
       "            self._inh_slice_area = [bool(getattr(a, \"inh_slice\", False)) for a in self.areas]\n"
       "            oi = 0\n"
       "            self._pool_i_col0 = []\n"
       "            for a in self.areas:\n"
       "                self._pool_i_col0.append(self.within_inh_index[oi:oi + a.n_inh])\n"
       "                oi += a.n_inh\n"),
      ("                    if kind == 0 and self._inh_rep_area[i]:     # 版57：抑制の体の受け手だけ引き直す\n",
       "                    if kind == 1 and self._inh_slice_area[i]:   # 版64：表現・誤差の受け手は抑制の体の1歩前の列から\n"
       "                        mr = m & ~self.is_inh_r[self.e_recv[cuts].long()]\n"
       "                        kr = int(mr.sum())\n"
       "                        if kr:\n"
       "                            pc = self._pool_i_col0[i]\n"
       "                            new[mr] = pc[torch.randint(0, len(pc), (kr,), device=self.dev, generator=self.gen)]\n"
       "                    if kind == 0 and self._inh_rep_area[i]:     # 版57：抑制の体の受け手だけ引き直す\n")]
BB = [("VERSION = 63          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (62,)",
       "VERSION = 64          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()"),
      ("#  63  周回の中で撃った抑制の体の発火も次の周で届ける（配線は版62 のまま）【ヒト】Szegedi 2017。v62b で運動の抑制が10周とも同じ【み】10-11\n",
       "#  63  周回の中で撃った抑制の体の発火も次の周で届ける（配線は版62 のまま）【ヒト】Szegedi 2017。v62b で運動の抑制が10周とも同じ【み】10-11\n"
       "#  64  運動の表現・誤差の抑制の相手を、抑制の体そのもの（1歩前の列）から 体の数 × 23% を1回ずつ【ヒト】Jiang 2012【み】10-11「Aで様子を見よう」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py")


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
    if babble.VERSION != 63:
        sys.exit(f"⚠️ babble.py が版63 ではない（版 {babble.VERSION}）。先に apply_v63.py")
    out = [edit("torch_brain.py", BRAIN), edit("torch_area.py", AREA), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版64 を当てた（torch_brain・torch_area・torch_group2・babble）。運動の表現・誤差の抑制は抑制の体そのものから 23% を1回ずつ")
    print("   ⚠️ 前の state.pt は読めない。健診は0から")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "64" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['64'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版64 でも回るようにした")
