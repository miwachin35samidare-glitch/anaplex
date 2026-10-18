"""版93：中脳と手綱・視床下核の辺縁・境の核を、撃ち方 B・周ごと・学ぶ形に（核の作り）
（~/anaplex で。版92 の上に当てる。印が1つでも合わなければ何も書かずに止まる。midbrain_v93.py を同じ場所に置く）
  python3 apply_v93.py
設計：handoff/docs/設計案_基底核と小脳.md（10-16〜10-18）／ 試し台 bench_v93a〜h・d2 ／ 新しい midbrain の頭の説明
変えるもの
  midbrain.py：まるごと置き換え（各 256体・撃ち方 B・周ごとに 10周・NMDA の尾・受け手1体は別々の 64体から・学ぶ）
  torch_brain.py
    1 中脳の核の体数を 256 に・線の張り方を新しい形に（前頭前野の表現・側坐核の表現＋誤差・上丘の誤差・PPN）
    2 PPN（脳幹）は周ごとに撃つ（1周の確率 ＝ 1歩の確率を 10周に分けたもの。CL へは1歩のどこかで撃ったか）【み】10-18 16:27
    3 中脳へは、領野が撃った周のまま渡す（Group2 の _last_fire）【み】10-18 16:18
    4「来なかった誤差」（torch_brain が誤差の素子の中の値から計算していた値）を中脳へ渡すのをやめる【み】10-18 16:57
  brainstem.py：版40 の名残（DA_LAT・DA_DUR・DA_CAP・da_buf）を消す
  babble.py：VERSION 93（SAME_NET に 92。中脳の重さは新しく保存する）・履歴の行・中脳の重さの保存と読み込み
⚠️ 値が変わる（中脳の作り・PPN のくじの並び）。網の領野の作りは同じ ⇒ 版92 の state.pt は読める（中脳は初めの重さから）
"""
import ast, pathlib, re, sys

TB = [
    ("#   黒質・視床下核の辺縁・境の細胞・外側手綱核・吻内側被蓋核は midbrain.py（学ばない小さな核）\n",
     "#   黒質・視床下核の辺縁・境の細胞・外側手綱核・吻内側被蓋核は midbrain.py（版93：撃ち方 B を周ごとに・入るシナプスは学ぶ）\n"),
    ('N_MB = {"黒質": 56, "視床下核": 64, "境": 64, "外側手綱核": 64, "吻内側被蓋核": 64}\n',
     '# 版93：各 256体（基底核の核と同じ。動く最低数は 128 から t ＞ 3、bench_v93f 版6）【み】10-18 12:44\n'
     'N_MB = {"黒質": 256, "視床下核": 256, "境": 256, "外側手綱核": 256, "吻内側被蓋核": 256}\n'),
    ("P_PPN = 23.2 * 10.67 / 1000.0      # 【ヒト】\n",
     "P_PPN = 23.2 * 10.67 / 1000.0      # 【ヒト】\n"
     "# 版93：PPN は周ごとに撃つ（1歩に撃つ確率は同じ P_PPN）【み】10-18 16:27\n"
     "P_PPN_PASS = 1.0 - (1.0 - P_PPN) ** (1.0 / 10)\n"),
    ("        # 中脳の小さな核（学ばない）\n"
     "        import midbrain as MB\n"
     "        self.mb = MB.Midbrain(N_MB, dev=dev, gen=g)\n"
     "        self.mb.wire(nn[\"上丘\"], nn[\"側坐核\"], N_OMIT)\n"
     "        self.omit = torch.zeros(N_OMIT, device=dev)\n"
     "        self.gate = torch.ones((), device=dev)\n",
     "        # 中脳の小さな核（版93：撃ち方 B・周ごと・学ぶ）。外から来る送り手：前頭前野の表現・側坐核の表現＋誤差・上丘の誤差・PPN\n"
     "        #   ⟲ 版43b〜92 は「来なかった誤差」も渡していた（ニューロンの発火ではない計算の値）【み】10-18 16:57\n"
     "        import midbrain as MB\n"
     "        self.mb = MB.Midbrain(N_MB, dev=dev, gen=g)\n"
     "        self.mb.wire({\"前頭前野\": nn[\"前頭前野\"], \"側坐核\": 2 * nn[\"側坐核\"], \"上丘\": nn[\"上丘\"], \"PPN\": N_PPN})\n"
     "        self.gate = torch.ones((), device=dev)\n"),
    ("        if self.bu_on:                                         # 版75：残りの道を1回で（入力は版74 と同じもの）\n"
     "            vin = {",
     "        # 版93：PPN（脳幹）は周ごとに撃つ。CL へは1歩のどこかで撃ったか、中脳の核へは撃った周のまま\n"
     "        self._ppn10 = (torch.rand(self.g.recur, N_PPN, device=self.dev, generator=getattr(self, \"cl_gen\", self.gen))\n"
     "                       < P_PPN_PASS).float()\n"
     "        self._ppn = self._ppn10.amax(0)\n"
     "        if self.bu_on:                                         # 版75：残りの道を1回で（入力は版74 と同じもの）\n"
     "            vin = {"),
    ("            if \"PPN→CL\" in self.bu:                              # PPN のくじ（cl_gen）を CL の放出より前に引く\n"
     "                self._ppn = (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()\n"
     "                vin[\"PPN→CL\"] = self._ppn\n",
     "            if \"PPN→CL\" in self.bu:                              # 版93：PPN のくじは上で（周ごと）\n"
     "                vin[\"PPN→CL\"] = self._ppn\n"),
    ("        # 版43b：前頭前野の「来なかった誤差」（頭 N_OMIT 体）＝ 予測 − 下からの入力 が THETA を越えた\n"
     "        #   【ヒト】上側頭回の後ろ・外側眼窩前頭（別の群れ）／【マカク】眼窩前頭 3.6〜7.1%\n"
     "        #   誤差のニューロンの予測 p と的（下からの入力で撃つか）は、音の誤差と同じものを使う（向きだけ逆）\n"
     "        pe, xt = outs[2], self.g.last_xt\n"
     "        lo = int(self.off_ch[NAMES.index(\"前頭前野\")])\n"
     "        self.omit = ((pe[lo:lo + N_OMIT] - xt[lo:lo + N_OMIT]) > THETA).float()\n"
     "        # 中脳：1歩前の 上丘の誤差・側坐核の誤差・来なかった誤差 を受けて進む。門は脳幹が使う\n"
     "        self.gate = self.mb.step(self.up[\"上丘\"], self.up[\"側坐核\"], self.omit)\n",
     "        # 版93：中脳の核へ、領野が撃った周のまま渡す（Group2 の「最後に撃った周」。1歩に2回以上撃った体は最後の1回、今の _any と同じ）\n"
     "        #   ⟲ 版43b〜92 の「来なかった誤差」（予測 − 下からの入力 を torch_brain が計算した値）は渡さない【み】10-18 16:57\n"
     "        #      ⇒ 来ないことに撃つ細胞を本物の素子として作り直す項目（設計案 10-18 17:10）\n"
     "        R = self.g.recur\n"
     "        pc, lf = self.g._pc_t, self.g._last_fire\n"
     "        ps_rep = (lf[0] - (pc - R + 1)).clamp(0, R - 1)\n"
     "        ps_err = (lf[1] - (pc - R + 1)).clamp(0, R - 1) if len(lf[1]) == len(err) else torch.zeros_like(err, dtype=torch.long)\n"
     "        mb = self.mb\n"
     "        ext = {\"前頭前野\": mb.ext_passes(cut(\"前頭前野\", rep), cut(\"前頭前野\", ps_rep)),\n"
     "               \"側坐核\": mb.ext_passes(torch.cat([cut(\"側坐核\", rep), cut(\"側坐核\", err)]),\n"
     "                                      torch.cat([cut(\"側坐核\", ps_rep), cut(\"側坐核\", ps_err)])),\n"
     "               \"上丘\": mb.ext_passes(cut(\"上丘\", err), cut(\"上丘\", ps_err)),\n"
     "               \"PPN\": self._ppn10}\n"
     "        self.gate = mb.step(ext, learn=learn)        # 門は脳幹が使う\n"),
]
BS = [
    ("DA_LAT = int(round(150.0 / DT_MS))       # 14歩\n"
     "DA_DUR = int(round(225.0 / DT_MS))       # 21歩\n"
     "DA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）\n",
     "#   ⟲ 版93：版40 の名残 DA_LAT・DA_DUR・DA_CAP を消した（門は midbrain が出す）\n"),
    ("        self.hist = torch.zeros(max(SURPRISE_LAT, DA_LAT) + 1, device=self.dev)   # 版40：中脳の遅れの分まで持つ\n"
     "        self.da_buf = torch.zeros(DA_DUR, device=self.dev)   # 中脳の短い発火が続く長さ（版40）\n",
     "        self.hist = torch.zeros(SURPRISE_LAT + 1, device=self.dev)   # 版93：中脳の遅れの分（版40 の名残）は消した\n"),
    ("        self.da_buf.zero_()\n", ""),
]
BB = [
    ("VERSION = 92          # コードの版。網の作りを変えたら上げる\n"
     "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91)",
     "VERSION = 93          # コードの版。網の作りを変えたら上げる\n"
     "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92)"),
    ("#  92  組む途中と刈り込みの並べ直しで、番号を 32bit・写しを1本ずつ（値は同じ）【み】10-15「いいよ」【測】版91 山 10.16GB\n",
     "#  92  組む途中と刈り込みの並べ直しで、番号を 32bit・写しを1本ずつ（値は同じ）【み】10-15「いいよ」【測】版91 山 10.16GB\n"
     "#  93  中脳の核を撃ち方 B・周ごと・学ぶ形に（各 256体）・PPN は周ごと・領野の撃った周のまま渡す・来なかった誤差は渡さない【み】10-18\n"),
    ("    if getattr(b, \"cl_on\", False):     # 版67：CL → 皮質 のシナプス\n"
     "        d[\"cl\"] = {",
     "    d[\"mb\"] = b.mb.state_dict()        # 版93：中脳の核の重さ\n"
     "    if getattr(b, \"cl_on\", False):     # 版67：CL → 皮質 のシナプス\n"
     "        d[\"cl\"] = {"),
    ("    return int(d[\"done\"]), list(d.get(\"log\", []))\n",
     "    if \"mb\" in d:                       # 版93（版92 までの state には無い ⇒ 初めの重さ）\n"
     "        b.mb.load_state_dict(d[\"mb\"])\n"
     "    return int(d[\"done\"]), list(d.get(\"log\", []))\n"),
]
CHK = [   # 健診（measure_check・measure_check2）の中脳の率：版93 は周ごとに撃つので「1歩のどこかで撃ったか」では速い核を数え落とす ⇒ 核が数えた率で
    ('MB_HZ = {"黒質": 4.56, "視床下核": 36.0, "境": 33.0, "外側手綱核": 30.0, "吻内側被蓋核": 17.8}',
     'MB_HZ = {"黒質": 4.56, "視床下核": 19.3, "境": 33.0, "外側手綱核": 30.0, "吻内側被蓋核": 17.8}   # 版93：視床下核の辺縁 19.3【ヒト】'),
    ('    f"{nm} {mb[nm]/k*1000/T.DT_MS:.1f}（{hz}）" for nm, hz in MB_HZ.items()), flush=True)',
     '    f"{nm} {b.mb.rates_hz()[nm]:.1f}（{hz}）" for nm, hz in MB_HZ.items()), flush=True)   # 版93：周ごとの発火を核が数えた率'),
]
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
    if babble.VERSION != 92:
        sys.exit(f"⚠️ babble.py が版92 ではない（版 {babble.VERSION}）")
    old = pathlib.Path("midbrain.py").read_text()
    if "（版43b）" not in old:
        sys.exit("⚠️ midbrain.py が版43b の形ではない。何も書いていない")
    new_mb = pathlib.Path(__file__).with_name("midbrain_v93.py")
    if not new_mb.exists():
        new_mb = pathlib.Path("midbrain_v93.py")
    if not new_mb.exists():
        sys.exit("⚠️ midbrain_v93.py が見つからない（apply_v93.py と同じ場所に置く）。何も書いていない")
    mbs = new_mb.read_text(); ast.parse(mbs)
    out = [edit("torch_brain.py", TB), edit("brainstem.py", BS), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    pathlib.Path("midbrain.py").write_text(mbs)
    print("✅ 版93 を当てた（midbrain 置き換え・torch_brain・brainstem・babble）", flush=True)
    for f in ("measure_check.py", "measure_check2.py"):
        if pathlib.Path(f).exists():
            p_, s_ = edit(f, CHK); p_.write_text(s_)
            print(f"   {f}：中脳の率を核が数えた率に（視床下核 19.3Hz）", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "93" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['93'])}):" + s[m.end():]
            q.write_text(s)
            try:
                print(f"   {f}：版93 でも回るようにした", flush=True)
            except BrokenPipeError:
                pass
