"""版69：下から上への道を学ぶシナプスにする（皮質・皮質下の11本の道。海馬の中・運動前野 → 運動・蹴りはまだ）
（~/anaplex で。版68 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v69.py
【み】10-12「人のシナプスは全域で学ぶんだよね？」「土台になると思うから、全体をシナプスにしよう」
【み】10-12「1本は1本」「確かめて作った今の本数を置き換えるんじゃないわけ？」「切片を領野にして繋いでいるイメージ」（10-06）
【み】10-12「アでいいよ」：時刻の情報が要る所は回数（時間細胞）、大きさだけ足りない所は gain。まず下からの道をシナプスに（gain は入口に残す）
設計：handoff/docs/設計案_全体シナプス.md
組み方
  今の道の1本1本を、そのまま網の辺（シナプス）に置き換える。どこからどこへ何本かは変えない（確かめて作った本数）
    視床 → 聴覚（自分の帯の 10本）／ 聴覚 → 連合（約211本）／ 連合 → 高次（峰 約8本）／
    高次・背内側視床 → 前頭前野 ／ 高次・前頭前野 → 嗅内浅 ／ CA1 → 嗅内深 ／ CA1・嗅内深 → 側坐核 ／ 下丘 → 上丘（各 8本）
  1本 ＝ 人の繋がり1本：重さ 対数正規 中央 0.41・平均 1（1.12mV ＝ 1）【ヒト】Planert 2025・Hunt 2023（版67 の CL と同じ当て方【C】）
  短期の増減：人の ex→ex の表（Allen）／ 学び：三つの因子（版67 の CL → 皮質 と同じ）【ヒト】Frontiers Cell Neurosci 2021・Verhoog 2013
    受け手：その道が届く領野の表現の体が「この歩のどこかで撃ったか」【C】（版67 と同じ）
  駆動 ＝ Σ（重さ × 放出）。領野の入口の gain（x_gain）は今のまま掛かる（ハードの制約のノブ。人の形ではない）
  誤差の的（下からだけで閾値を越えたか）も、この重さの和で決まる ⇒ 的も学びで動く
  ANAPLEX_BU=0 で版68 と同じ振る舞い
まだ：海馬の中（貫通路・苔状線維・シャッファー。人の短期の増減・学びが未）／ 下丘 → 視床（視床の閾値が人の値でない）／ 運動前野 → 運動・PPN → CL ／ 蹴り（保留）
保存：下からのシナプスの重さ・短期の増減の状態を state.pt に書く。Group2 の配線は同じ ⇒ 版66〜68 の state.pt は読める（下からの重さは初めから）
"""
import ast, pathlib, re, sys

BRAIN = [
    # 1 道の表と、1本の道の入れ物（クラス）
    ("class Brain:\n",
     "# 版69：下から上への道（学ぶシナプス）。（名前, 受け手の領野）\n"
     "BU_PATHS = ((\"視床→聴覚\", \"聴覚\"), (\"聴覚→連合\", \"連合\"), (\"連合→高次\", \"高次\"),\n"
     "            (\"高次→前頭前野\", \"前頭前野\"), (\"背内側視床→前頭前野\", \"前頭前野\"),\n"
     "            (\"高次→嗅内浅\", \"嗅内浅\"), (\"前頭前野→嗅内浅\", \"嗅内浅\"), (\"CA1→嗅内深\", \"嗅内深\"),\n"
     "            (\"CA1→側坐核\", \"側坐核\"), (\"嗅内深→側坐核\", \"側坐核\"), (\"下丘→上丘\", \"上丘\"))\n\n\n"
     "class _BUSyn:\n"
     "    \"\"\"版69：下から上への道1本ぶん（送り手 → 受け手の領野の体）。重さ・短期の増減・三つの因子の学び（版67 の CL と同じ形）\"\"\"\n\n"
     "    def __init__(self, src, tgt, n_tgt, ch0, stp_path, dev, gen):\n"
     "        import math as _m\n"
     "        self.src, self.tgt, self.n_tgt, self.ch0 = src.long(), tgt.long(), int(n_tgt), int(ch0)\n"
     "        sig = _m.sqrt(2.0 * _m.log(CL_W_MEAN / CL_W_MED))\n"
     "        self.w = torch.exp(_m.log(CL_W_MED) + sig * torch.randn(len(self.src), device=dev, generator=gen))\n"
     "        z = torch.zeros(len(self.src), dtype=torch.bool, device=dev)\n"
     "        self.stp = _STP_CL(len(self.src), z, z, stp_path, dev, gen)   # 【ヒト】ex→ex の表\n"
     "        self.t = 0\n"
     "        self.pre = None\n"
     "        self.pre_prev = None\n"
     "        self.post_prev = torch.zeros(self.n_tgt, device=dev)\n\n"
     "    def drive(self, v):\n"
     "        \"\"\"届いた重さ × 放出 の和（受け手の体ごと）\"\"\"\n"
     "        self.pre = v\n"
     "        out = torch.zeros(self.n_tgt, device=v.device)\n"
     "        e = torch.nonzero(v[self.src] > 0).squeeze(1)\n"
     "        if len(e):\n"
     "            out.index_add_(0, self.tgt[e], self.w[e] * self.stp.release_at(e, self.t))\n"
     "        self.t += 1\n"
     "        return out\n\n"
     "    def learn(self, post, d):\n"
     "        \"\"\"post：受け手の体がこの歩のどこかで撃ったか ／ d：学びの門（0〜1）\"\"\"\n"
     "        if self.pre_prev is None:\n"
     "            self.pre_prev = torch.zeros_like(self.pre)\n"
     "        pre_now = self.pre[self.src] > 0\n"
     "        pre_e = pre_now | (self.pre_prev[self.src] > 0)\n"
     "        post_e = post[self.tgt] > 0\n"
     "        ln_pp = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS\n"
     "        pf = (self.post_prev[self.tgt] > 0) & ~post_e & pre_now\n"
     "        self.w = self.w * torch.exp(ln_pp * (pre_e & post_e).float() + _CL_LN_POST / CL_PAIRS * pf.float())\n"
     "        self.pre_prev = self.pre.clone()\n"
     "        self.post_prev = post.clone()\n\n\n"
     "class Brain:\n"),
    # 2 組み立て（CL の組み立ての後）
    ("            self.g.cl_drive = torch.zeros(self.g.n_ch_all, device=dev)\n",
     "            self.g.cl_drive = torch.zeros(self.g.n_ch_all, device=dev)\n"
     "        # 版69：下から上への道を学ぶシナプスに（ANAPLEX_BU=0 で版68 と同じ）。本数・どこからどこへ は今の道のまま\n"
     "        self.bu_on = _os_cl.environ.get(\"ANAPLEX_BU\", \"1\") != \"0\"\n"
     "        self.bu = {}\n"
     "        if self.bu_on:\n"
     "            gb = torch.Generator(device=dev); gb.manual_seed(seed + 69)\n"
     "            import config as _C_bu\n"
     "            nf = self.n[\"視床\"] // _C_bu.N_MELS                     # 1帯の視床の体数（繊維の段）\n"
     "            ia = torch.arange(self.n[\"聴覚\"], device=dev)\n"
     "            band = (ia.repeat_interleave(nf) // nf) * nf + torch.arange(nf, device=dev).repeat(len(ia))\n"
     "            src_of = {\"視床→聴覚\": (band, ia.repeat_interleave(nf), self.n[\"聴覚\"]),\n"
     "                      \"聴覚→連合\": self.a2c, \"連合→高次\": self.c2h,\n"
     "                      \"高次→前頭前野\": self.p_h2pfc, \"背内側視床→前頭前野\": self.p_md2pfc,\n"
     "                      \"高次→嗅内浅\": self.p_h2ecs, \"前頭前野→嗅内浅\": self.p_pfc2ecs, \"CA1→嗅内深\": self.p_ca12ecd,\n"
     "                      \"CA1→側坐核\": self.p_ca12nac, \"嗅内深→側坐核\": self.p_ecd2nac, \"下丘→上丘\": self.p_ic2sc}\n"
     "            for key, nm in BU_PATHS:\n"
     "                p = src_of[key]\n"
     "                n_tgt = p[2] if isinstance(p[2], int) else len(p[2])\n"
     "                self.bu[key] = _BUSyn(p[0], p[1], n_tgt, self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n"),
    # 3 学び（CL の学びの後）
    ("            self.cl_post_prev = post\n",
     "            self.cl_post_prev = post\n"
     "        if self.bu_on and learn:                                   # 版69：下からのシナプスの学び（三つの因子）\n"
     "            anyr = self.g._any[0]\n"
     "            d_bu = (self.g.third / 2.0).clamp(0.0, 1.0)             # 学びの門（版40、CL と同じ）⚠️ ノブ【C】\n"
     "            for s in self.bu.values():\n"
     "                s.learn(anyr[s.ch0:s.ch0 + s.n_tgt], d_bu)\n"),
    # 4 毎歩の入口（今の本数の和 → シナプスの和）
    ("self._band_sum(self.up[\"視床\"])", "self._sx(\"視床→聴覚\", self.up[\"視床\"], None)"),
    ("self._proj(self.up[\"聴覚\"], self.a2c)", "self._sx(\"聴覚→連合\", self.up[\"聴覚\"], self.a2c)"),
    ("self._proj(self.up[\"連合\"], self.c2h)", "self._sx(\"連合→高次\", self.up[\"連合\"], self.c2h)"),
    ("self._proj(self.up[\"高次\"], self.p_h2pfc)", "self._sx(\"高次→前頭前野\", self.up[\"高次\"], self.p_h2pfc)"),
    ("self._proj(self.md_out, self.p_md2pfc)", "self._sx(\"背内側視床→前頭前野\", self.md_out, self.p_md2pfc)"),
    ("self._proj(self.up[\"高次\"], self.p_h2ecs)", "self._sx(\"高次→嗅内浅\", self.up[\"高次\"], self.p_h2ecs)"),
    ("self._proj(self.up[\"前頭前野\"], self.p_pfc2ecs)", "self._sx(\"前頭前野→嗅内浅\", self.up[\"前頭前野\"], self.p_pfc2ecs)"),
    ("self._proj(self.ca1_out, self.p_ca12ecd)", "self._sx(\"CA1→嗅内深\", self.ca1_out, self.p_ca12ecd)"),
    ("self._proj(self.ca1_out, self.p_ca12nac)", "self._sx(\"CA1→側坐核\", self.ca1_out, self.p_ca12nac)"),
    ("self._proj(self.pred[\"嗅内深\"], self.p_ecd2nac)", "self._sx(\"嗅内深→側坐核\", self.pred[\"嗅内深\"], self.p_ecd2nac)"),
    ("self._proj(ic, self.p_ic2sc)", "self._sx(\"下丘→上丘\", ic, self.p_ic2sc)"),
    # 5 入口の関数
    ("    def _proj(self, v, proj):\n",
     "    def _sx(self, key, v, proj):\n"
     "        \"\"\"版69：下から上への道。シナプスなら重さ × 放出の和、ANAPLEX_BU=0 なら前の本数の和（proj が None なら視床 → 聴覚の帯の和）\"\"\"\n"
     "        if self.bu_on:\n"
     "            return self.bu[key].drive(v)\n"
     "        return self._band_sum(v) if proj is None else self._proj(v, proj)\n\n"
     "    def _proj(self, v, proj):\n"),
]
BB = [("VERSION = 68          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67)",
       "VERSION = 69          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68)"),
      ("#      【測】measure_v67b・c・d【み】10-12「a」「イ」「いいよ」\n",
       "#      【測】measure_v67b・c・d【み】10-12「a」「イ」「いいよ」\n"
       "#  69  下から上への道（皮質・皮質下の11本）を学ぶシナプスに。今の本数のまま1本 ＝ 人の1本（対数正規・短期の増減・三つの因子）。\n"
       "#      gain は入口に残す（ハードの制約のノブ）【み】10-12「全体をシナプスにしよう」「1本は1本」「アでいいよ」\n"),
      ("    if getattr(b, \"cl_on\", False):     # 版67：CL → 皮質 のシナプス\n",
       "    if getattr(b, \"bu_on\", False):     # 版69：下からのシナプス\n"
       "        d[\"bu\"] = {k: {\"w\": s.w, \"t\": s.t, \"stp\": {kk: getattr(s.stp, kk) for kk in (\"row\", \"state\", \"fac\", \"last\", \"deplete\")}}\n"
       "                   for k, s in b.bu.items()}\n"
       "    if getattr(b, \"cl_on\", False):     # 版67：CL → 皮質 のシナプス\n"),
      ("    if getattr(b, \"cl_on\", False) and \"cl\" in d and len(d[\"cl\"][\"w\"]) == len(b.cl_w):",
       "    if getattr(b, \"bu_on\", False) and \"bu\" in d:   # 版69（版68 までの state には無い ⇒ 初めの重さ）\n"
       "        for k, s in b.bu.items():\n"
       "            v = d[\"bu\"].get(k)\n"
       "            if v is not None and len(v[\"w\"]) == len(s.w):\n"
       "                s.w, s.t = v[\"w\"], v[\"t\"]\n"
       "                for kk, vv in v[\"stp\"].items():\n"
       "                    setattr(s.stp, kk, vv)\n"
       "    if getattr(b, \"cl_on\", False) and \"cl\" in d and len(d[\"cl\"][\"w\"]) == len(b.cl_w):")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py")


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
    if babble.VERSION != 68:
        sys.exit(f"⚠️ babble.py が版68 ではない（版 {babble.VERSION}）。先に apply_v68.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版69 を当てた（torch_brain・babble）。下から上への道 11本を学ぶシナプスに（本数はそのまま・gain は入口に残す）")
    print("   Group2 の配線は版68 のまま（版66〜68 の state.pt は読める。下からの重さは初めから）。ANAPLEX_BU=0 で版68 と同じ振る舞い")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "69" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['69'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版69 でも回るようにした")
