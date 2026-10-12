"""版67（シナプスの版）：上行の覚醒の道（脳幹の橋の被蓋 → 視床の中心外側核 CL → 皮質の表現の体を撃たせる側）。皮質の自発のくじを外す
（~/anaplex で。版66 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v67.py
⟲ 10-12 に渡した前の apply_v67.py（当てていない）は CL → 皮質 が「毎歩 足し込む索引の表」で、シナプスになっていなかった
  【み】10-12「シナプスにしてよ」「そもそもなんでシナプスじゃなくなってる？」「可塑性のシナプスによってTransformerができないリアルタイム学習をするって話」
  【み】10-12 Claude Code「人の形は一つなのにどうして選択になる？」「はい」⇒ 人の形（学ぶシナプス）で組む。値の無い所はノブ
設計：claude/設計案_自発.md（§1 人の形・§6）／ 学びの人の値は 01_現在地.md の 10-12 12時47分〜13時台
人
  【ヒト】切片の錐体は入力なしでは撃たない ／ 半球離断で皮質下から切り離された皮質は起きていても徐波（10人）／ 橋の被蓋の病変で昏睡（Fischer 2016）
  【ヒト】脚橋被蓋核 23.2Hz（JNNP 2010、4人）／ 視床の中心 CL 約11Hz（Wang 2025、意識障害 29人）
  【ヒト】人の CL は matrix 型（calbindin）／【マカク】matrix → 皮質の浅い層 I〜III上（Jones 1998）
  【ヒト】起きている錐体 3.3Hz（Verma 2024）⇒ 本数のノブの合わせ先
  【ヒト iPS】視床 → 皮質 で LTP・LTD が起きる（thalamocortical assembloids 2024）
  【ヒト】大人の新皮質 V層、入力の 10ms 後に撃たせる組 75回：ドーパミンなし 78.7% ／ あり 121.8%（Frontiers Cell Neurosci 2021、18人、Ca 2mM）
  【ヒト】Verhoog 2013：撃ってから入力（受け手が先）で強まる（約121%）。どちらも NMDA が要る。向きはげっ歯類と逆
組み方
  脳幹の橋の被蓋（脚橋被蓋核）N_PPN 体：1歩ごとに 23.2Hz の確率で撃つ【ヒト】
  PPN → CL：網の辺（1体 K_PC 本・重さ 1）。届いた和が M_PC 以上で撃つ ⇒ 約10.4Hz【計算】（人の視床の閾値なし。CL 11Hz に合わせたノブ）
    ⚠️ PPN → CL は学ばない・短期の増減なし（人の値なし。今回は CL → 皮質 をシナプスにする版）
  CL → 皮質の表現の体（聴覚・運動・視覚・連合・高次・前頭前野）：網の辺（シナプス）1体 K_CL 78本
    重さ（1本 ＝ 繋がり1本。1.12mV ＝ 1）：対数正規 中央値 0.41（0.46mV【ヒト】Planert 2025 錐体どうし）・平均 1（1.12mV【ヒト】Hunt 2023）
      ⚠️ 人の CL → 皮質 の1本の大きさは無い。錐体どうしの分布を当てた【C】。中央値と平均は別の研究
    短期の増減：人の ex→ex の表（synapse_model_human.json、Allen）【ヒト 側頭葉の錐体どうしの代用】
    学び（三つの因子）
      入力が先・撃つのが後（同じ歩か、入力が1歩前）：1組ごとに ×exp(((1−d)·ln0.787 ＋ d·ln1.218) ÷ 75)【ヒト V層 10ms】
      撃つのが先・入力が後（受け手が1歩前に撃ち、この歩は撃っていない）：1組ごとに ×1.21^(1/75)【ヒト Verhoog 2013 の向き】
      d ＝ 学びの門（版40、黒質の率 ÷ ふだん）÷ 2 を 0〜1 に。ふだん（門 1）で d 0.5 ⚠️ ノブ【C】（「あり／なし」と門の値の対応は人から出ない）
      ⚠️ ノブ：窓の形（1歩より細かくは分けない）・1組の回数 75 を受け手が先の側にも当てた・アセチルコリン（模型に無い）
    駆動：届いた重さ × 放出 の和を、表現の体の駆動に足す（÷(1＋抑制)）。誤差の的（下からの入力）には入れない
  皮質の表現の体の自発のくじ（0.3Hz）を外す（ほかの領野のくじは残す）
  ANAPLEX_CL=0 で版66 と同じ振る舞い
保存：CL → 皮質 の重さ・短期の増減の状態を state.pt に書く。版66 の state.pt は読める（CL は初めの重さから）⇒ SAME_NET (66,)
まだ入れていない：CL → I層（調整する側）／ 眠りでの率（状態ごとの人の率なし。day.py の眠りを回す前に決める）
"""
import ast, pathlib, re, sys

BRAIN = [("P_INH_CONN = 0.23\n",
          "P_INH_CONN = 0.23\n"
          "# 版67：上行の覚醒の道（脳幹の橋の被蓋 → 視床の中心外側核 CL → 皮質）。CL → 皮質 は学ぶシナプス\n"
          "#   【ヒト】脚橋被蓋核 23.2Hz（JNNP 2010）／ CL 約11Hz（Wang 2025、意識障害）／ 起きている錐体 3.3Hz（Verma 2024）\n"
          "#   【み】10-12「シナプスにしてよ」「人の形は一つなのにどうして選択になる？」\n"
          "CL_CORTEX = (\"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"前頭前野\")\n"
          "N_PPN, N_CL = 512, 4096            # ⚠️ ハードの側のノブ\n"
          "P_PPN = 23.2 * 10.67 / 1000.0      # 【ヒト】\n"
          "K_PC, M_PC = 8, 4                  # PPN → CL：8本受けて4本で撃つ ⇒ 約10.4Hz【計算】（CL 11Hz に合わせたノブ）\n"
          "K_CL = 78                          # CL → 皮質の表現の体：CL だけ・抑制 0 で 3.3Hz【計算】ノブ\n"
          "CL_W_MED, CL_W_MEAN = 0.46 / 1.12, 1.0   # 1本の重さ（1.12mV ＝ 1）【ヒト】Planert 2025 中央値・Hunt 2023 平均を当てた【C】\n"
          "CL_PAIRS = 75.0                    # 【ヒト】1組の回数（Frontiers Cell Neurosci 2021）\n"
          "CL_F_NODA, CL_F_DA, CL_F_POST = 0.787, 1.218, 1.21   # 【ヒト】入力が先：DA なし・あり ／ 受け手が先（Verhoog 2013）\n")]
BRAIN2 = [("        self.bs = BS.Brainstem(self)\n",
           "        self.bs = BS.Brainstem(self)\n"
           "        # 版67：上行の覚醒の道（ANAPLEX_CL=0 で版66 と同じ）\n"
           "        import os as _os_cl, math as _m_cl\n"
           "        self.cl_on = _os_cl.environ.get(\"ANAPLEX_CL\", \"1\") != \"0\"\n"
           "        self.cl_rate = None                                        # 測る道具用（この歩の CL の撃った割合）\n"
           "        if self.cl_on:\n"
           "            gc = torch.Generator(device=dev); gc.manual_seed(seed + 67)\n"
           "            self.cl_gen = gc\n"
           "            self.cl_ppn = torch.randint(0, N_PPN, (N_CL, K_PC), device=dev, generator=gc)   # PPN → CL の辺（重さ 1）\n"
           "            idx = torch.cat([torch.arange(int(self.off_ch[NAMES.index(nm)]), int(self.off_ch[NAMES.index(nm)]) + self.n[nm],\n"
           "                                          device=dev) for nm in CL_CORTEX])\n"
           "            self.cl_ch = idx                                    # 皮質の表現の体（n_ch_all の並び）\n"
           "            nr = len(idx)\n"
           "            self.cl_src = torch.randint(0, N_CL, (nr * K_CL,), device=dev, generator=gc)   # CL → 皮質 の辺（シナプス）\n"
           "            self.cl_recv = torch.arange(nr, device=dev).repeat_interleave(K_CL)\n"
           "            sig = _m_cl.sqrt(2.0 * _m_cl.log(CL_W_MEAN / CL_W_MED))\n"
           "            self.cl_w = torch.exp(_m_cl.log(CL_W_MED) + sig * torch.randn(nr * K_CL, device=dev, generator=gc))\n"
           "            z = torch.zeros(nr * K_CL, dtype=torch.bool, device=dev)\n"
           "            self.cl_stp = _STP_CL(nr * K_CL, z, z, stp_path, dev, gc)   # 【ヒト】ex→ex の表\n"
           "            self.cl_t = 0\n"
           "            self.cl_spk = torch.zeros(N_CL, device=dev)\n"
           "            self.cl_pre_prev = torch.zeros(N_CL, device=dev)\n"
           "            self.cl_post_prev = torch.zeros(nr, device=dev)\n"
           "            sp = self.g.spont_p.clone(); sp[idx] = 0.0             # 皮質の表現の体の自発のくじを外す\n"
           "            self.g.spont_p = sp\n"
           "            self.g.spont_p_pass = 1.0 - (1.0 - sp) ** (1.0 / self.g.recur)\n"
           "            self.g.cl_drive = torch.zeros(self.g.n_ch_all, device=dev)\n"),
          ("        outs = self.g.step(x, e, learn)\n",
           "        if self.cl_on:                                             # 版67：脳幹 → CL → 皮質（シナプス）\n"
           "            ppn = (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()\n"
           "            cl = (ppn[self.cl_ppn].sum(1) >= M_PC).float()\n"
           "            self.cl_spk = cl\n"
           "            self.cl_rate = cl.mean()\n"
           "            self.g.cl_drive.zero_()\n"
           "            ce = torch.nonzero(cl[self.cl_src] > 0).squeeze(1)\n"
           "            if len(ce):\n"
           "                amp = self.cl_w[ce] * self.cl_stp.release_at(ce, self.cl_t)\n"
           "                self.g.cl_drive[self.cl_ch] = torch.zeros(len(self.cl_ch), device=self.dev).index_add_(0, self.cl_recv[ce], amp)\n"
           "            self.cl_t += 1\n"
           "        outs = self.g.step(x, e, learn)\n"
           "        if self.cl_on and learn:                                   # 版67：CL → 皮質 の学び（三つの因子）\n"
           "            post = self.g._any[0][self.cl_ch]                      # この歩のどこかで撃ったか\n"
           "            pre_e = (self.cl_spk + self.cl_pre_prev)[self.cl_src] > 0\n"
           "            post_e = post[self.cl_recv] > 0\n"
           "            d = (self.g.third / 2.0).clamp(0.0, 1.0)                # 学びの門（版40）⚠️ ノブ【C】\n"
           "            ln_pp = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS\n"
           "            pf = (self.cl_post_prev[self.cl_recv] > 0) & ~post_e & (self.cl_spk[self.cl_src] > 0)\n"
           "            self.cl_w = self.cl_w * torch.exp(ln_pp * (pre_e & post_e).float() + _CL_LN_POST / CL_PAIRS * pf.float())\n"
           "            self.cl_pre_prev = self.cl_spk\n"
           "            self.cl_post_prev = post\n")]
BRAIN3 = [("class Brain:\n",
           "from torch_area import STP as _STP_CL   # 版67\n"
           "_CL_LN_NODA, _CL_LN_DA, _CL_LN_POST = (__import__(\"math\").log(v) for v in (CL_F_NODA, CL_F_DA, CL_F_POST))   # 版67\n\n\n"
           "class Brain:\n")]
G2 = [("        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])\n",
       "        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])\n"
       "        if getattr(self, \"cl_drive\", None) is not None:      # 版67：CL からの駆動（シナプス。誤差の的には入れない）\n"
       "            drive = drive + self.cl_drive / (1.0 + ihn[self.rep_i])\n")]
BB = [("VERSION = 66          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (65,)",
       "VERSION = 67          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66,)"),
      ("#  66  指令の写しが聴覚の抑制の体を撃たせる（配線は版65 のまま）【マーモセット】Eliades & Wang 2003【み】10-12「まぁアでいいよ」「人じゃなくて霊長類」\n",
       "#  66  指令の写しが聴覚の抑制の体を撃たせる（配線は版65 のまま）【マーモセット】Eliades & Wang 2003【み】10-12「まぁアでいいよ」「人じゃなくて霊長類」\n"
       "#  67  上行の覚醒の道（脳幹の橋の被蓋 → CL → 皮質の表現の体）。CL → 皮質 は学ぶシナプス（重み・短期の増減・三つの因子）。\n"
       "#      皮質の自発のくじを外す【ヒト】設計案_自発・Frontiers Cell Neurosci 2021・Verhoog 2013【み】10-12「シナプスにしてよ」「人の形は一つ」\n"),
      ("    if g.stp is not None:\n        d[\"stp\"] = {k: getattr(g.stp, k) for k in (\"row\", \"state\", \"fac\", \"last\", \"deplete\")}\n",
       "    if g.stp is not None:\n        d[\"stp\"] = {k: getattr(g.stp, k) for k in (\"row\", \"state\", \"fac\", \"last\", \"deplete\")}\n"
       "    if getattr(b, \"cl_on\", False):     # 版67：CL → 皮質 のシナプス\n"
       "        d[\"cl\"] = {\"w\": b.cl_w, \"src\": b.cl_src, \"t\": b.cl_t,\n"
       "                   \"stp\": {k: getattr(b.cl_stp, k) for k in (\"row\", \"state\", \"fac\", \"last\", \"deplete\")}}\n"),
      ("    if g.stp is not None and \"stp\" in d:\n        for k, vv in d[\"stp\"].items():\n            setattr(g.stp, k, vv)\n",
       "    if g.stp is not None and \"stp\" in d:\n        for k, vv in d[\"stp\"].items():\n            setattr(g.stp, k, vv)\n"
       "    if getattr(b, \"cl_on\", False) and \"cl\" in d and len(d[\"cl\"][\"w\"]) == len(b.cl_w):   # 版67（版66 の state には無い ⇒ 初めの重さ）\n"
       "        b.cl_w, b.cl_src, b.cl_t = d[\"cl\"][\"w\"], d[\"cl\"][\"src\"], d[\"cl\"][\"t\"]\n"
       "        for k, vv in d[\"cl\"][\"stp\"].items():\n"
       "            setattr(b.cl_stp, k, vv)\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py")


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
    if babble.VERSION != 66:
        sys.exit(f"⚠️ babble.py が版66 ではない（版 {babble.VERSION}）。先に apply_v66.py")
    out = [edit("torch_brain.py", BRAIN + BRAIN2 + BRAIN3), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版67 を当てた（torch_brain・torch_group2・babble）。脳幹 → CL → 皮質の表現の体（CL → 皮質 は学ぶシナプス）。皮質の自発のくじを外した")
    print("   Group2 の配線は版66 のまま（版66 の state.pt は読める。CL は初めの重さから）。ANAPLEX_CL=0 で版66 と同じ振る舞い")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "67" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['67'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版67 でも回るようにした")
