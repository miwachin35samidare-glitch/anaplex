"""版67：上行の覚醒の道を足す（脳幹の橋の被蓋 → 視床の中心外側核 CL → 皮質の表現の体を撃たせる側）。皮質の自発のくじを外す
（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v67.py      （版66 の上に当てる）
【み】10-12 12時02分「ウ」「a1で数値見て必要ならa2へ」「4-3は人の値でいい」「ハード…動くラインでいいよ」
設計：claude/設計案_自発.md（§1 人の形・§6 人で調べ直した分）
人
  【ヒト】切片の錐体は入力なしでは撃たない ／ 半球離断で皮質下から切り離された皮質は起きていても徐波（10人）／ 橋の被蓋の病変で昏睡（Fischer 2016）
  【ヒト】脚橋被蓋核 23.2Hz（JNNP 2010、4人）／ 視床の中心 CL 約11Hz（Wang 2025、意識障害 29人）
  【ヒト】人の CL は matrix 型（calbindin）／【マカク】matrix → 皮質の浅い層 I〜III上（Jones 1998）
  【ヒト】起きている錐体 3.3Hz（Verma 2024。模型の RATE_HUMAN と同じ値）⇒ 本数のノブの合わせ先
組み方（この版）
  脳幹の橋の被蓋（脚橋被蓋核）N_PPN 体：1歩ごとに 23.2Hz の確率で撃つ【ヒト】
  視床の中心 CL N_CL 体：PPN から K_PC 本受け、M_PC 本以上届いたら撃つ ⇒ 約10.4Hz（【ヒト】11Hz に合わせたノブ。人の視床の閾値は無い）【計算】
  皮質の表現の体（聴覚・運動・視覚・連合・高次・前頭前野）：CL から K_CL 本受け、届いた数を下からの駆動に足す（撃たせる側＝ II・III層の分）
    K_CL 78：入力が CL だけ・抑制 0 のとき 1体 3.3Hz（【ヒト】Verma）になる本数【計算 二項】。1本 ＝ 繋がり1本の重さ 1【C】
    誤差の的（下からの入力）には入れない（予測する中身ではない）
  皮質の表現の体の自発のくじ（0.3Hz）を外す（ほかの領野のくじは残す）
  ANAPLEX_CL=0 で版66 と同じ振る舞い
まだ入れていない（設計案 4-1 ウ の残り）：CL → I層（調整する側）。重さの決め方を【み】に聞いている
⚠️ 体数 N_PPN 512・N_CL 4096 はハードの側のノブ（皮質の体どうしの入力が重なりすぎないように多めに）
⚠️ 眠り（N3・REM）でも同じ率で撃つ（状態ごとの人の率なし。day.py の眠りを回す前に決める）
⚠️ 配線は版66 のまま（学ぶ辺ではなく、毎歩 足し込む入力）⇒ SAME_NET (66,)
"""
import ast, pathlib, re, sys

BRAIN = [("P_INH_CONN = 0.23\n",
          "P_INH_CONN = 0.23\n"
          "# 版67：上行の覚醒の道（脳幹の橋の被蓋 → 視床の中心外側核 CL → 皮質）【み】10-12「ウ」「a1で数値見て」「人の値でいい」\n"
          "#   【ヒト】脚橋被蓋核 23.2Hz（JNNP 2010）／ CL 約11Hz（Wang 2025、意識障害）／ 起きている錐体 3.3Hz（Verma 2024）\n"
          "CL_CORTEX = (\"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"前頭前野\")\n"
          "N_PPN, N_CL = 512, 4096            # ⚠️ ハードの側のノブ\n"
          "P_PPN = 23.2 * 10.67 / 1000.0      # 【ヒト】\n"
          "K_PC, M_PC = 8, 4                  # PPN → CL：8本受けて4本で撃つ ⇒ 約10.4Hz【計算】（人の視床の閾値なし。CL 11Hz に合わせたノブ）\n"
          "K_CL = 78                          # CL → 皮質の表現の体（撃たせる側）：CL だけ・抑制 0 で 3.3Hz【計算】ノブ\n"
          "W_CL = 1.0                         # 1本 ＝ 繋がり1本【C】\n")]
BRAIN2 = [("        self.bs = BS.Brainstem(self)\n",
           "        self.bs = BS.Brainstem(self)\n"
           "        # 版67：上行の覚醒の道（ANAPLEX_CL=0 で版66 と同じ）\n"
           "        import os as _os_cl\n"
           "        self.cl_on = _os_cl.environ.get(\"ANAPLEX_CL\", \"1\") != \"0\"\n"
           "        if self.cl_on:\n"
           "            gc = torch.Generator(device=dev); gc.manual_seed(seed + 67)\n"
           "            self.cl_ppn = torch.randint(0, N_PPN, (N_CL, K_PC), device=dev, generator=gc)\n"
           "            idx = torch.cat([torch.arange(int(self.off_ch[NAMES.index(nm)]), int(self.off_ch[NAMES.index(nm)]) + self.n[nm],\n"
           "                                          device=dev) for nm in CL_CORTEX])\n"
           "            self.cl_ch = idx                                    # 皮質の表現の体（n_ch_all の並び）\n"
           "            self.cl_src = torch.randint(0, N_CL, (len(idx), K_CL), device=dev, generator=gc)\n"
           "            self.cl_gen = gc\n"
           "            sp = self.g.spont_p.clone(); sp[idx] = 0.0             # 皮質の表現の体の自発のくじを外す\n"
           "            self.g.spont_p = sp\n"
           "            self.g.spont_p_pass = 1.0 - (1.0 - sp) ** (1.0 / self.g.recur)\n"
           "            self.g.cl_drive = torch.zeros(self.g.n_ch_all, device=dev)\n"
           "        self.cl_rate = None                                        # 測る道具用（この歩の CL の撃った割合）\n"),
          ("        outs = self.g.step(x, e, learn)\n",
           "        if self.cl_on:                                             # 版67：脳幹 → CL → 皮質（毎歩 新しく引く）\n"
           "            ppn = (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()\n"
           "            cl = (ppn[self.cl_ppn].sum(1) >= M_PC).float()\n"
           "            self.cl_rate = cl.mean()\n"
           "            self.g.cl_drive.zero_()\n"
           "            self.g.cl_drive[self.cl_ch] = W_CL * cl[self.cl_src].sum(1)\n"
           "        outs = self.g.step(x, e, learn)\n")]
G2 = [("        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])\n",
       "        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])\n"
       "        if getattr(self, \"cl_drive\", None) is not None:      # 版67：CL からの駆動（誤差の的には入れない）\n"
       "            drive = drive + self.cl_drive / (1.0 + ihn[self.rep_i])\n")]
BB = [("VERSION = 66          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (65,)",
       "VERSION = 67          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66,)"),
      ("#  66  指令の写しが聴覚の抑制の体を撃たせる（配線は版65 のまま）【マーモセット】Eliades & Wang 2003【み】10-12「まぁアでいいよ」「人じゃなくて霊長類」\n",
       "#  66  指令の写しが聴覚の抑制の体を撃たせる（配線は版65 のまま）【マーモセット】Eliades & Wang 2003【み】10-12「まぁアでいいよ」「人じゃなくて霊長類」\n"
       "#  67  上行の覚醒の道（脳幹の橋の被蓋 → CL → 皮質の表現の体を撃たせる側）・皮質の自発のくじを外す【ヒト】設計案_自発【み】10-12\n")]
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
    out = [edit("torch_brain.py", BRAIN + BRAIN2), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版67 を当てた（torch_brain・torch_group2・babble）。脳幹 → CL → 皮質の表現の体。皮質の自発のくじを外した")
    print("   配線は版66 のまま（版66 の state.pt は読める）。ANAPLEX_CL=0 で版66 と同じ振る舞い")
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
