"""版71：シナプスにしてずれた3つのノブを、それぞれの定義で出し直す（版70 の上）
（~/anaplex で。版70 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v71.py
【み】10-12「全部適用してずれた所を直していく」「ウで駄目ならイ」「もっと探してみて」
元【計算】版70 で足した道は「撃った数」で率を合わせたノブの上にあった。シナプスの短期の増減（人の ex→ex の表）で1発の放出が下がる
  1発あたりの放出（続けて撃つときの定常、コンテナで表を回した）：3.3Hz 0.97 ／ 10.4Hz 0.80 ／ 23.2Hz 0.61 ／ 35Hz 0.51 ／ 85Hz 0.29
直すもの（シナプスのとき＝ANAPLEX_BU が 0 でないときだけ。ANAPLEX_BU=0 は版68 のまま）
  1 K_PM 27 → 53（定義：指令の体が【ヒト】話すあいだの 35Hz で撃つ本数）
      放出 0.51 ⇒ 駆動の平均を版68 と合わせる本数 27 ÷ 0.51 ≈ 53【計算】⚠️ 網では measure_v67b で確かめる
  2 M_PC 4 → 2.7（定義：CL が【ヒト】約11Hz で撃つ和。ノブ）
      PPN 23.2Hz・放出 0.61・重さ 対数正規・8本 ⇒ 和 2.7 以上で 1歩 約11%【計算】
  3 視床の閾値 0.3 → 0.07（ノブ。合わせ先は【マカク】の中継の効き）
      【ヒト】視床の中継の閾値・太いシナプスの値は見つからない（10-12 夜、2回探した）
      【マカク】外側膝状体（生体、麻酔）：網膜の1発が中継を撃たせる割合（効き）平均 約46%（Sincich 2007）・約50%（Lee 1985、Weyand の引用）
               1発だけではまず撃たず、30〜40ms 内に2発重なると撃つ ／ 撃ったあと効きが約70% に上がり、そのあと弱まる
      模型：下丘 → 視床（1対1、重さ 対数正規、ex→ex の表）。続く音（耳の繊維の 92% が次の歩も撃つ）で効き 50% になる閾値 ≈ 0.07【計算】（0.3 では 26%）
      ⚠️ 視覚（外側膝状体）の値を聴覚（内側膝状体）に当てた ／ 「2発重なると撃つ」時間の和は模型に無い（1歩ごとの判定）／ 重さの分布・短期の増減は皮質の値のまま【C】
SAME_NET (66, 67, 68, 69, 70)（Group2 の配線は同じ）
"""
import ast, pathlib, re, sys

BRAIN = [
    ("K_PM = 27\n",
     "K_PM = 27\n"
     "# 版71：シナプスのとき（ANAPLEX_BU が 0 でない）の値。短期の増減で1発の放出が下がるぶんを、それぞれの定義で出し直した\n"
     "K_PM_SYN = 53            # 指令の体 35Hz：放出 0.51（35Hz の定常）⇒ 27 ÷ 0.51【計算】⚠️ measure_v67b で確かめる\n"
     "M_PC_SYN = 2.7           # CL 約11Hz：PPN 23.2Hz・放出 0.61・対数正規 8本 ⇒ 和 2.7 以上で 約11%【計算】ノブ\n"
     "FIRE_TH_THAL_SYN = 0.07  # 視床：続く音で中継の効き 50%【マカク】Sincich 2007・Lee 1985 ⇒ 0.07【計算】ノブ（人の値なし）\n"),
    ("                    \"運動前野→運動\": (ar(nm_ * K_PM), ar(nm_).repeat_interleave(K_PM), nm_, \"運動\")}\n",
     "                    \"運動前野→運動\": (ar(nm_ * K_PM_SYN), ar(nm_).repeat_interleave(K_PM_SYN), nm_, \"運動\")}   # 版71\n"),
    ("            cl = ((self.bu[\"PPN→CL\"].drive(ppn) if self.bu_on else ppn[self.cl_ppn].sum(1)) >= M_PC).float()   # 版70\n",
     "            cl = ((self.bu[\"PPN→CL\"].drive(ppn) >= M_PC_SYN) if self.bu_on else (ppn[self.cl_ppn].sum(1) >= M_PC)).float()   # 版70・71\n"),
    ("            if self.cl_on:                                           # PPN → CL（受け手は CL の体。領野の外）\n",
     "            o_t = int(self.off_ch[NAMES.index(\"視床\")])                # 版71：視床の閾値（シナプスのとき）\n"
     "            ft = self.g.fire_th.clone(); ft[o_t:o_t + self.n[\"視床\"]] = FIRE_TH_THAL_SYN\n"
     "            self.g.fire_th = ft\n"
     "            if self.cl_on:                                           # PPN → CL（受け手は CL の体。領野の外）\n"),
]
BB = [("VERSION = 70          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69)",
       "VERSION = 71          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70)"),
      ("#      海馬の短期の増減・学びは人の値なし ⇒ 皮質の値を当てた【C】【み】10-12「全部適用してずれた所を直していくほうがよくないか」\n",
       "#      海馬の短期の増減・学びは人の値なし ⇒ 皮質の値を当てた【C】【み】10-12「全部適用してずれた所を直していくほうがよくないか」\n"
       "#  71  シナプスの短期の増減でずれた3つのノブを定義で出し直し：K_PM 27 → 53（35Hz）・M_PC 4 → 2.7（CL 11Hz）・\n"
       "#      視床の閾値 0.3 → 0.07（中継の効き 50%【マカク】Sincich 2007・Lee 1985）【み】10-12「ウで駄目ならイ」「もっと探してみて」\n")]
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
    if babble.VERSION != 70:
        sys.exit(f"⚠️ babble.py が版70 ではない（版 {babble.VERSION}）。先に apply_v70.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版71 を当てた（torch_brain・babble）。シナプスのとき K_PM 53・M_PC 2.7・視床の閾値 0.07（ANAPLEX_BU=0 は版68 のまま）")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "71" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['71'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版71 でも回るようにした")
