"""版68：運動の表現の撃ちすぎを直す（K_CL 78 → 40 ／ K_PM 115 → 27）
（~/anaplex で。版67 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v68.py
【み】10-12「a」（的はそのまま：指令の体 35Hz・指令の無い体 3.3Hz）「イ」（内訳を測ってから一緒に直す）「いいよ」
なぜ
  【測】版67 の健診：運動の表現 31〜33% ⇒ 口が型の2倍を越えて撃つと黙る ⇒ 言えた 0/43
  【測】measure_v67c：指令の無い体を撃たせていたのは CL（黙り・聞く で撃った体の 85% が CL だけで閾値を越えた）
  K_CL：定義「CL だけ・抑制 0 で【ヒト】3.3Hz」（Verma 2024）
    🔴 78 は重さ 1 のまま数えた本数（Claude の数え違い）。版67 で入れた対数正規の重さ（中央 0.41・平均 1）で数え直すと 40【計算】
    【測】measure_v67d（網の中、K_PM 20）：K_CL 40 で 運動の指令の無い体（黙り・聞く）3.5%
  K_PM：定義「指令の体が【ヒト】話すあいだの 35Hz で撃つ本数」
    115 は版48 の抑制（ihn 約2）の上で出した値。版64 で抑制が人の形（ihn 約0.1）になり前提が崩れていた
    【測】measure_v67b（K_CL 40）：K_PM 20 で 17.5%・30 で 47.1% ⇒ 37%（35Hz）は約27【計算：直線で】。言えた 30 で 7/7
    ⚠️ 35Hz の出どころ Stavisky 2019 は背側（手の領域）。腹側の発話運動野の集団の率は見つからない（10-12 夜、5本を読んだ）
       【ヒト】Nat Commun 2026 補足 図4 の例の体：合図の前後 6〜21Hz・発話の山 50〜75Hz（選んだ例・麻痺の人）
変えるもの：torch_brain.py の K_CL・K_PM の2つの値と注 ／ babble.py の VERSION・SAME_NET・履歴
保存：Group2 の配線は変わらない ⇒ 版66・67 の state.pt は読める。CL → 皮質 の本数が変わるので CL の重さは初めから（版67 の読み込みの長さの確かめで落ちる）
"""
import ast, pathlib, re, sys

BRAIN = [("K_CL = 78                          # CL → 皮質の表現の体：CL だけ・抑制 0 で 3.3Hz【計算】ノブ\n",
          "K_CL = 40                          # CL → 皮質の表現の体：CL だけ・抑制 0 で 3.3Hz【計算】ノブ\n"
          "#   版68：対数正規の重さで数え直した（78 は重さ 1 のまま数えた Claude の数え違い）【測】measure_v67d で運動の指令の無い体 3.5%\n"),
         ("#   K_PM ⚠️ ノブ【C】：版48 の指令の歩の ihn 約 2 で、指令の体が【ヒト】話すあいだの 35Hz で撃つ本数\n"
          "K_PM = 115\n",
          "#   K_PM ⚠️ ノブ【C】：指令の体が【ヒト】話すあいだの 35Hz で撃つ本数\n"
          "#   ⟲ 版49〜67 は 115（版48 の指令の歩の ihn 約2 の上の値。版64 で抑制が人の形になり前提が崩れた）\n"
          "#   版68：今の抑制と CL（K_CL 40）の上で出し直した【測】measure_v67b：20 で 17.5%・30 で 47.1% ⇒ 37% は約27【計算】\n"
          "K_PM = 27\n")]
BB = [("VERSION = 67          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66,)",
       "VERSION = 68          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67)"),
      ("#      皮質の自発のくじを外す【ヒト】設計案_自発・Frontiers Cell Neurosci 2021・Verhoog 2013【み】10-12「シナプスにしてよ」「人の形は一つ」\n",
       "#      皮質の自発のくじを外す【ヒト】設計案_自発・Frontiers Cell Neurosci 2021・Verhoog 2013【み】10-12「シナプスにしてよ」「人の形は一つ」\n"
       "#  68  K_CL 78 → 40（対数正規の重さで数え直し）・K_PM 115 → 27（今の抑制と CL の上で 35Hz）。版67 は運動の表現が撃ちすぎて言えた 0/43\n"
       "#      【測】measure_v67b・c・d【み】10-12「a」「イ」「いいよ」\n")]
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
    if babble.VERSION != 67:
        sys.exit(f"⚠️ babble.py が版67 ではない（版 {babble.VERSION}）。先に apply_v67.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版68 を当てた（torch_brain・babble）。K_CL 78 → 40 ／ K_PM 115 → 27")
    print("   Group2 の配線は版67 のまま（版66・67 の state.pt は読める。CL → 皮質 の重さは初めから）")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "68" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['68'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版68 でも回るようにした")
