"""版74：下からのシナプスの学びに、ドーパミンの門を全部の窓に掛ける（版73 の上）
（~/anaplex で。版73 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v74.py
【み】10-13「眠りで抑えるってレベルの増幅じゃないでしょ 起きてる時にどうか調べて」「一旦暴走してるとこに入れよう」
元【ヒト】起きているあいだ、撃つだけでは強まらない（学びには門がある）
  Ziemann 2004：同じ回数の親指の動きでも、ゆっくりやって学ばないと、あとの PAS は変わらない
  Louth 2021（Frontiers Cell Neurosci）：大人の人の L5 錐体、入力が先 10ms を 75組 ⇒ ドーパミンなし 78.7%・あり 121.8%
  薬（8人、プラセボ対照）：ハロペリドール・プラゾシン・ビペリデンで PAS の強まりが下がる
  ⇒ 版73 は門が「入力が先（皮質）」にしか無く、受け手が先の窓（Verhoog）と海馬の窓（Testa-Silva）は毎歩そのまま学んでいた
直すもの（版69〜73 の下からのシナプスだけ。CL → 皮質（版67）はまだ前の形）
  開き g ＝ (2d − 1) を 0〜1 に（d ＝ third／2 ＝ 学びの門。third は黒質のドーパミンの率 ÷ ふだんの率）
    ふだんのドーパミン（third 1、d 0.5）で g ＝ 0：変わらない ／ ふだんの 2倍（d 1）で g ＝ 1：人の窓のまま
    【C】入力が先（皮質）の表と同じ当て方：d 1 ⇔ ドーパミンを足した 121.8%、d 0.5 ⇔ 78.7% と 121.8% の間（×0.98、ほぼ変わらない）
    【測】コンテナの喃語 3回（368歩）：third 中央 1.03（1〜99%：0.81〜1.23）⇒ g は多くの歩で 0、開いても 0.2 ほど
  皮質の受け手が先（Verhoog の窓）× g
  海馬の窓（入力が先 +72%・受け手が先 +58%／−55%）× g【C】人の海馬のドーパミンの値なし。皮質の門を当てた
  皮質の入力が先は今までどおり（門 d で 0.787〜1.218。もともと門がある）
⚠️ 受け手が先の強まり（Verhoog 2013、L2/3 の切片）はドーパミンを足さずに出ている。層の違いか切片の条件かは分からない【C】
SAME_NET (66, 67, 68, 69, 70, 71, 72, 73)
"""
import ast, pathlib, re, sys

BRAIN = [
    ("WIN_MAXK = 12\n",
     "WIN_MAXK = 12\n"
     "# 版74：門の開き g ＝ (2d − 1) を 0〜1 に。ふだんのドーパミン（d 0.5）で 0、2倍（d 1）で 1【C】\n"
     "#   【ヒト】撃つだけでは強まらない（Ziemann 2004 の対照・Louth 2021・PAS の薬）。受け手が先・海馬の窓に掛ける\n"),
    ("        if self.hippo:                                              # 入力が先【ヒト】Testa-Silva\n"
     "            ln_pp = torch.full_like(d, LN_PRE_HIP) if torch.is_tensor(d) else LN_PRE_HIP\n",
     "        g_on = (2.0 * torch.as_tensor(d, dtype=self.w.dtype, device=self.w.device) - 1.0).clamp(0.0, 1.0)   # 版74：門の開き\n"
     "        if self.hippo:                                              # 入力が先【ヒト】Testa-Silva（版74：× 門）\n"
     "            ln_pp = LN_PRE_HIP * g_on\n"),
    ("        pf = self._tab[self.hippo][k] * (pre_now & ~post_e).float()   # 受け手が先\n",
     "        pf = self._tab[self.hippo][k] * (pre_now & ~post_e).float() * g_on   # 受け手が先（版74：× 門）\n"),
]
BB = [("VERSION = 73          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72)",
       "VERSION = 74          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73)"),
      ("#      ⟲ 版72 の皮質の弱まる窓（海馬の値を当てた誤り）【み】10-12「どうして根幹に関わるシナプスの学びを外そうとする？」\n",
       "#      ⟲ 版72 の皮質の弱まる窓（海馬の値を当てた誤り）【み】10-12「どうして根幹に関わるシナプスの学びを外そうとする？」\n"
       "#  74  下からのシナプスの受け手が先の窓・海馬の窓にもドーパミンの門（開き 2d − 1）【ヒト】Ziemann 2004・Louth 2021【C】当て方。\n"
       "#      【み】10-13「起きてる時にどうか調べて」「一旦暴走してるとこに入れよう」\n")]
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
    if babble.VERSION != 73:
        sys.exit(f"⚠️ babble.py が版73 ではない（版 {babble.VERSION}）。先に apply_v73.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版74 を当てた（torch_brain・babble）。下からのシナプスの受け手が先の窓・海馬の窓にもドーパミンの門")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "74" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['74'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版74 でも回るようにした")
