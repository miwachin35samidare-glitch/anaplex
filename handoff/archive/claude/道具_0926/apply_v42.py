"""版41 → 版42：驚きで、その歩の学ぶ速さを上げる（青斑核）を戻す（arkhe の ~/anaplex で  python3 apply_v42.py ）

版40 で Claude が外した機構を戻す（【み】09-26「脳幹にある機能なら驚きは入れて」で版38 に入れたもの）
  【ヒト】驚くと新しい情報で考えを大きく改める（学ぶ速さが上がる。瞳孔、Nassar 2012、行動の段）
  【マカク】青斑核は予期しない合図に 約90ms で短く撃つ ⇒ 8歩前の誤差で、その歩の学ぶ速さを上げる
  学ぶ速さ ＝ 1 ＋ LC_K ×（8歩前の皮質の誤差 ÷ いつも − 1 の正の分）。上限なし（版38〜39 と同じ）
  ⚠️ 入口を皮質の誤差にするのと LC_K ＝ 1 は【C】のノブ（版38 のまま）
【測】09-28 版41 の網で 学ぶ速さへの驚きを 切る（base）34.4% ／ 戻す（v39）12.9%（喃語100回・自分の誤差）
学びの門（中脳・版40）と覚醒（感度・眠気の覆い・版41）はそのまま並べる。門の膨らみ（ほぼずっと開く）はまだ直していない
印が1つでも見つからなければ何も書かずに止まる。回すときは  python3 measure_v42.py full 100 ／ k0 100
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


def maybe(s, old, new):                    # 注釈だけ。見つからなくても止めない
    return s.replace(old, new) if s.count(old) == 1 else s


NAMES = ("torch_group2.py", "brainstem.py", "babble.py")
F = {k: open(k, encoding="utf-8").read() for k in NAMES}
if "VERSION = 41 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版41 ではない。何も書いていない（先に apply_v41.py）")
if "LC_K" in F["brainstem.py"]:
    sys.exit("⚠️ もう版42 になっている。何も書いていない")

# ── brainstem.py ──
s = F["brainstem.py"]
k = "DA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）\n"
s = once(s, k, k +
         "# 驚きで学ぶ速さを上げる（青斑核。版38 の形を版42 で戻した）【ヒト】Nassar 2012 ⚠️ 量は【C】のノブ\n"
         "LC_K = 1.0\n", "脳幹の定数")
s = once(s,
         "            self.g.eta_gain = torch.ones((), device=self.dev)\n"
         "            # 覚醒（版41）",
         "            # 驚きで、その歩の学ぶ速さを上げる（青斑核・8歩遅れ。版42 で戻した。版40 で外したのは【C】の誤り）\n"
         "            self.g.eta_gain = 1.0 + LC_K * self.surprise\n"
         "            # 覚醒（版41）", "起きているあいだの学ぶ速さ")
s = maybe(s, "学ぶ速さに直接は掛けない（版40）", "学ぶ速さにも掛ける（版42 で戻した）")

# ── torch_group2.py（注釈だけ）──
g = maybe(F["torch_group2.py"], "# eta_gain は版40 から 1（驚きは覚醒の側へ）",
          "# eta_gain：驚きで上がる（青斑核。版42 で戻した）")

# ── babble.py ──
b = F["babble.py"]
b = once(b, "VERSION = 41 ", "VERSION = 42 ", "VERSION")
k = "#      表現の閾値を最大 21% 下げる（【ラット】3.4mV ÷【ヒト】16mV）／ 眠るのは 眠りの圧 ≥ A のとき\n"
b = once(b, k, k +
         "#  42  驚きでその歩の学ぶ速さを上げる（青斑核、版38〜39 の形）を戻した。版40 で外したのは誤り\n", "版の履歴")

out = {"torch_group2.py": g, "brainstem.py": s, "babble.py": b}
for x in out.values():
    ast.parse(x)
for k, x in out.items():
    shutil.copy(k, k + ".v41")
    open(k, "w", encoding="utf-8").write(x)
print("版42 にした（元は *.v41 に残した）")
print("回すときは  python3 measure_v42.py full 100  →  python3 measure_v42.py k0 100")
