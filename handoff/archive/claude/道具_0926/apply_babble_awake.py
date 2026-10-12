"""babble.py の起きているあいだの恒常性を、day.py と揃える（arkhe の ~/anaplex で  python3 apply_babble_awake.py ）

いま  day.py は起きているあいだ下方調整を止める（b.g.scaling_on = False）
      babble.py は止めていない ⇒ 同じ「起きて喃語」なのに、回す道具で網の動きが違う
      torch_group2.py の注記も「起きているあいだは止める（babble 側で切る）」
人    【ヒト】TMS の応答は起きている時間とともに大きくなり、眠ると元に戻る（sleep.py の注記）
⇒ babble.py でも起きているあいだは止める。網の作りは変えないので、版の番号は上げない（state はそのまま使える）

⚠️ 印が見つからなければ、何も書かずに止まる
"""
import ast
import shutil
import sys

P = "babble.py"
s = open(P, encoding="utf-8").read()
old = "    rng = np.random.default_rng(0)\n    if fake:\n        inv = make_inventory("
if s.count(old) != 1:
    sys.exit(f"⚠️ 印が {s.count(old)} 回。何も書いていない")
if "b.g.scaling_on = False" in s:
    sys.exit("⚠️ もう入っている。何も書いていない")
new = ("    b.g.scaling_on = False      # 起きているあいだは下方調整を止める（day.py と同じ。【ヒト】眠ると戻る）\n"
       "    #   ⟲ 09-26 まで babble.py だけ止めていなかった（版30 までの babble の結果は、起きているあいだも下げていた）\n"
       + old)
s = s.replace(old, new)
ast.parse(s)
shutil.copy(P, P + ".before_awake")
open(P, "w", encoding="utf-8").write(s)
print("babble.py：起きているあいだは下方調整を止める（元は babble.py.before_awake）")
