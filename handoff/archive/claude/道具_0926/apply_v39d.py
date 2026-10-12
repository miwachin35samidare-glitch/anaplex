"""版39c → 版39d：眠くなる早さから疲れを外す（arkhe の ~/anaplex で  python3 apply_v39d.py ）

【ヒト】Elmenhorst 2017 PNAS（15人、52時間の断眠 → 14時間の回復の眠り、PET）
  断眠で A1 アデノシン受容体の空きが増え、回復の眠りで 10〜14% 下がって元に戻る
  眠りの圧は起きている時間とともに飽和する指数でたまり、眠ると指数で下がる（二過程モデル）と合う
  ⚠️「人でアデノシンそのものが起きている時間で増えることは、まだ示されていない」（本文）。
     てんかんの患者の微小透析では有意に増えなかった
  ⇒ 人で分かっているのは「眠くなる早さは起きている時間」まで。活動や学んだ量で早く眠くなる人の測定は無い
【ヒト】学んだ場所で、その夜の徐波が局所に強まる（Huber 2004）⇒ 学んだ量が効くのは深さ（局所）
⇒ 版39c の「(1 ＋ 疲れ) で早くたまる」は Claude の形で、人の構造ではなかった。眠りの圧は起きている時間だけでたまる
   疲れは測って表示だけする（深さは、いまの下方調整が目標より上の分だけ強く下げる形のまま）
"""
import ast
import shutil
import sys

s = open("brainstem.py", encoding="utf-8").read()
old = "        self.pressure += n * (1.0 + self.fatigue) / self.wake_steps()\n"
if s.count(old) != 1:
    sys.exit(f"⚠️ 眠りの圧の行が {s.count(old)} 回。何も書いていない（版39c になっているか確かめる）")
s = s.replace(old,
              "        # 版39d：眠りの圧は起きている時間だけでたまる（【ヒト】Elmenhorst 2017。疲れで早くなる人の測定は無い）\n"
              "        #   疲れ（学んだ量）が効くのは深さ（【ヒト】Huber 2004）。いまの下方調整が目標より上の分だけ強く下げる\n"
              "        self.pressure += n / self.wake_steps()\n")
ast.parse(s)
shutil.copy("brainstem.py", "brainstem.py.v39c")
open("brainstem.py", "w", encoding="utf-8").write(s)
print("版39d にした（元は brainstem.py.v39c に残した）。疲れは day.py の表示に残る")
