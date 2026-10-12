"""版36 → 版37 に直す（arkhe の ~/anaplex で  python3 apply_v37.py ）

直すもの：上の段から下へ送るものを、上の表現の「予測 p」から「発火」にする（論点3）
  人      別の領野へ届くのは、送る細胞が撃ったスパイクだけ。上からの入力は樹状突起に来て撃たせない側
  決め    09-11「段の出力は表現のニューロンの発火（予測を出力に使わない）」（予測は履歴で決まり、入力が違っても同じものが出た）
          上→下 だけ p のまま残っていた。09-26 午後の確かめ直しで「発火を送るのが人の形」と戻してあった
  変えないもの  下の樹状突起に入る場所（調整する入力。撃たせない）／ CA3・CA1 の完成（自分の p を足す。別の配線）

⚠️ 印が1つでも見つからなければ、何も書かずに止まる
⚠️ 版36 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
"""
import ast
import shutil
import sys

T, B = "torch_brain.py", "babble.py"


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


t = open(T, encoding="utf-8").read()
b = open(B, encoding="utf-8").read()
if "VERSION = 36 " not in b:
    sys.exit("⚠️ babble.py が版36 ではない。何も書いていない（先に apply_v36_b.py）")

t = once(t,
         '        self.pred["連合"] = self.areas["連合"].pred_rep\n'
         '        self.pred["高次"] = self.areas["高次"].pred_rep\n',
         '        # 上から下へ送るのは、上の表現の発火（版37）。【ヒト】別の領野へ届くのは撃ったスパイクだけ\n'
         '        #   ⟲ 版36 までは上の表現の予測 p を 0.5 で切って送っていた\n'
         '        self.pred["連合"] = cut("連合", rep)\n'
         '        self.pred["高次"] = cut("高次", rep)\n',
         "上から送るもの")
# 冒頭の注記（あれば直す。無くても止めない）
t = t.replace('  上 → 下   表現のニューロンの予測（調整する入力。樹状突起へ。撃たせない）\n',
              '  上 → 下   表現のニューロンの発火（調整する入力。樹状突起へ。撃たせない）（版37。前は予測 p）\n')

b = once(b, "VERSION = 36 ", "VERSION = 37 ", "VERSION")
h_old = "#      ⚠️ 脳幹（覚醒系・調整）はまだ無い。値はそれを加味して読む\n"
b = once(b, h_old, h_old +
         "#  37  上の段から下へ送るものを、上の表現の予測 p から発火に（【ヒト】別の領野へ届くのは撃ったスパイクだけ。\n"
         "#      09-11「段の出力は表現の発火」の形。上→下 だけ p のまま残っていた）\n", "版の履歴")

for s in (t, b):
    ast.parse(s)
shutil.copy(T, T + ".v36")
shutil.copy(B, B + ".v36")
open(T, "w", encoding="utf-8").write(t)
open(B, "w", encoding="utf-8").write(b)
print("版37 にした（元は torch_brain.py.v36 ／ babble.py.v36 に残した）")
print("回すときは  python3 babble.py 100 --fresh")
