"""版35 → 版36 に直す（arkhe の ~/anaplex で  python3 apply_v36_b.py ）

直すもの（どちらも「決めたが、まだ入れていない」ものを入れるだけ。論点台帳 1）
  1 興奮の重みも 0 以上                【み】09-26「2は生物じゃなくて人の形ならそれにして」
                                       【ヒト】繋がりの符号は送り手の型で決まる
  2 表現の的を「自分がこの歩に撃ったか」  【み選】09-26（09-11「3つ入れた」の形。版4 でずれていた）
     そのために表現の発火の計算を、学びの前に移す（学びはこの周の exc・ihn を変えないので、発火の値は同じ）

⚠️ 印が1つでも見つからなければ、何も書かずに止まる
⚠️ 版35 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
⚠️ 脳幹（覚醒系・調整）はまだ無い。出た値はそれを加味して読む（【み】09-26）
"""
import ast
import shutil
import sys

G, B = "torch_group2.py", "babble.py"


def take(s, a, b, name):
    """s の中の a から b の手前までを切り出す。どちらも1回だけ出てくること"""
    if s.count(a) != 1 or s.count(b) != 1:
        sys.exit(f"⚠️ {name} の印が見つからない（または2つ以上ある）。何も書いていない")
    i, j = s.index(a), s.index(b)
    if i >= j:
        sys.exit(f"⚠️ {name} の印の順が違う。何も書いていない")
    return i, j


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


g = open(G, encoding="utf-8").read()
b = open(B, encoding="utf-8").read()
if "VERSION = 35 " not in b:
    sys.exit("⚠️ babble.py が版35 ではない。何も書いていない")

# ── 表現の発火のかたまりを、学びの前へ移す ──
A0 = "        pr = p[self.is_rep]\n        # 表現のニューロンも、シナプスの合計が閾値を越えたら撃つ\n"
A1 = "        inh_spk = (p[self.is_inh_r] > 0.5).float()\n"
i, j = take(g, A0, A1, "表現の発火のかたまり")
block = g[i:j]
if "rep = torch.maximum(rep, (torch.rand(" not in block:
    sys.exit("⚠️ 表現の発火のかたまりの中身が想定と違う。何も書いていない")
g = g[:i] + g[j:]
T0 = "        # 的：誤差と表現は自分の入口、抑制はその領野の入口の平均（ループなしで一度に）\n"
g = once(g, T0,
         "        # 表現の発火は学びの前に出す（版36。的に使うため）。学びはこの周の exc・ihn を変えないので値は同じ\n"
         + block + "\n" + T0, "的の見出し")

# ── 表現の的は自分の発火 ──
t_old = "        tgt = torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], tgt)\n"
g = once(g, t_old, t_old +
         "        # 表現の的は「自分がこの歩に撃ったか」（版36）。樹状突起が自分の細胞体を予測する（2区画の系統）\n"
         "        #   09-11「3つ入れた」の形（そのときは来たもので撃つので、自分の発火 ＝ 下からの入力だった）\n"
         "        #   版4 で合計で撃つにしたとき、的だけ下からの入力のまま残っていた\n"
         "        tgt = tgt.clone()\n"
         "        tgt[self.is_rep] = rep\n", "表現の的")

# ── 興奮の重みも 0 以上 ──
c_old = "            self.w[e] = torch.where(self.e_inh[e], self.w[e].clamp(min=0.0), self.w[e])\n"
g = once(g, c_old,
         "            # 興奮も抑制も 0 以上（版36。【ヒト】繋がりの符号は送り手の型で決まる。前は興奮が負になれた）\n"
         "            self.w[e] = self.w[e].clamp(min=0.0)\n", "重みの切り")

# ── babble の版 ──
b = once(b, "VERSION = 35 ", "VERSION = 36 ", "VERSION")
h_old = "#  35  版33・34 を取り下げて版32 に戻した（根拠の MRS はシナプスの抑制ではなかった。学ばない抑制は根幹を外す）\n"
b = once(b, h_old, h_old +
         "#  36  興奮の重みも 0 以上（【ヒト】符号は送り手の型で決まる。【み】09-26）／\n"
         "#      表現の的を「自分がこの歩に撃ったか」に（09-11 の形。版4 でずれていた。【み選】09-26）\n"
         "#      ⚠️ 脳幹（覚醒系・調整）はまだ無い。値はそれを加味して読む\n", "版の履歴")

for name, s in ((G, g), (B, b)):
    ast.parse(s)
for name, s in ((G, g), (B, b)):
    shutil.copy(name, name + ".v35")
    open(name, "w", encoding="utf-8").write(s)
print("版36 にした（元は torch_group2.py.v35 ／ babble.py.v35 に残した）")
print("回すときは  python3 babble.py 100 --fresh")
