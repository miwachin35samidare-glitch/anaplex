"""版47 → 版48：形B を直す（arkhe の ~/anaplex で  python3 apply_v48.py ）

形B：抑制の体が p ＝ σ(焼いた見込み(自分の履歴) ＋ 興奮 ÷ (1＋抑制)) ＞ 0.5 で撃ち、撃つと見込みが上がって撃ち続けた
  【測】版47 arkhe measure_v47 30：抑制の体の率が 10回ごとに上がり続けた（聴覚 9 → 16% ／ 視床 28 → 59% ／ 上丘 71 → 95%）
  【測】コンテナ simB（運動1つ・錐体 3.52%/歩・200歩）：版47 11.6%・続けて最長 193歩 ／ 見込みを静止に固定 0.9%・最長 2歩
直すもの（版46 の撃ち方を戻す。版46 が壊れたのは相手の数 1,080 のためで、撃ち方ではなかった【測】simB 10-03）
  抑制の体は (届いた興奮 − 上からの戻り) ÷ (1＋抑制) ＞ INH_TH で撃つ。焼いた見込み・p は使わない
  INH_TH ＝ 20.9mV（【ヒト】速い抑制細胞の静止から閾値まで、Wilbers 2023）を版45 の単位に ＝ 1.98（K_MV【C】）
  【ヒト】速い抑制細胞は 1歩の刻みでは履歴にほぼ左右されない（Lee・Dalley 2023：膜の時定数 12ms・順応指数 0.03）
  抑制の体は焼いた網を使わない（【み】10-02「焼かないのはいい」「いいよ」。体の中は凍結のまま）
  【み】10-03「AnaPlex全体の目的とマイルストーン…ズレてなければ入れていいよ」
変えないもの  相手の数 180 ／ 194・重みの初めと弱まり（版47）／ 背内側視床の抑制の割合（版46 の 37% は入れない）
⚠️ コンテナの最小構成では、抑制の体の率は人の覚醒（5.0%）より低い（約 1%）。180 は「抑制がかからない」前提の見積もり
⚠️ 印が1つでも見つからなければ、何も書かずに止まる
⚠️ 版47 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
"""
import ast
import shutil
import sys

A, G, B = "torch_area.py", "torch_group2.py", "babble.py"


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


F = {k: open(k, encoding="utf-8").read() for k in (A, G, B)}
if "VERSION = 47 " not in F[B]:
    sys.exit("⚠️ babble.py が版47 ではない。何も書いていない（先に apply_v47.py）")
if "INH_TH = " in F[A]:
    sys.exit("⚠️ もう版48 になっている。何も書いていない")

a = F[A]
a = once(a, "INH_LTD_TH = 4.21 * K_MV\n",
         "INH_LTD_TH = 4.21 * K_MV\n"
         "# 版48：抑制の体の撃つ閾値。【ヒト】速い抑制細胞の静止から閾値まで 20.9mV（Wilbers 2023）を版45 の単位に（＝ 1.98）\n"
         "INH_TH = 20.9 * K_MV\n", "area の INH_TH")

g = F[G]
g = once(g, "from torch_area import INH_W_SIG, INH_W_MU, INH_W_MEAN, INH_LTD_TH, INH_LTD_F, INH_LTD_WIN, INH_LTD_N   # 版47\n",
         "from torch_area import INH_W_SIG, INH_W_MU, INH_W_MEAN, INH_LTD_TH, INH_LTD_F, INH_LTD_WIN, INH_LTD_N   # 版47\n"
         "from torch_area import INH_TH   # 版48\n", "group2 の import")
g = once(g, "        inh_spk = (p[self.is_inh_r] > 0.5).float()\n",
         "        # 版48：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ。焼いた見込み・p は使わない\n"
         "        #   ⟲ 版47 まで p ＞ 0.5【C】。撃つと見込みが上がって撃ち続けた（形B）\n"
         "        #   【ヒト】速い抑制細胞は 1歩の刻みでは履歴にほぼ左右されない（Lee・Dalley 2023）\n"
         "        d_inh = (exc - exc_mod)[self.is_inh_r] / (1.0 + ihn[self.is_inh_r])\n"
         "        inh_spk = (d_inh > INH_TH).float()\n", "group2 の抑制の撃ち方")

b = F[B]
b = once(b, "VERSION = 47 ", "VERSION = 48 ", "VERSION")
h = "#      抑制の体の興奮性の相手 180（【ヒト】覚醒の率から【C】。海馬・嗅内 194）【み選】B・【み】10-02\n"
b = once(b, h, h +
         "#  48  形B の直し：抑制の体は届いた興奮（上からの戻りを除く）÷ (1＋抑制) ＞ 20.9mV ぶんで撃つ。焼いた見込みを使わない\n"
         "#      （【ヒト】Wilbers 2023・Lee・Dalley 2023。版46 の撃ち方。【み】10-02・10-03）\n", "版の履歴")

out = {A: a, G: g, B: b}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".pre48")
    open(k, "w", encoding="utf-8").write(v)
print("版48 にした（元は *.pre48 に残した）")
print("回すときは  python3 measure_v48.py 100（学びながら。0から）")
