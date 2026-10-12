"""版37 → 版38：脳幹（覚醒系）を入れる（arkhe の ~/anaplex で  python3 apply_v38.py ）

同じ場所に brainstem.py と sleep.py（新しい版）を置いてから回す
  brainstem.py   新しいファイル。状態（起きている ／ N3 ／ REM）を持って全領野へ配る
  sleep.py       置き換え。N3 と REM をくり返す。状態の切り替えは脳幹に任せる

このスクリプトが直すもの（印が1つでも見つからなければ、何も書かずに止まる）
  torch_group2.py  送り手ごとの重さ send_gain（上からの戻り）と、学ぶ速さの掛け算 eta_gain（驚き）
  torch_brain.py   脳幹を組み、毎歩の終わりに observe を呼ぶ
  day.py           眠りを 5000歩（仮）→ 脳幹の長さ（約10万歩）。起きているときの切り替えを脳幹へ。記録に実際に眠った歩数
  babble.py        起きているときの切り替えを脳幹へ。版38。state に脳幹の基準を保存
⚠️ 版37 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
"""
import ast
import os
import re
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


for f in ("brainstem.py", "sleep.py.v38"):
    if not os.path.exists(f):
        sys.exit(f"⚠️ {f} が無い。brainstem.py と、新しい sleep.py を sleep.py.v38 という名前で置いてから回す")

F = {k: open(k, encoding="utf-8").read() for k in ("torch_group2.py", "torch_brain.py", "day.py", "babble.py", "sleep.py")}
if "VERSION = 37 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版37 ではない。何も書いていない（先に apply_v37.py）")
if "b.g.scaling_on = False      # 起きているあいだは下方調整を止める" not in F["babble.py"]:
    sys.exit("⚠️ babble.py に起きているあいだの揃え（apply_babble_awake.py）が入っていない。何も書いていない")
if "いまは N3 だけを入れる" not in F["sleep.py"]:
    sys.exit("⚠️ sleep.py が想定した版（版37 まで）ではない。何も書いていない")

# ── torch_group2.py ──
g = F["torch_group2.py"]
m = re.findall(r"^        self\.scaling_on = True[^\n]*\n", g, flags=re.M)
if len(m) != 1:
    sys.exit(f"⚠️ group2 の scaling_on の行が {len(m)} 回。何も書いていない")
g = once(g, m[0],
         "        self.scaling_on = True          # 脳幹（brainstem.py）が状態で切り替える（版38）\n"
         "        self.send_gain = torch.ones(self.n_send_all, device=dev)   # 送り手ごとの重さ。眠りで上からの戻りを落とす（版38）\n"
         "        self.eta_gain = 1.0             # 学ぶ速さの掛け算。起きているあいだの驚き（版38）\n",
         "group2 の scaling_on")
g = once(g, "            # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）\n",
         "            amp = amp * self.send_gain[self.e_src[e]]   # 脳幹：眠りで上からの戻りを落とす（版38）\n"
         "            # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）\n",
         "group2 の足し込み")
g = once(g, "            gg = self.eta * err_sig[r]\n",
         "            gg = self.eta * self.eta_gain * err_sig[r]   # 脳幹：驚きで学ぶ速さを上げる（版38）\n",
         "group2 の学び")

# ── torch_brain.py ──
t = F["torch_brain.py"]
t = once(t, "from torch_group2 import Group2\n", "from torch_group2 import Group2\nimport brainstem as BS\n", "brain の import")
t = once(t, '        self.kick_abs = int(self.off_ch[NAMES.index("運動")]) + self.kick_to\n',
         '        self.kick_abs = int(self.off_ch[NAMES.index("運動")]) + self.kick_to\n'
         '        # 脳幹（版38）：いまの状態（起きている ／ N3 ／ REM）を持って全領野へ配る。起きているから始める\n'
         '        self.bs = BS.Brainstem(self)\n',
         "brain の組み立ての終わり")
t = once(t, '        return cut("聴覚", rep), cut("聴覚", err), cut("運動", rep), cut("運動", err)\n',
         '        self.bs.observe(self)                  # 脳幹：驚き・起こすか（版38）\n'
         '        return cut("聴覚", rep), cut("聴覚", err), cut("運動", rep), cut("運動", err)\n',
         "brain の step の終わり")

# ── day.py ──
d = F["day.py"]
d = once(d, "import sleep as S\n", "import sleep as S\nimport brainstem as BS\n", "day の import")
d = once(d, "SLEEP = 5000        # 1回の眠りの歩数（仮。測って決める）\n",
         "SLEEP = BS.SLEEP_STEPS   # 1回の眠り（版38）。起きる:眠る を【ヒト】乳児の約2.5:1 に。N3 と REM をくり返す\n",
         "day の SLEEP")
d = once(d, "    b.g.scaling_on = False                    # 起きているあいだは下方調整を止める（【ヒト】）\n",
         "    b.bs.set_state(BS.WAKE)                   # 起きている（版38：恒常性・上からの戻りは脳幹が決める）\n",
         "day の scaling_on")
d = once(d, "                   f\" 眠り {n_sleep}歩  supp {s:.2f} → {s2:.2f}\"\n",
         "                   f\" 眠り {b.bs.last_slept}/{n_sleep}歩  supp {s:.2f} → {s2:.2f}\"\n",
         "day の記録")

# ── babble.py ──
b = F["babble.py"]
b = once(b, "import torch_area as T\n", "import torch_area as T\nimport brainstem as BS\n", "babble の import")
b = once(b, "    b.g.scaling_on = False      # 起きているあいだは下方調整を止める（day.py と同じ。【ヒト】眠ると戻る）\n",
         "    b.bs.set_state(BS.WAKE)     # 起きている（版38：恒常性・上からの戻りは脳幹が決める。day.py と同じ）\n",
         "babble の scaling_on")
b = once(b, "VERSION = 37 ", "VERSION = 38 ", "VERSION")
h = "#      09-11「段の出力は表現の発火」の形。上→下 だけ p のまま残っていた）\n"
b = once(b, h, h +
         "#  38  脳幹（覚醒系）を入れた（brainstem.py）。状態 起きている ／ N3 ／ REM を持って全領野へ配る\n"
         "#      眠りは N3 と REM をくり返す（前は N3 だけ。REM を落とす承認は無かった）\n"
         "#      眠りで上からの戻りを落とす（N3 0.18 倍 ／ REM 0.33 倍。【ヒト】Hayat 2022【み選】）\n"
         "#      起きているあいだ、驚き（皮質の誤差がいつもより多い）で学ぶ速さを上げる（量はノブ【C】）\n"
         "#      眠りの長さを【ヒト】乳児の比に（約10万歩、1周 約6万歩の半分が REM【み選】）\n", "版の履歴")
b = once(b, '"up": b.up, "pred": b.pred, "ca3_prev": b.ca3_prev}',
         '"up": b.up, "pred": b.pred, "ca3_prev": b.ca3_prev, "bs": b.bs.state_dict()}', "babble の save")
b = once(b, '    b.up, b.pred, b.ca3_prev = d["up"], d["pred"], d["ca3_prev"]\n',
         '    b.up, b.pred, b.ca3_prev = d["up"], d["pred"], d["ca3_prev"]\n'
         '    if "bs" in d:                          # 版38 から\n'
         '        b.bs.load_state_dict(d["bs"])\n', "babble の load")

new_sleep = open("sleep.py.v38", encoding="utf-8").read()
out = {"torch_group2.py": g, "torch_brain.py": t, "day.py": d, "babble.py": b, "sleep.py": new_sleep}
for s in list(out.values()) + [open("brainstem.py", encoding="utf-8").read()]:
    ast.parse(s)
for k, s in out.items():
    shutil.copy(k, k + ".v37")
    open(k, "w", encoding="utf-8").write(s)
os.remove("sleep.py.v38")
print("版38 にした（元は *.v37 に残した）")
print("回すときは  python3 babble.py 100 --fresh  ／ 眠りまで回すなら  python3 day.py 1 --fresh")
