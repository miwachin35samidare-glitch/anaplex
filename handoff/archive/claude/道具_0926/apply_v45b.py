"""版45 → 版45b：恒常性の目標を、起きている人の率に直す（arkhe の ~/anaplex で  python3 apply_v45b.py ）

直すもの（10-02【み】「間違ってるなら直して」）
  RATE_HUMAN 0.45%/歩 は人の値ではなかった
    元は前の作りの 1.5Hz（【ラット・サル】。08-08 当時 人の値が見つからず承認）× 1歩 3ms ＝ 0.45%/歩
    1歩が 10.67ms に変わっても 0.45%/歩 のまま残り、約 0.42Hz に意味が変わった
    版20 で【ヒト】の札、版39b で Peyrache 2012 の「RS 0.5Hz 未満」を後から当てた
    Peyrache は一晩通し・状態を分けない値。同じ記録の覚醒の値は RS 2.6Hz（Le Van Quyen 2016 表S3）
  ⇒ 皮質の表現・誤差 3.30Hz ＝ 3.52%/歩 ／ 抑制 4.70Hz ＝ 5.01%/歩
    【ヒト】Verma 2024（PNAS。bioRxiv v2 本文で確認）覚醒・リップルの外。Utah、II/III層、3人（上・中側頭回 ／ 中心前回）、錐体 437・抑制 72体
    もう1つの覚醒の値：Le Van Quyen 2016 表S3（中側頭回 II/III、2人）RS 2.6 ／ FS 6.4Hz。体数の多い Verma を取る
  誤差は表現と同じ（【C 当てはめ】のまま）。視床 18.8Hz・運動を下げない・海馬は眠りで下げない、は変えない

起きているあいだは下方調整が止まっている（脳幹）⇒ 喃語の中身は変わらない ⇒ VERSION は 45 のまま（state.pt はそのまま使える）
🔴 眠りは変わる：模型の皮質の表現は 1歩 0.3〜1.3%（09-26 の測り）で新しい目標より低い ⇒ 眠ると下げずに上げる側に回る
   「眠りで両向き（上げる向きの人の根拠なし）」は【み】保留のまま。day.py・sleep を回す前に決める
印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


F = {k: open(k, encoding="utf-8").read() for k in ("torch_area.py", "torch_group2.py", "torch_brain.py", "babble.py")}
if "VERSION = 45 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版45 ではない。何も書いていない")
if "Verma 2024" in F["torch_area.py"]:
    sys.exit("⚠️ もう版45b になっている。何も書いていない")

a = F["torch_area.py"]
a = once(a,
         "HUMAN_INPUT_PER_STEP = 135.0   # 【ヒト】毎歩 受けている入力の本数（30,000シナプス × 0.45%/歩）\n"
         "RATE_HUMAN = 0.0045             # 【ヒト】皮質の細胞が 1歩（10.67ms）に撃つ確率 0.45%（約 0.4Hz）\n",
         "# 【ヒト】起きている皮質の錐体 3.30 ± 2.52Hz（Verma 2024、覚醒・リップルの外、Utah II/III層、3人・437体）\n"
         "#   ⇒ 1歩（10.67ms）に 3.52%。版45b（10-02）\n"
         "#   ⟲ 版45 までは 0.45%/歩。前の作りの 1.5Hz（【ラット・サル】）× 1歩 3ms が、1歩 10.67ms になっても残った数\n"
         "#     （約 0.42Hz に意味が変わっていた）。後から当てた Peyrache 2012 の RS 0.5Hz 未満は一晩通し・状態を分けない値\n"
         "#   もう1つの覚醒の値：Le Van Quyen 2016 表S3 RS 2.6Hz（中側頭回 II/III、2人）\n"
         "RATE_HUMAN = 3.30 * DT_MS / 1000.0\n"
         "HUMAN_INPUT_PER_STEP = 30000 * RATE_HUMAN   # 毎歩 受けている入力の本数（30,000シナプス × RATE_HUMAN ≈ 1,056）。どこからも使っていない\n",
         "torch_area の RATE_HUMAN")

g = F["torch_group2.py"]
g = once(g, "            #   目標は【ヒト】0.45%/歩。撃ちすぎなら下げ、足りなければ上げる\n",
         "            #   目標は受け手ごとの【ヒト】の起きているときの率（target_recv。版45b）。撃ちすぎなら下げ、足りなければ上げる\n",
         "group2 の注記")

t = F["torch_brain.py"]
t = once(t,
         "        #   皮質  表現・誤差 0.45%（【ヒト】Peyrache 2012 RS 0.5Hz 未満。誤差は表現と同じ【C 当てはめ】）／ 抑制 2Hz（FS）\n",
         "        #   皮質  表現・誤差 3.30Hz ＝ RATE_HUMAN（【ヒト】Verma 2024 覚醒の錐体。誤差は表現と同じ【C 当てはめ】）\n"
         "        #         抑制 4.70Hz（【ヒト】Verma 2024 覚醒の抑制）。版45b\n"
         "        #         ⟲ 版45 までは 0.45% ／ 2Hz（Peyrache 2012。一晩通し・状態を分けない値）\n",
         "brain の目標の注記")
t = once(t, "                gg.target_recv[ih] = hz(2.0)\n",
         "                gg.target_recv[ih] = hz(4.70)\n", "brain の抑制の目標")

for k, s in (("torch_area.py", a), ("torch_group2.py", g), ("torch_brain.py", t)):
    ast.parse(s)
for k, s in (("torch_area.py", a), ("torch_group2.py", g), ("torch_brain.py", t)):
    shutil.copy(k, k + ".v45")
    open(k, "w", encoding="utf-8").write(s)
print("版45b にした（元は *.py.v45 に残した）。VERSION は 45 のまま")
print("🔴 眠り（day.py・sleep）は回す前に：目標より低い受け手は眠りで上げる側に回る")
