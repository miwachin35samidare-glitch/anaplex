"""版48 → 版49：語の指令を運動前野から皮質どうしの繋がりで入れる ／ 口は筋の形でならして読む（arkhe の ~/anaplex で）

【み】10-03「まぁいいよ」（指令を運動前野の送り手の集団から本数で入れる ／ 口を筋の形で読む）
人の形
  【ヒト】一次運動野・腹側中心前回（6v1〜3）に第4層は無い（Bakken 2021 ／ Comm Biol 2025）
  【ヒト】下前頭回 ⇄ 口・顔の運動野は直の皮質どうしの繋がり（Greenlee 2004、14人、最初の振れ 13ms）
  【ヒト】話すとき ブローカ野（準備）→ 運動野（実行）（Flinker 2015、7人。本文未読）
  ⇒ 語の指令は下（視床・第4層）からではなく、運動前野から多くの弱い繋がりで来る
直すもの
  1 運動の領野の入口：補助輪の指令 1本 × 28.6（閾値の2倍【C】）をやめ、運動前野の送り手 K_PM 本の発火の数 × 1（人の繋がり1本）
      指令が入っている体の送り手は 35Hz（【ヒト】話すあいだの運動野の細胞の山 30〜40Hz、Stavisky 2019）
      入っていない体の送り手は 3.30Hz（【ヒト】起きている錐体、Verma 2024）
      K_PM ＝ 115 ⚠️ ノブ【C】。人の本数は見つからない。arkhe の版48 で測った指令の歩の ihn 約 2 のとき、
        指令の体が人の話すあいだの率（35Hz ＝ 37%/歩）で撃つ本数（二項分布の計算）
        ihn 1 ⇒ ほぼ 100% ／ ihn 3 ⇒ ほぼ 0%（横の興奮が無いとき）
      ⟲ 版48 まで：28.6 は「抑制なしで1本で越える」前提。抑制が人の率になって ihn 約 2 では指令だけで越えなかった（10-03）
  2 蹴り（運動の視床）は今のまま：撃った歩に 28.6（KICK_W ＝ 前の x_gain）。⚠️ ノブのまま（視床の本数は人に無い）
  3 口の読み方：窓の半分以上の歩で撃った体【C】をやめ、発火を筋の形でならした力で読む
      【ヒト】Ito 2004：唇・舌の力は発火を 2次の系でならしたもの（6.1Hz・ζ 0.69・遅れ 15ms。上唇8人・下唇8人・舌2人）
      読む時刻は窓の終わり（VOICEVOX が丸ごと受け取るため【C】のまま）
      READ_TH ＝ 0.20 ⚠️【C】（力は撃ち続けたときの発火率に寄る。人の話すあいだ 37% と ふだん 3.5% の間）
変えないもの  抑制（版45〜48）／ 聴覚・高次の入口 ／ 写し（運動の領野の表現）
⚠️ 印が1つでも見つからなければ、何も書かずに止まる
⚠️ 版48 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
"""
import ast
import re
import shutil
import sys

R, M, B = "torch_brain.py", "mouth.py", "babble.py"


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


def once_re(s, pat, rep, name):
    n = len(re.findall(pat, s, flags=re.M))
    if n != 1:
        sys.exit(f"⚠️ {name} の印が {n} 回。何も書いていない")
    return re.sub(pat, rep, s, flags=re.M)


F = {k: open(k, encoding="utf-8").read() for k in (R, M, B)}
if "VERSION = 48 " not in F[B]:
    sys.exit("⚠️ babble.py が版48 ではない。何も書いていない（先に apply_v48.py）")
if "K_PM" in F[R]:
    sys.exit("⚠️ もう版49 になっている。何も書いていない")

# ── torch_brain.py ──
r = F[R]
r = once(r, '"運動": 2 * HUMAN_TH, ', '"運動": 1.0, ', "brain の X_GAIN 運動")
r = once_re(r, r"^(KICK_FRAC = 0\.75 .*)$",
            r"\1\n"
            "# 版49：語の指令は運動前野から、皮質どうしの繋がりで来る（【ヒト】Greenlee 2004 ／ M1・6v に第4層なし Bakken 2021・Comm Biol 2025）\n"
            "#   運動の領野の1体に運動前野の送り手 K_PM 本。1本 ＝ 人の繋がり1本（重さ 1）。届いた数がそのまま入口\n"
            "#   K_PM ⚠️ ノブ【C】：版48 の指令の歩の ihn 約 2 で、指令の体が【ヒト】話すあいだの 35Hz で撃つ本数\n"
            "K_PM = 115\n"
            "R_PM_ON = 35.0 * 10.67 / 1000.0     # 指令が入っている体の送り手【ヒト】話すあいだの運動野の細胞の山 30〜40Hz（Stavisky 2019）\n"
            "R_PM_OFF = 3.30 * 10.67 / 1000.0    # 入っていない体の送り手【ヒト】起きている錐体（Verma 2024）\n"
            "KICK_W = 2 * HUMAN_TH               # 蹴りの1発の重さ ⚠️ ノブ（前の X_GAIN 運動。視床の本数は人に無い）",
            "brain の KICK_FRAC")
r = once_re(r, r"^(\s+)cmd,(\s+# 口の指令 → 運動の領野（版43d で戻した）)$",
            r"\1self._premotor(cmd),\2 ／ 版49：運動前野の送り手の数",
            "brain の x の cmd")
r = once(r, "x[self.kick_abs] = (torch.rand(", "x[self.kick_abs] = x[self.kick_abs] + KICK_W * (torch.rand(", "brain の蹴り")
r = once(r, "    def _fit(self, v, n):\n",
         "    def _premotor(self, cmd):\n"
         "        \"\"\"版49：運動前野の送り手 K_PM 本のうち、この歩に撃った数（指令が入っている体は R_PM_ON、ほかは R_PM_OFF）\"\"\"\n"
         "        p = torch.where(cmd > 0, torch.full_like(cmd, R_PM_ON), torch.full_like(cmd, R_PM_OFF))\n"
         "        return torch.binomial(torch.full_like(p, float(K_PM)), p, generator=self.gen)\n\n"
         "    def _fit(self, v, n):\n", "brain の _premotor")

# ── mouth.py ──
m = F[M]
m = once_re(m, r"^(MATCH_MIN = 0\.5\s+#.*)$",
         r"\1\n"
         "# 版49：口は運動の領野の発火を筋の形でならして読む\n"
         "#   【ヒト】Ito 2004 J Appl Physiol：電気刺激1発への力を 2次の系で当てはめ\n"
         "#     上唇 6.10Hz・ζ 0.714・15ms（8人）／ 下唇 6.05Hz・0.682・16ms（8人）／ 舌 6.11Hz・0.675・17ms（2人）\n"
         "#   喉頭の筋は人の値が無い（【ヒト】発火から声の高さの山まで TA 5〜20ms・CT 6〜75ms、Larson 1987）⇒ 唇・舌の値で代用【C】\n"
         "MUSCLE_FN = 6.1\n"
         "MUSCLE_ZETA = 0.69\n"
         "MUSCLE_DELAY_MS = 15.0\n"
         "READ_TH = 0.20            # ⚠️【C】力（撃ち続けたときの発火率に寄る）がこれを越えた体を「撃った」と読む",
         "mouth の定数")
m = once(m, "class Mouth:\n",
         "def muscle_force(spikes, dt_ms=10.67):\n"
         "    \"\"\"版49：発火の列 [歩, 体] → 最後の歩の力 [体]。2次の系（直流の利得 1：撃ち続けると発火率に寄る）\"\"\"\n"
         "    spikes = np.asarray(spikes, np.float64)\n"
         "    w = 2.0 * np.pi * MUSCLE_FN\n"
         "    d = int(round(MUSCLE_DELAY_MS / dt_ms))\n"
         "    sub = 20\n"
         "    h = dt_ms / 1000.0 / sub\n"
         "    y = np.zeros(spikes.shape[1]); v = np.zeros_like(y)\n"
         "    for t in range(len(spikes)):\n"
         "        u = spikes[t - d] if t >= d else 0.0\n"
         "        for _ in range(sub):\n"
         "            v += (w * w * (u - y) - 2.0 * MUSCLE_ZETA * w * v) * h\n"
         "            y += v * h\n"
         "    return y\n\n\n"
         "class Mouth:\n", "mouth の muscle_force")

# ── babble.py ──
b = F[B]
b = once(b, "    said, v = mo.decode((torch.stack(reps).mean(0) >= 0.5).float().cpu().numpy()[None])\n",
         "    # 版49：口は発火を筋の形でならした力で読む（【ヒト】Ito 2004）。⟲ 版48 まで「窓の半分以上の歩」【C】\n"
         "    import mouth as MO\n"
         "    force = MO.muscle_force(torch.stack(reps).cpu().numpy())\n"
         "    said, v = mo.decode((force >= MO.READ_TH).astype(np.float32)[None])\n", "babble の口の読み方")
b = once(b, "VERSION = 48 ", "VERSION = 49 ", "VERSION")
h = "#      （【ヒト】Wilbers 2023・Lee・Dalley 2023。版46 の撃ち方。【み】10-02・10-03）\n"
b = once(b, h, h +
         "#  49  語の指令を運動前野の送り手 115本（ノブ）の発火の数で入れる（1本 ＝ 人の繋がり1本。指令 35Hz・ふだん 3.3Hz【ヒト】）／\n"
         "#      口は発火を筋の形（【ヒト】Ito 2004 6.1Hz・ζ 0.69・15ms）でならした力 ≥ 0.20【C】で読む【み】10-03\n", "版の履歴")

out = {R: r, M: m, B: b}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".pre49")
    open(k, "w", encoding="utf-8").write(v)
print("版49 にした（元は *.pre49 に残した）")
print("回すときは  python3 measure_v49.py 100（学びながら。0から）")
