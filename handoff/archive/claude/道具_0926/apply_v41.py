"""版40 → 版41：覚醒（感度と眠気の覆い）（arkhe の ~/anaplex で  python3 apply_v41.py ）

設計と出どころは claude/設計案_脳幹.md §9。印が1つでも見つからなければ何も書かずに止まる
  覚醒 A ＝ 青斑核のふだんの発火の高さ（起きているあいだ）
    青斑核 ＝ 皮質の誤差（予期しない入力の量、8歩遅れ）。その短い移動平均 ÷ 起きているあいだの長い平均（ふだん 1）
    ⚠️ 入口を皮質の誤差にするのは【C】（前は 1 ＋ 驚き にしたが、揺れが 0.99〜1.11 しか出ず、退屈で下がらなかった【測】）
    【リスザル】起きているあいだに約10倍揺れる：ぼんやり 0.5Hz 〜 好物 15Hz（ふだん 2.45Hz）⇒ A を 0.20〜6.12 に収める【み】許可
    何も起きないと下がり、何か起きると上がる【み】退屈・待ち時間で眠くなる
    眠りのあいだは段の比（N3 0.10 ／ REM 0.01【ネズミ】【み選】09-26。下限 0.20）
  感度：表現の閾値を下げる
    【ラット】ノルアドレナリンで錐体細胞が 3.4mV 脱分極（Grzelka 2017）÷【ヒト】静止から閾値まで 16mV ⇒ 閾値の 21%【み】許可
    A が最大（6.12）で 21% 下がり、ふだん（1）で 0、ぼんやり（0.20）で 3% 上がる（青斑核の率に比例【C】）
    【ヒト】覚醒で強いものがより強く、弱いものがより弱く（Lee 2018）⇒ 閾値近くまで来ていた入力だけ越える
    ⚠️ 誤差の的（下からの入力で撃つか）は変えない（的は学ぶ量や状態に依存させない）
  眠気の覆い：眠るのは 眠りの圧 ≥ A のとき（眠りの側と覚醒の側が押し合い、強い方に倒れる）
    【ヒト】視索前野に当たる中間核の細胞数が多い人ほど眠りがまとまる（Lim 2014）。押し合う回路は【ネズミ】
⚠️ ノブ：A の短い平均の長さ（SURPRISE_TAU と同じ 2000歩）／ 長い平均の長さ（起きていられる歩数）／ A と脱分極の対応（比例）
起きているあいだの撃ち方が変わるので版41。⇒ python3 babble.py 100 --fresh
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


NAMES = ("torch_group2.py", "brainstem.py", "babble.py", "day.py")
F = {k: open(k, encoding="utf-8").read() for k in NAMES}
if "VERSION = 40 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版40 ではない。何も書いていない（先に apply_v40.py）")
if "th_gain" in F["torch_group2.py"]:
    sys.exit("⚠️ もう版41 になっている。何も書いていない")

# ── torch_group2.py ──
g = F["torch_group2.py"]
g = once(g, "        self.third = torch.ones((), device=dev)\n",
         "        self.third = torch.ones((), device=dev)\n"
         "        self.th_gain = torch.ones((), device=dev)   # 覚醒で表現の閾値に掛ける倍率（版41。脳幹が決める）\n", "group2 の init")
g = once(g, "        rep = (drive > self.fire_th).float()\n",
         "        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n", "表現の撃ち方")
g = once(g, "            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th) & (x_all > 0)\n",
         "            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th * self.th_gain) & (x_all > 0)\n",
         "完成の撃ち方")

# ── brainstem.py ──
s = F["brainstem.py"]
k = "DA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）\n"
s = once(s, k, k +
         "# 覚醒（版41）設計案 §9\n"
         "#   【リスザル】青斑核は起きているあいだ ぼんやり 0.5Hz 〜 好物 15Hz、ふだん 2.45Hz（Foote 1980）【み】09-27 許可\n"
         "LC_MIN, LC_MAX = 0.5 / 2.45, 15.0 / 2.45\n"
         "#   【ラット】NA で 3.4mV 脱分極（Grzelka 2017）÷【ヒト】静止から閾値 16mV（Hunt 2023）【み】09-27 許可\n"
         "NA_DEPOL = 3.4 / 16.0\n"
         "AROUSAL_TAU = SURPRISE_TAU               # ⚠️ ノブ：覚醒の短い平均の長さ（歩）\n",
         "脳幹の定数")
s = once(s, "        self.surprise = torch.zeros((), device=self.dev)\n",
         "        self.surprise = torch.zeros((), device=self.dev)\n"
         "        self.lc_s = torch.ones((), device=self.dev)       # 青斑核の短い平均（版41）\n"
         "        self.lc_l = torch.ones((), device=self.dev)       # 起きているあいだの長い平均\n"
         "        self.arousal = torch.ones((), device=self.dev)    # 覚醒 A（ふだん 1）\n"
         "        self.lc_on = False\n", "脳幹の init")
s = once(s, "        self.da_buf.zero_()\n        self.wake_request = False\n",
         "        self.da_buf.zero_()\n"
         "        # 覚醒（版41）：眠りは段の比（【ネズミ】【み選】09-26）。起きたら observe が決める\n"
         "        self.arousal = torch.tensor(max(LC_RATE[s], LC_MIN), device=self.dev)\n"
         "        self._set_gain()\n"
         "        self.wake_request = False\n", "脳幹の set_state")
k = "    def gate_mean(self):\n"
s = once(s, k,
         "    def _set_gain(self):\n"
         "        \"\"\"覚醒 A に応じて表現の閾値を下げる（最大で【ラット】3.4mV ÷【ヒト】16mV。青斑核の率に比例【C】）\"\"\"\n"
         "        a = (self.arousal - 1.0) / (LC_MAX - 1.0)\n"
         "        self.g.th_gain = 1.0 - NA_DEPOL * a\n"
         "\n" + k, "脳幹の _set_gain")
s = once(s,
         "            self.g.eta_gain = torch.ones((), device=self.dev)\n"
         "            # 学びの門（版40）",
         "            self.g.eta_gain = torch.ones((), device=self.dev)\n"
         "            # 覚醒（版41）：青斑核のふだんの高さ ＝ 皮質の誤差（予期しない入力の量。8歩遅れ）の短い平均 ÷ 長い平均\n"
         "            #   何も起きない・聞き慣れた音ばかりだと下がり、予期しない音が続くと上がる（入口を誤差にするのは【C】）\n"
         "            if ready:\n"
         "                lc = self.hist[-1 - SURPRISE_LAT]\n"
         "                if not self.lc_on:                       # 基準ができた歩で、どちらも「いつも」から始める\n"
         "                    self.lc_s, self.lc_l, self.lc_on = self.base.clone(), self.base.clone(), True\n"
         "                self.lc_s = self.lc_s + (lc - self.lc_s) / AROUSAL_TAU\n"
         "                self.lc_l = self.lc_l + (lc - self.lc_l) / max(self.wake_steps(), 1)\n"
         "                self.arousal = (self.lc_s / self.lc_l.clamp(min=1e-6)).clamp(LC_MIN, LC_MAX)\n"
         "            else:\n"
         "                self.arousal = torch.ones((), device=self.dev)\n"
         "            self._set_gain()\n"
         "            # 学びの門（版40）", "脳幹の覚醒")
s = once(s, "    @property\n    def sleep_due(self):\n        return self.pressure >= 1.0\n",
         "    @property\n    def sleep_due(self):\n"
         "        \"\"\"眠りの圧が覚醒を上回ったら眠る（版41。落ち着く・退屈で A が下がると早く、驚きが続くと遅く）\"\"\"\n"
         "        return self.pressure >= float(self.arousal)\n", "脳幹の sleep_due")
s = once(s, '                "r0": self.r0, "k_wake": self.k_wake}\n',
         '                "r0": self.r0, "k_wake": self.k_wake, "lc_s": self.lc_s, "lc_l": self.lc_l, "lc_on": self.lc_on}\n', "脳幹の保存")
s = once(s, '        self.r0, self.k_wake = d.get("r0"), d.get("k_wake", 0)\n',
         '        self.r0, self.k_wake = d.get("r0"), d.get("k_wake", 0)\n'
         '        if "lc_s" in d:\n'
         '            self.lc_s, self.lc_l, self.lc_on = d["lc_s"], d["lc_l"], d.get("lc_on", True)\n', "脳幹の読み込み")

# ── babble.py ──
b = F["babble.py"]
b = once(b, "VERSION = 40 ", "VERSION = 41 ", "VERSION")
k = "#      驚きは学ぶ速さから外した（覚醒の側へ）\n"
b = once(b, k, k +
         "#  41  覚醒 A（青斑核のふだんの高さ ＝ 驚きの短い平均 ÷ 長い平均。【リスザル】0.20〜6.12）で\n"
         "#      表現の閾値を最大 21% 下げる（【ラット】3.4mV ÷【ヒト】16mV）／ 眠るのは 眠りの圧 ≥ A のとき\n", "版の履歴")
b = once(b, "            print(f\"  喃語 {done} 回  門 {b.bs.gate_mean():.3f}  経過",
         "            print(f\"  喃語 {done} 回  門 {b.bs.gate_mean():.3f}  覚醒 {float(b.bs.arousal):.2f}  経過", "babble の表示")

# ── day.py ──
d = F["day.py"]
d = once(d, "                      f\"眠りの圧 {b.bs.pressure:.2f}  疲れ {b.bs.fatigue:.2f}  \"\n",
         "                      f\"眠りの圧 {b.bs.pressure:.2f}  覚醒 {float(b.bs.arousal):.2f}  疲れ {b.bs.fatigue:.2f}  \"\n", "day の表示")

out = {"torch_group2.py": g, "brainstem.py": s, "babble.py": b, "day.py": d}
for x in out.values():
    ast.parse(x)
for k, x in out.items():
    shutil.copy(k, k + ".v40")
    open(k, "w", encoding="utf-8").write(x)
print("版41 にした（元は *.v40 に残した）")
print("回すときは  python3 babble.py 100 --fresh   （表示の「覚醒」はふだん 1 前後）")
