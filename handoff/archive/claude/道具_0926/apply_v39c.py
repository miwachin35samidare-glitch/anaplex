"""版39b → 版39c：眠りの圧（arkhe の ~/anaplex で  python3 apply_v39c.py ）

【み選】09-27  C「時間 ＋ 疲れ」／ B「年齢は脳幹が持つ（回した分だけ歳をとる）」／ B「疲れは 深さ と 眠くなる早さ の両方」
脳幹が持つもの
  年齢       喃語が出るころ 6か月から始め、起きている・眠っている歩ごとに進む（1歩 10.67ms × 4 ＝ 人の時間に戻す）
             ⚠️ 夜の眠りは入っていない（模型の1日は昼寝と昼寝のあいだだけ）
  起きている長さ・1回の眠り   年齢の表から。【ヒト】6〜9か月 起きている 2.5〜3.5時間（3.0）／ 昼寝 1〜1.5時間（1.25）（生物台帳）
             人の約1/4 に縮める（【み選】09-26、版38 の眠りの長さと同じ縮め方）⇒ 起きている 約25.3万歩 ／ 眠り 約10.5万歩
             ⚠️ 表は 6〜9か月の1行だけ。ほかの月齢の起きている長さは未取得（範囲の外は端の値のまま）
  眠りの圧   起きているあいだ、1歩ごとに (1 ＋ 疲れ) ÷ 起きている長さ だけたまり、1 に届いたら眠る。起きたら 0 に戻す
             【ヒト】眠りの圧は起きている長さとともにたまり、眠ると下がる（二過程モデル）
  疲れ       皮質（聴覚・連合・高次）の受け手の「率 ÷ 目標」の平均が、起きてからどれだけ増えたか（増えた割合）
             【ヒト】起きていると皮質の興奮性が上がる（Huber 2013・Kuhn 2016）
             ⚠️ 疲れで早く眠くなる人の測定は無い。(1 ＋ 疲れ) の形は【C】（疲れ 0 なら人の時間どおり）
             起きた直後の率は眠りの値を引きずる（約2000歩の移動平均）ので、基準は起きて 2000歩たってから取る
  深さ       いまの下方調整がそのまま「目標より上の分だけ強く下げる」ので、疲れた受け手ほど強く下がる（足すものなし）
             ⚠️ 【ヒト】学んだ場所で徐波が強まる（Huber 2004）は、徐波を外から掛けている今の形では足していない
day.py      起きる長さを「800回」から「眠りの圧が 1 に届くまで」に。眠る長さは脳幹の年齢から
            --wake N ／ --sleep N を付けたときは今までどおり回数・歩数で決める
起きているあいだの学びは変わらない ⇒ VERSION は 39 のまま（state.pt はそのまま使える。年齢は 6か月から）
印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


F = {k: open(k, encoding="utf-8").read() for k in ("brainstem.py", "day.py", "torch_group2.py")}
if "target_recv" not in F["torch_group2.py"]:
    sys.exit("⚠️ 版39b になっていない（先に apply_v39b.py）。何も書いていない")
if "AGE_TABLE" in F["brainstem.py"]:
    sys.exit("⚠️ もう版39c になっている。何も書いていない")

s = F["brainstem.py"]
s = once(s, "import torch\n", "import torch\nfrom torch_area import DT_MS\n", "脳幹の import")
k = "WAKE_CHECK = 94              # 起こすかを見るのは約1秒ごと（毎歩 GPU から値を出すと遅くなるため）\n"
s = once(s, k, k +
         "# 眠りの圧（版39c）【み選】09-27 C（時間 ＋ 疲れ）・年齢は脳幹が持つ・疲れは深さと眠くなる早さの両方\n"
         "COMPRESS = 4.0              # 起きる・眠るを人の約1/4 に縮めた（【み選】09-26。版38 の眠りの長さと同じ）\n"
         "AGE0_MONTH = 6.0            # 喃語が出るころ（【ヒト】6〜9か月）\n"
         "# (月齢から, まで, 起きている時間, 1回の眠りの時間)\n"
         "#   【ヒト】6〜9か月 起きている 2.5〜3.5時間 ／ 昼寝 1〜1.5時間（生物台帳）。ほかの月齢は未取得\n"
         "AGE_TABLE = ((6.0, 9.0, 3.0, 1.25),)\n"
         "FATIGUE_AREAS = (\"聴覚\", \"連合\", \"高次\")   # 疲れを読む皮質（運動は目標なし・視覚は目を閉じている）\n"
         "_STEP_DAYS = DT_MS * COMPRESS / 1000.0 / 86400.0\n"
         "\n"
         "\n"
         "def _hours_to_steps(h):\n"
         "    return int(h * 3600.0 * 1000.0 / DT_MS / COMPRESS)\n",
         "脳幹の定数")
k = "        self.n_wake = 0              # 起きていた歩数（基準ができたかを見る。版39）\n"
s = once(s, k, k +
         "        self.age_days = AGE0_MONTH * 30.44   # 年齢（版39c）。起きている・眠っている歩ごとに進む\n"
         "        self.pressure = 0.0          # 眠りの圧。1 に届いたら眠る\n"
         "        self.fatigue = 0.0           # 疲れ（皮質の 率÷目標 が起きてから増えた割合）\n"
         "        self.r0 = None               # 疲れの基準（起きて homeo_tau 歩たってからの 率÷目標）\n"
         "        self.k_wake = 0              # 起きてからの歩数\n"
         "        self._fmask = None\n",
         "脳幹の init")
s = once(s, "        if s == self.state:\n            return\n        self.state = s\n",
         "        if s == self.state:\n            return\n"
         "        if s == WAKE and self.state in (N3, REM):   # 眠りから起きた（版39c）：眠りの圧を下ろし、疲れを測り直す\n"
         "            self.pressure, self.fatigue, self.r0, self.k_wake = 0.0, 0.0, None, 0\n"
         "        self.state = s\n", "脳幹の set_state")
k = "        s = torch.cat([b.up[nm] for nm in CORTEX]).mean()\n"
s = once(s, k,
         "        self.age_days += _STEP_DAYS\n"
         "        if self.state == WAKE:                  # 眠りの圧（版39c）。値を出すのは約1秒ごと\n"
         "            self.k_wake += 1\n"
         "            if self.k_wake % WAKE_CHECK == 0:\n"
         "                self._press(WAKE_CHECK)\n" + k, "脳幹の observe")
k = "    def sleep_state(self, k):\n"
s = once(s, k,
         "    def _row(self):\n"
         "        m = self.age_days / 30.44\n"
         "        for lo, hi, w, sl in AGE_TABLE:\n"
         "            if lo <= m < hi:\n"
         "                return w, sl\n"
         "        row = AGE_TABLE[0] if m < AGE_TABLE[0][0] else AGE_TABLE[-1]   # ⚠️ 表の外は端の値\n"
         "        return row[2], row[3]\n"
         "\n"
         "    def wake_steps(self):\n"
         "        \"\"\"いまの年齢で、疲れが無いときに起きていられる歩数\"\"\"\n"
         "        return _hours_to_steps(self._row()[0])\n"
         "\n"
         "    def sleep_steps(self):\n"
         "        \"\"\"いまの年齢の1回の眠りの歩数\"\"\"\n"
         "        return _hours_to_steps(self._row()[1])\n"
         "\n"
         "    @property\n"
         "    def sleep_due(self):\n"
         "        return self.pressure >= 1.0\n"
         "\n"
         "    def _ratio(self):\n"
         "        g = self.g\n"
         "        if self._fmask is None:\n"
         "            m = torch.zeros(g.n_recv_all, dtype=torch.bool, device=self.dev)\n"
         "            for i, nm in enumerate(self.b.areas.keys()):\n"
         "                if nm in FATIGUE_AREAS:\n"
         "                    m[int(g.off_recv[i]):int(g.off_recv[i + 1])] = True\n"
         "            self._fmask = m & ~g.scale_skip\n"
         "        return float((g.rate_ema[self._fmask] / g.target_recv[self._fmask]).mean())\n"
         "\n"
         "    def _press(self, n):\n"
         "        \"\"\"起きているあいだの眠りの圧。(1 ＋ 疲れ) ÷ 起きていられる歩数 ずつたまる\"\"\"\n"
         "        if self.k_wake >= self.g.homeo_tau:     # 率は約2000歩の移動平均。起きた直後は眠りの値を引きずる\n"
         "            r = self._ratio()\n"
         "            if self.r0 is None:\n"
         "                self.r0 = r\n"
         "            self.fatigue = max(0.0, r / max(self.r0, 1e-9) - 1.0)\n"
         "        self.pressure += n * (1.0 + self.fatigue) / self.wake_steps()\n"
         "\n" + k, "脳幹の sleep_state")
k = '        return {"state": self.state, "base": self.base, "base_aud": self.base_aud, "n_wake": self.n_wake}\n'
s = once(s, k,
         '        return {"state": self.state, "base": self.base, "base_aud": self.base_aud, "n_wake": self.n_wake,\n'
         '                "age_days": self.age_days, "pressure": self.pressure, "fatigue": self.fatigue,\n'
         '                "r0": self.r0, "k_wake": self.k_wake}\n', "脳幹の保存")
k = '        self.n_wake = d.get("n_wake", 0)\n'
s = once(s, k, k +
         '        self.age_days = d.get("age_days", self.age_days)\n'
         '        self.pressure, self.fatigue = d.get("pressure", 0.0), d.get("fatigue", 0.0)\n'
         '        self.r0, self.k_wake = d.get("r0"), d.get("k_wake", 0)\n', "脳幹の読み込み")

d = F["day.py"]
d = once(d, '    wake = arg("--wake", WAKE)\n    n_sleep = arg("--sleep", SLEEP)\n',
         '    wake = arg("--wake", None)                # 付けなければ、眠りの圧が 1 に届くまで起きている（版39c）\n'
         '    n_sleep = arg("--sleep", None)            # 付けなければ、脳幹の年齢から（版39c）\n', "day の引数")
d = once(d,
         "        for i in range(wake):\n"
         "            mel, cmd, enc = speak(rng)\n"
         "            B.run(b, cmd, enc(mel), True, True)\n"
         "            done += 1\n"
         "            if (i + 1) % 100 == 0:\n"
         "                el = time.time() - t0\n"
         "                print(f\"  [{cyc+1}/{n_cycle}] 起きて {i+1}/{wake}回  経過 {el/60:.1f}分  \"\n"
         "                      f\"残り {el/(i+1)*(wake-i-1)/60:.1f}分\", flush=True)\n",
         "        i = 0\n"
         "        while (i < wake) if wake is not None else not b.bs.sleep_due:\n"
         "            mel, cmd, enc = speak(rng)\n"
         "            B.run(b, cmd, enc(mel), True, True)\n"
         "            done += 1\n"
         "            i += 1\n"
         "            if i % 100 == 0:\n"
         "                el = time.time() - t0\n"
         "                print(f\"  [{cyc+1}/{n_cycle}] 起きて {i}回  経過 {el/60:.1f}分  \"\n"
         "                      f\"眠りの圧 {b.bs.pressure:.2f}  疲れ {b.bs.fatigue:.2f}  \"\n"
         "                      f\"年齢 {b.bs.age_days/30.44:.2f}か月\", flush=True)\n"
         "        n_wake = i\n"
         "        ns = n_sleep if n_sleep is not None else b.bs.sleep_steps()\n",
         "day の起きている")
d = once(d, "        print(f\"  [{cyc+1}] 眠る（{n_sleep}歩 ＝ {n_sleep*10.67/1000/60:.1f}分ぶん）\", flush=True)\n"
            "        S.sleep(b, n_step=n_sleep, report_every=max(1, n_sleep // 4))\n",
         "        fat = b.bs.fatigue                        # 起きると 0 に戻るので、眠る前に控える\n"
         "        print(f\"  [{cyc+1}] 眠る（{ns}歩 ＝ {ns*10.67/1000/60:.1f}分ぶん）  起きていた {n_wake}回  \"\n"
         "              f\"眠りの圧 {b.bs.pressure:.2f}  疲れ {b.bs.fatigue:.2f}\", flush=True)\n"
         "        S.sleep(b, n_step=ns, report_every=max(1, ns // 4))\n", "day の眠る")
d = once(d, "喃語 {done}回（+{wake}）", "喃語 {done}回（+{n_wake}）", "day の記録（回数）")
d = once(d, "眠り {b.bs.last_slept}/{n_sleep}歩", "眠り {b.bs.last_slept}/{ns}歩  疲れ {fat:.2f}", "day の記録（眠り）")

out = {"brainstem.py": s, "day.py": d}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".v39b")
    open(k, "w", encoding="utf-8").write(v)
print("版39c にした（元は *.v39b に残した）")
print("回すときは  python3 day.py 1   （眠りの圧が 1 に届くまで起きて、脳幹の年齢の長さで眠る）")
