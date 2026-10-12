"""版60（表現の体の f-I）を外して版59 に戻す（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 revert_v60.py
【み】10-11 16時10分「戻して無音を調べる」
  16時07分【み】「要は人の自然な反応じゃなくて、電極刺して生体にありえない電流を流して測った数字ってことでしょ？」
  ⇒ f-I（Barzó・Szöts 2024）は切片に一定の電流を流した値。模型の駆動に見立てたのは【C の当てはめ】で、人の自然な反応の根拠が無い
  不応期 3周（【ヒト】Ison 2011：起きている人の単一細胞）は残す
戻すもの：torch_group2.py の FI_SLOPE・FI_RHEO・PASS_MS と、_pass の f-I の行 ／ babble.py の VERSION を 59 に
  SAME_NET に 60 を入れる（配線は版60 と同じ ⇒ 版60 で育てた state.pt も読める）
"""
import ast, pathlib, sys

G2 = [
    ("REFRACT_PASSES = 3\n"
     "# 版60：表現の体の f-I。【ヒト】大人の L2/3 錐体 f-I の傾き 0.127〜0.176Hz/pA・閾値の電流 208〜263pA（Barzó・Szöts 2024）\n"
     "FI_SLOPE, FI_RHEO = 0.15, 230.0\n"
     "PASS_MS = 1.102                     # 1周 ＝ torch_area.SYN_DELAY_MS（【ヒト】シナプス1本の遅れ）\n",
     "REFRACT_PASSES = 3\n"
     "# ⟲ 版60 の f-I は外した（10-11【み】「電極刺して生体にありえない電流を流して測った数字ってことでしょ？」）\n"),
    ("        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n"
     "        d_fi = drive\n"
     "        if self._complete_any:           # 入力が無いところは予測が足しになる（撃たせはしない）。版52 B\n"
     "            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th * self.th_gain) & (x_all > 0)\n"
     "            rep = torch.where(fill, torch.ones_like(rep), rep)\n"
     "            d_fi = torch.where(fill, drive + self.pred_boost * pr, drive)\n"
     "        # 版60：f-I。前に撃ってからの時間 × f(r) ≧ 1 のときだけ撃つ（r ＝ この周の駆動 ÷ 閾値）\n"
     "        #   ⟲ 版59 まで越えれば r によらず不応期明けごと（約300Hz）。1歩に2発 93〜100%（人 17〜25%）【測】measure_v59b\n"
     "        if getattr(self, \"_last_fire\", None) is not None:\n"
     "            f_hz = FI_SLOPE * FI_RHEO * (d_fi / (self.fire_th * self.th_gain) - 1.0).clamp(min=0.0)\n"
     "            gap_ms = (self._pc + 1 - self._last_fire[0]).float() * PASS_MS\n"
     "            rep = rep * (gap_ms * f_hz >= 1000.0).float()\n",
     "        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n"
     "        if self._complete_any:           # 入力が無いところは予測が足しになる（撃たせはしない）。版52 B\n"
     "            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th * self.th_gain) & (x_all > 0)\n"
     "            rep = torch.where(fill, torch.ones_like(rep), rep)\n"),
]
BB = [("VERSION = 60          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (57, 58, 59)",
       "VERSION = 59          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (57, 58, 60)"),
      ("#  60  表現の体に f-I（撃ったあと 1/f(r) 待つ。【ヒト】Barzó・Szöts 2024 0.15Hz/pA・230pA）。自発は不応期だけ【み】10-11\n",
       "#  60  表現の体に f-I（撃ったあと 1/f(r) 待つ。【ヒト】Barzó・Szöts 2024 0.15Hz/pA・230pA）。自発は不応期だけ【み】10-11\n"
       "#  ⟲ 10-11 版60 の f-I を外して版59 に戻した（切片に電流を流した値の当てはめで、人の自然な反応の根拠が無い【み】）\n")]


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    import babble
    if babble.VERSION != 60:
        sys.exit(f"⚠️ babble.py が版60 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版60 の f-I を外し、版59 に戻した（torch_group2・babble）。不応期 3周はそのまま")
    print("   測る道具（measure_check2・v59a・v59b）は版59 でそのまま回る。measure_v60a は版60 用（回らない）")
