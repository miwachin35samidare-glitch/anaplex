"""版59（コンテナの写し用。編集は claude/道具_0926/apply_v59.py と同じ）"""
import ast, pathlib, sys

G2 = [
    ("LONG_TAU = 3600 * 1000.0 / DT_MS\n",
     "LONG_TAU = 3600 * 1000.0 / DT_MS\n"
     "# 版59：不応期（周の数）。【ヒト】起きている人の内側側頭葉の単一細胞で 3ms 以内の間隔は 0.10%（Ison 2011）⇒ 3ms ÷ 1.102ms ≒ 3周\n"
     "REFRACT_PASSES = 3\n"),
    ("        inh_spk = (d_inh > INH_TH).float()\n"
     "        if learn:                                  # 細胞自身がこの歩に撃ったか（恒常性の観測）\n"
     "            fired = torch.zeros(self.n_recv_all, device=self.dev)\n"
     "            fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = err, rep, inh_spk\n",
     "        inh_spk = (d_inh > INH_TH).float()\n"
     "        # 版59：不応期。撃った周から REFRACT_PASSES 周たつまで撃たない（歩をまたいでも数える）\n"
     "        if first:\n"
     "            self._learn_step = learn\n"
     "            if getattr(self, \"_last_fire\", None) is None:\n"
     "                self._pc = 0\n"
     "                self._last_fire = [torch.full(v.shape, -10 ** 6, dtype=torch.long, device=self.dev)\n"
     "                                   for v in (rep, err, inh_spk)]\n"
     "        self._pc += 1\n"
     "        rep, err, inh_spk = [v * ((self._pc - lf) >= REFRACT_PASSES).float()\n"
     "                             for v, lf in zip((rep, err, inh_spk), self._last_fire)]\n"
     "        self._last_fire = [torch.where(v > 0, torch.full_like(lf, self._pc), lf)\n"
     "                           for v, lf in zip((rep, err, inh_spk), self._last_fire)]\n"
     "        # 版59：この歩のどこかで撃ったか（周の発火の重ね）\n"
     "        self._any = [rep, err, inh_spk] if first else [torch.maximum(a, v) for a, v in zip(self._any, (rep, err, inh_spk))]\n"
     "        if advance and getattr(self, \"_learn_step\", False):   # 細胞自身がこの歩に撃ったか（恒常性の観測。版59：歩のどこか）\n"
     "            fired = torch.zeros(self.n_recv_all, device=self.dev)\n"
     "            fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any\n"),
    ("        if advance:\n"
     "            # 版50：行（領野ごとに 誤差・表現・抑制）と同じ並びで足す\n",
     "        if advance:\n"
     "            rep, err, inh_spk = self._any     # 版59：最後の周ではなく、この歩のどこかで撃ったか\n"
     "            # 版50：行（領野ごとに 誤差・表現・抑制）と同じ並びで足す\n"),
]
BB = [("VERSION = 58          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (57,)",
       "VERSION = 59          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (57, 58)"),
      ("#  58  自発のくじを周ごとのまま1周ぶんの確率に（1周 ＝ 人のシナプス1本の遅れ）。前は届く自発が約10倍【み】10-10\n",
       "#  58  自発のくじを周ごとのまま1周ぶんの確率に（1周 ＝ 人のシナプス1本の遅れ）。前は届く自発が約10倍【み】10-10\n"
       "#  59  不応期 3周（【ヒト】Ison 2011：3ms 以内 0.10%）。履歴・時間細胞・返す発火・恒常性は「この歩のどこかで撃ったか」【み】10-10\n")]


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版59 を当てた")
