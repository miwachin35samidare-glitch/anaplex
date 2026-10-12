"""版58（コンテナの写し用。編集は claude/道具_0926/apply_v58.py と同じ）"""
import ast, pathlib, sys

G2 = [
    ("        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])\n",
     "        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])\n"
     "        # 版58：自発のくじは周ごと（1周 ＝ 【ヒト】シナプス1本の遅れ 1.1ms）。1回あたりは1周ぶんの確率にする\n"
     "        #   ⟲ 版57 まで1歩ぶんの確率で周ごと10回引き、相手に届く自発が置いた率の約10倍だった【測】measure_v57b\n"
     "        self.spont_p_pass = 1.0 - (1.0 - self.spont_p) ** (1.0 / self.recur)\n"),
    ("                                             generator=self.gen) < self.spont_p).float())\n",
     "                                             generator=self.gen) < self.spont_p_pass).float())   # 版58\n"),
]
BB = [("VERSION = 57          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()   # 版57 で海馬の抑制の体の配線を引き直した ⇒ 版56 までの state.pt は読めない",
       "VERSION = 58          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (57,)   # 版57 で海馬の抑制の体の配線を引き直した ⇒ 版56 までの state.pt は読めない（版58 は自発の確率だけ）"),
      ("#  57  海馬の抑制の体の興奮の相手 194 を錐体（遅れの列）から引く（194 の前提どおり）。版56 で抑制の体が止まった【み】10-10\n",
       "#  57  海馬の抑制の体の興奮の相手 194 を錐体（遅れの列）から引く（194 の前提どおり）。版56 で抑制の体が止まった【み】10-10\n"
       "#  58  自発のくじを周ごとのまま1周ぶんの確率に（1周 ＝ 人のシナプス1本の遅れ）。前は届く自発が約10倍【み】10-10\n")]


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
    print("✅ 版58 を当てた")
