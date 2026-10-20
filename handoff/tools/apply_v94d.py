"""版94d：視床の中心外側核 CL（4,096体）を素子に（網の領野として入れる）（版94c の上に当てる。印が合わなければ何も書かずに止まる）
  python3 apply_v94d.py
【み】10-19「まずニューロンを素子に直して」／ 10-20 19:37「イ」（PPN は版95 で内節・網様部と一緒に。先に入力がそろっているものから）
⟲ 版67〜94c の CL は「PPN からの和が M_PC_SYN 以上なら撃つ」決まりで、素子ではなかった（予測・誤差・自分の素子の型なし）
変えるもの（torch_brain.py）
  1 NAMES の後ろに "CL"（N_CL 4,096体のまま。⚠️ ハードの側のノブ）。素子の型は「視床」（CL は髄板内核 ＝ 視床）
  2 時間細胞・傾きの細胞なし・中継どうしの興奮なし（版61【マカク】と同じ）・抑制の体 27.5%（【ヒト】視床の抑制性 25〜30%。CL の値ではない）
  3 撃つ閾値は M_PC_SYN 2.7 のまま（前の決まりの閾値を素子の閾値へ写す。【計算】ノブ）・入口の重さ 1
  4 PPN → CL のシナプスの受け手を網の CL の表現の体に（学びは網の発火で。前は CL の決まりの発火で）
  5 CL → 皮質 は1歩前の CL の表現の発火から（ほかの領野どうしと同じ1歩。⟲ 前は同じ歩）
⚠️ 数値は合わせない。⚠️ 網の辺が増える（CL の表現・誤差 8,192体 × 抑制の相手 約2,500 ⇒ 約2,000万本）⇒ arkhe で VRAM を測る
"""
import ast, pathlib, sys

TB = [
    ('         "上丘") + NUC                                              # 版43b ／ 版94：中脳の核\n',
     '         "上丘") + NUC + ("CL",)                                    # 版43b ／ 版94：中脳の核 ／ 版94d：CL\n'),
    ('"側坐核": "側坐核", "上丘": "上丘", **{k: k for k in NUC}}',
     '"側坐核": "側坐核", "上丘": "上丘", "CL": "視床", **{k: k for k in NUC}}'),
    ('N_PPN, N_CL = 512, 4096            # ⚠️ ハードの側のノブ\n',
     'N_PPN, N_CL = 512, 4096            # ⚠️ ハードの側のノブ\n'
     'N_NEW["CL"] = N_CL                 # 版94d：CL は網の領野\n'),
    ('FIRE_TH_THAL_SYN = 0.09',
     'FIRE_TH["CL"] = M_PC_SYN           # 版94d：前の決まりの閾値を素子の閾値へ（【計算】ノブ）\n'
     'X_GAIN["CL"] = 1.0\n'
     'FIRE_TH_THAL_SYN = 0.09'),
    ('TGT_BY_FIRE = TGT_BY_FIRE + NUC   # 版94\n',
     'TGT_BY_FIRE = TGT_BY_FIRE + NUC + ("CL",)   # 版94 ／ 版94d\n'),
    ('                          **{k: 0 for k in NUC})       # 版94：中脳の核は細胞体へだけ\n',
     '                          **{k: 0 for k in NUC}, CL=0)   # 版94：中脳の核は細胞体へだけ ／ 版94d：CL も\n'),
    ('            n_cell=(0 if nm in NUC else n_cell), n_ramp=(0 if nm in NUC else 8),\n',
     '            n_cell=(0 if nm in NUC + ("CL",) else n_cell), n_ramp=(0 if nm in NUC + ("CL",) else 8),   # 版94d：CL も\n'),
    ('            relay_exc=(RELAY_EXC if nm == "視床" else 0 if nm in NUC else None),',
     '            relay_exc=(RELAY_EXC if nm in ("視床", "CL") else 0 if nm in NUC else None),'),
    ('            inh_frac=(0.275 if nm in ("視床", "背内側視床") else 0.075',
     '            inh_frac=(0.275 if nm in ("視床", "背内側視床", "CL") else 0.075'),
    ('                self.bu["PPN→CL"] = _BUSyn(self.cl_ppn.reshape(-1), ar(N_CL).repeat_interleave(K_PC), N_CL, -1, stp_path, dev, gb)\n',
     '                self.bu["PPN→CL"] = _BUSyn(self.cl_ppn.reshape(-1), ar(N_CL).repeat_interleave(K_PC), N_CL,\n'
     '                                          self.off_ch[NAMES.index("CL")], stp_path, dev, gb)   # 版94d：受け手は網の CL\n'),
    ('            self._bo["視床下核→黒質"] + self._bo["前頭前野→黒質"] + self._bo["上丘→黒質"] + self._bo["PPN→黒質"]])\n',
     '            self._bo["視床下核→黒質"] + self._bo["前頭前野→黒質"] + self._bo["上丘→黒質"] + self._bo["PPN→黒質"],\n'
     '            self._bo["PPN→CL"]])                                                                              # 版94d：CL\n'),
    ('            cl = (self._bo["PPN→CL"] >= M_PC_SYN).float()      # 版70・71・75（PPN のくじは表の前で引いた）\n',
     '            cl = self.cl_prev                                    # 版94d：CL は網の領野（素子）。1歩前の表現の発火\n'),
    ('            self.but.learn(anyr, self.cl_spk if "PPN→CL" in self.bu else None, d_bu)   # 版75：表で1回\n',
     '            self.but.learn(anyr, None, d_bu)   # 版75：表で1回 ／ 版94d：PPN → CL の受け手も網の発火\n'),
    ('        for nm in NUC:\n'
     '            self.nuc[nm] = cut(nm, rep)\n',
     '        for nm in NUC:\n'
     '            self.nuc[nm] = cut(nm, rep)\n'
     '        self.cl_prev = cut("CL", rep)                 # 版94d\n'),
    ('            self.cl_spk = torch.zeros(N_CL, device=dev)\n',
     '            self.cl_spk = torch.zeros(N_CL, device=dev)\n'
     '            self.cl_prev = torch.zeros(N_CL, device=dev)          # 版94d\n'),
]
BB = [
    ("#  94c 前のスレッドの核の値を戻す",
     "#  94d CL を素子に（網の領野、型「視床」・閾値 M_PC_SYN のまま）【み】10-20「イ」\n#  94c 前のスレッドの核の値を戻す"),
]


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    if "94c" not in pathlib.Path("babble.py").read_text():
        sys.exit("⚠️ babble.py が版94c ではない（apply_v94c.py のあとに当てる）")
    out = [edit("torch_brain.py", TB), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版94d を当てた（CL を素子に：網の領野・型「視床」）", flush=True)
