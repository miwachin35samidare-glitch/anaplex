"""版95：基底核を組む（線条体の型・外節・内節・網様部・腹側淡蒼球・CM/Pf・PPN を素子に）＋ 核どうしを周の遅れに
（~/anaplex で。版94d の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v95.py
【み】10-21 23:37「いいよ、やってみても」（引き継ぎ_版93 §32）／ 10-22 0:33「聞くまでもなくi0やん」
値の出どころ：docs/設計案_版95の組み表.md（1つの表。下の NUC_LINES・N95・NUC_VINF・NUC_TAIL に写した）
入れるもの
  1 領野：尾状核 986・被殻 1,302（型「線条体」）・外節・内節・網様部・腹側淡蒼球 各 256・PPN 512（核）・CM・Pf 各 64（型「視床」）
     側坐核は型「線条体」へ（⟲ 版43a〜94 は皮質の型）／ 境は型「内節」へ（内節の一部【み】10-18 0:15）
  2 線条体の型：抑制の体 22.5%【ヒト】20〜25% ／ 中の興奮の横なし ／ 受け手1体の相手 3,800【ヒト】/ 時間細胞なし
  3 核どうしの線を周（1.1ms）の遅れに（⟲ 版94 は1歩の遅れ。10-16【み】「核は周回ごと。核どうしの遅れは周の数」）
     torch_group2.NucNet：核どうし 1周・線条体から 2周【み】10-16 20:20。興奮は受け手の表現の駆動へ（x と同じ入り方）・
     抑制は受け手の表現・誤差の抑制へ（÷(1＋抑制)）。興奮の NMDA の尾は周ごと（版93 の式）
     学び：決め A【み】10-15 23:51（送り手の強い入力のあとに強まる × 門・受け手の発火は条件にしない。量・強い入力の数・上限はノブ【C】）
  4 領野 → 線条体（下からのシナプス・学ぶ）：運動 → 被殻・前頭前野 → 尾状核 各 2,700本 ／ CM → 被殻・Pf → 尾状核 各 1,100本
     【ヒト】棘 3,800（10-15 23:39）×【マカク】皮質 51.9%・視床 21.7%（Raju 2008）【計算】／ 送り手の切片より多い分は同じ体を何度も数える
  5 CM・Pf の入力（運動 → CM・前頭前野 → Pf、K_NEW 8）／ 核 → 領野の抑制（網様部 → 上丘・内節 → CM・網様部 → Pf・腹側淡蒼球 → 背内側視床、64体）
  6 PPN：くじを消して素子に（i0 で自分で撃つ【み】10-22）。PPN → CL は PPN の表現の発火（1歩前）から
⚠️ 数値は合わせない（10-18 22:08【み】全部組んでから）。i0 は版93・teacher_v95 の撃ち方 B の値の写し・PPN は 23.2Hz からの【計算】
⚠️ 側坐核は「中の興奮の横なし」にしたので、前頭前野 → 側坐核（調整する側・ext）は届かなくなる（版95 の作りの限り【C】）
"""
import ast, pathlib, re, sys

# ───────── torch_brain.py ─────────
TB = [
    ('NUC = ("視床下核", "境", "外側手綱核", "吻内側被蓋核", "黒質")   # 版94：中脳の核（網の領野・素子）\n',
     'NUC = ("視床下核", "境", "外側手綱核", "吻内側被蓋核", "黒質",   # 版94：中脳の核（網の領野・素子）\n'
     '       "外節", "内節", "網様部", "腹側淡蒼球", "PPN")             # 版95：基底核の核・PPN\n'
     'STR = ("尾状核", "被殻")              # 版95：背側の線条体（型「線条体」。側坐核も同じ型）\n'
     'THAL_IL = ("CM", "Pf")                # 版95：髄板内核（型「視床」）\n'
     'STR_ALL = STR + ("側坐核",)\n'),
    ('              "側坐核": "側坐核", "上丘": "上丘", "CL": "視床", **{k: k for k in NUC}}   # 版94b：核ごとの型（⟲ 版94 は5核で「中脳」1つ）\n'
     'UNBAKED_OK = NUC   # 版94b：率を人に合わせてから録って焼くまで prior_v1（⟲ 版94 の列で焼いた prior_v2_中脳 は使わない）\n',
     '              "側坐核": "側坐核", "上丘": "上丘", "CL": "視床", **{k: k for k in NUC}}   # 版94b：核ごとの型（⟲ 版94 は5核で「中脳」1つ）\n'
     'PRIOR_TYPE.update({"境": "内節", "側坐核": "線条体", **{k: "線条体" for k in STR}, "CM": "視床", "Pf": "視床"})   # 版95\n'
     'UNBAKED_OK = tuple(sorted({PRIOR_TYPE[k] for k in NUC} | {"線条体"}))   # 版94b ／ 版95：焼くまで prior_v1\n'),
    ('         "上丘") + NUC + ("CL",)                                    # 版43b ／ 版94：中脳の核 ／ 版94d：CL\n',
     '         "上丘") + NUC + ("CL",) + STR + THAL_IL                    # 版43b ／ 版94：中脳の核 ／ 版94d：CL ／ 版95\n'),
    ('            {"視床下核": 0.0337, "境": 0.0616, "外側手綱核": 0.0117, "吻内側被蓋核": 0.02699, "黒質": 0.00917}.items()}\n',
     '            {"視床下核": 0.0337, "境": 0.0616, "外側手綱核": 0.0117, "吻内側被蓋核": 0.02699, "黒質": 0.00917,\n'
     '             # 版95：teacher_v95（撃ち方 B の閉じた輪で率に合わせた i0。⚠️ 写し・合わせ直す）【マカク】自分で撃つ Kita 2005・10-17 0:54\n'
     '             "外節": 0.0896, "内節": 0.1050, "網様部": 0.0811,\n'
     '             "腹側淡蒼球": 0.0}.items()}   # 版95：腹側淡蒼球は自分で撃つ根拠を調べていない ⇒ i0 なし（無い性質は足さない【み】10-16 23:33）\n'
     '# 版95：PPN は 23.2Hz【ヒト】で自分で撃つ形の行き着く先【計算】（間隔 T で閾値に届く：1 ÷ (1 − e^(−T/τm))）【み】10-22「i0」\n'
     'NUC_VINF["PPN"] = 1.0 / (1.0 - __import__("math").exp(-(1000.0 / 23.2) / 40.0))\n'),
    ('            "吻内側被蓋核": ((75.5, 1.0), (75.5, 0.0), 0.5), "黒質": ((75.5, 1.0), (75.5, 0.0), 0.5)}\n',
     '            "吻内側被蓋核": ((75.5, 1.0), (75.5, 0.0), 0.5), "黒質": ((75.5, 1.0), (75.5, 0.0), 0.5),\n'
     '            # 版95：外節・内節は淡蒼球の決め【み】10-17 12:20 ／ 網様部・腹側淡蒼球は値なし ⇒ 淡蒼球を当てる【C】／ PPN は尾なし【C】\n'
     '            "外節": ((21.0, 0.5), (1340.0, 0.5), 0.4), "内節": ((21.0, 0.5), (1340.0, 0.5), 0.4),\n'
     '            "網様部": ((21.0, 0.5), (1340.0, 0.5), 0.4), "腹側淡蒼球": ((21.0, 0.5), (1340.0, 0.5), 0.4),\n'
     '            "PPN": ((10.0, 1.0), (10.0, 0.0), 0.0)}\n'),
    ('N_NEW["CL"] = N_CL                 # 版94d：CL は網の領野\n',
     'N_NEW["CL"] = N_CL                 # 版94d：CL は網の領野\n'
     '# 版95：基底核（設計案_版95の組み表 §1）\n'
     '#   尾状核 986・被殻 1,302【ヒト】Wegiel 2014 × 片側 100万体あたり 20 ／ 基底核の核 256【み】10-17 10:11 ／ CM・Pf 64（下限）\n'
     'N_NEW.update({"PPN": N_PPN, "尾状核": 986, "被殻": 1302, "CM": 64, "Pf": 64})\n'
     'N_PARTNER_STR = 3800              # 線条体の受け手1体の相手【ヒト】棘 約3,800（PLOS Comput Biol 2025）【み】10-15 23:39\n'
     'INH_FRAC_STR = 0.225              # 線条体の介在【ヒト】20〜25%（Cicchetti 2000）の真ん中【C】\n'
     'K_CX_STR, K_TH_STR = 2700, 1100   # 皮質・視床から【マカク】Raju 2008 の比 × 3,800【計算】\n'),
    ('                          **{k: 0 for k in NUC}, CL=0)   # 版94：中脳の核は細胞体へだけ ／ 版94d：CL も\n',
     '                          **{k: 0 for k in NUC}, CL=0,   # 版94：中脳の核は細胞体へだけ ／ 版94d：CL も\n'
     '                          **{k: 0 for k in STR + THAL_IL})   # 版95：背側の線条体・CM・Pf も細胞体へだけ\n'),
    ('            n_partner=N_PARTNER.get(nm, n_partner), stp_path=stp_path, dev=dev, seed=seed + i,\n',
     '            n_partner=(N_PARTNER_STR if nm in STR_ALL else N_PARTNER.get(nm, n_partner)), stp_path=stp_path, dev=dev, seed=seed + i,\n'),
    ('            n_cell=(0 if nm in NUC + ("CL",) else n_cell), n_ramp=(0 if nm in NUC + ("CL",) else 8),   # 版94d：CL も\n',
     '            n_cell=(0 if nm in NUC + ("CL",) + STR_ALL + THAL_IL else n_cell),   # 版94d：CL も ／ 版95：線条体・CM・Pf も\n'
     '            n_ramp=(0 if nm in NUC + ("CL",) + STR_ALL + THAL_IL else 8),\n'),
    ('            inh_frac=(0.275 if nm in ("視床", "背内側視床", "CL") else 0.075 if nm == "視床下核" else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n',
     '            inh_frac=(0.275 if nm in ("視床", "背内側視床", "CL") + THAL_IL else 0.075 if nm == "視床下核"\n'
     '                      else INH_FRAC_STR if nm in STR_ALL else 0.16),   # 【ヒト】視床の抑制性 25〜30% ／ 版95：線条体 22.5%\n'),
    ('            relay_exc=(RELAY_EXC if nm in ("視床", "CL") else 0 if nm in NUC else None),   # 版51 ／ 版94：核の中の興奮の横なし【C】\n',
     '            relay_exc=(RELAY_EXC if nm in ("視床", "CL") + THAL_IL else 0 if nm in NUC + STR_ALL else None),   # 版51 ／ 版94：核の中の興奮の横なし【C】／ 版95：線条体も【み】10-16 22:55\n'),
    ('TGT_BY_FIRE = TGT_BY_FIRE + NUC + ("CL",)   # 版94 ／ 版94d\n',
     'TGT_BY_FIRE = TGT_BY_FIRE + NUC + ("CL",) + STR + THAL_IL   # 版94 ／ 版94d ／ 版95\n'),
    # BU の道：核どうしの5本は NucNet へ（周の遅れ）。足すのは 領野 → 線条体・CM・Pf
    ('            ("前頭前野→視床下核", "視床下核"), ("視床下核→境", "境"), ("境→外側手綱核", "外側手綱核"),\n'
     '            ("外側手綱核→吻内側被蓋核", "吻内側被蓋核"), ("視床下核→黒質", "黒質"), ("前頭前野→黒質", "黒質"),\n'
     '            ("上丘→黒質", "黒質"), ("PPN→黒質", "黒質"))\n',
     '            ("前頭前野→視床下核", "視床下核"), ("前頭前野→黒質", "黒質"), ("上丘→黒質", "黒質"),\n'
     '            # 版95：核どうし（視床下核 → 境・境 → 外側手綱核・外側手綱核 → 吻内側被蓋核・視床下核 → 黒質・PPN → 黒質）は NucNet へ（周の遅れ）\n'
     '            ("運動→被殻", "被殻"), ("前頭前野→尾状核", "尾状核"), ("CM→被殻", "被殻"), ("Pf→尾状核", "尾状核"),\n'
     '            ("運動→CM", "CM"), ("前頭前野→Pf", "Pf"))\n'),
    ('                      "前頭前野→視床下核": self.p_pfc2stn, "視床下核→境": self.p_stn2gpb, "境→外側手綱核": self.p_gpb2lhb,\n'
     '                      "外側手綱核→吻内側被蓋核": self.p_lhb2rmtg, "視床下核→黒質": self.p_stn2sn,\n'
     '                      "前頭前野→黒質": self.p_pfc2sn, "上丘→黒質": self.p_sc2sn, "PPN→黒質": self.p_ppn2sn}   # 版94\n',
     '                      "前頭前野→視床下核": self.p_pfc2stn, "前頭前野→黒質": self.p_pfc2sn, "上丘→黒質": self.p_sc2sn,   # 版94\n'
     '                      **self.p95}                                                                   # 版95\n'),
    # 核への抑制の線（1歩ごと）：核どうしの2本は NucNet へ。核 → 領野の4本を足す
    ('            for key, (p, nm) in (("側坐核→境", (self.p_nac2gpb, "境")), ("吻内側被蓋核→黒質", (self.p_rmtg2sn, "黒質"))):\n',
     '            # 版95：核どうし（側坐核 → 境・吻内側被蓋核 → 黒質）は NucNet へ（周の遅れ）。ここは 核 → 領野（1歩）\n'
     '            for key, (p, nm) in ((k_, (self.p95_inh[k_], t_)) for k_, (_s, t_) in INH_TO_AREA.items()):\n'),
    ('            self.g.ext_ihn = torch.zeros(self.g.n_recv_all, device=dev)\n'
     '            self.bu_pack()                                           # 版75：道をまとめた表\n',
     '            self.g.ext_ihn = torch.zeros(self.g.n_recv_all, device=dev)\n'
     '            self._nucnet_build(g)                                    # 版95：核どうしの線（周の遅れ）\n'
     '            self.bu_pack()                                           # 版75：道をまとめた表\n'),
    # step：PPN のくじを消す
    ('        self._ppn10 = (torch.rand(self.g.recur, N_PPN, device=self.dev, generator=getattr(self, "cl_gen", self.gen))\n'
     '                       < P_PPN_PASS).float()\n'
     '        self._ppn = self._ppn10.amax(0)\n',
     '        # 版95：PPN は網の素子（⟲ 版93〜94d のくじ）。1歩前の表現の発火\n'
     '        self._ppn = self.nuc["PPN"]\n'),
    ('                   # 版94：中脳の核（1歩前の発火）\n'
     '                   "前頭前野→視床下核": self.pred["前頭前野"], "視床下核→境": self.nuc["視床下核"],\n'
     '                   "境→外側手綱核": self.nuc["境"], "外側手綱核→吻内側被蓋核": self.nuc["外側手綱核"],\n'
     '                   "視床下核→黒質": self.nuc["視床下核"], "前頭前野→黒質": self.pred["前頭前野"],\n'
     '                   "上丘→黒質": self.up["上丘"], "PPN→黒質": self._ppn}\n',
     '                   # 版94：中脳の核（1歩前の発火）\n'
     '                   "前頭前野→視床下核": self.pred["前頭前野"], "前頭前野→黒質": self.pred["前頭前野"],\n'
     '                   "上丘→黒質": self.up["上丘"],\n'
     '                   # 版95：領野 → 線条体・CM・Pf（1歩前の表現の発火）\n'
     '                   "運動→被殻": self.motor_rep, "前頭前野→尾状核": self.pred["前頭前野"],\n'
     '                   "CM→被殻": self.cmpf["CM"], "Pf→尾状核": self.cmpf["Pf"],\n'
     '                   "運動→CM": self.motor_rep, "前頭前野→Pf": self.pred["前頭前野"]}\n'),
    ('            # 版94：中脳の核（NUC の順）\n'
     '            self._bo["前頭前野→視床下核"], self._bo["視床下核→境"], self._bo["境→外側手綱核"],\n'
     '            self._bo["外側手綱核→吻内側被蓋核"],\n'
     '            self._bo["視床下核→黒質"] + self._bo["前頭前野→黒質"] + self._bo["上丘→黒質"] + self._bo["PPN→黒質"],\n'
     '            self._bo["PPN→CL"]])                                                                              # 版94d：CL\n',
     '            # 版94：中脳の核 ／ 版95：基底核の核・PPN（NUC の順。核どうしは NucNet で周ごとに入る）\n'
     '            *[sum((self._bo[k_] for k_ in NUC_BU.get(nm_, ())), torch.zeros(self.n[nm_], device=self.dev)) for nm_ in NUC],\n'
     '            self._bo["PPN→CL"],                                                                               # 版94d：CL\n'
     '            self._bo["前頭前野→尾状核"] + self._bo["Pf→尾状核"], self._bo["運動→被殻"] + self._bo["CM→被殻"],    # 版95：尾状核・被殻\n'
     '            self._bo["運動→CM"], self._bo["前頭前野→Pf"]])                                                    # 版95：CM・Pf\n'),
    ('            for key, v in (("側坐核→境", self.nac_out), ("吻内側被蓋核→黒質", self.nuc["吻内側被蓋核"])):\n',
     '            for key, (sn_, _t) in INH_TO_AREA.items():          # 版95：核 → 領野（1歩前の表現の発火）\n'
     '                v = self.nuc[sn_]\n'),
    ('            self.but.learn(anyr, None, d_bu)   # 版75：表で1回 ／ 版94d：PPN → CL の受け手も網の発火\n',
     '            self.but.learn(anyr, None, d_bu)   # 版75：表で1回 ／ 版94d：PPN → CL の受け手も網の発火\n'
     '            if getattr(self.g, "nucnet", None) is not None:      # 版95：核どうしの線の学び（決め A）\n'
     '                self.g.nucnet.learn(torch.cat([self.g._any[0], self.g._any[1]]), d_bu)\n'),
    ('        self.cl_prev = cut("CL", rep)                 # 版94d\n',
     '        self.cl_prev = cut("CL", rep)                 # 版94d\n'
     '        self.cmpf = {k_: cut(k_, rep) for k_ in THAL_IL}   # 版95：CM・Pf は表現を送る（視床の中継）\n'),
    ('        self.nuc = {k: torch.zeros(nn[k], device=dev) for k in NUC}   # 1歩前の核の表現の発火\n',
     '        self.nuc = {k: torch.zeros(nn[k], device=dev) for k in NUC}   # 1歩前の核の表現の発火\n'
     '        self.cmpf = {k: torch.zeros(nn[k], device=dev) for k in THAL_IL}   # 版95\n'
     '        # 版95：領野 → 線条体・CM・Pf（下からのシナプス）と、核 → 領野の抑制（1歩）\n'
     '        def rpr(ns, nt, k):                     # 受け手1体が k 本（送り手の切片より多ければ同じ体を何度も数える）\n'
     '            if k <= ns:\n'
     '                return rpf(ns, nt, k)\n'
     '            src = torch.randint(0, ns, (nt, k), device=dev, generator=g)\n'
     '            return (src.reshape(-1), torch.arange(nt, device=dev).repeat_interleave(k), torch.full((nt,), float(k), device=dev))\n'
     '        self.p95 = {"運動→被殻": rpr(nn["運動"], nn["被殻"], K_CX_STR), "前頭前野→尾状核": rpr(nn["前頭前野"], nn["尾状核"], K_CX_STR),\n'
     '                    "CM→被殻": rpr(nn["CM"], nn["被殻"], K_TH_STR), "Pf→尾状核": rpr(nn["Pf"], nn["尾状核"], K_TH_STR),\n'
     '                    "運動→CM": rp(nn["運動"], nn["CM"]), "前頭前野→Pf": rp(nn["前頭前野"], nn["Pf"])}\n'
     '        self.p95_inh = {k_: rpf(nn[s_], nn[t_], K_NUC) for k_, (s_, t_) in INH_TO_AREA.items()}\n'),
]

# NucNet を組む（Brain のメソッド）と、線の表（モジュールの頭）
TB_TABLE = '''
# ───── 版95：核どうしの線（設計案_版95の組み表 §2。周の遅れ【み】10-16 20:20）─────
#   (送り手, 受け手, 符号, 受け手1体の送り手の数, 線を出す送り手の割合, 遅れ（周）, 送る体)
#   送る体："表誤" ＝ 表現と誤差（側坐核 → 境 の決め【み】10-18 13:36 に合わせた【C】）／ "表" ＝ 表現
#   線を出す割合【マカク】Sato 2000a・b（10-17 0:51）／ 線条体 → 淡蒼球 100体【マカク】Percheron（10-16 20:20）／ 核どうし 64体【み】10-17
#   ⚠️ D1・D2 は分けない（霊長類では細胞で分かれきっていない：全部が外節へ・82% が内節・網様部へも Lévesque & Parent 2005）【C】
STR_D = ("尾状核", "被殻")      # 背側の線条体（送り手を1つの池に）
NUC_LINES = [
    (STR_D, "外節", -1, 100, 1.00, 2, "表誤"), (STR_D, "内節", -1, 100, 0.82, 2, "表誤"), (STR_D, "網様部", -1, 100, 0.82, 2, "表誤"),
    (("側坐核",), "境", -1, 64, 1.00, 2, "表誤"), (("側坐核",), "腹側淡蒼球", -1, 100, 1.00, 2, "表誤"),
    (("外節",), "視床下核", -1, 64, 0.842, 1, "表"), (("外節",), "内節", -1, 64, 0.316, 1, "表"), (("外節",), "境", -1, 64, 0.316, 1, "表"),
    (("外節",), "網様部", -1, 64, 0.658, 1, "表"), (("外節",), "外節", -1, 64, 1.00, 1, "表"),
    (("視床下核",), "外節", +1, 64, 0.827, 1, "表"), (("視床下核",), "内節", +1, 64, 0.693, 1, "表"), (("視床下核",), "境", +1, 64, 0.693, 1, "表"),
    (("視床下核",), "網様部", +1, 64, 0.240, 1, "表"), (("視床下核",), "黒質", +1, 64, 1.00, 1, "表"), (("視床下核",), "腹側淡蒼球", +1, 64, 1.00, 1, "表"),
    (("境",), "外側手綱核", +1, 64, 1.00, 1, "表"), (("腹側淡蒼球",), "外側手綱核", -1, 64, 1.00, 1, "表"),
    (("外側手綱核",), "吻内側被蓋核", +1, 64, 1.00, 1, "表"), (("吻内側被蓋核",), "黒質", -1, 64, 1.00, 1, "表"),
    (("PPN",), "黒質", +1, 64, 1.00, 1, "表"), (("内節",), "PPN", -1, 64, 1.00, 1, "表"), (("網様部",), "PPN", -1, 64, 1.00, 1, "表"),
    # 線条体の棘細胞どうし（抑制の側枝【み】10-16 22:55。本数は人・霊長類なし ⇒ 64【C】）
    (("尾状核",), "尾状核", -1, 64, 1.00, 1, "表"), (("被殻",), "被殻", -1, 64, 1.00, 1, "表"), (("側坐核",), "側坐核", -1, 64, 1.00, 1, "表"),
]
# 抑制の強さの比（受け手ごと）【リスザル】Shink & Smith 1995 の樹状突起の終末：線条体 ＞80%・外節 ＜5%（10-17 0:51）。
#   受け手1体の抑制の合計は「64本 × W_INH_BU」（版94c の核の抑制の線と同じ単位）を、比で分ける【C】
INH_SHARE = {"外節": {STR_D: 0.80, ("外節",): 0.05}, "内節": {STR_D: 0.80, ("外節",): 0.05},
             "境": {("側坐核",): 0.80, ("外節",): 0.05}, "網様部": {STR_D: 0.80, ("外節",): 0.05},
             "PPN": {("内節",): 0.5, ("網様部",): 0.5}}
# 核 → 領野の抑制（1歩。下からの抑制の線、学ばない）【み】10-16 23:12・20:25 ／ 本数の決め無し ⇒ 64【C】
INH_TO_AREA = {"網様部→上丘": ("網様部", "上丘"), "内節→CM": ("内節", "CM"), "網様部→Pf": ("網様部", "Pf"),
               "腹側淡蒼球→背内側視床": ("腹側淡蒼球", "背内側視床")}
# 核へ領野から来る下からのシナプス（1歩）
NUC_BU = {"視床下核": ("前頭前野→視床下核",), "黒質": ("前頭前野→黒質", "上丘→黒質")}
'''

TB_BUILD = '''
    def _nucnet_build(self, g):
        """版95：核どうしの線を網の NucNet に（周の遅れ）。値は NUC_LINES・INH_SHARE（設計案_版95の組み表）"""
        import math as _m
        from torch_group2 import NucNet
        dev, nn, gg = self.dev, self.n, self.g
        n_ch = gg.n_ch_all
        ch0 = lambda nm: int(self.off_ch[NAMES.index(nm)])
        sig = _m.sqrt(2.0 * _m.log(CL_W_MEAN / CL_W_MED))
        S_, D_, W_, G_, L_ = [], [], [], [], []
        for srcs, tgt, sg, k, frac, dl, kind in NUC_LINES:
            pool = []
            for s in srcs:                                     # 線を出す送り手（体ごと）・送る体
                cells = torch.randperm(nn[s], device=dev, generator=g)[:max(1, int(round(nn[s] * frac)))] + ch0(s)
                pool.append(cells)
                if kind == "表誤":
                    pool.append(cells + n_ch)                  # 誤差の体（送り手の並びは [表現 | 誤差]）
            pool = torch.cat(pool)
            nt = nn[tgt]
            kk = min(k, len(pool))
            if srcs == (tgt,):                                 # 自分の核の中：自分からは取らない
                pass
            src = torch.stack([pool[torch.randperm(len(pool), device=dev, generator=g)[:kk]] for _ in range(nt)]).reshape(-1)
            dst = (torch.arange(nt, device=dev) + ch0(tgt)).repeat_interleave(kk)
            if srcs == (tgt,):
                keep = (src % n_ch) != dst
                src, dst = src[keep], dst[keep]
            if sg > 0:
                w = torch.exp(_m.log(CL_W_MED) + sig * torch.randn(len(src), device=dev, generator=g))
            else:
                sh = INH_SHARE.get(tgt, {}).get(srcs, None)
                tot = sum(INH_SHARE[tgt].values()) if sh is not None else None
                w = torch.full((len(src),), W_INH_BU * K_NUC * ((sh / tot) if sh is not None else 1.0) / kk, device=dev)
            S_.append(src); D_.append(dst); W_.append(w)
            G_.append(torch.full((len(src),), sg, dtype=torch.int8, device=dev))
            L_.append(torch.full((len(src),), dl, dtype=torch.int8, device=dev))
        # 興奮の NMDA の尾（周ごと。版93 の式）：受け手の核の体ごと
        tl = {k_: torch.zeros(n_ch, device=dev) for k_ in ("a1", "m1", "a2", "m2", "f")}
        tl["a1"].fill_(0.0); tl["a2"].fill_(0.0)
        for k_ in NUC:
            (t1, w1), (t2, w2), f_ = NUC_TAIL[k_]
            o, n_ = ch0(k_), nn[k_]
            for kk_, v in (("a1", _m.exp(-1.067 / t1)), ("m1", w1), ("a2", _m.exp(-1.067 / t2)), ("m2", w2), ("f", f_)):
                tl[kk_][o:o + n_] = float(v)
        gg.nucnet = NucNet(torch.cat(S_), torch.cat(D_), torch.cat(W_), torch.cat(G_), torch.cat(L_),
                           n_ch, gg.n_recv_all, gg.rep_i, gg.err_i, tl, dev)
'''

# ───────── torch_group2.py ─────────
TG = [
    ('                fire_th, th_gain, complete, pred_boost, spont_u, spont_p_pass, ext_ihn, n_recv: int):\n',
     '                fire_th, th_gain, complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_recv: int):\n', 1),
    ('    if ext_ihn is not None:                  # 版94：外からの抑制の線（中脳の核へ）。学ばない\n'
     '        ihn = ihn + ext_ihn\n',
     '    if ext_ihn is not None:                  # 版94：外からの抑制の線（中脳の核へ）。学ばない\n'
     '        ihn = ihn + ext_ihn\n'
     '    if nuc_ihn is not None:                  # 版95：核どうしの抑制（周ごと。NucNet）\n'
     '        ihn = ihn + nuc_ihn\n', 1),
    ('    drive = ((exc - exc_mod)[rep_i] + x_all * x_gain) / (1.0 + ihn[rep_i])\n',
     '    xx = x_all if nuc_x is None else x_all + nuc_x   # 版95：核どうしの興奮（周ごと。x と同じ入り方。的には入れない）\n'
     '    drive = ((exc - exc_mod)[rep_i] + xx * x_gain) / (1.0 + ihn[rep_i])\n', 1),
    # _fast_body の1周目
    ('            X["pred_boost"], F["spont1"], X["spont_p_pass"], opt("ext_ihn"), self.n_recv_all)\n',
     '            X["pred_boost"], F["spont1"], X["spont_p_pass"], opt("ext_ihn"), *self._nuc_in(), self.n_recv_all)\n', 1),
    ('            _out_core, err_sig, exc, exc_mod, ihn, self.err_i, self.inh_i, opt("v_carry_inh"), rep,\n'
     '            S["lf0"], S["lf1"], S["lf2"], S["pc"], None, None, None)\n',
     '            _out_core, err_sig, exc, exc_mod, ihn, self.err_i, self.inh_i, opt("v_carry_inh"), rep,\n'
     '            S["lf0"], S["lf1"], S["lf2"], S["pc"], None, None, None)\n'
     '        if getattr(self, "nucnet", None) is not None:      # 版95：核どうしの線（次の周・その次の周へ）\n'
     '            self.nucnet.after(S, rep, err)\n', 1),
    # _gp_body（2周目から）
    ('            X["pred_boost"], S["spont"], X["spont_p_pass"], opt("ext_ihn"), self.n_recv_all)\n',
     '            X["pred_boost"], S["spont"], X["spont_p_pass"], opt("ext_ihn"), *self._nuc_in(), self.n_recv_all)\n', 1),
    ('            S["lf0"], S["lf1"], S["lf2"], S["pc"], S["any0"], S["any1"], S["any2"])\n',
     '            S["lf0"], S["lf1"], S["lf2"], S["pc"], S["any0"], S["any1"], S["any2"])\n'
     '        if getattr(self, "nucnet", None) is not None:      # 版95\n'
     '            self.nucnet.after(S, rep, err)\n', 1),
    # _pass（周ごと・eager）
    ('            spont_u, self.spont_p_pass, getattr(self, "ext_ihn", None), self.n_recv_all)\n',
     '            spont_u, self.spont_p_pass, getattr(self, "ext_ihn", None), *self._nuc_in(), self.n_recv_all)\n', 1),
    ('            rep, *self._last_fire, self._pc_t, *an)\n'
     '        self._dinh_last = d_inh\n',
     '            rep, *self._last_fire, self._pc_t, *an)\n'
     '        self._dinh_last = d_inh\n'
     '        if getattr(self, "nucnet", None) is not None:      # 版95\n'
     '            self.nucnet.after(self._gp_state(), rep, err)\n', 1),
    # 周をまたぐ置き場所
    ('        self._gS, self._gX, self._graphs, self._gpins = S, {}, {}, None\n',
     '        if getattr(self, "nucnet", None) is not None:      # 版95：核どうしの線の周ごとの置き場所\n'
     '            S.update(self.nucnet.state(self.n_ch_all, self.n_recv_all))\n'
     '        self._gS, self._gX, self._graphs, self._gpins = S, {}, {}, None\n', 1),
    # 撮ったグラフが読む配列（重さが置き換わったら撮り直す）
    ('                self.rate_ema, self.rate_long, self.scale_skip, self.is_hippo, F["E"])\n',
     '                self.rate_ema, self.rate_long, self.scale_skip, self.is_hippo, F["E"]) + self._nuc_pins()\n', 1),
    ('                self.within_index, self.within_inh_index, self._amp_buf if self.stp is not None else None)\n',
     '                self.within_index, self.within_inh_index, self._amp_buf if self.stp is not None else None) + self._nuc_pins()\n', 1),
]
TG_METH = '''
    def _nuc_in(self):
        """版95：この周に届く核どうしの興奮（受け手の体の駆動へ）・抑制（表現・誤差の行へ）。NucNet が無ければ None"""
        if getattr(self, "nucnet", None) is None:
            return None, None
        S = self._gp_state()
        return S["nuc_x"], S["nuc_ihn"]

    def _nuc_pins(self):
        nn_ = getattr(self, "nucnet", None)
        return () if nn_ is None else (nn_.w, nn_.src, nn_.dst)
'''
TG_CLASS = '''

class NucNet:
    """版95：核どうしの線（周 1.1ms ごとに届く。核どうし 1周・線条体から 2周）【み】10-16「核は周回ごと。核どうしの遅れは周の数」
    送り手 ＝ 網の表現・誤差の体（[表現 n_ch | 誤差 n_ch] の並び）。興奮は受け手の表現の駆動へ（x と同じ入り方・誤差の的には入れない）、
    抑制は受け手の表現・誤差の行の抑制へ（÷(1＋抑制)）。興奮には周ごとの NMDA の尾（版93 の式）
    学び（決め A【み】10-15 23:51）：送り手の強い入力（直近 NUC_A_WIN 歩のうち NUC_A_N 歩以上撃った）のあとに強まる × 門。
      受け手の発火は条件にしない。⚠️ 量 NUC_A_ETA・強い入力の数・上限 NUC_A_WMAX はノブ【C】（弱まる側は人の値なし）"""
    NUC_A_WIN, NUC_A_N, NUC_A_ETA, NUC_A_WMAX = 10, 8, 1e-3, 4.0

    def __init__(self, src, dst, w, sign, delay, n_ch, n_recv, rep_i, err_i, tail, dev):
        self.src, self.dst, self.w = src.long(), dst.long(), w.float()
        self.sign, self.delay = sign, delay
        self.w0 = self.w.clone()
        self.n_ch, self.n_recv, self.dev = n_ch, n_recv, dev
        ex = sign > 0
        self.ie1 = torch.nonzero(ex & (delay == 1)).squeeze(1)
        self.ie2 = torch.nonzero(ex & (delay == 2)).squeeze(1)
        ii1 = torch.nonzero(~ex & (delay == 1)).squeeze(1)
        ii2 = torch.nonzero(~ex & (delay == 2)).squeeze(1)
        # 抑制は表現の行と誤差の行の両方へ（版94 の核への抑制の線と同じ）
        self.ii1 = torch.cat([ii1, ii1]); self.ii2 = torch.cat([ii2, ii2])
        self.ti1 = torch.cat([rep_i[self.dst[ii1]], err_i[self.dst[ii1]]]).long()
        self.ti2 = torch.cat([rep_i[self.dst[ii2]], err_i[self.dst[ii2]]]).long()
        self.tail = tail
        self.hist = torch.zeros(self.NUC_A_WIN, n_ch * 2, dtype=torch.uint8, device=dev)
        self.cnt = torch.zeros(n_ch * 2, dtype=torch.int16, device=dev)
        self.hi = 0

    def state(self, n_ch, n_recv):
        z = lambda n: torch.zeros(n, device=self.dev)
        return {"nuc_x": z(n_ch), "nuc_x2": z(n_ch), "nuc_ihn": z(n_recv), "nuc_ihn2": z(n_recv),
                "nuc_s1": z(n_ch), "nuc_s2": z(n_ch)}

    def after(self, S, rep, err):
        """この周の発火 → 次の周（遅れ1）・その次の周（遅れ2）に届く分。置き場所は S（グラフの中で書き換える）"""
        v = torch.cat([rep, err])
        c = self.w * v[self.src]
        n_ch, n_recv = self.n_ch, self.n_recv
        x1 = torch.zeros(n_ch, device=self.dev).index_add_(0, self.dst[self.ie1], c[self.ie1])
        x2 = torch.zeros(n_ch, device=self.dev).index_add_(0, self.dst[self.ie2], c[self.ie2])
        h1 = torch.zeros(n_recv, device=self.dev).index_add_(0, self.ti1, c[self.ii1])
        h2 = torch.zeros(n_recv, device=self.dev).index_add_(0, self.ti2, c[self.ii2])
        raw = S["nuc_x2"] + x1                                   # 次の周に届く興奮
        T = self.tail
        s1 = S["nuc_s1"] * T["a1"] + raw * (1 - T["a1"])
        s2 = S["nuc_s2"] * T["a2"] + raw * (1 - T["a2"])
        S["nuc_s1"].copy_(s1); S["nuc_s2"].copy_(s2)
        S["nuc_x"].copy_((1 - T["f"]) * raw + T["f"] * (T["m1"] * s1 + T["m2"] * s2))
        S["nuc_x2"].copy_(x2)
        S["nuc_ihn"].copy_(S["nuc_ihn2"] + h1)
        S["nuc_ihn2"].copy_(h2)

    @torch.no_grad()
    def learn(self, any_src, d):
        """1歩に1回。any_src：送り手の体がこの歩のどこかで撃ったか（[表現 | 誤差]）／ d：学びの門（0〜1）"""
        now = (any_src > 0).to(torch.uint8)
        self.cnt += now.to(torch.int16) - self.hist[self.hi].to(torch.int16)
        self.hist[self.hi] = now
        self.hi = (self.hi + 1) % self.NUC_A_WIN
        g_on = (2.0 * torch.as_tensor(d, dtype=self.w.dtype, device=self.dev) - 1.0).clamp(0.0, 1.0)
        strong = (self.cnt[self.src] >= self.NUC_A_N).float()
        self.w.mul_(torch.exp(self.NUC_A_ETA * g_on * strong))
        torch.minimum(self.w, self.w0 * self.NUC_A_WMAX, out=self.w)

    def state_dict(self):
        return {"w": self.w, "cnt": self.cnt, "hist": self.hist, "hi": self.hi}

    def load_state_dict(self, d):
        if d.get("w") is None or d["w"].shape != self.w.shape:
            print("⚠️ 核どうしの線の重さは形が違うので読まない（初めの重さから）", flush=True)
            return
        self.w.copy_(d["w"]); self.cnt.copy_(d["cnt"]); self.hist.copy_(d["hist"]); self.hi = int(d["hi"])
'''

# ───────── babble.py ─────────
BB = [
    ("VERSION = 94          # コードの版。網の作りを変えたら上げる\n"
     "SAME_NET = ()         # 版94：中脳の核を領野にした（網の形が変わる）⇒ 版93 までの state.pt は読めない\n",
     "VERSION = 95          # コードの版。網の作りを変えたら上げる\n"
     "SAME_NET = ()         # 版95：基底核・CM/Pf・PPN を領野に（網の形が変わる）⇒ 版94 までの state.pt は読めない\n"),
    ("#  94c 前のスレッドの核の値を戻す",
     "#  95  基底核（線条体の型・外節・内節・網様部・腹側淡蒼球）・CM/Pf・PPN を素子に・核どうしを周の遅れに（NucNet）【み】10-21 23:37\n"
     "#  94c 前のスレッドの核の値を戻す"),
    ('    if getattr(b, "cl_on", False):     # 版67：CL → 皮質 のシナプス\n'
     '        d["cl"] = {',
     '    if getattr(b.g, "nucnet", None) is not None:   # 版95：核どうしの線\n'
     '        d["nucnet"] = b.g.nucnet.state_dict()\n'
     '    if getattr(b, "cl_on", False):     # 版67：CL → 皮質 のシナプス\n'
     '        d["cl"] = {'),
    ('    return int(d["done"]), list(d.get("log", []))\n',
     '    if getattr(b.g, "nucnet", None) is not None and "nucnet" in d:   # 版95\n'
     '        b.g.nucnet.load_state_dict(d["nucnet"])\n'
     '    return int(d["done"]), list(d.get("log", []))\n'),
]
TOOLS = ("measure_check.py", "measure_speed75.py", "rec_streams.py", "watch_vram2.py", "measure_v93a.py")


def edit(s, pairs, path):
    for pr in pairs:
        a, b = pr[0], pr[1]
        n = pr[2] if len(pr) > 2 else 1
        if s.count(a) != n:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（{n} 個のはず）：{a[:70]!r}。何も書いていない")
        s = s.replace(a, b)
    return s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 94 or "版94d" not in pathlib.Path("torch_brain.py").read_text():
        sys.exit("⚠️ 版94d ではない（apply_v94 → 94b → 94c → 94d のあとに当てる）")
    s_tb = edit(pathlib.Path("torch_brain.py").read_text(), TB, "torch_brain.py")
    k = s_tb.find("\nclass _MBView:")
    if k < 0:
        sys.exit("⚠️ torch_brain.py に class _MBView が見つからない。何も書いていない")
    s_tb = s_tb[:k] + TB_TABLE + s_tb[k:]
    k = s_tb.find("\n    def _send_rows(self, area, first, n):")
    if k < 0:
        sys.exit("⚠️ torch_brain.py に _send_rows が見つからない。何も書いていない")
    s_tb = s_tb[:k] + TB_BUILD + s_tb[k:]
    s_tg = edit(pathlib.Path("torch_group2.py").read_text(), TG, "torch_group2.py")
    k = s_tg.find("\n    def _fuse_ok(self):")
    if k < 0:
        sys.exit("⚠️ torch_group2.py に _fuse_ok が見つからない。何も書いていない")
    s_tg = s_tg[:k] + TG_METH + s_tg[k:]
    s_tg = s_tg + TG_CLASS
    s_bb = edit(pathlib.Path("babble.py").read_text(), BB, "babble.py")
    for s in (s_tb, s_tg, s_bb):
        ast.parse(s)
    pathlib.Path("torch_brain.py").write_text(s_tb)
    pathlib.Path("torch_group2.py").write_text(s_tg)
    pathlib.Path("babble.py").write_text(s_bb)
    print("✅ 版95 を当てた（torch_brain・torch_group2・babble）。基底核・CM/Pf・PPN を素子に・核どうしを周の遅れに", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "95" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['95'])}):" + s[m.end():]
            q.write_text(s)
            print(f"   {f}：版95 でも回るようにした", flush=True)
