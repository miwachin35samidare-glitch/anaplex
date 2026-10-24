"""版96：VA・VLa・補足運動野・前補足運動野を足し、蹴りを消して合図を補足運動野へ（設計案_版96の組み表 §4〜§8）
（~/anaplex で。版95d の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v96.py
【み】10-16（決め）・10-22 22:12「それ終わったら続きの版をやってね」・10-23「人で偏りがあるならそのとおりに」「2 は駆動で」
     10-23「一旦それでいい、動かなかったら増やして」（視床の型の体数は人の比）・10-24「横で作って」「それで進めて」
入れるもの（値はすべて組み表から。⚠️ 数値は合わせない ⇒ 全部組んでから人の率に合わせる【み】10-18 22:08）
  1 領野：補足 81・前補足 146（皮質の型）／ VA 56・VLa 54（視床の型、時間細胞なし）
     CM 64 → 19・Pf 64 → 4・CL 64 → 5（決め2 A：人の片側 100万体あたり 20体。⟲ 下限 64 の取り違え）【測】measure_thal_min：4体でも届く
  2 抑制：内節 → VLa・VA ／ 網様部 → VA（核 → 領野、64体）【ヒト】Gallay 2008・Mai 2019
  3 視床 ⇄ 皮質（駆動、K_NEW 8）：VLa → 補足 ／ VA → 前補足・前頭前野 ／ 補足 → VLa ／ 前補足・前頭前野 → VA
     背内側視床 → 補足・前補足【マカク】Matelli 1996・Morel 2005
  4 皮質どうし（マーモセットの段：補足 2.94 ＜ 運動 6Va 3.97 ≒ 前補足 4.05、前頭前野 46D 3.78）
     補足 → 前補足・運動：上り（補足の誤差 → 細胞体）／ 前補足・運動 → 補足：下り（表現 → 樹状突起）
     前補足 ⇄ 運動・前補足 ⇄ 前頭前野：横（誤差 → 細胞体 と 表現 → 樹状突起 の両方）。半分ずつの比は組んでから合わせる
     ⚠️ 前頭前野 ⇄ 補足 は張らない（【マカク】Luppino 1993・マーモセット 細胞 49）
  5 線条体（2,700本の割り振り）【マカク】逆行性の細胞の割合（同じ側）
     被殻：運動 1,962・補足 728・前補足 10（Borra 2021 表3 の口に近い注入 71l：F1 18.6・F3 6.9・F6 0.1）
     尾状核：前頭前野 1,953・前補足 398・補足 349（Borra 2022 図5 の3回の注入の平均。前頭前野 ＝ 9/8B・46d・46v・8・10・11〜14・12・45・25 の合計）
       ⚠️ 図の棒を目で読んだ値（±0.3% ほど）
  6 視床下核：補足・前補足 → 視床下核（64体、前頭前野 → 視床下核と同じ）。割合は人・霊長類に数が無い ⇒ 組んでから人の率に合わせる
     （他の模型も同じ扱い：Liénard・Cos・Girard 2016「縛るデータを知らない」25〜5000 シナプスの範囲を発火率で決める）
  7 CM・Pf：補足 → CM ／ 前補足 → Pf（K_NEW 8）【マカク】Wiesendanger 1985
  8 CL → 補足・前補足（覚醒の道）【マカク】Rouiller 1999・Morel 2005
  9 蹴りを消す。合図は補足へ（学ぶ間の足場）。形は人の補足運動野の撃ち方
     【ヒト】Fried 2011：補足の 23% が動く前に率を変え、上がるのは 43% ⇒ 約 10% の体へ ／ 例 4 → 12Hz（上がり 8Hz）
     【ヒト】頭蓋内脳波 115人：刺激の 約200ms 後に立ち上がり・声の 約300ms 前に山・約100ms 前に下がる
     ⇒ 準備 GO_LEAD 103歩の 19歩目から上げ、75歩目（声の 28歩前）で山、94歩目（声の 9歩前）で 0
     ⚠️ 下がる側の細胞（57%）は足場では作らない。上がりの量 8Hz と重さ（KICK_W のまま）は組んでから合わせる
"""
import ast, pathlib, sys

TB = [
    # ── 1 領野の名前・型 ──
    ('THAL_IL = ("CM", "Pf")                # 版95：髄板内核（型「視床」）\n',
     'THAL_IL = ("CM", "Pf")                # 版95：髄板内核（型「視床」）\n'
     'SMA2 = ("補足", "前補足")            # 版96：補足運動野（SMA, 6mp）・前補足運動野（pre-SMA, 6ma）（型「皮質」）\n'
     'MTH = ("VA", "VLa")                  # 版96：運動の視床（淡蒼球を受ける側、型「視床」）\n'),
    ('UNBAKED_OK = tuple(sorted({PRIOR_TYPE[k] for k in NUC} | {"線条体"}))   # 版94b ／ 版95：焼くまで prior_v1\n',
     'UNBAKED_OK = tuple(sorted({PRIOR_TYPE[k] for k in NUC} | {"線条体"}))   # 版94b ／ 版95：焼くまで prior_v1\n'
     'PRIOR_TYPE.update({"補足": "皮質", "前補足": "皮質", "VA": "視床", "VLa": "視床"})   # 版96\n'),
    ('         "上丘") + NUC + ("CL",) + STR + THAL_IL                    # 版43b ／ 版94：中脳の核 ／ 版94d：CL ／ 版95\n',
     '         "上丘") + NUC + ("CL",) + STR + THAL_IL + SMA2 + MTH       # 版43b ／ 版94：中脳の核 ／ 版94d：CL ／ 版95 ／ 版96\n'),
    ('TGT_BY_FIRE = TGT_BY_FIRE + NUC + ("CL",) + STR + THAL_IL   # 版94 ／ 版94d ／ 版95\n',
     'TGT_BY_FIRE = TGT_BY_FIRE + NUC + ("CL",) + STR + THAL_IL + SMA2 + MTH   # 版94 ／ 版94d ／ 版95 ／ 版96\n'),
    ('CL_CORTEX = ("聴覚", "運動", "視覚", "連合", "高次", "前頭前野")\n',
     'CL_CORTEX = ("聴覚", "運動", "視覚", "連合", "高次", "前頭前野", "補足", "前補足")   # 版96：補足・前補足も【マカク】Rouiller 1999・Morel 2005\n'),
    ('N_PPN, N_CL = 512, 64              # ⚠️ PPN はハードの側のノブ ／ 版95c：CL は人の決め（20/100万・下限 64）【計算】⟲ 版67〜95b 4,096\n',
     'N_PPN, N_CL = 512, 5               # ⚠️ PPN はハードの側のノブ ／ 版96：CL は人の比（23万 × 20/100万 ≈ 5）【計算】⟲ 版95c 64（下限 64 は中脳の核の作りで測った数）\n'),
    ('N_NEW.update({"PPN": N_PPN, "尾状核": 986, "被殻": 1302, "CM": 64, "Pf": 64})\n',
     'N_NEW.update({"PPN": N_PPN, "尾状核": 986, "被殻": 1302, "CM": 19, "Pf": 4})   # 版96：CM・Pf は人の比（⟲ 版95 64）\n'
     '# 版96：補足 81・前補足 146【み】10-16 21:45（Julich 6mp・6ma × チンパンジーの密度比 × 人の A1）／ VA 56・VLa 54【み】10-16 20:57・10-23\n'
     'N_NEW.update({"補足": 81, "前補足": 146, "VA": 56, "VLa": 54})\n'
     '# 版96：線条体の受け手1体の皮質の 2,700本を送り手で分ける【マカク】逆行性の細胞の割合（同じ側）\n'
     '#   被殻：Borra 2021 表3 の口に近い注入 71l（F1 18.6・F3 6.9・F6 0.1）⇒ 72.7% : 27.0% : 0.4%\n'
     '#   尾状核：Borra 2022 図5 の3回の注入の平均（前頭前野 65.4 : F6 13.3 : F3 11.7）⇒ 72.3% : 14.7% : 12.9%（⚠️ 図の棒を目で読んだ）\n'
     'K_CX = {"運動→被殻": 1962, "補足→被殻": 728, "前補足→被殻": 10,\n'
     '        "前頭前野→尾状核": 1953, "前補足→尾状核": 398, "補足→尾状核": 349}\n'
     '# 版96：合図（学ぶ間の足場）を補足へ。形は【ヒト】の補足運動野の撃ち方（apply_v96 の頭を見る）\n'
     'CUE_FRAC = 0.23 * 0.43               # 動く前に上がる補足の細胞【ヒト】Fried 2011（23% が変わり、そのうち 43% が上がる）\n'
     'CUE_ON, CUE_PEAK, CUE_OFF = 19, 75, 94   # 準備の歩（GO_LEAD 103 のうち）【ヒト】頭蓋内脳波：+200ms 立ち上がり・声の −300ms 山・−100ms 下がる\n'
     'CUE_HZ = 8.0                         # 山での上がり（4 → 12Hz の例）【ヒト】Fried 2011 ⚠️ 例1つ。組んでから合わせる\n'),
    # ── 2 線の表 ──
    ('            ("運動→CM", "CM"), ("前頭前野→Pf", "Pf"))\n',
     '            ("運動→CM", "CM"), ("前頭前野→Pf", "Pf"),\n'
     '            # 版96：補足・前補足・VA・VLa（下からのシナプス・学ぶ）\n'
     '            ("補足→前補足", "前補足"), ("補足→運動", "運動"),                       # 上り（誤差 → 細胞体）\n'
     '            ("前補足→運動", "運動"), ("運動→前補足", "前補足"),                     # 横の細胞体の側（誤差）\n'
     '            ("前補足→前頭前野", "前頭前野"), ("前頭前野→前補足", "前補足"),\n'
     '            ("VLa→補足", "補足"), ("VA→前補足", "前補足"), ("VA→前頭前野", "前頭前野"),   # 視床 → 皮質（駆動）\n'
     '            ("背内側視床→補足", "補足"), ("背内側視床→前補足", "前補足"),\n'
     '            ("補足→VLa", "VLa"), ("前補足→VA", "VA"), ("前頭前野→VA", "VA"),       # 皮質 → 視床（駆動【み】10-23）\n'
     '            ("補足→被殻", "被殻"), ("前補足→被殻", "被殻"), ("前補足→尾状核", "尾状核"), ("補足→尾状核", "尾状核"),\n'
     '            ("補足→視床下核", "視床下核"), ("前補足→視床下核", "視床下核"),\n'
     '            ("補足→CM", "CM"), ("前補足→Pf", "Pf"))\n'),
    ('INH_TO_AREA = {"網様部→上丘": ("網様部", "上丘"), "内節→CM": ("内節", "CM"), "網様部→Pf": ("網様部", "Pf"),\n'
     '               "腹側淡蒼球→背内側視床": ("腹側淡蒼球", "背内側視床")}\n',
     'INH_TO_AREA = {"網様部→上丘": ("網様部", "上丘"), "内節→CM": ("内節", "CM"), "網様部→Pf": ("網様部", "Pf"),\n'
     '               "腹側淡蒼球→背内側視床": ("腹側淡蒼球", "背内側視床"),\n'
     '               # 版96【ヒト】死後脳：内節 → VApc・VLa（Gallay 2008）／ 網様部 → VA の内側（Mai 2019）\n'
     '               "内節→VLa": ("内節", "VLa"), "内節→VA": ("内節", "VA"), "網様部→VA": ("網様部", "VA")}\n'),
    ('NUC_BU = {"視床下核": ("前頭前野→視床下核",), "黒質": ("前頭前野→黒質", "上丘→黒質")}\n',
     'NUC_BU = {"視床下核": ("前頭前野→視床下核", "補足→視床下核", "前補足→視床下核"),   # 版96：超直接（割合は組んでから合わせる）\n'
     '          "黒質": ("前頭前野→黒質", "上丘→黒質")}\n'),
    # ── 3 樹状突起に入るもの（ext）──
    ('                          運動=n_high + nn["前頭前野"],   # 高次からの予測 ＋ 前頭前野から（版43a）\n',
     '                          運動=n_high + nn["前頭前野"] + nn["前補足"],   # 高次からの予測 ＋ 前頭前野から（版43a）＋ 前補足（横、版96）\n'),
    ('                          前頭前野=nn["嗅内深"],       # 嗅内深 から\n',
     '                          前頭前野=nn["嗅内深"] + nn["前補足"],       # 嗅内深 から ＋ 前補足（横、版96）\n'),
    ('                          **{k: 0 for k in STR + THAL_IL})   # 版95：背側の線条体・CM・Pf も細胞体へだけ\n',
     '                          **{k: 0 for k in STR + THAL_IL + MTH},   # 版95：背側の線条体・CM・Pf も細胞体へだけ ／ 版96：VA・VLa も\n'
     '                          補足=nn["前補足"] + n_cmd,          # 版96：前補足・運動から（下り）\n'
     '                          前補足=n_cmd + nn["前頭前野"])     # 版96：運動・前頭前野から（横の樹状突起の側）\n'),
    ('                             ("高次", nn["高次"] + nn["運動"], nn["前頭前野"]))  # 前頭前野から（版43a）\n',
     '                             ("高次", nn["高次"] + nn["運動"], nn["前頭前野"]),  # 前頭前野から（版43a）\n'
     '                             # 版96：下り・横の樹状突起の側（調整する側。眠りでも落とす）\n'
     '                             ("運動", nn["運動"] + nn["高次"] + nn["前頭前野"], nn["前補足"]),\n'
     '                             ("前頭前野", nn["前頭前野"] + nn["嗅内深"], nn["前補足"]),\n'
     '                             ("補足", nn["補足"], nn["前補足"] + n_cmd),\n'
     '                             ("前補足", nn["前補足"], n_cmd + nn["前頭前野"]))\n'),
    # ── 4 領野の作り ──
    ('            n_cell=(0 if nm in NUC + ("CL",) + STR_ALL + THAL_IL else n_cell),   # 版94d：CL も ／ 版95：線条体・CM・Pf も\n'
     '            n_ramp=(0 if nm in NUC + ("CL",) + STR_ALL + THAL_IL else 8),\n',
     '            n_cell=(0 if nm in NUC + ("CL",) + STR_ALL + THAL_IL + MTH else n_cell),   # 版94d：CL も ／ 版95：線条体・CM・Pf も ／ 版96：VA・VLa も\n'
     '            n_ramp=(0 if nm in NUC + ("CL",) + STR_ALL + THAL_IL + MTH else 8),\n'),
    ('                       else TAU["高次"] if nm in ("高次", "前頭前野", "嗅内浅", "嗅内深", "側坐核")\n',
     '                       else TAU["高次"] if nm in ("高次", "前頭前野", "嗅内浅", "嗅内深", "側坐核") + SMA2   # 版96：前頭前野と同じ\n'),
    ('            inh_frac=(0.275 if nm in ("視床", "背内側視床", "CL") + THAL_IL else 0.075 if nm == "視床下核"\n',
     '            inh_frac=(0.275 if nm in ("視床", "背内側視床", "CL") + THAL_IL + MTH else 0.075 if nm == "視床下核"\n'),
    ('            relay_exc=(RELAY_EXC if nm in ("視床", "CL") + THAL_IL else 0 if nm in NUC + STR_ALL else None),',
     '            relay_exc=(RELAY_EXC if nm in ("視床", "CL") + THAL_IL + MTH else 0 if nm in NUC + STR_ALL else None),'),
    # ── 5 蹴りを消して合図 ──
    ('        # 運動視床：合図のあと、語によらない大きな型を数歩だけ出す（【ヒト】BrainGate の「蹴り」）\n'
     '        #   ⟲ 移したとき「出力の75%」が「32体」に変わっていた\n'
     '        self.kick_to = torch.randperm(n_cmd, device=dev, generator=g)[:max(1, int(n_cmd * KICK_FRAC))]\n'
     '        self.kick_left = 0\n',
     '        # 版96：合図（学ぶ間の足場）は補足の CUE_FRAC の体へ。⟲ 版95d までの蹴り（運動の 75% へ 18.8Hz）は消した【み】10-16 21:28\n'
     '        self.cue_to = torch.randperm(N_NEW["補足"], device=dev, generator=g)[:max(1, int(round(N_NEW["補足"] * CUE_FRAC)))]\n'
     '        self.cue_t = None\n'),
    ('        self.p95_inh = {k_: rpf(nn[s_], nn[t_], K_NUC) for k_, (s_, t_) in INH_TO_AREA.items()}\n',
     '        # 版96：補足・前補足・VA・VLa の線（下からのシナプス）\n'
     '        self.p96 = {"補足→前補足": rp(nn["補足"], nn["前補足"]), "補足→運動": rp(nn["補足"], nn["運動"]),\n'
     '                    "前補足→運動": rp(nn["前補足"], nn["運動"]), "運動→前補足": rp(nn["運動"], nn["前補足"]),\n'
     '                    "前補足→前頭前野": rp(nn["前補足"], nn["前頭前野"]), "前頭前野→前補足": rp(nn["前頭前野"], nn["前補足"]),\n'
     '                    "VLa→補足": rp(nn["VLa"], nn["補足"]), "VA→前補足": rp(nn["VA"], nn["前補足"]),\n'
     '                    "VA→前頭前野": rp(nn["VA"], nn["前頭前野"]),\n'
     '                    "背内側視床→補足": rp(nn["背内側視床"], nn["補足"]), "背内側視床→前補足": rp(nn["背内側視床"], nn["前補足"]),\n'
     '                    "補足→VLa": rp(nn["補足"], nn["VLa"]), "前補足→VA": rp(nn["前補足"], nn["VA"]),\n'
     '                    "前頭前野→VA": rp(nn["前頭前野"], nn["VA"]),\n'
     '                    "補足→視床下核": rpf(nn["補足"], nn["視床下核"], K_NUC), "前補足→視床下核": rpf(nn["前補足"], nn["視床下核"], K_NUC),\n'
     '                    "補足→CM": rp(nn["補足"], nn["CM"]), "前補足→Pf": rp(nn["前補足"], nn["Pf"]),\n'
     '                    **{k_: rpr(nn[k_.split("→")[0]], nn[k_.split("→")[1]], K_CX[k_])\n'
     '                       for k_ in ("補足→被殻", "前補足→被殻", "前補足→尾状核", "補足→尾状核")}}\n'
     '        self.p95_inh = {k_: rpf(nn[s_], nn[t_], K_NUC) for k_, (s_, t_) in INH_TO_AREA.items()}\n'),
    ('        self.p95 = {"運動→被殻": rpr(nn["運動"], nn["被殻"], K_CX_STR), "前頭前野→尾状核": rpr(nn["前頭前野"], nn["尾状核"], K_CX_STR),\n',
     '        self.p95 = {"運動→被殻": rpr(nn["運動"], nn["被殻"], K_CX["運動→被殻"]),   # 版96：2,700本を送り手で分けた\n'
     '                    "前頭前野→尾状核": rpr(nn["前頭前野"], nn["尾状核"], K_CX["前頭前野→尾状核"]),\n'),
    ('        self.cmpf = {k: torch.zeros(nn[k], device=dev) for k in THAL_IL}   # 版95\n',
     '        self.cmpf = {k: torch.zeros(nn[k], device=dev) for k in THAL_IL}   # 版95\n'
     '        self.mth = {k: torch.zeros(nn[k], device=dev) for k in MTH}       # 版96：VA・VLa の表現（1歩前）\n'),
    ('        self.pred = {nm: torch.zeros(self.n[nm], device=dev) for nm in ("連合", "高次", "前頭前野", "嗅内深")}\n',
     '        self.pred = {nm: torch.zeros(self.n[nm], device=dev) for nm in ("連合", "高次", "前頭前野", "嗅内深") + SMA2}   # 版96：補足・前補足\n'),
    ('                      **self.p95}                                                                   # 版95\n',
     '                      **self.p95, **self.p96}                                                       # 版95 ／ 版96\n'),
    ('        self.kick_abs = int(self.off_ch[NAMES.index("運動")]) + self.kick_to\n',
     '        self.cue_abs = int(self.off_ch[NAMES.index("補足")]) + self.cue_to   # 版96：合図の体（n_ch_all の並び）\n'),
    ('            elif nm in ("聴覚", "視覚", "連合", "高次", "前頭前野"):\n',
     '            elif nm in ("聴覚", "視覚", "連合", "高次", "前頭前野") + SMA2:   # 版96：補足・前補足も皮質の目標\n'),
    ('    def go(self, n=3):\n'
     '        """合図（話し始める）。運動視床が蹴りを数歩だけ出す"""\n'
     '        self.kick_left = n\n',
     '    def go(self, n=None):\n'
     '        """合図（話し始める）。版96：補足へ、人の補足運動野の形（CUE_ON → CUE_PEAK で上げ CUE_OFF で 0）⟲ 版95d まで運動視床の蹴り"""\n'
     '        self.cue_t = 0\n'),
    ('        if speaking:\n'
     '            self.go(C.GO_LEAD - C.CMD_LEAD)     # 版43d：語の指令が入る前に終える（config.CMD_LEAD）\n',
     '        if speaking:\n'
     '            self.go()                           # 版96：補足への合図（準備のはじめから。形は CUE_*）\n'),
    # ── 6 step：入力 ──
    ('                   "運動→CM": self.motor_rep, "前頭前野→Pf": self.pred["前頭前野"]}\n',
     '                   "運動→CM": self.motor_rep, "前頭前野→Pf": self.pred["前頭前野"],\n'
     '                   # 版96：上り・横の細胞体の側は誤差 ／ 視床・皮質 → 視床・線条体・視床下核は表現（1歩前）\n'
     '                   "補足→前補足": self.up["補足"], "補足→運動": self.up["補足"],\n'
     '                   "前補足→運動": self.up["前補足"], "運動→前補足": self.up["運動"],\n'
     '                   "前補足→前頭前野": self.up["前補足"], "前頭前野→前補足": self.up["前頭前野"],\n'
     '                   "VLa→補足": self.mth["VLa"], "VA→前補足": self.mth["VA"], "VA→前頭前野": self.mth["VA"],\n'
     '                   "背内側視床→補足": self.md_out, "背内側視床→前補足": self.md_out,\n'
     '                   "補足→VLa": self.pred["補足"], "前補足→VA": self.pred["前補足"], "前頭前野→VA": self.pred["前頭前野"],\n'
     '                   "補足→被殻": self.pred["補足"], "前補足→被殻": self.pred["前補足"],\n'
     '                   "前補足→尾状核": self.pred["前補足"], "補足→尾状核": self.pred["補足"],\n'
     '                   "補足→視床下核": self.pred["補足"], "前補足→視床下核": self.pred["前補足"],\n'
     '                   "補足→CM": self.pred["補足"], "前補足→Pf": self.pred["前補足"]}\n'),
    ('            self._pm(cmd),                                               # 口の指令 → 運動の領野（版43d で戻した） ／ 版49：運動前野の送り手の数\n',
     '            self._pm(cmd) + self._bo["補足→運動"] + self._bo["前補足→運動"],   # 口の指令 → 運動の領野（版43d で戻した） ／ 版49 ／ 版96：補足（上り）・前補足（横）\n'),
    ('            self._sx("高次→前頭前野", self.up["高次"], self.p_h2pfc) + self._sx("背内側視床→前頭前野", self.md_out, self.p_md2pfc),   # 前頭前野\n',
     '            self._sx("高次→前頭前野", self.up["高次"], self.p_h2pfc) + self._sx("背内側視床→前頭前野", self.md_out, self.p_md2pfc)\n'
     '            + self._bo["前補足→前頭前野"] + self._bo["VA→前頭前野"],   # 前頭前野 ／ 版96：前補足（横）・VA\n'),
    ('            self._bo["前頭前野→尾状核"] + self._bo["Pf→尾状核"], self._bo["運動→被殻"] + self._bo["CM→被殻"],    # 版95：尾状核・被殻\n'
     '            self._bo["運動→CM"], self._bo["前頭前野→Pf"]])                                                    # 版95：CM・Pf\n',
     '            self._bo["前頭前野→尾状核"] + self._bo["Pf→尾状核"] + self._bo["前補足→尾状核"] + self._bo["補足→尾状核"],   # 版95：尾状核 ／ 版96\n'
     '            self._bo["運動→被殻"] + self._bo["CM→被殻"] + self._bo["補足→被殻"] + self._bo["前補足→被殻"],             # 版95：被殻 ／ 版96\n'
     '            self._bo["運動→CM"] + self._bo["補足→CM"], self._bo["前頭前野→Pf"] + self._bo["前補足→Pf"],                # 版95：CM・Pf ／ 版96\n'
     '            # 版96：補足・前補足・VA・VLa（NAMES の順）\n'
     '            self._bo["VLa→補足"] + self._bo["背内側視床→補足"],\n'
     '            self._bo["補足→前補足"] + self._bo["運動→前補足"] + self._bo["前頭前野→前補足"] + self._bo["VA→前補足"] + self._bo["背内側視床→前補足"],\n'
     '            self._bo["前補足→VA"] + self._bo["前頭前野→VA"],\n'
     '            self._bo["補足→VLa"]])\n'),
    ('        if self.kick_left > 0:                                 # 運動視床の蹴り\n'
     '            # 視床の細胞が人の速さで撃つスパイクの列（毎歩 1 ではない）\n'
     '            x[self.kick_abs] = x[self.kick_abs] + KICK_W * (torch.rand(len(self.kick_abs), device=self.dev,\n'
     '                                           generator=self.gen) < KICK_P).float()\n'
     '            self.kick_left -= 1\n',
     '        if self.cue_t is not None:                            # 版96：補足への合図（⟲ 版95d まで運動視床の蹴り）\n'
     '            t_ = self.cue_t\n'
     '            if CUE_ON <= t_ < CUE_OFF:\n'
     '                f_ = (t_ - CUE_ON) / (CUE_PEAK - CUE_ON) if t_ < CUE_PEAK else (CUE_OFF - t_) / (CUE_OFF - CUE_PEAK)\n'
     '                p_ = CUE_HZ * f_ * 10.67 / 1000.0\n'
     '                x[self.cue_abs] = x[self.cue_abs] + KICK_W * (torch.rand(len(self.cue_abs), device=self.dev,\n'
     '                                               generator=self.gen) < p_).float()\n'
     '            self.cue_t = t_ + 1 if t_ + 1 < CUE_OFF else None\n'),
    # ── 7 step：樹状突起に入るもの ──
    ('                       ph, pf,                                 # 運動（＋ 前頭前野。版43a）\n',
     '                       ph, pf, self.pred["前補足"],            # 運動（＋ 前頭前野。版43a ／ ＋ 前補足 横、版96）\n'),
    ('                       ed,                                     # 前頭前野（嗅内深 から）\n',
     '                       ed, self.pred["前補足"],                # 前頭前野（嗅内深 から ／ ＋ 前補足 横、版96）\n'),
    ('                       pf])                                    # 側坐核（前頭前野から）。上丘は ext なし\n',
     '                       pf,                                     # 側坐核（前頭前野から）。上丘は ext なし\n'
     '                       self.pred["前補足"], self.motor_rep,     # 版96：補足（前補足・運動から、下り）\n'
     '                       self.motor_rep, pf])                    # 版96：前補足（運動・前頭前野から、横）\n'),
    # ── 8 step：出力 ──
    ('        self.cmpf = {k_: cut(k_, rep) for k_ in THAL_IL}   # 版95：CM・Pf は表現を送る（視床の中継）\n',
     '        self.cmpf = {k_: cut(k_, rep) for k_ in THAL_IL}   # 版95：CM・Pf は表現を送る（視床の中継）\n'
     '        self.mth = {k_: cut(k_, rep) for k_ in MTH}         # 版96：VA・VLa も表現を送る\n'
     '        for k_ in SMA2:                                     # 版96：補足・前補足の表現（下り・横の樹状突起の側、線条体・視床へ）\n'
     '            self.pred[k_] = cut(k_, rep)\n'),
]
BB = [("VERSION = 95          # コードの版。網の作りを変えたら上げる",
       "VERSION = 96          # コードの版。網の作りを変えたら上げる"),
      ("#  95d 速さ：", "#  96  補足・前補足・VA・VLa を足す・蹴りを消して合図を補足へ・CM 19・Pf 4・CL 5（設計案_版96の組み表）【み】10-24\n#  95d 速さ：")]


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:70]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    src = pathlib.Path("babble.py").read_text()
    if "VERSION = 95 " not in src or "95d" not in src or "#  96 " in src:
        sys.exit("⚠️ 版95d ではない（apply_v95d.py のあとに1回だけ当てる）")
    out = [edit("torch_brain.py", TB), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版96 を当てた（補足・前補足・VA・VLa・蹴りを消して合図を補足へ・CM 19・Pf 4・CL 5）。⚠️ 網の形が変わるので state.pt は 0 から", flush=True)
