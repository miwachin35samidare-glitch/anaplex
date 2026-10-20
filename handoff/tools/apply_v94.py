"""版94：中脳の5核を素子に（網の領野として入れる）
（~/anaplex で。版93 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v94.py
【み】10-19 7:42「まずニューロンを素子に直して」
決まったこと（デフォルメ版_決まったこと.md）
  素子：焼いて凍結した小さな網。受け取るのは他の素子の発火だけ。何の領野かは配線が決める
  疎：入力があってから動く。入力が無ければ素子は休む
  種類：表現と誤差。どちらも同じ焼いた素子で、違うのは撃つ条件だけ
⟲ 版93 の midbrain.py（撃ち方 B・i0・NMDA の尾・Harnett と決め A の学び）は素子ではなかった（版43b〜92 の確率の形も同じ）
  i0（入力なしで撃つ）は「入力が無ければ素子は休む」と食い違っていた
変えるもの
  torch_brain.py
    1 NAMES の後ろに 視床下核・境・外側手綱核・吻内側被蓋核・黒質（各 256体）。素子の型は「中脳」
      prior_v2_中脳.npz が無いあいだは prior_v1 で（録って焼くまで。焼くのは運用の前【み】10-10）
    2 核に時間細胞・傾きの細胞は置かない（素子でない細胞を持ち込まない）・核の中の興奮の横は置かない（relay_exc 0）【C】
    3 入る線は下からのシナプス（_BUSyn。ほかの下からの道と同じ学び【み】10-16「同じなら合わせて1つに」）
       前頭前野 → 視床下核 ／ 視床下核 → 境 ／ 境 → 外側手綱核 ／ 外側手綱核 → 吻内側被蓋核 ／
       視床下核・前頭前野・上丘の誤差・PPN → 黒質。受け手1体が K_NEW 本（ほかの足した領野と同じ。ノブ）
    4 抑制の線（側坐核 → 境 ／ 吻内側被蓋核 → 黒質）は学ばない（版45 と同じ）。1本 ＝ 領野の中の抑制と同じ強さ
       受け手の表現・誤差の体の抑制（÷(1＋抑制)）に足す
    5 門 ＝ 黒質の表現の発火の割合を 225ms でならして ÷ 4.56Hz【ヒト】（版93 と同じ）
    6 b.mb は測る道具のための見かけだけ（核の発火・率を数える。計算はしない）
  torch_group2.py：外からの抑制 ext_ihn を、周回の抑制に足す（_drive_core。グラフの道も同じ）
  babble.py：VERSION 94（SAME_NET 空：領野が増えて網の形が変わる）・中脳の重さの保存をやめる
  測る道具：版94 でも回るように
⚠️ 数値は合わせない（10-18 22:08【み】全部組んでから）。midbrain.py・midbrain_i0.json は使わなくなる（消さない）
"""
import ast, pathlib, re, sys

NUC_DEF = 'NUC = ("視床下核", "境", "外側手綱核", "吻内側被蓋核", "黒質")   # 版94：中脳の核（網の領野・素子）\n'

TB = [
    ('# 版55：焼いた網を型ごとに（【み】10-10「型ごとに分けるでいいよ」「いいよ」）。bake_prior.py で焼いたもの\n',
     NUC_DEF + '# 版55：焼いた網を型ごとに（【み】10-10「型ごとに分けるでいいよ」「いいよ」）。bake_prior.py で焼いたもの\n'),
    ('              "側坐核": "側坐核", "上丘": "上丘"}\n',
     '              "側坐核": "側坐核", "上丘": "上丘", **{k: "中脳" for k in NUC}}   # 版94\n'
     'UNBAKED_OK = ("中脳",)   # 版94：焼く前は prior_v1（録って焼くまで）\n'),
    ('        if not os.path.exists(f):\n'
     '            raise SystemExit(f"⚠️ {f} が無い。bake_prior.py で焼いてから（ANAPLEX_PRIOR=v1 で前の網）")\n',
     '        if not os.path.exists(f):\n'
     '            if t in UNBAKED_OK:\n'
     '                print(f"⚠️ {f} が無い ⇒ 型「{t}」は焼く前の素子（prior_v1）で。rec_streams → bake_prior で焼く", flush=True)\n'
     '                continue\n'
     '            raise SystemExit(f"⚠️ {f} が無い。bake_prior.py で焼いてから（ANAPLEX_PRIOR=v1 で前の網）")\n'),
    ('from torch_area import INH_N_EXC, INH_N_EXC_HIPPO   # 版47\n',
     'from torch_area import INH_N_EXC, INH_N_EXC_HIPPO   # 版47\n'
     'from torch_area import W_INH, DT_MS as _DT_MS      # 版94\n'),
    ('         "上丘")                                                    # 版43b\n',
     '         "上丘") + NUC                                              # 版43b ／ 版94：中脳の核\n'),
    ('#   黒質・視床下核の辺縁・境の細胞・外側手綱核・吻内側被蓋核は midbrain.py（版93：撃ち方 B を周ごとに・入るシナプスは学ぶ）\n',
     '#   黒質・視床下核の辺縁・境の細胞・外側手綱核・吻内側被蓋核：版94 から網の領野（素子）。⟲ 版93 の midbrain.py は素子でなかった\n'),
    ('N_MB = {"黒質": 256, "視床下核": 256, "境": 256, "外側手綱核": 256, "吻内側被蓋核": 256}\n',
     'N_MB = {k: 256 for k in NUC}\n'
     'N_NEW.update(N_MB)                 # 版94：ほかの足した領野と同じ扱い（閾値 HUMAN_TH・入口の重さ X_NEW）\n'
     'DA_TAU = int(round(225.0 / _DT_MS))   # 版94：門のならし【ヒト】Zaghloul 2009（版93 と同じ）\n'
     'W_INH_BU = W_INH * 4.0             # 版94：核への抑制の線 1本（領野の中の抑制 1本 ＝ 4.0 × W_INH と同じ）。学ばない\n'),
    ('N_PARTNER = {"CA3": 16100}\n',
     'TGT_BY_FIRE = TGT_BY_FIRE + NUC   # 版94\n'
     'N_PARTNER = {"CA3": 16100}\n'),
    ('                          上丘=0)                      # 版43b：下丘から細胞体へだけ\n',
     '                          上丘=0,                      # 版43b：下丘から細胞体へだけ\n'
     '                          **{k: 0 for k in NUC})       # 版94：中脳の核は細胞体へだけ\n'),
    ('            n_partner=N_PARTNER.get(nm, n_partner), n_cell=n_cell, stp_path=stp_path, dev=dev, seed=seed + i,\n',
     '            n_partner=N_PARTNER.get(nm, n_partner), stp_path=stp_path, dev=dev, seed=seed + i,\n'
     '            # 版94：中脳の核は時間細胞・傾きの細胞を持たない（素子でない細胞を持ち込まない）\n'
     '            n_cell=(0 if nm in NUC else n_cell), n_ramp=(0 if nm in NUC else 8),\n'),
    ('            relay_exc=(RELAY_EXC if nm == "視床" else None),   # 版51\n',
     '            relay_exc=(RELAY_EXC if nm == "視床" else 0 if nm in NUC else None),   # 版51 ／ 版94：核の中の興奮の横なし【C】\n'),
    ('        # 中脳の小さな核（版93：撃ち方 B・周ごと・学ぶ）。外から来る送り手：前頭前野の表現・側坐核の表現＋誤差・上丘の誤差・PPN\n'
     '        #   ⟲ 版43b〜92 は「来なかった誤差」も渡していた（ニューロンの発火ではない計算の値）【み】10-18 16:57\n'
     '        import midbrain as MB\n'
     '        self.mb = MB.Midbrain(N_MB, dev=dev, gen=g)\n'
     '        self.mb.wire({"前頭前野": nn["前頭前野"], "側坐核": 2 * nn["側坐核"], "上丘": nn["上丘"], "PPN": N_PPN})\n'
     '        self.gate = torch.ones((), device=dev)\n',
     '        # 版94：中脳の核へ入る線（受け手1体が K_NEW 本。⚠️ ノブ）。出どころは版93 と同じ線（設計案_基底核と小脳 10-16〜18）\n'
     '        #   ⟲ 版93 の midbrain.py（受け手1体が 64本・前頭前野 → 黒質 8本・側坐核は表現＋誤差）\n'
     '        self.p_pfc2stn = rp(nn["前頭前野"], nn["視床下核"])\n'
     '        self.p_stn2gpb = rp(nn["視床下核"], nn["境"])\n'
     '        self.p_gpb2lhb = rp(nn["境"], nn["外側手綱核"])\n'
     '        self.p_lhb2rmtg = rp(nn["外側手綱核"], nn["吻内側被蓋核"])\n'
     '        self.p_stn2sn = rp(nn["視床下核"], nn["黒質"])\n'
     '        self.p_pfc2sn = rp(nn["前頭前野"], nn["黒質"])\n'
     '        self.p_sc2sn = rp(nn["上丘"], nn["黒質"])\n'
     '        self.p_ppn2sn = rp(N_PPN, nn["黒質"])\n'
     '        self.p_nac2gpb = rp(nn["側坐核"], nn["境"])          # 抑制\n'
     '        self.p_rmtg2sn = rp(nn["吻内側被蓋核"], nn["黒質"])  # 抑制\n'
     '        self.nuc = {k: torch.zeros(nn[k], device=dev) for k in NUC}   # 1歩前の核の表現の発火\n'
     '        self.nac_rep = torch.zeros(nn["側坐核"], device=dev)\n'
     '        self.p0 = 4.56 * _DT_MS / 1000.0                       # 黒質のふだん【ヒト】Ramayya 2014\n'
     '        self.da = torch.tensor(self.p0, device=dev)\n'
     '        self.gate = torch.ones((), device=dev)\n'
     '        self.mb = _MBView(nn, dev)                             # 測る道具のための見かけ\n'),
    ('                      "CA1→側坐核": self.p_ca12nac, "嗅内深→側坐核": self.p_ecd2nac, "下丘→上丘": self.p_ic2sc}\n',
     '                      "CA1→側坐核": self.p_ca12nac, "嗅内深→側坐核": self.p_ecd2nac, "下丘→上丘": self.p_ic2sc,\n'
     '                      "前頭前野→視床下核": self.p_pfc2stn, "視床下核→境": self.p_stn2gpb, "境→外側手綱核": self.p_gpb2lhb,\n'
     '                      "外側手綱核→吻内側被蓋核": self.p_lhb2rmtg, "視床下核→黒質": self.p_stn2sn,\n'
     '                      "前頭前野→黒質": self.p_pfc2sn, "上丘→黒質": self.p_sc2sn, "PPN→黒質": self.p_ppn2sn}   # 版94\n'),
    ('            self.bu_pack()                                           # 版75：道をまとめた表\n',
     '            # 版94：核への抑制の線（学ばない。版45 と同じ）。表の外で足す\n'
     '            self.bu_inh = {}\n'
     '            for key, (p, nm) in (("側坐核→境", (self.p_nac2gpb, "境")), ("吻内側被蓋核→黒質", (self.p_rmtg2sn, "黒質"))):\n'
     '                s = _BUSyn(p[0], p[1], len(p[2]), self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n'
     '                s.w = torch.full_like(s.w, W_INH_BU)\n'
     '                one = torch.ones(len(s.w), dtype=torch.bool, device=dev)\n'
     '                s.stp = _STP_CL(len(s.w), one, ~one, stp_path, dev, gb)    # 【ヒト】抑制 → 興奮 の表\n'
     '                o = int(self.off_ch[NAMES.index(nm)])\n'
     '                ch = torch.arange(o, o + self.n[nm], device=dev)\n'
     '                self.bu_inh[key] = (s, torch.cat([self.g.rep_i[ch], self.g.err_i[ch]]))\n'
     '            self.g.ext_ihn = torch.zeros(self.g.n_recv_all, device=dev)\n'
     '            self.bu_pack()                                           # 版75：道をまとめた表\n'),
    ('                   "下丘→上丘": ic}\n',
     '                   "下丘→上丘": ic,\n'
     '                   # 版94：中脳の核（1歩前の発火）\n'
     '                   "前頭前野→視床下核": self.pred["前頭前野"], "視床下核→境": self.nuc["視床下核"],\n'
     '                   "境→外側手綱核": self.nuc["境"], "外側手綱核→吻内側被蓋核": self.nuc["外側手綱核"],\n'
     '                   "視床下核→黒質": self.nuc["視床下核"], "前頭前野→黒質": self.pred["前頭前野"],\n'
     '                   "上丘→黒質": self.up["上丘"], "PPN→黒質": self._ppn}\n'),
    ('            self._sx("下丘→上丘", ic, self.p_ic2sc)])                                                        # 上丘（版43b）\n',
     '            self._sx("下丘→上丘", ic, self.p_ic2sc),                                                         # 上丘（版43b）\n'
     '            # 版94：中脳の核（NUC の順）\n'
     '            self._bo["前頭前野→視床下核"], self._bo["視床下核→境"], self._bo["境→外側手綱核"],\n'
     '            self._bo["外側手綱核→吻内側被蓋核"],\n'
     '            self._bo["視床下核→黒質"] + self._bo["前頭前野→黒質"] + self._bo["上丘→黒質"] + self._bo["PPN→黒質"]])\n'
     '        if getattr(self, "bu_inh", None):                    # 版94：核への抑制の線（1歩前の発火）\n'
     '            self.g.ext_ihn.zero_()\n'
     '            for key, v in (("側坐核→境", self.nac_rep), ("吻内側被蓋核→黒質", self.nuc["吻内側被蓋核"])):\n'
     '                s, rows = self.bu_inh[key]\n'
     '                d = s.drive(v)\n'
     '                self.g.ext_ihn.index_add_(0, rows, torch.cat([d, d]))\n'),
]
TB.append(('            ("CA1→側坐核", "側坐核"), ("嗅内深→側坐核", "側坐核"), ("下丘→上丘", "上丘"))\n',
           '            ("CA1→側坐核", "側坐核"), ("嗅内深→側坐核", "側坐核"), ("下丘→上丘", "上丘"),\n'
           '            # 版94：中脳の核へ（興奮の線）\n'
           '            ("前頭前野→視床下核", "視床下核"), ("視床下核→境", "境"), ("境→外側手綱核", "外側手綱核"),\n'
           '            ("外側手綱核→吻内側被蓋核", "吻内側被蓋核"), ("視床下核→黒質", "黒質"), ("前頭前野→黒質", "黒質"),\n'
           '            ("上丘→黒質", "黒質"), ("PPN→黒質", "黒質"))\n'))
TB_MB_OLD_HEAD = "        # 版93：中脳の核へ、領野が撃った周のまま渡す"
TB_MB_OLD_TAIL = "        self.gate = mb.step(ext, learn=learn)        # 門は脳幹が使う\n"
TB_MB_NEW = ('        # 版94：中脳の核は網の領野（素子）。外へ送るのは表現の発火（ほかの領野と同じ）\n'
             '        #   ⟲ 版93 は midbrain.step（撃ち方 B）に撃った周つきで渡していた\n'
             '        for nm in NUC:\n'
             '            self.nuc[nm] = cut(nm, rep)\n'
             '        self.nac_rep = cut("側坐核", rep)\n'
             '        self.da = self.da + (self.nuc["黒質"].mean() - self.da) / DA_TAU\n'
             '        self.gate = self.da / self.p0                 # 門は脳幹が使う\n'
             '        self.mb.see(self.nuc)\n')
MBVIEW = '''

class _MBView:
    """版94：測る道具のための見かけ（版93 の b.mb の spk・rates_hz・state_dict だけ）。中脳の核は網の領野で、ここは数えるだけ"""

    def __init__(self, n, dev):
        self.spk = {k: torch.zeros(n[k], device=dev) for k in NUC}
        self.rate_sum = torch.zeros(len(NUC), device=dev)
        self.steps = 0

    def see(self, nuc):
        self.spk = dict(nuc)
        self.rate_sum += torch.stack([nuc[k].mean() for k in NUC])
        self.steps += 1

    def rates_hz(self):
        t = max(self.steps, 1) * _DT_MS / 1000.0
        return {k: float(self.rate_sum[i]) / t for i, k in enumerate(NUC)}

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass
'''
TG = [
    ('                fire_th, th_gain, complete, pred_boost, spont_u, spont_p_pass, n_recv: int):\n',
     '                fire_th, th_gain, complete, pred_boost, spont_u, spont_p_pass, ext_ihn, n_recv: int):\n', 1),
    ('    exc = exc * (scale * scale_up); ihn = ihn * scale\n',
     '    exc = exc * (scale * scale_up); ihn = ihn * scale\n'
     '    if ext_ihn is not None:                  # 版94：外からの抑制の線（中脳の核へ）。学ばない\n'
     '        ihn = ihn + ext_ihn\n', 1),
    ('                   cl_drive=getattr(self, "cl_drive", None), v_carry=vc,\n',
     '                   cl_drive=getattr(self, "cl_drive", None), v_carry=vc, ext_ihn=getattr(self, "ext_ihn", None),\n', 2),
    ('            X["pred_boost"], F["spont1"], X["spont_p_pass"], self.n_recv_all)\n',
     '            X["pred_boost"], F["spont1"], X["spont_p_pass"], opt("ext_ihn"), self.n_recv_all)\n', 1),
    ('            X["pred_boost"], S["spont"], X["spont_p_pass"], self.n_recv_all)\n',
     '            X["pred_boost"], S["spont"], X["spont_p_pass"], opt("ext_ihn"), self.n_recv_all)\n', 1),
    ('            spont_u, self.spont_p_pass, self.n_recv_all)\n',
     '            spont_u, self.spont_p_pass, getattr(self, "ext_ihn", None), self.n_recv_all)\n', 1),
]
BB = [
    ("VERSION = 93          # コードの版。網の作りを変えたら上げる\n"
     "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92)",
     "VERSION = 94          # コードの版。網の作りを変えたら上げる\n"
     "SAME_NET = ()         # 版94：中脳の核を領野にした（網の形が変わる）⇒ 版93 までの state.pt は読めない\n"
     "_SAME_NET_93 = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92)"),
    ("#  93  中脳の核を撃ち方 B・周ごと・学ぶ形に（各 256体）・PPN は周ごと・領野の撃った周のまま渡す・来なかった誤差は渡さない【み】10-18\n",
     "#  93  中脳の核を撃ち方 B・周ごと・学ぶ形に（各 256体）・PPN は周ごと・領野の撃った周のまま渡す・来なかった誤差は渡さない【み】10-18\n"
     "#  94  中脳の5核を素子に（網の領野、型「中脳」）。入る線は下からのシナプス・抑制の線は学ばない【み】10-19「まずニューロンを素子に直して」\n"),
    ("    d[\"mb\"] = b.mb.state_dict()        # 版93：中脳の核の重さ\n", ""),
    ("    if \"mb\" in d:                       # 版93（版92 までの state には無い ⇒ 初めの重さ）\n"
     "        b.mb.load_state_dict(d[\"mb\"])\n", ""),
]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py", "measure_speed75.py", "rec_streams.py")


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for pr in pairs:
        a, b = pr[0], pr[1]
        n = pr[2] if len(pr) > 2 else 1
        if s.count(a) != n:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（{n} 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    return p, s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 93:
        sys.exit(f"⚠️ babble.py が版93 ではない（版 {babble.VERSION}）")
    p_tb, s_tb = edit("torch_brain.py", TB)
    i = s_tb.find(TB_MB_OLD_HEAD); j = s_tb.find(TB_MB_OLD_TAIL)
    if i < 0 or j < 0 or s_tb.count(TB_MB_OLD_HEAD) != 1 or s_tb.count(TB_MB_OLD_TAIL) != 1:
        sys.exit("⚠️ torch_brain.py の版93 の中脳の段が見つからない。何も書いていない")
    s_tb = s_tb[:i] + TB_MB_NEW + s_tb[j + len(TB_MB_OLD_TAIL):]
    k = s_tb.find("\nclass _BUSyn:")
    if k < 0:
        sys.exit("⚠️ torch_brain.py に class _BUSyn が見つからない。何も書いていない")
    s_tb = s_tb[:k] + MBVIEW + s_tb[k:]
    if "import midbrain" in s_tb or "mb.step(" in s_tb:
        sys.exit("⚠️ torch_brain.py に midbrain の呼び出しが残った。何も書いていない")
    p_tg, s_tg = edit("torch_group2.py", TG)
    p_bb, s_bb = edit("babble.py", BB)
    for s in (s_tb, s_tg, s_bb):
        ast.parse(s)
    for p, s in ((p_tb, s_tb), (p_tg, s_tg), (p_bb, s_bb)):
        p.write_text(s)
    print("✅ 版94 を当てた（torch_brain・torch_group2・babble）。中脳の5核を素子に（網の領野）", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "94" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['94'])}):" + s[m.end():]
            q.write_text(s)
            print(f"   {f}：版94 でも回るようにした", flush=True)
