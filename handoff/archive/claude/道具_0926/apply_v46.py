"""版45 → 版46（arkhe の ~/anaplex で  python3 apply_v46.py ）

抑制の体（速い籠細胞）を人の形で撃たせる【み】10-02「いいよ」
  撃ち方    (届いた興奮 − 上からの戻り) ÷ (1＋抑制) ＞ 12.67。焼いた見込み・p ＞ 0.5 は使わない
            12.67 ＝ 20.9mV ÷ 1.65mV【ヒト】速い抑制細胞の静止から閾値まで ÷ 錐体1本の EPSP（Wilbers 2023）
            ⟲ 版45 までは p ＝ σ(見込み(自分の履歴) ＋ 興奮 ÷ (1＋抑制)) ＞ 0.5【C】。撃つと次も撃ちやすくなり、
               5歩続くと自分の履歴だけで撃ち続けた（形B）。人の速い抑制細胞は 1歩の刻みでは履歴にほぼ左右されない
               （【ヒト】Lee・Dalley 2023：膜の時定数 12ms・順応指数 0.03・撃ったあとの谷は1歩で戻る）
            ⇒ 抑制の体は焼いた網を持たない（焼かずに人の値を与える。前の作りの 決-108・757 と同じ形）。体の中は凍結
  相手の数  抑制の体の興奮性の相手 1,080（ノブ【み】10-02「見つからないならノブで」）。抑制性の相手は inh_frac を保つ
            錐体が人の率 0.45%/歩 で撃つとき、抑制の体が【ヒト】約2Hz（Peyrache 2012）になる数（モンテカルロ、抑制なし）
  重みの初め 錐体 → 抑制の体 は対数正規、平均 1（＝1.65mV）・変動係数 0.96（【ヒト】1.65 ± 1.59mV、Wilbers 2023）
            1本の重さは「繋がり1つ」の量なので、接触の数（c_exc）で割って w に入れる（amp ＝ c_exc × w）
            刈り込みで開通した辺も同じ分布から
  学び      錐体 → 抑制の体 は予測誤差では学ばない（⟲ 版45 までは的 ＝ 領野の入口の平均【C】で学び、重みが上下した：形A）
            大きい繋がり（4.21mV 超 ＝ 2.55）だけ、送り手が直近10歩に5回以上撃った歩ごとに ×0.984（0.52 の 40乗根）
            （【ヒト】Molnár 2016：40Hz の連発で 0.52倍。弱まりは大きい繋がりに限る。強まりは人で示されていない）
  入れていないもの  弱い繋がりが多く一緒に撃つときの弱まり（【ヒト】0.72倍。何本一緒かが人に無い）
                    覚醒で閾値を下げる（版41）は表現の体だけのまま
  領野ごと（【み】10-02「まぁいいよそれで」）
    海馬・嗅内   興奮性の相手 194（【ヒト】Le Van Quyen 2008：錐体 3.1Hz・抑制 5.1Hz。閾値・1本は新皮質の代用【C】）
    背内側視床   抑制の割合 27.5% → 37%（【ヒト】Abitz 2007。細胞の大きさで分けた値）
    視床・側坐核・上丘  新皮質の値の代用【C】。撃ち方の形は【ネコ】視床の介在・【マカク】線条体の速い細胞と合う

⚠️ 印が1つでも見つからなければ、何も書かずに止まる
⚠️ 版45 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
"""
import ast
import shutil
import sys

A, G, B, R = "torch_area.py", "torch_group2.py", "babble.py", "torch_brain.py"


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


F = {k: open(k, encoding="utf-8").read() for k in (A, G, B, R)}
if "VERSION = 45 " not in F[B]:
    sys.exit("⚠️ babble.py が版45 ではない。何も書いていない")
if "INH_TH" in F[A]:
    sys.exit("⚠️ もう版46 になっている。何も書いていない")

# ── torch_area.py ──
a = F[A]
a = once(a, "import numpy as np\nimport torch\n", "import math\nimport numpy as np\nimport torch\n", "area の import")
a = once(a, "w_int8=False, fire_th=1.0, x_gain=1.0, tgt_by_fire=False, contacts_exc=3.3):\n",
         "w_int8=False, fire_th=1.0, x_gain=1.0, tgt_by_fire=False, contacts_exc=3.3,\n"
         "                 inh_n_exc=None):\n", "area の引数")
a = once(a, "W_INH = 0.02648 / 4.0\n", """W_INH = 0.02648 / 4.0

# 版46：抑制の体（速い籠細胞）は人の値を与えて撃たせる。焼いた網は持たない【み】10-02
#   撃つ：(届いた興奮 − 上からの戻り) ÷ (1＋抑制) ＞ INH_TH
#   【ヒト】Wilbers 2023（Sci Adv）：速い抑制細胞 静止 −63mV・閾値 −42.2mV（約 20.9mV）、錐体1本の EPSP 1.65 ± 1.59mV
INH_TH = 20.9 / 1.65
#   相手の数：興奮性 1,080（ノブ【み】10-02。人の抑制細胞の入力の数は見つからない）
#   錐体が 0.45%/歩 で撃つとき、抑制の体が【ヒト】約2Hz（Peyrache 2012）になる数（コンテナのモンテカルロ）
INH_N_EXC = 1080
#   海馬・嗅内：【ヒト】Le Van Quyen 2008（てんかん 11人、深部微小電極）ふだん 錐体 3.1Hz ／ 抑制 5.1Hz。海馬と嗅内に差なし
#   錐体が 3.1Hz で撃つとき抑制の体が 5.1Hz になる数 ⇒ 194【み】10-02「まぁいいよそれで」（閾値・1本は新皮質の代用【C】）
INH_N_EXC_HIPPO = 194
#   視床・側坐核・上丘は人の値が無く、新皮質の値の代用【C】【み】10-02。撃ち方の形は【ネコ】視床の介在・【マカク】線条体の速い細胞と合う
#   錐体 → 抑制の体 の1本：対数正規、平均 1・変動係数 0.96（【ヒト】1.65 ± 1.59mV を 1.65mV で割った）
INH_W_SIG = math.sqrt(math.log(1.0 + 0.96 ** 2))
INH_W_MU = -INH_W_SIG ** 2 / 2
#   弱まり：大きい繋がり（【ヒト】4.21mV 超、Molnár 2016）だけ、送り手が 10歩に5回以上撃った歩ごとに
#   0.52 の 40乗根（【ヒト】40Hz の連発 40回で 0.52倍）。弱い繋がりは弱まらない
INH_LTD_TH = 4.21 / 1.65
INH_LTD_F = 0.52 ** (1.0 / 40.0)
INH_LTD_WIN, INH_LTD_N = 10, 5
""", "W_INH")
a = once(a, """        n_partner = min(n_partner, self.n_send)
        self.n_partner = n_partner
""", """        n_partner = min(n_partner, self.n_send)
        self.n_partner = n_partner
        # 版46：抑制の体の相手は 興奮性 INH_N_EXC ＋ 抑制性（inh_frac を保つ）
        n_ie = INH_N_EXC if inh_n_exc is None else inh_n_exc
        n_pi = min(int(round(n_ie / (1 - inh_frac))), self.n_send)
        self.n_partner_inh = n_pi
        self.n_partner_vec = torch.cat([torch.full((n_ch * 2,), float(n_partner), device=dev),
                                        torch.full((self.n_inh,), float(n_pi), device=dev)])
""", "area の相手の数")
a = once(a, """        k_inh = int(round(n_partner * inh_frac))
        src = torch.empty((self.n_recv, n_partner), dtype=torch.long, device=dev)
        src[:, :k_inh] = inh_pool[torch.randint(0, len(inh_pool), (self.n_recv, k_inh),
                                                device=dev, generator=g)]
        src[:, k_inh:] = exc_pool[torch.randint(0, len(exc_pool), (self.n_recv, n_partner - k_inh),
                                                device=dev, generator=g)]
        src = src.reshape(-1)
        recv = torch.arange(self.n_recv, device=dev).repeat_interleave(n_partner)
""", """        def pick(n_rows, k_all):
            k_inh = int(round(k_all * inh_frac))
            s = torch.empty((n_rows, k_all), dtype=torch.long, device=dev)
            s[:, :k_inh] = inh_pool[torch.randint(0, len(inh_pool), (n_rows, k_inh),
                                                  device=dev, generator=g)]
            s[:, k_inh:] = exc_pool[torch.randint(0, len(exc_pool), (n_rows, k_all - k_inh),
                                                  device=dev, generator=g)]
            return s.reshape(-1)
        # 表現・誤差は n_partner、抑制の体は n_pi（版46）
        src = torch.cat([pick(n_ch * 2, n_partner), pick(self.n_inh, n_pi)])
        recv = torch.cat([torch.arange(n_ch * 2, device=dev).repeat_interleave(n_partner),
                          (torch.arange(self.n_inh, device=dev) + n_ch * 2).repeat_interleave(n_pi)])
""", "area の辺の相手")
a = once(a, "        self.budget = torch.full((self.n_recv,), float(n_partner), device=dev)\n",
         "        self.budget = self.n_partner_vec.clone()      # 版46：受け手の種類ごとの相手の数\n",
         "area の budget")

# ── torch_group2.py ──
g = F[G]
g = once(g, "from torch_area import TAUS, THETA, DT_MS, W_INH\n",
         "from torch_area import TAUS, THETA, DT_MS, W_INH\n"
         "from torch_area import INH_TH, INH_W_SIG, INH_W_MU, INH_LTD_TH, INH_LTD_F, INH_LTD_WIN, INH_LTD_N   # 版46\n",
         "group2 の import")
g = once(g, """        self.n_partner = torch.cat([torch.full((a.n_recv,), float(a.n_partner), device=dev)
                                    for a in self.areas])
""", """        self.n_partner = torch.cat([a.n_partner_vec for a in self.areas])   # 版46：抑制の体は相手が少ない
""", "group2 の相手の数")
g = once(g, """        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
""", """        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
        # 版46：錐体 → 抑制の体 の重みは人の分布で始める（予測誤差では学ばない）
        ei = self.is_inh_r[self.e_recv.long()] & ~self.e_inh
        self.w[ei] = self._inh_w(self.e_recv[ei].long())
        # 送り手ごとの直近 INH_LTD_WIN 歩の発火（弱まりの条件を見るため）
        self.hist = torch.zeros((INH_LTD_WIN, self.n_send_all), dtype=torch.uint8, device=dev)
        self.hist_n = torch.zeros(self.n_send_all, dtype=torch.int16, device=dev)
        self.hist_i = 0
""", "group2 の重みの初め")
g = once(g, """    def _starts(self):
""", """    def _inh_w(self, r):
        \"\"\"版46：錐体 → 抑制の体 の1本（繋がり1つ ＝ 平均 1.65mV を 1）を接触の数で割って w に\"\"\"
        z = torch.randn(len(r), device=self.dev, generator=self.gen)
        return torch.exp(INH_W_MU + INH_W_SIG * z) / self.c_exc[r]

    def _starts(self):
""", "group2 の _inh_w")
g = once(g, """        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        self.elig[cuts] = 0.0
""", """        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        ei = cuts[self.is_inh_r[self.e_recv[cuts].long()] & ~self.e_inh[cuts]]
        if len(ei):                    # 版46：開通した 錐体 → 抑制の体 も人の分布から
            self.w[ei] = self._inh_w(self.e_recv[ei].long())
        self.elig[cuts] = 0.0
""", "group2 の rewire")
g = once(g, """    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
""", """    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
        if first and self._base is not None:   # 版46：送り手ごとの直近 INH_LTD_WIN 歩の発火を数える
            now = (self._base > 0).to(torch.uint8)
            self.hist_n += now.to(torch.int16) - self.hist[self.hist_i].to(torch.int16)
            self.hist[self.hist_i] = now
            self.hist_i = (self.hist_i + 1) % INH_LTD_WIN
""", "group2 の発火の数え")
g = once(g, """            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            self.elig[e] += torch.where(self.e_inh[e], torch.zeros_like(gg), gg)
""", """            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            # 版46：錐体 → 抑制の体 も予測誤差では学ばない（下の弱まりだけ）
            to_inh = self.is_inh_r[r]
            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg)
            # 版46：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）
            se = self.e_src[e].long()
            ltd = to_inh & ~self.e_inh[e] & (self.hist_n[se] >= INH_LTD_N) & \\
                (self.w[e] * self.c_exc[r] > INH_LTD_TH)
            if bool(ltd.any()):
                self.w[e[ltd]] *= INH_LTD_F
""", "group2 の学び")
g = once(g, """        inh_spk = (p[self.is_inh_r] > 0.5).float()
""", """        # 版46：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ。焼いた見込み・p は使わない
        #   ⟲ 版45 まで p ＞ 0.5【C】。撃つと次も撃ちやすくなり、自分の履歴だけで撃ち続けた（形B）
        d_inh = (exc - exc_mod)[self.is_inh_r] / (1.0 + ihn[self.is_inh_r])
        inh_spk = (d_inh > INH_TH).float()
""", "group2 の抑制の撃ち方")

# ── torch_brain.py ──
r = F[R]
r = once(r, "from torch_area import Prior, Area, steps, THETA\n",
         "from torch_area import Prior, Area, steps, THETA\n"
         "from torch_area import INH_N_EXC, INH_N_EXC_HIPPO   # 版46\n", "brain の import")
r = once(r, "            inh_frac=(0.275 if nm in (\"視床\", \"背内側視床\") else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n",
         "            inh_frac=(0.37 if nm == \"背内側視床\" else 0.275 if nm == \"視床\" else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n"
         "            #   版46：背内側視床 37%【ヒト】Abitz 2007（成人8体。小さい細胞を介在とみなす、大きさだけで分けた）【み】10-02\n",
         "brain の抑制の割合")
r = once(r, "            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),\n",
         "            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),\n"
         "            # 版46：抑制の体の興奮性の相手。海馬・嗅内は【ヒト】の率（錐体 3.1Hz・抑制 5.1Hz）から 194\n"
         "            inh_n_exc=(INH_N_EXC_HIPPO if nm in (\"歯状回\", \"CA3\", \"CA1\", \"嗅内浅\", \"嗅内深\") else INH_N_EXC),\n",
         "brain の相手の数")

# ── babble.py ──
b = F[B]
b = once(b, "VERSION = 45 ", "VERSION = 46 ", "VERSION")
h = "#      喃語の高さを中心 300Hz・発話ごと 3.3半音・発話の中 4.47半音に（設計案_脳幹.md §12-63〜73）\n"
b = once(b, h, h +
         "#  45  抑制の繋がりを人の強さで固定し、学ばない【み】10-01\n"
         "#  46  抑制の体を人の値で撃たせる（届いた興奮 ÷ (1＋抑制) ＞ 12.67。焼いた見込みを使わない）／\n"
         "#      抑制の体の相手 興奮性 1,080（ノブ。海馬・嗅内は【ヒト】の率から 194）／ 背内側視床の抑制 37%【ヒト】／\n"
         "#      錐体 → 抑制の体 は人の分布で始め、予測誤差では学ばず、\n"
         "#      大きい繋がりだけ連発で弱まる（【ヒト】Wilbers 2023・Peyrache 2012・Molnár 2016。【み】10-02）\n",
         "版の履歴")

out = {A: a, G: g, B: b, R: r}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".v45")
    open(k, "w", encoding="utf-8").write(v)
print("版46 にした（元は *.v45 に残した）")
print("回すときは  python3 babble.py 100 --fresh")
