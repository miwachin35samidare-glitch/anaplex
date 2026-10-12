"""版45 → 版47：形A を直す（arkhe の ~/anaplex で  python3 apply_v47.py ）  ※版46 は戻したので番号は飛ばす

形A：錐体 → 抑制の体 を予測誤差で学び（的 ＝ 領野の入口の平均【C】）、重みが上下した。高い時に蹴りで抑制の体が撃ち、指令を割った
直すもの（【み選】B 10-02「錐体 → 抑制の体 は人の弱まりだけ」【み】10-02 23時「Aを直して」）
  重みの初め 錐体 → 抑制の体 は人の分布で始める：対数正規 1.65 ± 1.59mV（【ヒト】Wilbers 2023）
            版45 の撃ち方の単位に直す 1mV ＝ 1.98/20.9（静止の見込み −1.98【焼いた素子】を人の 20.9mV に当てる【C】）
            1本は「繋がり1つ」の量なので接触の数（c_exc）で割って w に。刈り込みで開通した辺も同じ分布から
  学び      予測誤差では学ばない。大きい繋がり（4.21mV 超）だけ、送り手が直近10歩に5回以上撃った歩ごとに ×0.984
            （【ヒト】Molnár 2016：40Hz の連発 40回で 0.52倍。弱まりは大きい繋がりに限る。強まりは人で示されていない）
  相手の数  抑制の体の興奮性の相手 180（ノブ【C】の見積もり。人の抑制細胞の入力の数は見つからない）
            錐体が【ヒト】覚醒 3.30Hz で撃つとき抑制の体が【ヒト】覚醒 4.70Hz になる数（Verma 2024。mc_inh_partner.py）
            （Le Van Quyen 2016 の覚醒 2.6 ／ 6.4Hz なら 248）。抑制性の相手は inh_frac を保つ
            海馬・嗅内 194（【ヒト】Le Van Quyen 2008 錐体 3.1Hz ／ 抑制 5.1Hz）
            ⟲ 人の重みのまま相手 9,100 だと、抑制の体が 84〜92% 撃って指令が 0%（コンテナの最小構成 simA、10-02）
  変えないもの 抑制の体の撃ち方 p ＞ 0.5（形B は別）／ 背内側視床の抑制の割合
コンテナの最小構成（運動の領野1つ・600体・相手 9,100、200歩ごとに 10歩 30体へ指令）
  錐体を 0.45%/歩 で撃たせる  抑制の体 0.1〜1% ／ 指令の体 100%
  錐体を 3.52%/歩（人の覚醒）  抑制の体 3〜8%（人 5.0%）／ 指令の体 100%
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
if "INH_N_EXC" in F[A]:
    sys.exit("⚠️ もう版46／47 の形になっている。何も書いていない")

# ── torch_area.py ──
a = F[A]
a = once(a, "import numpy as np\nimport torch\n", "import math\nimport numpy as np\nimport torch\n", "area の import")
a = once(a, "w_int8=False, fire_th=1.0, x_gain=1.0, tgt_by_fire=False, contacts_exc=3.3):\n",
         "w_int8=False, fire_th=1.0, x_gain=1.0, tgt_by_fire=False, contacts_exc=3.3,\n"
         "                 inh_n_exc=None):\n", "area の引数")
a = once(a, "W_INH = 0.02648 / 4.0\n", """W_INH = 0.02648 / 4.0

# 版47：錐体 → 抑制の体（形A の直し）【み選】10-02 B ／【み】10-02「Aを直して」
#   版45 の撃ち方の単位で 1mV：静止の見込み −1.98（焼いた素子の「何も撃っていない」）を
#   【ヒト】速い抑制細胞の静止から閾値まで 20.9mV（Wilbers 2023）に当てる【C】
K_MV = 1.98 / 20.9
#   相手の数：興奮性 180（ノブ【C】。人の抑制細胞の入力の数は見つからない）
#   錐体が【ヒト】覚醒 3.30Hz で撃つとき抑制の体が【ヒト】覚醒 4.70Hz になる数（Verma 2024。EPSP・閾値は Wilbers 2023）
#   ⟲ 版46（戻した）は 1,080：錐体 0.45%/歩・抑制 2Hz（Peyrache 2012）で出した数。0.45%/歩 は人の値ではなかった
INH_N_EXC = 180
#   海馬・嗅内：【ヒト】Le Van Quyen 2008（てんかん 11人、深部微小電極）ふだん 錐体 3.1Hz ／ 抑制 5.1Hz ⇒ 194
INH_N_EXC_HIPPO = 194
#   錐体 → 抑制の体 の1本：対数正規 1.65 ± 1.59mV（【ヒト】Wilbers 2023）を K_MV で版45 の単位に
INH_W_SIG = math.sqrt(math.log(1.0 + 0.96 ** 2))
INH_W_MU = -INH_W_SIG ** 2 / 2
INH_W_MEAN = 1.65 * K_MV
#   弱まり：大きい繋がり（【ヒト】4.21mV 超、Molnár 2016）だけ、送り手が 10歩に5回以上撃った歩ごとに
#   0.52 の 40乗根（【ヒト】40Hz の連発 40回で 0.52倍）。弱い繋がりは弱まらない
INH_LTD_TH = 4.21 * K_MV
INH_LTD_F = 0.52 ** (1.0 / 40.0)
INH_LTD_WIN, INH_LTD_N = 10, 5
""", "W_INH")
a = once(a, """        n_partner = min(n_partner, self.n_send)
        self.n_partner = n_partner
""", """        n_partner = min(n_partner, self.n_send)
        self.n_partner = n_partner
        # 版47：抑制の体の相手は 興奮性 INH_N_EXC ＋ 抑制性（inh_frac を保つ）
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
        # 表現・誤差は n_partner、抑制の体は n_pi（版47）
        src = torch.cat([pick(n_ch * 2, n_partner), pick(self.n_inh, n_pi)])
        recv = torch.cat([torch.arange(n_ch * 2, device=dev).repeat_interleave(n_partner),
                          (torch.arange(self.n_inh, device=dev) + n_ch * 2).repeat_interleave(n_pi)])
""", "area の辺の相手")
a = once(a, "        self.budget = torch.full((self.n_recv,), float(n_partner), device=dev)\n",
         "        self.budget = self.n_partner_vec.clone()      # 版47：受け手の種類ごとの相手の数\n",
         "area の budget")

# ── torch_group2.py ──
g = F[G]
g = once(g, "from torch_area import TAUS, THETA, DT_MS, W_INH\n",
         "from torch_area import TAUS, THETA, DT_MS, W_INH\n"
         "from torch_area import INH_W_SIG, INH_W_MU, INH_W_MEAN, INH_LTD_TH, INH_LTD_F, INH_LTD_WIN, INH_LTD_N   # 版47\n",
         "group2 の import")
g = once(g, """        self.n_partner = torch.cat([torch.full((a.n_recv,), float(a.n_partner), device=dev)
                                    for a in self.areas])
""", """        self.n_partner = torch.cat([a.n_partner_vec for a in self.areas])   # 版47：抑制の体は相手が少ない
""", "group2 の相手の数")
g = once(g, """        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
""", """        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
        # 版47：錐体 → 抑制の体 の重みは人の分布で始める（予測誤差では学ばない。形A の直し）
        ei = self.is_inh_r[self.e_recv.long()] & ~self.e_inh
        self.w[ei] = self._inh_w(self.e_recv[ei].long())
        # 送り手ごとの直近 INH_LTD_WIN 歩の発火（弱まりの条件を見るため）
        self.hist = torch.zeros((INH_LTD_WIN, self.n_send_all), dtype=torch.uint8, device=dev)
        self.hist_n = torch.zeros(self.n_send_all, dtype=torch.int16, device=dev)
        self.hist_i = 0
""", "group2 の重みの初め")
g = once(g, """    def _starts(self):
""", """    def _inh_w(self, r):
        \"\"\"版47：錐体 → 抑制の体 の1本（繋がり1つ。平均 1.65mV を版45 の単位にした INH_W_MEAN）を接触の数で割って w に\"\"\"
        z = torch.randn(len(r), device=self.dev, generator=self.gen)
        return torch.exp(INH_W_MU + INH_W_SIG * z) * INH_W_MEAN / self.c_exc[r]

    def _starts(self):
""", "group2 の _inh_w")
g = once(g, """        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        self.elig[cuts] = 0.0
""", """        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        ei = cuts[self.is_inh_r[self.e_recv[cuts].long()] & ~self.e_inh[cuts]]
        if len(ei):                    # 版47：開通した 錐体 → 抑制の体 も人の分布から
            self.w[ei] = self._inh_w(self.e_recv[ei].long())
        self.elig[cuts] = 0.0
""", "group2 の rewire")
g = once(g, """    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
""", """    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
        if first and self._base is not None:   # 版47：送り手ごとの直近 INH_LTD_WIN 歩の発火を数える
            now = (self._base > 0).to(torch.uint8)
            self.hist_n += now.to(torch.int16) - self.hist[self.hist_i].to(torch.int16)
            self.hist[self.hist_i] = now
            self.hist_i = (self.hist_i + 1) % INH_LTD_WIN
""", "group2 の発火の数え")
g = once(g, """            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            self.elig[e] += torch.where(self.e_inh[e], torch.zeros_like(gg), gg)
""", """            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            # 版47：錐体 → 抑制の体 も予測誤差では学ばない（下の弱まりだけ。⟲ 版45 は的 ＝ 入口の平均で学び上下した：形A）
            to_inh = self.is_inh_r[r]
            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg)
            # 版47：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）
            se = self.e_src[e].long()
            ltd = to_inh & ~self.e_inh[e] & (self.hist_n[se] >= INH_LTD_N) & \\
                (self.w[e] * self.c_exc[r] > INH_LTD_TH)
            if bool(ltd.any()):
                self.w[e[ltd]] *= INH_LTD_F
""", "group2 の学び")

# ── torch_brain.py ──
r = F[R]
r = once(r, "from torch_area import Prior, Area, steps, THETA\n",
         "from torch_area import Prior, Area, steps, THETA\n"
         "from torch_area import INH_N_EXC, INH_N_EXC_HIPPO   # 版47\n", "brain の import")
r = once(r, "            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),\n",
         "            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),\n"
         "            # 版47：抑制の体の興奮性の相手 180（【ヒト】覚醒の率から【C】）。海馬・嗅内は【ヒト】の率（錐体 3.1Hz・抑制 5.1Hz）から 194\n"
         "            inh_n_exc=(INH_N_EXC_HIPPO if nm in (\"歯状回\", \"CA3\", \"CA1\", \"嗅内浅\", \"嗅内深\") else INH_N_EXC),\n",
         "brain の相手の数")

# ── babble.py ──
b = F[B]
b = once(b, "VERSION = 45 ", "VERSION = 47 ", "VERSION")
h = "#      喃語の高さを中心 300Hz・発話ごと 3.3半音・発話の中 4.47半音に（設計案_脳幹.md §12-63〜73）\n"
b = once(b, h, h +
         "#  45  抑制の繋がりを人の強さで固定し、学ばない【み】10-01\n"
         "#  46  （戻した）抑制の体を閾値で撃たせる形。相手 1,080 が多すぎて言えなくなった\n"
         "#  47  形A の直し：錐体 → 抑制の体 を人の分布で始め、予測誤差では学ばず、大きい繋がりだけ連発で弱まる／\n"
         "#      抑制の体の興奮性の相手 180（【ヒト】覚醒の率から【C】。海馬・嗅内 194）【み選】B・【み】10-02\n",
         "版の履歴")

out = {A: a, G: g, B: b, R: r}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".pre47")
    open(k, "w", encoding="utf-8").write(v)
print("版47 にした（元は *.pre47 に残した）")
print("回すときは  python3 babble.py 100 --fresh")
