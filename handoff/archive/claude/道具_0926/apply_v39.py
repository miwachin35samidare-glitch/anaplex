"""版38 → 版39（arkhe の ~/anaplex で  python3 apply_v39.py ）

09-27 夜中の切り分け（measure_versions.py）で決めたもの。どれも「印が1つでも見つからなければ何も書かずに止まる」
  1 表現の的を下からの入力に戻す（版36 の B を外す）                    【み】09-27「まぁいいよそれで」
      B のもとでは、予測するシナプスが自分の的（自分の発火）を作れ、「いつも撃つ・いつも撃つと予測」が誤差0の解になる
      （論点台帳 論点2 の読み解き。版36 で表現の発火 50〜64%）
      ⚠️ 的を何にするかは人からは決まらない【C】。区画（予測する網と駆動する網）を作る段で見直す
      興奮の重み 0 以上（版36）は残す
  2 傾きの細胞の発火を 30%/歩 → 【ヒト】皮質の発火率 0.3Hz（1歩 0.32%）
      30% は chains.py の既定値で出どころの記録が無かった【C】。【ヒト】皮質の単一細胞 0.3〜0.9Hz の 30〜90倍
  3 調整する側の経路 4本は、予測 p には入るが表現を撃たせない（a、【み選】09-27「一旦aで」）
      連合→聴覚・高次→運動・高次→連合（上からの戻り）／ 指令の写し（運動→聴覚）
      【ヒト】遠い樹状突起は細胞体をほとんど興奮させない（Beaulieu-Laroche 2018）／ 話すと上側頭回に来る入力が減る（Chan 2014）
      ⚠️ ほかの経路（海馬・CA3 の再帰・運動の誤差→高次・皮質→歯状回・CA1 の戻り）は未決のまま撃たせる側
  4 驚きの不具合を直す（版38 の Claude の作りの誤り）
      組んだ直後は誤差が全部 0 なので「いつもの誤差」が 0 から始まり、学ぶ速さが最初の数千歩 数十〜280倍になっていた
      ⇒ 起きてからの歩数が SURPRISE_TAU に届くまでは、基準を起きてからの平均で作り、驚きは効かせない【C】
⚠️ 版38 の state.pt は使えない（版の番号で弾かれる）。--fresh で0から
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


F = {k: open(k, encoding="utf-8").read()
     for k in ("torch_group2.py", "torch_brain.py", "brainstem.py", "babble.py")}
if "VERSION = 38 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版38 ではない。何も書いていない（先に apply_v38.py）")

# ── torch_group2.py ──
g = F["torch_group2.py"]
g = once(g,
         "        # 表現の的は「自分がこの歩に撃ったか」（版36）。樹状突起が自分の細胞体を予測する（2区画の系統）\n"
         "        #   09-11「3つ入れた」の形（そのときは来たもので撃つので、自分の発火 ＝ 下からの入力だった）\n"
         "        #   版4 で合計で撃つにしたとき、的だけ下からの入力のまま残っていた\n"
         "        tgt = tgt.clone()\n"
         "        tgt[self.is_rep] = rep\n",
         "        # 表現の的は下からの入力（版39。版36 の「自分の発火」を外した）\n"
         "        #   ⟲ 自分の発火を的にすると、学んだシナプスがそのまま駆動に入るので「いつも撃つ」が誤差0の解になり、\n"
         "        #      全領野が撃ちっぱなしになった（版36〜38、表現 50〜64%）\n"
         "        #   ⚠️ 何を的にするかは人からは決まらない【C】。区画（予測する網と駆動する網）を作る段で見直す\n",
         "表現の的（版36）")
g = once(g, "from torch_area import TAUS, THETA\n",
         "from torch_area import TAUS, THETA, DT_MS\n"
         "\n"
         "# 傾きの細胞の発火（版39）：【ヒト】皮質の単一細胞の発火率 0.3Hz を1歩に（0.32%）\n"
         "#   ⟲ 版38 までは 30%/歩（chains.py の既定値。出どころの記録なし【C】。人の 30〜90倍）\n"
         "RAMP_P = 0.3 * DT_MS / 1000.0\n", "group2 の import")
g = once(g, "< y * 0.3\n", "< y * RAMP_P\n", "傾きの細胞（上がる側）")
g = once(g, "< (1 - y) * 0.3\n", "< (1 - y) * RAMP_P\n", "傾きの細胞（下がる側）")
g = once(g, "        self.send_gain = torch.ones(self.n_send_all, device=dev)",
         "        self.mod_send = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 調整する側の送り手（版39。torch_brain が印を付ける）\n"
         "        self.send_gain = torch.ones(self.n_send_all, device=dev)", "group2 の send_gain")
g = once(g, "        ihn = torch.zeros(self.n_recv_all, device=self.dev)\n",
         "        ihn = torch.zeros(self.n_recv_all, device=self.dev)\n"
         "        exc_mod = torch.zeros(self.n_recv_all, device=self.dev)   # 調整する側から来た興奮（版39）\n", "group2 の ihn")
g = once(g, "            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n",
         "            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n"
         "            mm = self.mod_send[self.e_src[e]] & ~self.e_inh[e]\n"
         "            exc_mod = exc_mod.index_add(0, r[mm], amp[mm])\n", "group2 の足し込み")
g = once(g, "        exc = exc * self.scale; ihn = ihn * self.scale\n",
         "        exc = exc * self.scale; ihn = ihn * self.scale\n"
         "        exc_mod = exc_mod * self.scale\n", "group2 の scale")
g = once(g, "            exc = exc * self.exc_gain\n",
         "            exc = exc * self.exc_gain\n"
         "            exc_mod = exc_mod * self.exc_gain\n", "group2 の exc_gain")
g = once(g, "        drive = (exc[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])\n",
         "        # 調整する側（上からの戻り・指令の写し）は予測 p には入るが、表現を撃たせない（版39）\n"
         "        drive = ((exc - exc_mod)[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])\n",
         "表現の駆動")

# ── torch_brain.py ──
t = F["torch_brain.py"]
t = once(t, "        self.bs = BS.Brainstem(self)\n",
         "        self.bs = BS.Brainstem(self)\n"
         "        # 調整する側の経路（版39）：予測 p には入るが、表現を撃たせない\n"
         "        #   【ヒト】遠い樹状突起は細胞体をほとんど興奮させない（Beaulieu-Laroche 2018）／ 上からは調整する入力（09-11）\n"
         "        #   【ヒト】話すと上側頭回に来る入力が減る（Chan 2014）⇒ 指令の写しも調整する側\n"
         "        #   ⚠️ ほかの経路（海馬・CA3 の再帰・運動の誤差→高次・皮質→歯状回・CA1 の戻り）は未決のまま撃たせる側\n"
         "        n = self.n\n"
         "        for area, first, cnt in ((\"聴覚\", n[\"聴覚\"], n[\"運動\"]),                  # 指令の写し\n"
         "                                 (\"聴覚\", n[\"聴覚\"] + n[\"運動\"], n[\"連合\"]),      # 連合から\n"
         "                                 (\"運動\", n[\"運動\"], n[\"高次\"]),                  # 高次から\n"
         "                                 (\"連合\", n[\"連合\"], n[\"高次\"])):                 # 高次から（CA1 の戻りは含めない）\n"
         "            self.g.mod_send[self._send_rows(area, first, cnt)] = True\n",
         "brain の脳幹の後")
t = once(t, "    @property\n    def n_cells(self):\n",
         "    def _send_rows(self, area, first, n):\n"
         "        \"\"\"その領野の送り元の行 first〜first+n が持つ送り手の番号（時間細胞・傾きの細胞の分と、短い遅れの分）\"\"\"\n"
         "        g = self.g\n"
         "        i = NAMES.index(area)\n"
         "        buf_off = sum(self.areas[nm].n_src * self.areas[nm].D for nm in NAMES[:i])\n"
         "        a = self.areas[area]\n"
         "        per = a.ch.n_cell + a.ch.n_ramp\n"
         "        rows = torch.arange(first, first + n, device=self.dev)\n"
         "        tc = int(g.off_send[i]) + rows.unsqueeze(1) * per + torch.arange(per, device=self.dev)\n"
         "        bf = int(g.off_send[-1]) + buf_off + rows.unsqueeze(1) * a.D + torch.arange(a.D, device=self.dev)\n"
         "        return torch.cat([tc.reshape(-1), bf.reshape(-1)])\n"
         "\n"
         "    @property\n    def n_cells(self):\n", "brain の n_cells")

# ── brainstem.py（驚きの不具合）──
s = F["brainstem.py"]
s = once(s, "        self.base_aud = None         # 同じく聴覚だけ\n",
         "        self.base_aud = None         # 同じく聴覚だけ\n"
         "        self.n_wake = 0              # 起きていた歩数（基準ができたかを見る。版39）\n", "脳幹の init")
s = once(s,
         "            if self.base is None:\n"
         "                self.base, self.base_aud = s.clone(), aud.clone()\n"
         "            a = 1.0 / SURPRISE_TAU\n",
         "            # ⟲ 版38 は最初の歩の誤差（組んだ直後は 0）を基準にしたので、学ぶ速さが最初の数千歩 数十〜280倍になった\n"
         "            #   ⇒ SURPRISE_TAU 歩までは起きてからの平均で基準を作り、驚きは効かせない【C】（版39）\n"
         "            if self.base is None:\n"
         "                self.base, self.base_aud = torch.zeros_like(s), torch.zeros_like(aud)\n"
         "            self.n_wake += 1\n"
         "            a = max(1.0 / SURPRISE_TAU, 1.0 / self.n_wake)\n", "脳幹の基準")
s = once(s,
         "            self.g.eta_gain = 1.0 + SURPRISE_K * (ratio - 1.0).clamp(min=0.0)\n",
         "            if self.n_wake < SURPRISE_TAU:\n"
         "                self.g.eta_gain = torch.ones((), device=self.dev)\n"
         "            else:\n"
         "                self.g.eta_gain = 1.0 + SURPRISE_K * (ratio - 1.0).clamp(min=0.0)\n", "脳幹の驚き")
s = once(s, '        return {"state": self.state, "base": self.base, "base_aud": self.base_aud}\n',
         '        return {"state": self.state, "base": self.base, "base_aud": self.base_aud, "n_wake": self.n_wake}\n',
         "脳幹の保存")
s = once(s, '        self.base, self.base_aud = d.get("base"), d.get("base_aud")\n',
         '        self.base, self.base_aud = d.get("base"), d.get("base_aud")\n'
         '        self.n_wake = d.get("n_wake", 0)\n', "脳幹の読み込み")

# ── babble.py ──
b = F["babble.py"]
b = once(b, "VERSION = 38 ", "VERSION = 39 ", "VERSION")
h = "#      眠りの長さを【ヒト】乳児の比に（約10万歩、1周 約6万歩の半分が REM【み選】）\n"
b = once(b, h, h +
         "#  39  表現の的を下からの入力に戻した（版36 の「自分の発火」は、学んだシナプスが駆動に入る今の撃ち方では\n"
         "#      「いつも撃つ」が誤差0の解になり撃ちっぱなしになった。的は人から決まらない【C】）／\n"
         "#      傾きの細胞を【ヒト】皮質の発火率 0.3Hz に（前は 30%/歩【C】）／\n"
         "#      上からの戻り3本と指令の写しは表現を撃たせない（予測 p には入る）／ 驚きの基準が 0 から始まる不具合を直した\n",
         "版の履歴")

out = {"torch_group2.py": g, "torch_brain.py": t, "brainstem.py": s, "babble.py": b}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".v38")
    open(k, "w", encoding="utf-8").write(v)
print("版39 にした（元は *.v38 に残した）")
print("回すときは  python3 babble.py 100 --fresh")
