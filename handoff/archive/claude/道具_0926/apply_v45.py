"""版44 → 版45：抑制は学ばない。人の強さで始まり、そのまま
  arkhe の ~/anaplex で  python3 apply_v45.py

設計と出どころは claude/設計案_脳幹.md §12-80〜83
  【み】10-01「まぁいいよ、学ばない形にしても」
  torch_area.py   W_INH ＝ 0.02648 ÷ 4.0（【ヒト】側頭葉の抑制性 → 興奮性 18組 g_syn/g_input、1接続。Campagnola 2022）
                  抑制の辺は W_INH で始まる ／ 学びで変えない ／ 刈り込みで開通した抑制も W_INH
  torch_group2.py 同じ（学びの跡に抑制の分を入れない。開通した抑制は W_INH）
  babble.py       VERSION 45（重みの作りが変わる。版44 の state.pt は読めない）
先に apply_v44.py を当てておくこと。印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import json
import shutil
import sys

HUNKS = json.loads(r'''{
"torch_area.py": [
[
"import numpy as np\nimport torch\n\nDT_MS = 10.67\n",
"import numpy as np\nimport torch\n\n# 抑制の1接続の強さ（学ばない。版45【み】10-01「学ばない形にしても」）\n#   【ヒト】側頭葉 MTG の抑制性 → 興奮性 18組：g_syn / g_input 平均 0.02648（1接続。Campagnola 2022 ／ synapse_model_human.json）\n#   接触 4.0 で割って1本の重さに（amp ＝ 4.0 × w）。短期可塑性も同じ表から（静止時の1発目 ＝ 1）\n#   ⚠️ 速い籠細胞だけなら 0.69nS ≈ 漏れの 0.07倍（Szegedi 2017）。こちらは型を混ぜた平均\n#   ⚠️ 人の抑制 → 錐体 のシナプスが学ぶかは測られていない（Mansvelder 2019）。学ばないのは【み】の判断\nW_INH = 0.02648 / 4.0\n\nDT_MS = 10.67\n"
],
[
"        self.is_inh_send = is_inh_send\n        self.e_inh = is_inh_send[self.e_src_sorted]\n        recv_inh = torch.zeros(self.n_recv, dtype=torch.bool, device=dev)\n        recv_inh[n_ch * 2:] = True\n",
"        self.is_inh_send = is_inh_send\n        self.e_inh = is_inh_send[self.e_src_sorted]\n        self.w[self.e_inh] = W_INH                 # 版45：抑制は人の強さで始まり、学ばない\n        recv_inh = torch.zeros(self.n_recv, dtype=torch.bool, device=dev)\n        recv_inh[n_ch * 2:] = True\n"
],
[
"            r = self.e_recv[e].long()\n            gg = self.eta * err_sig[r]\n            shunt = (exc / (1.0 + ihn) ** 2)[r]\n            self.w[e] += torch.where(self.e_inh[e], -gg * shunt, gg)\n            self.w[e] = torch.where(self.e_inh[e], self.w[e].clamp(min=0.0), self.w[e])\n            a = 1.0 / self.homeo_tau\n            cnt = torch.zeros(self.n_recv, device=self.dev)\n",
"            r = self.e_recv[e].long()\n            gg = self.eta * err_sig[r]\n            self.w[e] += torch.where(self.e_inh[e], torch.zeros_like(gg), gg)   # 版45：抑制は学ばない\n            a = 1.0 / self.homeo_tau\n            cnt = torch.zeros(self.n_recv, device=self.dev)\n"
],
[
"        self.e_recv, self.w = self.e_recv[o], self.w[o]\n        self.e_inh = self.is_inh_send[self.e_src_sorted]\n        if self.stp is not None:\n            self.stp.reorder(o)\n",
"        self.e_recv, self.w = self.e_recv[o], self.w[o]\n        self.e_inh = self.is_inh_send[self.e_src_sorted]\n        self.w[self.e_inh] = W_INH                 # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）\n        if self.stp is not None:\n            self.stp.reorder(o)\n"
]
],
"torch_group2.py": [
[
"import math\nimport torch\nfrom torch_area import TAUS, THETA, DT_MS\n\n# 傾きの細胞の発火（版39）：【ヒト】皮質の単一細胞の発火率 0.3Hz を1歩に（0.32%）\n",
"import math\nimport torch\nfrom torch_area import TAUS, THETA, DT_MS, W_INH\n\n# 傾きの細胞の発火（版39）：【ヒト】皮質の単一細胞の発火率 0.3Hz を1歩に（0.32%）\n"
],
[
"        cuts = pos[cuts]\n        self.w[cuts] = 0.0             # 開通の値（決まったこと 09-11：開通は w0、学びの跡は 0）\n        self.elig[cuts] = 0.0\n        if self.stp is not None:\n",
"        cuts = pos[cuts]\n        self.w[cuts] = 0.0             # 開通の値（決まったこと 09-11：開通は w0、学びの跡は 0）\n        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）\n        self.elig[cuts] = 0.0\n        if self.stp is not None:\n"
],
[
"            r = self.e_recv[e].long()\n            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain：驚きで上がる（青斑核。版42 で戻した）\n            shunt = (exc / (1.0 + ihn) ** 2)[r]\n            # 学びの門（版40）：その歩の学びは跡にためる。重みを変えるのは脳幹の第三の信号の分だけ\n            #   【ヒト】皮質の古典的な対ではドーパミンを入れて初めて強まる（前の作り 生-217）\n            self.elig.mul_(self.elig_decay)\n            self.elig[e] += torch.where(self.e_inh[e], -gg * shunt, gg)\n            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）\n            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))\n",
"            r = self.e_recv[e].long()\n            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain：驚きで上がる（青斑核。版42 で戻した）\n            # 学びの門（版40）：その歩の学びは跡にためる。重みを変えるのは脳幹の第三の信号の分だけ\n            #   【ヒト】皮質の古典的な対ではドーパミンを入れて初めて強まる（前の作り 生-217）\n            self.elig.mul_(self.elig_decay)\n            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）\n            self.elig[e] += torch.where(self.e_inh[e], torch.zeros_like(gg), gg)\n            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）\n            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))\n"
]
],
"babble.py": [
[
"\nSTATE = \"state.pt\"\nVERSION = 44          # コードの版。網の作りを変えたら上げる\n#   1  最初の torch 版\n#   2  周回のあいだも発火が伝わるように直した\n",
"\nSTATE = \"state.pt\"\nVERSION = 45          # コードの版。網の作りを変えたら上げる\n#   1  最初の torch 版\n#   2  周回のあいだも発火が伝わるように直した\n"
]
]
}''')
NEW = {}
import os
for f in NEW:
    if os.path.exists(f):
        sys.exit(f"⚠️ {f} がもうある。何も書いていない（もう当ててあるか）")
out = {}
for f, hs in HUNKS.items():
    s = open(f, encoding="utf-8").read()
    for i, (old, new) in enumerate(hs):
        c = s.count(old)
        if c != 1:
            sys.exit(f"⚠️ {f} の {i+1} 番目の印が {c} 回。何も書いていない（もう当ててあるか、43d がまだ）\n--- 探した行 ---\n{old}")
        s = s.replace(old, new)
    ast.parse(s)
    out[f] = s
for f, s in out.items():
    shutil.copy(f, f + ".v44")
    open(f, "w", encoding="utf-8").write(s)
for f, s in NEW.items():
    ast.parse(s)
    open(f, "w", encoding="utf-8").write(s)
print("版45 にした（元は *.v44 に残した）")
print("次：python3 measure_v43d.py 100（抑制の重み/体 は一定のままになるはず）")
