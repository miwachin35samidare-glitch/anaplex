"""版43a の上に：辺の番号の持ち方を小さくする（arkhe の ~/anaplex で  python3 apply_v43m.py ）【み】09-29「2はやってもいい」

  辺の送り手・受け手の番号   64bit → 32bit（番号は1つもずれない）
  短期可塑性の行の番号       64bit → 16bit（表は172行）
  送り手ごとの始まりは数えて作る（64bit の写しを作らない）
【測】コンテナ（CPU、相手150本、210歩・刈り込み2回）で、前の持ち方と 重み・辺・短期可塑性・誤差の発火 が完全に一致
値も精度も変わらないので VERSION は 43 のまま（版43a と同じ網）
先に apply_v43a.py を当てておくこと。印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import json
import shutil
import sys

HUNKS = json.loads(r'''{
"torch_group2.py": [
[
"        self.pool_send = remap\n        o = torch.argsort(src)\n        self.e_src, self.e_recv, self.w, self.e_inh = src[o], recv[o], w[o], inh[o]\n        self.start = torch.zeros(self.n_send_all + 1, dtype=torch.long, device=dev)\n        self.start.scatter_add_(0, self.e_src + 1, torch.ones(len(o), dtype=torch.long, device=dev))\n        self.start = self.start.cumsum(0)\n\n        self.stp = None\n",
"        self.pool_send = remap\n        o = torch.argsort(src)\n        # 辺の番号は 32bit で持つ（版43a-m【み】09-29 許可。番号は1つもずれない。64bit の半分）\n        self.e_src, self.e_recv, self.w, self.e_inh = src[o].int(), recv[o].int(), w[o], inh[o]\n        del src, recv, w, inh\n        self.start = self._starts()\n\n        self.stp = None\n"
],
[
"        self._release_areas()\n\n    def _release_areas(self):\n        \"\"\"まとめ終わった領野ごとの配列を手放す（中身は group に移っている）\n",
"        self._release_areas()\n\n    def _starts(self):\n        \"\"\"送り手ごとの辺の始まり（e_src は並んでいる）。数えるだけなので 64bit の写しを作らない（版43a-m）\"\"\"\n        cnt = torch.bincount(self.e_src, minlength=self.n_send_all)\n        return torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev), cnt.cumsum(0)])\n\n    def _release_areas(self):\n        \"\"\"まとめ終わった領野ごとの配列を手放す（中身は group に移っている）\n"
],
[
"        self.n_cut = getattr(self, \"n_cut\", 0) + len(cuts)      # 何本 切ったか\n        # 生やす先：受け手と同じ領野の送り元から一様に（種類は送り元の割合のまま）\n        ar = self.area_of[self.e_recv[cuts]]\n        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)\n        for i in range(len(self.areas)):\n",
"        self.n_cut = getattr(self, \"n_cut\", 0) + len(cuts)      # 何本 切ったか\n        # 生やす先：受け手と同じ領野の送り元から一様に（種類は送り元の割合のまま）\n        ar = self.area_of[self.e_recv[cuts].long()]\n        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)\n        for i in range(len(self.areas)):\n"
],
[
"                new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev,\n                                            generator=self.gen)]\n        self.e_src[cuts] = new\n        o = torch.argsort(self.e_src)\n        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]\n        self.elig = self.elig[o]\n        self.e_inh = self.is_inh_all[self.e_src]\n        # ⟲ 版40 で直した：並べ直したあとに、並べ直す前の位置 cuts で 0 にしていた。\n        #    開通した辺は切った辺の重みを持ち越し、代わりに無関係の辺の重みが 100歩ごとに受け手の数だけ消えていた\n",
"                new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev,\n                                            generator=self.gen)]\n        self.e_src[cuts] = new.to(self.e_src.dtype)\n        o = torch.argsort(self.e_src)\n        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]\n        self.elig = self.elig[o]\n        self.e_inh = torch.index_select(self.is_inh_all, 0, self.e_src)\n        # ⟲ 版40 で直した：並べ直したあとに、並べ直す前の位置 cuts で 0 にしていた。\n        #    開通した辺は切った辺の重みを持ち越し、代わりに無関係の辺の重みが 100歩ごとに受け手の数だけ消えていた\n"
],
[
"        if self.stp is not None:\n            self.stp.reorder(o)\n        self.start = torch.zeros(self.n_send_all + 1, dtype=torch.long, device=self.dev)\n        self.start.scatter_add_(0, self.e_src + 1,\n                                torch.ones(len(self.w), dtype=torch.long, device=self.dev))\n        self.start = self.start.cumsum(0)\n\n    @torch.no_grad()\n",
"        if self.stp is not None:\n            self.stp.reorder(o)\n        self.start = self._starts()\n\n    @torch.no_grad()\n"
],
[
"        if len(e):\n            # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0\n            amp = torch.where(self.e_inh[e], 4.0, self.c_exc[self.e_recv[e]]) * self.w[e]\n            if self.stp is not None:\n                if first:\n",
"        if len(e):\n            # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0\n            re_ = self.e_recv[e].long()\n            se_ = self.e_src[e].long()\n            amp = torch.where(self.e_inh[e], 4.0, self.c_exc[re_]) * self.w[e]\n            if self.stp is not None:\n                if first:\n"
],
[
"                    self._amp[e] = self.stp.release_at(e, self.t)\n                amp = amp * self._amp[e]\n            amp = amp * self.send_gain[self.e_src[e]]   # 脳幹：眠りで上からの戻りを落とす（版38）\n            # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）\n            #   興奮 ＝ (和 ＋ |和|) / 2 にはできないので、抑制の総量を別に持つ必要がある\n            #   ⇒ 受け手の数だけの配列を2本用意して、1回の scatter_add にまとめる\n            r = self.e_recv[e]\n            both = torch.zeros(self.n_recv_all * 2, device=self.dev)\n            both.scatter_add_(0, r + self.n_recv_all * self.e_inh[e].long(), amp)\n            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n            mm = self.mod_send[self.e_src[e]] & ~self.e_inh[e]\n            exc_mod = exc_mod.index_add(0, r[mm], amp[mm])\n        exc = exc * self.scale; ihn = ihn * self.scale\n",
"                    self._amp[e] = self.stp.release_at(e, self.t)\n                amp = amp * self._amp[e]\n            amp = amp * self.send_gain[se_]   # 脳幹：眠りで上からの戻りを落とす（版38）\n            # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）\n            #   興奮 ＝ (和 ＋ |和|) / 2 にはできないので、抑制の総量を別に持つ必要がある\n            #   ⇒ 受け手の数だけの配列を2本用意して、1回の scatter_add にまとめる\n            r = re_\n            both = torch.zeros(self.n_recv_all * 2, device=self.dev)\n            both.scatter_add_(0, r + self.n_recv_all * self.e_inh[e].long(), amp)\n            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n            mm = self.mod_send[se_] & ~self.e_inh[e]\n            exc_mod = exc_mod.index_add(0, r[mm], amp[mm])\n        exc = exc * self.scale; ihn = ihn * self.scale\n"
],
[
"\n        if learn and len(e):\n            r = self.e_recv[e]\n            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain：驚きで上がる（青斑核。版42 で戻した）\n            shunt = (exc / (1.0 + ihn) ** 2)[r]\n",
"\n        if learn and len(e):\n            r = self.e_recv[e].long()\n            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain：驚きで上がる（青斑核。版42 で戻した）\n            shunt = (exc / (1.0 + ihn) ** 2)[r]\n"
]
],
"torch_area.py": [
[
"            if n:\n                row[m] = off[k] + torch.randint(0, len(d[\"行\"][k]), (n,), device=dev, generator=gen)\n        self.row = row\n        self.deplete = self.tbl[row, 1] < 0\n        self.state = torch.where(self.deplete, torch.ones(n_edge, device=dev),\n",
"            if n:\n                row[m] = off[k] + torch.randint(0, len(d[\"行\"][k]), (n,), device=dev, generator=gen)\n        self.row = row.to(torch.int16)       # 表は172行なので 16bit で足りる（版43a-m。使うときに広げる）\n        self.deplete = self.tbl[row, 1] < 0\n        self.state = torch.where(self.deplete, torch.ones(n_edge, device=dev),\n"
],
[
"    def release_at(self, e, t):\n        \"\"\"触った辺だけ、最後に触った歩からまとめて減らして放出する\"\"\"\n        r = self.row[e]\n        n = (t - self.last[e]).float()\n        dep = self.deplete[e]\n",
"    def release_at(self, e, t):\n        \"\"\"触った辺だけ、最後に触った歩からまとめて減らして放出する\"\"\"\n        r = self.row[e].long()\n        n = (t - self.last[e]).float()\n        dep = self.deplete[e]\n"
]
]
}''')
out = {}
for f, hs in HUNKS.items():
    s = open(f, encoding="utf-8").read()
    for i, (old, new) in enumerate(hs):
        c = s.count(old)
        if c != 1:
            sys.exit(f"⚠️ {f} の {i+1} 番目の印が {c} 回。何も書いていない（もう当ててあるか、ファイルが違う）\n--- 探した行 ---\n{old}")
        s = s.replace(old, new)
    ast.parse(s)
    out[f] = s
if "VERSION = 43 " not in open("babble.py", encoding="utf-8").read():
    sys.exit("⚠️ 先に apply_v43a.py を当てること。何も書いていない")
for f, s in out.items():
    shutil.copy(f, f + ".v43a")
    open(f, "w", encoding="utf-8").write(s)
print("辺の番号を 32bit・短期可塑性の行を 16bit にした（元は *.v43a に残した）")
print("次：python3 measure_size.py（取り置きの山を見る）")
