"""版92：網を組む途中と刈り込みの GPU のメモリの山を下げる（値・並び・くじは同じ）
（~/anaplex で。版91 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v92.py
【み】10-15「予算12gb」／「いいよ」（基底核・小脳を足す前に、中身を変えずに山を下げる）
【測】arkhe watch_vram2（版91）：始める前 1.10GB ／ 組む途中 9.40GB（Group2 を組むところ）／ 刈り込み 10.16GB（並べ直しの stp.reorder）
  基底核 A ＋ 小脳で辺 +25〜39% ⇒ このままだと 12.4〜13.7GB の見込み【計算】
直すもの（torch_group2）
  組む途中（Group2.__init__）
    1 受け手の番号を最初から 32bit で（前：64bit で作って並べたあと 32bit に）
    2 送り手の番号も 32bit にしてから並べ替え（値は 120万未満。GPU の並べ替えは同じ値を番号順に保つ【測】check_v91_gpu ⇒ 並びは同じ。check_v92_gpu で確かめる）
    3 並べたあとの写しを1本ずつ作って、元をすぐ手放す（前：4本の写しを全部作ってから手放した）
  刈り込み（rewire）
    4 一番弱い辺を選び終えたら inf の列を手放す
    5 並べ直しの番号 o と位置 pos を 32bit で（値は辺の数 7,195万 ＜ 21億）
    6 短期可塑性の5本（row・state・fac・last・deplete）を、同じ置き場所へ1本ずつ写す（前：5本の新しい写しを全部作ってから入れ替え）
      ⇒ 他の辺の配列（e_src ほか、版79）と同じやり方。グラフは短期可塑性の配列を覚えていない（放出はグラフの外）
⚠️ 網の作りは同じ（state.pt は読める）。VERSION 92（SAME_NET に 91）
"""
import ast, pathlib, re, sys

G2 = [
    ("        src = torch.cat([remap[i][a.e_src_sorted] for i, a in enumerate(self.areas)])\n"
     "        recv = torch.cat([a.e_recv.long() + self.off_recv[i] for i, a in enumerate(self.areas)])\n",
     "        # 版92：番号は最初から 32bit（値は同じ。組む途中の GPU の山を下げる）\n"
     "        src = torch.cat([remap[i][a.e_src_sorted].int() for i, a in enumerate(self.areas)])\n"
     "        recv = torch.cat([(a.e_recv.long() + self.off_recv[i]).int() for i, a in enumerate(self.areas)])\n"),
    ("        self.e_src, self.e_recv, self.w, self.e_inh = src[o].int(), recv[o].int(), w[o], inh[o]\n"
     "        del src, recv, w, inh\n",
     "        # 版92：写しを1本ずつ作り、元をすぐ手放す（⟲ 版91 まで4本の写しを全部作ってから）\n"
     "        self.e_src = src[o]; del src\n"
     "        self.e_recv = recv[o]; del recv\n"
     "        self.w = w[o]; del w\n"
     "        self.e_inh = inh[o]; del inh\n"),
    ("            for k in (\"row\", \"state\", \"fac\", \"last\", \"deplete\"):\n"
     "                setattr(g, k, torch.cat([getattr(a.stp, k) for a in self.areas])[o])\n"
     "            self.stp = g\n",
     "            for k in (\"row\", \"state\", \"fac\", \"last\", \"deplete\"):\n"
     "                setattr(g, k, torch.cat([getattr(a.stp, k) for a in self.areas])[o])\n"
     "            self.stp = g\n"
     "        del o                              # 版92\n"),
    ("                                                                    generator=self.gen), inf))   # でたらめな抑制1本\n",
     "                                                                    generator=self.gen), inf))   # でたらめな抑制1本\n"
     "        del inf                            # 版92：選び終えたら手放す\n"),
    ("        o = torch.argsort(self.e_src)\n"
     "        # 版79：並べ直しは同じ置き場所へ写す",
     "        o = torch.argsort(self.e_src).int()     # 版92：番号の列は 32bit で持つ（値は辺の数未満）\n"
     "        # 版79：並べ直しは同じ置き場所へ写す"),
    ("        pos = torch.empty_like(o)\n"
     "        pos[o] = torch.arange(len(o), device=self.dev)\n"
     "        cuts = pos[cuts]\n",
     "        pos = torch.empty_like(o)\n"
     "        pos[o] = torch.arange(len(o), device=self.dev, dtype=o.dtype)    # 版92：32bit\n"
     "        cuts = pos[cuts].long()\n"
     "        del pos\n"),
]
TA = [("    def reorder(self, o):\n"
       "        self.row, self.state, self.fac = self.row[o], self.state[o], self.fac[o]\n"
       "        self.last, self.deplete = self.last[o], self.deplete[o]\n",
       "    def reorder(self, o):\n"
       "        # 版92：同じ置き場所へ1本ずつ写す（⟲ 版91 まで5本の新しい写しを全部作ってから入れ替え ⇒ 刈り込みの山）\n"
       "        for t in (self.row, self.state, self.fac, self.last, self.deplete):\n"
       "            t.copy_(t[o])\n")]
BB = [("VERSION = 91          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90)",
       "VERSION = 92          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91)"),
      ("#  91  刈り込みの受け手ごとの一番小さい辺を並べ替えなしで（選ぶ辺は同じ。GPU の山 14.9GB → 下げる）【み】10-15「予算12gb」\n",
       "#  91  刈り込みの受け手ごとの一番小さい辺を並べ替えなしで（選ぶ辺は同じ。GPU の山 14.9GB → 下げる）【み】10-15「予算12gb」\n"
       "#  92  組む途中と刈り込みの並べ直しで、番号を 32bit・写しを1本ずつ（値は同じ）【み】10-15「いいよ」【測】版91 山 10.16GB\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py", "measure_speed75.py")


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
    import babble
    if babble.VERSION != 91:
        sys.exit(f"⚠️ babble.py が版91 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("torch_area.py", TA), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版92 を当てた（torch_group2・torch_area・babble）。組む途中と刈り込みのメモリの山を下げた（値は同じ）", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "92" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['92'])}):" + s[m.end():]
            q.write_text(s)
            try:
                print(f"   {f}：版92 でも回るようにした", flush=True)
            except BrokenPipeError:
                pass
