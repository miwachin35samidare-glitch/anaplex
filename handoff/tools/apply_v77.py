"""版77：torch.compile を戻す（番号の範囲の確かめを外す）＋ 版76 で足した CPU の待ちを消す（速さだけ。式は同じ）
（~/anaplex で。版76b を当てた上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v77.py
【測】probe_compile2（10-14）：番号で引く・書く計算を compile すると全部 起動できない（driver error 401）。番号なしは動く
【測】probe_compile3（10-14）：核の中の「番号が外に出ていないか」の assert を外すと、9つとも動く（dynamic・大きさを決める の両方）
  ⇒ assert_indirect_indexing を外す。assert は確かめのためだけで、計算の値には関わらない
直すもの
  1 _run：torch.compile（dynamic）で回す。版76 の「まとめられないときは元の形で」は GPU では戻れなかったので消した（止まれば止まる）
  2 学びの eta と 減り^(−k) を GPU の箱へ fill_ で入れる（版76 の torch.as_tensor は1歩1回 CPU が待っていた【測】measure_speed75）
"""
import ast, pathlib, re, sys

G2 = [
    ("def _run(fn, *a):\n"
     "    \"\"\"版76b：まとめない形で回す（GPU で Triton の核が起動できない【測】probe_compile 10-14。直したら戻す）\"\"\"\n"
     "    return fn(*a)\n",
     "# 版77：核の中の「番号が外に出ていないか」の assert を外す（RX 9070 XT で、assert の入った核が起動できない）\n"
     "#   【測】probe_compile2・3（10-14）：assert ありは番号で引く計算が全部 ✗（driver error 401）、外すと全部 ○。値には関わらない\n"
     "import torch._inductor.config as _ind_cfg\n"
     "_ind_cfg.assert_indirect_indexing = False\n\n\n"
     "def _run(fn, *a):\n"
     "    \"\"\"版77：torch.compile（dynamic）で回す\"\"\"\n"
     "    return _cc(fn)(*a)\n"),
    ("            g_ = torch.as_tensor(self.eta * self.eta_gain, dtype=self.w.dtype, device=self.dev)\n"
     "            pw_ = torch.as_tensor(self.elig_decay ** (-k), dtype=self.w.dtype, device=self.dev)\n",
     "            if getattr(self, \"_g_buf\", None) is None:     # 版77：値は GPU の箱へ fill_ で入れる（CPU が待たない。値は同じ）\n"
     "                self._g_buf = torch.zeros((), dtype=self.w.dtype, device=self.dev)\n"
     "                self._pw_buf = torch.zeros((), dtype=self.w.dtype, device=self.dev)\n"
     "            eg = self.eta * self.eta_gain\n"
     "            g_ = self._g_buf.copy_(eg) if torch.is_tensor(eg) else self._g_buf.fill_(eg)\n"
     "            pw_ = self._pw_buf.fill_(self.elig_decay ** (-k))\n"),
]
BB = [("VERSION = 76          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75)",
       "VERSION = 77          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76)"),
      ("#      網の辺の計算（足し込み・学び・まとめて足す）を torch.compile で1つに（式・足す順は同じ）【測】版75 1歩 33.0ms\n",
       "#      網の辺の計算（足し込み・学び・まとめて足す）を torch.compile で1つに（式・足す順は同じ）【測】版75 1歩 33.0ms\n"
       "#      ⟲ GPU で起動できなかった（driver error 401）⇒ 版76b で止めた（Claude の誤り：逃げ道が GPU では役に立たない作り）\n"
       "#  77  torch.compile を戻した：核の中の番号の assert を外す【測】probe_compile3 で9つとも動く ／ 学びの値は fill_ で（CPU が待たない）\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py", "measure_speed75.py")


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない（版76b を当てた？）")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 76:
        sys.exit(f"⚠️ babble.py が版76 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版77 を当てた（torch_group2・babble）。torch.compile を戻した（番号の assert を外す）・学びの値で CPU が待たない")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "77" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['77'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版77 でも回るようにした")
