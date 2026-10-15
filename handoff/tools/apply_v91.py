"""版91：刈り込みの「受け手ごとに一番小さい辺」を、並べ替えなしで塊ごとに（選ぶ辺は同じ。GPU のメモリの山を下げる）
（~/anaplex で。版90 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v91.py
【み】10-15「予算12gbって言ってあると思うんだけど」
【測】arkhe watch_vram2（版90・学んでいない網・喃語10回）：Linux 側の山 14.93GB ＝ 歩 1000 の刈り込み（rewire → _first_per_recv）
  _first_per_recv は 7,195万本の辺を2回 argsort（64bit の番号の列 約0.58GB × 4本 ＋ 並べ替えの作業場）⇒ ふだん 6.5GB から +8.4GB
  組む途中の山は 9.48GB（_starts）
直すもの（torch_group2._first_per_recv）
  前：key で並べ替え → 受け手で安定に並べ替え → 受け手ごとの先頭
      ⇒ 受け手ごとに key が一番小さい辺。同じ値が並ぶときは、key の並べ替えで前に来た辺
  後：辺を 1,677万本ずつの塊で
      1 受け手ごとの key の最小（scatter_reduce amin）
      2 key が最小と等しく有限の辺のうち、番号が一番小さい辺（scatter_reduce amin）
  ⚠️ 同じ値が並ぶとき（新しく生えた辺の w ＝ 0 など）、前は「key の並べ替えで前に来た辺」。GPU の並べ替えが安定（同じ値は番号順）なら
     一番小さい番号 ＝ 前と同じ。安定でなければ選ぶ辺が変わる ⇒ check_v91_gpu.py で、前と後の選んだ辺が全部同じかを arkhe で確かめる
  −0.0 と 0.0：並べ替えは −0.0 を前に置く・amin は等しいと見る ⇒ 違いが出たらここ（check_v91_gpu が数える）
⚠️ 網の作りは同じ（state.pt は読める）。VERSION は 91（SAME_NET に 90）
"""
import ast, pathlib, re, sys

G2 = [
    ("    def _first_per_recv(self, key):\n"
     "        \"\"\"受け手ごとに key が一番小さい辺（key が inf の辺しか無い受け手は −1）\"\"\"\n"
     "        o1 = torch.argsort(key)\n"
     "        o2 = torch.argsort(self.e_recv[o1].long(), stable=True)\n"
     "        order = o1[o2]\n"
     "        r_sorted = self.e_recv[order].long()\n"
     "        first = torch.ones_like(r_sorted, dtype=torch.bool)\n"
     "        first[1:] = r_sorted[1:] != r_sorted[:-1]\n"
     "        pick = torch.full((self.n_recv_all,), -1, dtype=torch.long, device=self.dev)\n"
     "        ok = first & torch.isfinite(key[order])\n"
     "        pick[r_sorted[ok]] = order[ok]\n"
     "        return pick\n",
     "    def _first_per_recv(self, key):\n"
     "        \"\"\"受け手ごとに key が一番小さい辺（key が inf の辺しか無い受け手は −1）。同じ値なら番号が一番小さい辺\n"
     "        版91：並べ替えなしで塊ごとに（⟲ 版90 まで argsort 2回 ⇒ 刈り込みで GPU の山 +8.4GB【測】watch_vram2）\"\"\"\n"
     "        n, m, step = self.n_recv_all, len(key), 1 << 24\n"
     "        mn = torch.full((n,), float(\"inf\"), dtype=key.dtype, device=self.dev)\n"
     "        for a in range(0, m, step):\n"
     "            b = min(m, a + step)\n"
     "            mn.scatter_reduce_(0, self.e_recv[a:b].long(), key[a:b], reduce=\"amin\")\n"
     "        big = torch.iinfo(torch.long).max\n"
     "        pick = torch.full((n,), big, dtype=torch.long, device=self.dev)\n"
     "        for a in range(0, m, step):\n"
     "            b = min(m, a + step)\n"
     "            r = self.e_recv[a:b].long()\n"
     "            k = key[a:b]\n"
     "            hit = (k == mn[r]) & torch.isfinite(k)\n"
     "            idx = torch.arange(a, b, device=self.dev)\n"
     "            pick.scatter_reduce_(0, r[hit], idx[hit], reduce=\"amin\")\n"
     "        pick[pick == big] = -1\n"
     "        return pick\n"),
]
BB = [("VERSION = 90          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89)",
       "VERSION = 91          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90)"),
      ("#  90  1周目から最後の周までを1つのグラフに（1周目の辺は Triton の核。くじ・放出・歩の終わりは外）【み】10-15「稼げるならやっておいて」\n",
       "#  90  1周目から最後の周までを1つのグラフに（1周目の辺は Triton の核。くじ・放出・歩の終わりは外）【み】10-15「稼げるならやっておいて」\n"
       "#  91  刈り込みの受け手ごとの一番小さい辺を並べ替えなしで（選ぶ辺は同じ。GPU の山 14.9GB → 下げる）【み】10-15「予算12gb」\n")]
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
    if babble.VERSION != 90:
        sys.exit(f"⚠️ babble.py が版90 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版91 を当てた（torch_group2・babble）。刈り込みの一番小さい辺を並べ替えなしで", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "91" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['91'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版91 でも回るようにした", flush=True)
