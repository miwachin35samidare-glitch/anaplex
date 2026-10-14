"""版84：歩の終わりと1周目の頭の小さな命令をまとめる・1周目で CPU が待つ所を 2 → 1回に（速さだけ。値は同じ）
（~/anaplex で。版83 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v84.py
【測】arkhe 版83：1歩 11.6ms（実時間の 1.1倍）／ GPU の核 8.9ms ／ CPU が待つ 3回/歩（CL・1周目の撃った送り手・その辺の数）
  ⇒ 残りは CPU が命令を1つずつ渡す時間。コンテナで数えた命令：歩の終わり 約200（領野ごとに入口を並べ直す・buf を1列ずらす）
直すもの（torch_group2）
  歩の終わり：入口の並べ直しを番号の表1つで（領野ごとの切り貼り 約45命令 → 2）
             buf（短い遅れの列）を1本の配列で持ち、「新しい列＋古い列を1つずらす」を番号の表1つで（約45 → 3）。self.bufs は今まで通り領野ごとの [送り元, D]（1本の配列の見え方）
  1周目の頭：時間細胞と buf の撃った辺を、撃った送り手を数えずに累積和から1回で（nonzero と辺の数の読み 2回 → 辺の数の読み 1回）
  どれも値を写す・並べるだけ（式は同じ）。辺の並びも同じ（送り手の順・送り手の中の順）
【確】コンテナ（CPU）版78 と 200歩 一致
"""
import ast, pathlib, re, sys

G2 = [
    ("            self._base_e = self._expand(torch.nonzero(self._base_on).squeeze(1))\n",
     "            if type(self)._expand is _G2_ORIG[\"_expand\"]:   # 版84：数えずに累積和から（CPU が待つのは辺の数の1回）\n"
     "                self._base_e = self._expand_mask(self._base_on)\n"
     "            else:\n"
     "                self._base_e = self._expand(torch.nonzero(self._base_on).squeeze(1))\n"),
    ("                self._base = torch.cat([self.ch.spikes()] + [b.reshape(-1) for b in self.bufs])\n",
     "                self._base = torch.cat([self.ch.spikes(), self._buf_flat_get()])   # 版84：buf は1本の配列で（並びは同じ）\n"),
    ("    def _fired_edges(self):\n",
     "    @torch.no_grad()\n"
     "    def _expand_mask(self, mask):\n"
     "        \"\"\"版84：撃った送り手（mask）→ その辺の番号。_expand(nonzero(mask)) と同じ並び。CPU が待つのは辺の数の1回\"\"\"\n"
     "        lens = torch.where(mask, self.start[1:] - self.start[:-1], torch.zeros_like(self.start[1:]))\n"
     "        csum = lens.cumsum(0)\n"
     "        tot = int(csum[-1])\n"
     "        if tot == 0:\n"
     "            return torch.empty(0, dtype=torch.long, device=self.dev)\n"
     "        return _run(_expand_mask_core, lens, csum, self.start, tot)\n"
     "\n"
     "    def _buf_flat_get(self):\n"
     "        \"\"\"版84：buf を1本の配列で（領野ごとの [送り元, D] を順に並べたもの。読み込みなどで置き換わったら作り直す）\"\"\"\n"
     "        refs = getattr(self, \"_buf_refs\", None)\n"
     "        if refs is None or len(refs) != len(self.bufs) or any(a is not b for a, b in zip(refs, self.bufs)):\n"
     "            self._buf_flat = torch.cat([b.reshape(-1) for b in self.bufs])\n"
     "            self._buf_refs = tuple(self.bufs)\n"
     "        return self._buf_flat\n"
     "\n"
     "    def _build_tail_idx(self, n_rep, n_ext, n_inh):\n"
     "        \"\"\"版84：歩の終わりの番号の表（入口の並べ直し・buf のずらし）。領野の形だけで決まるので1回だけ作る\"\"\"\n"
     "        dev = self.dev\n"
     "        si = torch.empty(self.n_src_all, dtype=torch.long, device=dev)\n"
     "        eo = io = 0\n"
     "        for i, a in enumerate(self.areas):        # cat([rep, ext, inh]) の中の番号（版79 の切り貼りと同じ並び）\n"
     "            lo = int(self.off_src[i]); c = int(self.off_ch[i])\n"
     "            si[lo:lo + a.n_ch] = torch.arange(c, c + a.n_ch, device=dev)\n"
     "            if a.n_ext:\n"
     "                si[lo + a.n_ch:lo + a.n_ch + a.n_ext] = n_rep + torch.arange(eo, eo + a.n_ext, device=dev)\n"
     "                eo += a.n_ext\n"
     "            si[lo + a.n_ch + a.n_ext:int(self.off_src[i + 1])] = n_rep + n_ext + torch.arange(io, io + a.n_inh, device=dev)\n"
     "            io += a.n_inh\n"
     "        bi, shapes, bo = [], [], 0\n"
     "        for i, b in enumerate(self.bufs):        # cat([src, 古い buf]) の中の番号：列 0 は新しい入口、列 d は古い列 d−1\n"
     "            n, d = b.shape\n"
     "            r = torch.arange(n, device=dev).unsqueeze(1)\n"
     "            col = torch.arange(d, device=dev).unsqueeze(0)\n"
     "            old = self.n_src_all + bo + r * d + (col - 1)\n"
     "            bi.append(torch.where(col == 0, int(self.off_src[i]) + r + 0 * col, old).reshape(-1))\n"
     "            shapes.append((bo, n, d))\n"
     "            bo += n * d\n"
     "        self._src_idx, self._buf_idx, self._buf_shapes = si, torch.cat(bi), shapes\n"
     "        self._tail_n = (n_rep, n_ext, n_inh)\n"
     "\n"
     "    def _fired_edges(self):\n"),
    ("        src = torch.empty(self.n_src_all, device=self.dev)\n"
     "        eo = io = 0\n"
     "        for i, a in enumerate(self.areas):        # 入口の並べ直しだけは領野の形に依る\n"
     "            lo = int(self.off_src[i]); c = int(self.off_ch[i])\n"
     "            src[lo:lo + a.n_ch] = rep[c:c + a.n_ch]\n"
     "            if a.n_ext:\n"
     "                src[lo + a.n_ch:lo + a.n_ch + a.n_ext] = ext_all[eo:eo + a.n_ext]\n"
     "                eo += a.n_ext\n"
     "            src[lo + a.n_ch + a.n_ext:int(self.off_src[i + 1])] = inh_spk[io:io + a.n_inh]\n"
     "            io += a.n_inh\n"
     "        self.ch.step(src)\n"
     "        for i, a in enumerate(self.areas):\n"
     "            lo, hi = int(self.off_src[i]), int(self.off_src[i + 1])\n"
     "            self.bufs[i] = torch.cat([src[lo:hi].unsqueeze(1), self.bufs[i][:, :-1]], 1)\n",
     "        # 版84：入口の並べ直しを番号の表1つで（領野ごとの切り貼りと同じ並び・同じ値）\n"
     "        nn = (len(rep), len(ext_all), len(inh_spk))\n"
     "        if getattr(self, \"_src_idx\", None) is None or self._tail_n != nn \\\n"
     "                or [tuple(b.shape) for b in self.bufs] != [(n, d) for _, n, d in self._buf_shapes]:\n"
     "            self._build_tail_idx(*nn)\n"
     "        src = torch.cat([rep.float(), ext_all.float(), inh_spk.float()])[self._src_idx]\n"
     "        self.ch.step(src)\n"
     "        # 版84：buf は1本の配列で1回に進める（領野ごとに 列 0 ＝ 新しい入口、列 d ＝ 古い列 d−1。値を写すだけ）\n"
     "        new = torch.cat([src, self._buf_flat_get()])[self._buf_idx]\n"
     "        self.bufs = [new[o:o + n * d].view(n, d) for o, n, d in self._buf_shapes]\n"
     "        self._buf_flat, self._buf_refs = new, tuple(self.bufs)\n"),
    ("def _flush_amp_core(",
     "def _expand_mask_core(lens, csum, start, C: int):\n"
     "    \"\"\"版84：累積和から辺の番号を C 本（C ＝ 辺の数ちょうど）。_expand と同じ並び（送り手の順・送り手の中の順）\"\"\"\n"
     "    j = torch.arange(C, device=csum.device)\n"
     "    s = torch.searchsorted(csum, j, right=True)\n"
     "    return start[s] + j - (csum[s] - lens[s])\n"
     "\n"
     "\n"
     "def _flush_amp_core("),
]
BB = [("VERSION = 83          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82)",
       "VERSION = 84          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83)"),
      ("#      【み】10-14「まだ詰めれるなら速さを」【測】版82 1歩 13.4ms ／ GPU 9.4ms\n",
       "#      【み】10-14「まだ詰めれるなら速さを」【測】版82 1歩 13.4ms ／ GPU 9.4ms\n"
       "#  84  歩の終わりの入口の並べ直し・buf のずらしを番号の表1つで ／ 1周目の撃った辺を累積和から（CPU の待ち 2 → 1回）。値は同じ\n"
       "#      【測】版83 1歩 11.6ms ／ GPU 8.9ms\n")]
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
    if babble.VERSION != 83:
        sys.exit(f"⚠️ babble.py が版83 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版84 を当てた（torch_group2・babble）。歩の終わりと1周目の頭の小さな命令をまとめる", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "84" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['84'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版84 でも回るようにした", flush=True)
