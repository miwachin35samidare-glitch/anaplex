"""版83：2周目から最後の周までを「1つの」グラフに（周ごとに辺の数を CPU で読まない。速さだけ。式・くじの順は同じ）
（~/anaplex で。版82 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v83.py
【み】10-14「まだ詰めれるなら速さを」
【測】arkhe 版82：1歩 13.4ms ／ GPU の核 9.4ms/歩（実時間の内側）⇒ 残りは CPU が命令を渡す時間と、CPU が GPU を待つ所（14回/歩、うち周ごとの辺の数の読み 9回）
  周ごとに数を読むと、そのたび CPU は GPU が追いつくのを待ち、次の歩の準備を先に進められない
直すもの（torch_group2。GPU・Triton のときだけ）
  2周目からの1周の「新しく撃った送り手の辺 → 学びの分を足す → 足し込み」を Triton の核1つに。辺の数は GPU の中で読む
    （核は決まった数だけ並べ、辺の数まで回る。箱の余りは無い）。足し込みは版82 と同じく下書き 64枚
  ⇒ 2周目から最後の周までが決まった形になり、1つのグラフで撮れる（9周を1回の再生で。CPU が待つのは 9 → 0回）
  自発のくじは9周ぶんを、今と同じ順に先に引いておく（その間に他のくじは引かない）
  ANAPLEX_ONEGRAPH=0 で版82 と同じ（周ごとのグラフ）
【確】コンテナ（CPU。Triton の核の代わりに同じ式の torch で）版78 と 200歩 一致。GPU は check_gpu.py ANAPLEX_ONEGRAPH 0 1
"""
import ast, pathlib, re, sys

G2 = [
    ("class Group2:\n",
     "# 版83：2周目から最後の周までを1つのグラフに（辺の数は GPU の中で読む）。ANAPLEX_ONEGRAPH=0 で版82 と同じ\n"
     "ONEGRAPH = _os.environ.get(\"ANAPLEX_ONEGRAPH\", \"1\") != \"0\"\n"
     "if _TRITON:\n"
     "    @triton.jit\n"
     "    def _pass_edges_k(csum_p, lens_p, W_p, start_p, k_start_p, elig_p, A_p, anow_p, kc1_p, w_p,\n"
     "                      e_recv_p, e_src_p, e_inh_p, c_exc_p, amp_p, send_gain_p, mod_send_p, mod_drive_inh_p, is_inh_r_p,\n"
     "                      both_p, mod_p, nW, n_recv, HAS_AMP: tl.constexpr, LOG: tl.constexpr, K: tl.constexpr,\n"
     "                      BLOCK: tl.constexpr):\n"
     "        \"\"\"版83：周回の中で新しく撃った送り手の辺（_edges_core）→ 学びの分を足す（_flush_vals）→ 足し込み（_sum_core_v）を1つに。\n"
     "        辺の数（csum の最後）は GPU の中で読み、決まった数の束で辺の数まで回る（箱の余りは無い ⇒ 1本の辺は1か所だけが書く）\"\"\"\n"
     "        pid = tl.program_id(0)\n"
     "        nprog = tl.num_programs(0)\n"
     "        tot = tl.load(csum_p + nW - 1).to(tl.int32)\n"
     "        a_now = tl.load(anow_p)\n"
     "        kc1 = tl.load(kc1_p)\n"
     "        rep = pid % K\n"
     "        for base in range(pid * BLOCK, tot, nprog * BLOCK):\n"
     "            j = base + tl.arange(0, BLOCK)\n"
     "            m = j < tot\n"
     "            lo = tl.zeros([BLOCK], dtype=tl.int32)\n"
     "            hi = tl.zeros([BLOCK], dtype=tl.int32) + nW\n"
     "            for _ in tl.static_range(LOG):          # csum[s] > j の最初の s（searchsorted right と同じ）\n"
     "                act = lo < hi\n"
     "                mid = (lo + hi) // 2\n"
     "                cv = tl.load(csum_p + mid, mask=m & act, other=0)\n"
     "                go_r = act & (cv <= j)\n"
     "                go_l = act & (cv > j)\n"
     "                lo = tl.where(go_r, mid + 1, lo)\n"
     "                hi = tl.where(go_l, mid, hi)\n"
     "            s = lo\n"
     "            cs = tl.load(csum_p + s, mask=m, other=0)\n"
     "            ln = tl.load(lens_p + s, mask=m, other=0)\n"
     "            snd = tl.load(W_p + s, mask=m, other=0)\n"
     "            e = tl.load(start_p + snd, mask=m, other=0) + j - (cs - ln)\n"
     "            ks = tl.load(k_start_p + e, mask=m, other=1).to(tl.int64)\n"
     "            ak = tl.load(A_p + ks - 1, mask=m, other=0.0)\n"
     "            pend = tl.load(elig_p + e, mask=m, other=0.0) * (a_now - ak).to(tl.float32)\n"
     "            w1 = tl.maximum(tl.load(w_p + e, mask=m, other=0.0) + pend, 0.0)\n"
     "            tl.store(w_p + e, w1, mask=m)\n"
     "            tl.store(k_start_p + e, (tl.zeros([BLOCK], dtype=tl.int32) + kc1.to(tl.int32)).to(k_start_p.dtype.element_ty), mask=m)\n"
     "            r = tl.load(e_recv_p + e, mask=m, other=0)\n"
     "            se = tl.load(e_src_p + e, mask=m, other=0)\n"
     "            ie = tl.load(e_inh_p + e, mask=m, other=0).to(tl.int32)\n"
     "            cx = tl.load(c_exc_p + r, mask=m, other=0.0)\n"
     "            amp = tl.where(ie != 0, 4.0, cx) * w1\n"
     "            if HAS_AMP:\n"
     "                amp = amp * tl.load(amp_p + e, mask=m, other=1.0)\n"
     "            amp = amp * tl.load(send_gain_p + se, mask=m, other=0.0)\n"
     "            ms = tl.load(mod_send_p + se, mask=m, other=0).to(tl.int32)\n"
     "            mdi = tl.load(mod_drive_inh_p + se, mask=m, other=0).to(tl.int32)\n"
     "            iir = tl.load(is_inh_r_p + r, mask=m, other=0).to(tl.int32)\n"
     "            mm = (ms != 0) & (ie == 0) & ((mdi == 0) | (iir == 0))\n"
     "            tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie, amp, mask=m, sem=\"relaxed\")\n"
     "            tl.atomic_add(mod_p + rep * n_recv + r, amp, mask=m & mm, sem=\"relaxed\")\n"
     "\n"
     "\n"
     "class Group2:\n"),
    ("        for i in range(1, self.recur):\n"
     "            tot = int(S[\"csum\"][-1])     # 周回の中で新しく撃った送り手の辺の数（1周に1回だけ待つ）\n",
     "        if self._one_ok():                 # 版83：9周を1つのグラフに（自発のくじは今と同じ順に先に引く）\n"
     "            sp = S.get(\"spont_all\")\n"
     "            if sp is None or sp.shape[0] != self.recur - 1:\n"
     "                sp = S[\"spont_all\"] = torch.zeros(self.recur - 1, self.n_ch_all, device=self.dev)\n"
     "                self._graphs.clear()\n"
     "            for i in range(self.recur - 1):\n"
     "                torch.rand(self.n_ch_all, device=self.dev, generator=self.gen, out=sp[i])\n"
     "            self._gp_run1(key)\n"
     "            self._pc += self.recur - 1\n"
     "        for i in (range(1, self.recur) if not self._one_ok() else ()):\n"
     "            tot = int(S[\"csum\"][-1])     # 周回の中で新しく撃った送り手の辺の数（1周に1回だけ待つ）\n"),
    ("    def _gp_body(self, C, key):\n",
     "    def _one_ok(self):\n"
     "        \"\"\"版83：9周を1つのグラフに（GPU は Triton の核で。CPU（eager の確かめ）は同じ式の torch で）\"\"\"\n"
     "        if not ONEGRAPH:\n"
     "            return False\n"
     "        return _TRITON or not str(self.dev).startswith(\"cuda\")\n"
     "\n"
     "    def _gp_run1(self, key):\n"
     "        \"\"\"版83：2周目から最後の周までを1回の再生で（1回だけ撮る）\"\"\"\n"
     "        if GRAPH == \"eager\" or not str(self.dev).startswith(\"cuda\"):\n"
     "            return self._gp_body1(key)\n"
     "        g = self._graphs.get((\"one\", key))\n"
     "        if g is None:\n"
     "            S = self._gS\n"
     "            snap = {k: v.clone() for k, v in S.items()}\n"
     "            s = torch.cuda.Stream()\n"
     "            s.wait_stream(torch.cuda.current_stream())\n"
     "            with torch.cuda.stream(s):\n"
     "                self._gp_body1(key)            # 慣らし。学びの分を足すのは同じ刻みでは2回目が 0\n"
     "            torch.cuda.current_stream().wait_stream(s)\n"
     "            for k, v in snap.items():\n"
     "                S[k].copy_(v)\n"
     "            if getattr(self, \"_gpool\", None) is None:\n"
     "                self._gpool = torch.cuda.graph_pool_handle()\n"
     "            g = torch.cuda.CUDAGraph()\n"
     "            with torch.cuda.graph(g, pool=self._gpool):\n"
     "                self._gp_body1(key)\n"
     "            self._graphs[(\"one\", key)] = g\n"
     "            self.n_capture = getattr(self, \"n_capture\", 0) + 1\n"
     "        g.replay()\n"
     "\n"
     "    def _gp_body1(self, key):\n"
     "        \"\"\"版83：2周目から最後の周まで（周ごとの中身は _gp_body と同じ式）\"\"\"\n"
     "        S = self._gS\n"
     "        for i in range(self.recur - 1):\n"
     "            S[\"spont\"].copy_(S[\"spont_all\"][i])\n"
     "            self._gp_body(None, key)\n"
     "\n"
     "    def _pass_edges(self):\n"
     "        \"\"\"版83：周回の中で新しく撃った送り手の辺の分（学びの分を足してから足し込む）。返り値は足し込みの増え分\"\"\"\n"
     "        S, X = self._gS, self._gX\n"
     "        nW = len(self._gW)\n"
     "        if str(self.dev).startswith(\"cuda\"):\n"
     "            K, BL = _SC_K, 256\n"
     "            b2 = torch.zeros(K, self.n_recv_all * 2, device=self.dev)\n"
     "            m2 = torch.zeros(K, self.n_recv_all, device=self.dev)\n"
     "            amp = self._amp_buf if self.stp is not None else self.w\n"
     "            _pass_edges_k[(512,)](S[\"csum\"], S[\"lens\"], self._gW, self.start, self.k_start, self.elig, self.A,\n"
     "                                  S[\"a_now\"], S[\"kcur1\"], self.w, self.e_recv, self.e_src, self.e_inh, self.c_exc,\n"
     "                                  amp, X[\"send_gain\"], self.mod_send, self.mod_drive_inh, self.is_inh_r,\n"
     "                                  b2, m2, nW, self.n_recv_all, HAS_AMP=self.stp is not None,\n"
     "                                  LOG=max(1, int(math.ceil(math.log2(nW + 1)))) + 1, K=K, BLOCK=BL)\n"
     "            return b2.sum(0), m2.sum(0)\n"
     "        # CPU（確かめ用）：版82 の _gp_body と同じ式。辺の数ちょうどの箱で\n"
     "        tot = int(S[\"csum\"][-1])\n"
     "        if tot == 0:\n"
     "            return None\n"
     "        e, valid = _run(_edges_core, S[\"lens\"], S[\"csum\"], self._gW, self.start, tot)\n"
     "        vals = _run(_flush_vals, e, self.k_start, self.elig, self.A, S[\"a_now\"], self.w)\n"
     "        self.w.index_put_((e,), vals)\n"
     "        self.k_start.index_put_((e,), S[\"kcur1\"])\n"
     "        return _run(_sum_core_v, e, valid, self.e_recv, self.e_src, self.e_inh, self.w, self.c_exc,\n"
     "                    self._amp_buf if self.stp is not None else None, X[\"send_gain\"], self.mod_send,\n"
     "                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)\n"
     "\n"
     "    def _gp_body(self, C, key):\n"),
    ("        if C:            # 周回の中で新しく撃った送り手の辺：学びの分を足してから足し込む\n",
     "        if C is None:    # 版83：辺の数は GPU の中で読む（1つのグラフの形）\n"
     "            inc = self._pass_edges()\n"
     "            if inc is None:\n"
     "                both, exc_mod = X[\"both_base\"], X[\"mod_base\"]\n"
     "            else:\n"
     "                both, exc_mod = X[\"both_base\"] + inc[0], X[\"mod_base\"] + inc[1]\n"
     "        elif C:          # 周回の中で新しく撃った送り手の辺：学びの分を足してから足し込む\n"),
]
BB = [("VERSION = 82          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81)",
       "VERSION = 83          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82)"),
      ("#      【み】10-14「その仕組みならいいよ」【測】probe_scatter_real2 2.37 → 1.10ms\n",
       "#      【み】10-14「その仕組みならいいよ」【測】probe_scatter_real2 2.37 → 1.10ms\n"
       "#  83  2周目から最後の周までを1つのグラフに（辺の数は GPU の中で読む。CPU が待つ 9 → 0回/歩）。式・くじの順は同じ\n"
       "#      【み】10-14「まだ詰めれるなら速さを」【測】版82 1歩 13.4ms ／ GPU 9.4ms\n")]
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
    if babble.VERSION != 82:
        sys.exit(f"⚠️ babble.py が版82 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版83 を当てた（torch_group2・babble）。2周目から最後の周までを1つのグラフに", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "83" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['83'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版83 でも回るようにした", flush=True)
