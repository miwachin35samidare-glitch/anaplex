"""版80：1周目と2周目の頭で、1歩ぶんの辺を読む回数を減らす（速さだけ。式・くじの順は同じ）
（~/anaplex で。版79b の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v80.py
【測】arkhe 版79b（10-14）：1歩 18.3ms ／ 網 15.1（1周目 9.71・2周目から 5.37 ＝ 1周 0.60）／ GPU 13.7ms/歩（GPU が律速）
  GPU の上位：1歩ぶんの辺（時間細胞と buf が撃った辺）の足し込み 2.18ms × 2回/歩 ／ 学び 1.76ms ／ binomial 1.28ms
  1歩ぶんの辺（数百万本）を、1周目に「学びの分を足す → 足し込み → 学ぶ」、2周目の頭に「学びの分を足す → 足し込み」と5回読んでいた
直すもの（torch_group2）
  _flush_sum_core：1周目の「学びの分を足す」と「足し込み」を1つに（放出のくじを先に引く。放出は重さを読まないので値・くじの順は同じ）
  _learn_sum_core：1周目の「学ぶ」と、2周目の頭の「学びの分を足す・足し込み」を1つに（A を先に1つ進める。A は跡を読まないので同じ）
    ⇒ 2周目の頭は作ってある足し込みを使う。学ばない歩・1周だけのとき・測る道具が _flush・_sum を差し替えているときは今まで通り
【確】コンテナ（CPU・相手200・放出のくじを期待値に・刈り込み 60歩ごと）：版78 と版80（グラフの形 eager／周ごと の両方）200歩 一致
"""
import ast, pathlib, sys

G2 = [
    ("    return both, mod\n"
     "\n"
     "\n"
     "class Group2:\n",
     "    return both, mod\n"
     "\n"
     "\n"
     "def _flush_sum_core(e, k_start, elig, A, a_now, w, kc1, e_recv, e_src, e_inh, c_exc, amp_s, send_gain, mod_send,\n"
     "                    mod_drive_inh, is_inh_r, n_recv: int):\n"
     "    \"\"\"版80：_flush_core（学びの分を足す）と _sum_core（足し込み）を1つに。式は同じ（足した重さをそのまま足し込みに使う）\"\"\"\n"
     "    ks = k_start[e].long()\n"
     "    w1 = (w[e] + elig[e] * (a_now - A[ks - 1]).float()).clamp(min=0.0)\n"
     "    w[e] = w1\n"
     "    k_start[e] = kc1\n"
     "    re_ = e_recv[e].long()\n"
     "    se_ = e_src[e].long()\n"
     "    ie = e_inh[e]\n"
     "    amp = torch.where(ie, 4.0, c_exc[re_]) * w1\n"
     "    if amp_s is not None:\n"
     "        amp = amp * amp_s[e]\n"
     "    amp = amp * send_gain[se_]\n"
     "    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)\n"
     "    both.scatter_add_(0, re_ + n_recv * ie.long(), amp)\n"
     "    mm = mod_send[se_] & ~ie\n"
     "    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])\n"
     "    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, re_, amp * mm)\n"
     "    return both, mod\n"
     "\n"
     "\n"
     "def _learn_sum_core(e, e_recv, e_src, e_inh, is_inh_r, err_sig, g, pw, elig, hist_n, w, c_exc, k_start, A, a_k, kc1,\n"
     "                    amp_s, send_gain, mod_send, mod_drive_inh, n_recv: int):\n"
     "    \"\"\"版80：_learn_core（学ぶ）と、2周目の頭の _flush_core・_sum_core を1つに。式は同じ（a_k ＝ 進めたあとの A[k]）\"\"\"\n"
     "    r = e_recv[e].long()\n"
     "    gg = g * err_sig[r]\n"
     "    to_inh = is_inh_r[r]\n"
     "    ie = e_inh[e]\n"
     "    el = elig[e] + torch.where(ie | to_inh, torch.zeros_like(gg), gg) * pw\n"
     "    elig[e] = el\n"
     "    se = e_src[e].long()\n"
     "    w0 = w[e]\n"
     "    ltd = to_inh & ~ie & (hist_n[se] >= INH_LTD_N) & (w0 * c_exc[r] > INH_LTD_TH)\n"
     "    wl = torch.where(ltd, w0 * INH_LTD_F, w0)\n"
     "    ks = k_start[e].long()\n"
     "    w2 = (wl + el * (a_k - A[ks - 1]).float()).clamp(min=0.0)\n"
     "    w[e] = w2\n"
     "    k_start[e] = kc1\n"
     "    cnt = torch.zeros(n_recv, device=w.device, dtype=w.dtype)\n"
     "    cnt.scatter_add_(0, r, torch.ones(e.shape[0], device=w.device, dtype=w.dtype))\n"
     "    amp = torch.where(ie, 4.0, c_exc[r]) * w2\n"
     "    if amp_s is not None:\n"
     "        amp = amp * amp_s[e]\n"
     "    amp = amp * send_gain[se]\n"
     "    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)\n"
     "    both.scatter_add_(0, r + n_recv * ie.long(), amp)\n"
     "    mm = mod_send[se] & ~ie\n"
     "    mm = mm & ~(mod_drive_inh[se] & to_inh)\n"
     "    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, r, amp * mm)\n"
     "    return cnt, both, mod\n"
     "\n"
     "\n"
     "class Group2:\n"),
    ("        self._within = None                  # 周回の中の発火（1歩の中だけ。歩はまたがない）\n"
     "        gp = self._gp_ok()                   # 版79：2周目からをグラフの形で\n",
     "        self._within = None                  # 周回の中の発火（1歩の中だけ。歩はまたがない）\n"
     "        self._pre2 = None                    # 版80：1周目の学びと一緒に作った2周目の頭の足し込み\n"
     "        gp = self._gp_ok()                   # 版79：2周目からをグラフの形で\n"),
    # 2周目の頭（グラフの形）
    ("        self._flush(self._base_e)\n"
     "        both_base, mod_base = self._sum(self._base_e, False)\n"
     "        self._both_base, self._mod_base = both_base, mod_base\n",
     "        if self._pre2 is not None:            # 版80：1周目の学びと一緒に作ってある（学びの分も足してある）\n"
     "            both_base, mod_base = self._pre2\n"
     "        else:\n"
     "            self._flush(self._base_e)\n"
     "            both_base, mod_base = self._sum(self._base_e, False)\n"
     "        self._both_base, self._mod_base = both_base, mod_base\n"),
    # 新しい関数（_gp_ok の前に）
    ("    def _gp_ok(self):\n",
     "    def _fuse_ok(self):\n"
     "        \"\"\"版80：辺を読む回数を減らした形で回すか（測る道具が _flush・_sum を差し替えているときは今まで通り）\"\"\"\n"
     "        return type(self)._flush is _G2_ORIG[\"_flush\"] and type(self)._sum is _G2_ORIG[\"_sum\"]\n"
     "\n"
     "    def _kc(self, v):\n"
     "        \"\"\"版80：刻みの数を GPU の箱で渡す（CPU が待たない）\"\"\"\n"
     "        if getattr(self, \"_kc_buf\", None) is None:\n"
     "            self._kc_buf = torch.zeros((), dtype=self.k_start.dtype, device=self.dev)\n"
     "        return self._kc_buf.fill_(v)\n"
     "\n"
     "    @torch.no_grad()\n"
     "    def _flush_sum(self, e):\n"
     "        \"\"\"版80：1周目の _flush(e) → _sum(e, True) と同じ（放出を先に。放出は重さを読まない）\"\"\"\n"
     "        if self.stp is not None:       # 放出は _sum の1周目と同じ\n"
     "            if self._amp_buf is None or len(self._amp_buf) != len(self.w):\n"
     "                self._amp_buf = torch.ones(len(self.w), device=self.dev)\n"
     "            elif self._amp_prev is not None:\n"
     "                self._amp_buf.index_fill_(0, self._amp_prev, 1.0)     # 版54b\n"
     "            self._amp_buf[e] = self.stp.release_at(e, self.t)\n"
     "            self._amp_prev = e\n"
     "            self._amp = self._amp_buf\n"
     "        return _run(_flush_sum_core, e, self.k_start, self.elig, self.A, self.A[self.kcur], self.w,\n"
     "                    self._kc(self.kcur + 1), self.e_recv, self.e_src, self.e_inh, self.c_exc,\n"
     "                    self._amp if self.stp is not None else None, self.send_gain, self.mod_send,\n"
     "                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)\n"
     "\n"
     "    def _gp_ok(self):\n"),
    # _pass：1周目
    ("        if first:\n"
     "            self._flush(e)\n"
     "            both, exc_mod = self._sum(e, True)\n"
     "            self._both_base = None\n",
     "        if first:\n"
     "            if len(e) and self._fuse_ok():      # 版80：学びの分を足すのと足し込みを1つに\n"
     "                both, exc_mod = self._flush_sum(e)\n"
     "            else:\n"
     "                self._flush(e)\n"
     "                both, exc_mod = self._sum(e, True)\n"
     "            self._both_base = None\n"),
    # _pass：2周目の頭（周ごと）
    ("            if self._both_base is None:\n"
     "                self._flush(self._base_e)\n"
     "                self._both_base, self._mod_base = self._sum(self._base_e, False)\n",
     "            if self._both_base is None:\n"
     "                if getattr(self, \"_pre2\", None) is not None:   # 版80：1周目の学びと一緒に作ってある\n"
     "                    self._both_base, self._mod_base = self._pre2\n"
     "                else:\n"
     "                    self._flush(self._base_e)\n"
     "                    self._both_base, self._mod_base = self._sum(self._base_e, False)\n"),
    # _pass：学び
    ("            cnt = _run(_learn_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig, g_, pw_,\n"
     "                       self.elig, self.hist_n, self.w, self.c_exc, self.n_recv_all)\n"
     "            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）\n"
     "            # 版54：全部の辺へ足す代わりに A を1つ進める（辺は読む前にまとめて足す）\n"
     "            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(()) * (self.elig_decay ** k)\n"
     "            self.kcur = k\n",
     "            # 版80：2周目の頭（学びの分を足す・1歩ぶんの辺の足し込み）も一緒に（A を先に進める。A は跡を読まない）\n"
     "            fuse = self.recur > 1 and e is getattr(self, \"_base_e\", None) and self._fuse_ok()\n"
     "            if not fuse:\n"
     "                cnt = _run(_learn_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig, g_, pw_,\n"
     "                           self.elig, self.hist_n, self.w, self.c_exc, self.n_recv_all)\n"
     "            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）\n"
     "            # 版54：全部の辺へ足す代わりに A を1つ進める（辺は読む前にまとめて足す）\n"
     "            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(()) * (self.elig_decay ** k)\n"
     "            if fuse:\n"
     "                cnt, b2, m2 = _run(_learn_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,\n"
     "                                   g_, pw_, self.elig, self.hist_n, self.w, self.c_exc, self.k_start, self.A,\n"
     "                                   self.A[k], self._kc(k + 1), self._amp if self.stp is not None else None,\n"
     "                                   self.send_gain, self.mod_send, self.mod_drive_inh, self.n_recv_all)\n"
     "                self._pre2 = (b2, m2)\n"
     "            self.kcur = k\n"),
]
BB = [("VERSION = 79          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78)",
       "VERSION = 80          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79)"),
      ("#  79b 箱の余りの辺を1か所に集めない（版79 1歩 21.5ms。余りが先頭の辺の受け手にぶつかっていた）。値は同じ\n",
       "#  79b 箱の余りの辺を1か所に集めない（版79 1歩 21.5ms。余りが先頭の辺の受け手にぶつかっていた）。値は同じ\n"
       "#  80  1周目と2周目の頭で1歩ぶんの辺を読む回数 5 → 3（学びの分を足す＋足し込み／学ぶ＋2周目の頭）。式・くじの順は同じ\n"
       "#      【測】版79b 1歩 18.3ms ／ GPU 13.7ms（1歩ぶんの辺の足し込み 2.18ms × 2回）\n")]
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
    import re
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 79:
        sys.exit(f"⚠️ babble.py が版79 ではない（版 {babble.VERSION}）")
    if "版79b" not in pathlib.Path("torch_group2.py").read_text():
        sys.exit("⚠️ 版79b が当たっていない（先に python3 apply_v79b.py）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版80 を当てた（torch_group2・babble）。1歩ぶんの辺を読む回数 5 → 3", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "80" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['80'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版80 でも回るようにした", flush=True)
