"""版81：1周目の足し込みで、要らない「ばらばらの所への小数の足し込み」を省く（速さだけ。値は同じ）
（~/anaplex で。版80 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v81.py
【測】probe_atomic（10-14、arkhe）：230万本を受け手 240万の所へ
  読むだけ 0.022ms ／ 小数の足し込み（float の atomic）0.67ms ／ 整数の足し込み（int32）0.158ms ／ Triton で直に 0.69ms
【測】measure_gpu80：1歩ぶんの辺 平均 231万本 ／ 学ぶ＋足し込み 2.47ms（足し込み3つ）／ 足す＋足し込み 1.96ms（足し込み2つ）
  ⇒ 1周目の重い2つは、ほとんどが小数の足し込み
直すもの（GPU のときだけ。CPU は版80 と同じ）
  調整する側の足し込み（mod）：調整する側でない辺（ほとんど）は 0 を足していた ⇒ 足さない（0 を足しても足さなくても同じ値）
  受け手ごとの辺の数（cnt）：1 を小数で足していた ⇒ 整数で数えてから小数に（数は 2^24 より小さいので同じ値）
  ⇒ 足し込みは Triton で直に書いた核1つ（印のある所だけ足す）。式の計算（重さ・学び）は今まで通り torch.compile
  ANAPLEX_SCATTER=0 で版80 と同じ（比べる用）
⚠️ GPU の小数の足し込みは、もともと足す順が毎回少し違う（丸めの範囲）。確かめは arkhe で check_gpu.py ANAPLEX_SCATTER 0 1
"""
import ast, pathlib, re, sys

G2 = [
    ("    return cnt, both, mod\n"
     "\n"
     "\n"
     "class Group2:\n",
     "    return cnt, both, mod\n"
     "\n"
     "\n"
     "# 版81：1周目の足し込みを Triton の核で（印のある所だけ足す・辺の数は整数で数える）。GPU のときだけ\n"
     "#   【測】probe_atomic（10-14）：230万本 小数の足し込み 0.67ms ／ 整数 0.158ms\n"
     "#   ANAPLEX_SCATTER=0 で版80 と同じ（比べる用）\n"
     "SCATTER = _os.environ.get(\"ANAPLEX_SCATTER\", \"1\") != \"0\"\n"
     "try:\n"
     "    import triton\n"
     "    import triton.language as tl\n"
     "\n"
     "    @triton.jit\n"
     "    def _scatter3_k(r_p, ie_p, amp_p, mm_p, both_p, mod_p, cnt_p, n, n_recv, DO_CNT: tl.constexpr, BLOCK: tl.constexpr):\n"
     "        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
     "        m = o < n\n"
     "        r = tl.load(r_p + o, mask=m, other=0)\n"
     "        ie = tl.load(ie_p + o, mask=m, other=0).to(tl.int32)\n"
     "        a = tl.load(amp_p + o, mask=m, other=0.0)\n"
     "        mm = tl.load(mm_p + o, mask=m, other=0).to(tl.int32)\n"
     "        tl.atomic_add(both_p + r + n_recv * ie, a, mask=m, sem=\"relaxed\")\n"
     "        tl.atomic_add(mod_p + r, a, mask=m & (mm != 0), sem=\"relaxed\")\n"
     "        if DO_CNT:\n"
     "            tl.atomic_add(cnt_p + r, tl.full([BLOCK], 1, tl.int32), mask=m, sem=\"relaxed\")\n"
     "    _TRITON = True\n"
     "except Exception:          # Triton が無い（CPU のコンテナなど）⇒ 版80 と同じ\n"
     "    _TRITON = False\n"
     "\n"
     "\n"
     "def _scatter3(r, ie, amp, mm, n_recv, do_cnt):\n"
     "    \"\"\"版81：both（興奮｜抑制）・mod（印 mm の辺だけ）・cnt（辺の数。整数で数えて小数に）\"\"\"\n"
     "    n = r.shape[0]\n"
     "    both = torch.zeros(n_recv * 2, device=amp.device, dtype=amp.dtype)\n"
     "    mod = torch.zeros(n_recv, device=amp.device, dtype=amp.dtype)\n"
     "    cnt = torch.zeros(n_recv if do_cnt else 1, device=amp.device, dtype=torch.int32)\n"
     "    if n:\n"
     "        _scatter3_k[(triton.cdiv(n, 1024),)](r, ie, amp, mm, both, mod, cnt, n, n_recv, DO_CNT=do_cnt, BLOCK=1024)\n"
     "    return both, mod, (cnt.to(amp.dtype) if do_cnt else None)\n"
     "\n"
     "\n"
     "def _flush_amp_core(e, k_start, elig, A, a_now, w, kc1, e_recv, e_src, e_inh, c_exc, amp_s, send_gain, mod_send,\n"
     "                    mod_drive_inh, is_inh_r):\n"
     "    \"\"\"版81：_flush_sum_core の足し込みの前まで（式は同じ）。足し込みは _scatter3\"\"\"\n"
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
     "    mm = mod_send[se_] & ~ie\n"
     "    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])\n"
     "    return re_.int(), ie, amp, mm\n"
     "\n"
     "\n"
     "def _learn_amp_core(e, e_recv, e_src, e_inh, is_inh_r, err_sig, g, pw, elig, hist_n, w, c_exc, k_start, A, a_k, kc1,\n"
     "                    amp_s, send_gain, mod_send, mod_drive_inh):\n"
     "    \"\"\"版81：_learn_sum_core の足し込みの前まで（式は同じ）。足し込み・辺の数は _scatter3\"\"\"\n"
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
     "    amp = torch.where(ie, 4.0, c_exc[r]) * w2\n"
     "    if amp_s is not None:\n"
     "        amp = amp * amp_s[e]\n"
     "    amp = amp * send_gain[se]\n"
     "    mm = mod_send[se] & ~ie\n"
     "    mm = mm & ~(mod_drive_inh[se] & to_inh)\n"
     "    return r.int(), ie, amp, mm\n"
     "\n"
     "\n"
     "class Group2:\n"),
    ("        return _run(_flush_sum_core, e, self.k_start, self.elig, self.A, self.A[self.kcur], self.w,\n"
     "                    self._kc(self.kcur + 1), self.e_recv, self.e_src, self.e_inh, self.c_exc,\n"
     "                    self._amp if self.stp is not None else None, self.send_gain, self.mod_send,\n"
     "                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)\n",
     "        if self._tri_ok():               # 版81：足し込みは Triton の核で（印のある所だけ）\n"
     "            r_, ie_, amp_, mm_ = _run(_flush_amp_core, e, self.k_start, self.elig, self.A, self.A[self.kcur], self.w,\n"
     "                                      self._kc(self.kcur + 1), self.e_recv, self.e_src, self.e_inh, self.c_exc,\n"
     "                                      self._amp if self.stp is not None else None, self.send_gain, self.mod_send,\n"
     "                                      self.mod_drive_inh, self.is_inh_r)\n"
     "            both, mod, _ = _scatter3(r_, ie_, amp_, mm_, self.n_recv_all, False)\n"
     "            return both, mod\n"
     "        return _run(_flush_sum_core, e, self.k_start, self.elig, self.A, self.A[self.kcur], self.w,\n"
     "                    self._kc(self.kcur + 1), self.e_recv, self.e_src, self.e_inh, self.c_exc,\n"
     "                    self._amp if self.stp is not None else None, self.send_gain, self.mod_send,\n"
     "                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)\n"
     "\n"
     "    def _tri_ok(self):\n"
     "        \"\"\"版81：足し込みを Triton の核で回すか（GPU・Triton あり・ANAPLEX_SCATTER が 0 でない）\"\"\"\n"
     "        return SCATTER and _TRITON and str(self.dev).startswith(\"cuda\")\n"),
    ("            if fuse:\n"
     "                cnt, b2, m2 = _run(_learn_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,\n"
     "                                   g_, pw_, self.elig, self.hist_n, self.w, self.c_exc, self.k_start, self.A,\n"
     "                                   self.A[k], self._kc(k + 1), self._amp if self.stp is not None else None,\n"
     "                                   self.send_gain, self.mod_send, self.mod_drive_inh, self.n_recv_all)\n"
     "                self._pre2 = (b2, m2)\n",
     "            if fuse and self._tri_ok():   # 版81：足し込み・辺の数は Triton の核で\n"
     "                r_, ie_, amp_, mm_ = _run(_learn_amp_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r,\n"
     "                                          err_sig, g_, pw_, self.elig, self.hist_n, self.w, self.c_exc, self.k_start,\n"
     "                                          self.A, self.A[k], self._kc(k + 1),\n"
     "                                          self._amp if self.stp is not None else None, self.send_gain,\n"
     "                                          self.mod_send, self.mod_drive_inh)\n"
     "                b2, m2, cnt = _scatter3(r_, ie_, amp_, mm_, self.n_recv_all, True)\n"
     "                self._pre2 = (b2, m2)\n"
     "            elif fuse:\n"
     "                cnt, b2, m2 = _run(_learn_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,\n"
     "                                   g_, pw_, self.elig, self.hist_n, self.w, self.c_exc, self.k_start, self.A,\n"
     "                                   self.A[k], self._kc(k + 1), self._amp if self.stp is not None else None,\n"
     "                                   self.send_gain, self.mod_send, self.mod_drive_inh, self.n_recv_all)\n"
     "                self._pre2 = (b2, m2)\n"),
]
BB = [("VERSION = 80          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79)",
       "VERSION = 81          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80)"),
      ("#      【測】版79b 1歩 18.3ms ／ GPU 13.7ms（1歩ぶんの辺の足し込み 2.18ms × 2回）\n",
       "#      【測】版79b 1歩 18.3ms ／ GPU 13.7ms（1歩ぶんの辺の足し込み 2.18ms × 2回）\n"
       "#  81  1周目の足し込みを Triton の核で：調整する側でない辺は 0 を足さない・辺の数は整数で数える（値は同じ）\n"
       "#      【測】版80 1歩 15.8ms ／ probe_atomic 小数の足し込み 0.67ms・整数 0.158ms（230万本）\n")]
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
    if babble.VERSION != 80:
        sys.exit(f"⚠️ babble.py が版80 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版81 を当てた（torch_group2・babble）。1周目の足し込みを Triton の核で（要らない足し込みを省く）", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "81" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['81'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版81 でも回るようにした", flush=True)
