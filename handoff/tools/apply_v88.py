"""版88：学び・恒常性の小さな命令を torch.compile でまとめる（まとめて4つ。速さだけ。式は同じ）
（~/anaplex で。版87 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v88.py
【み】10-15「また1個ずつ入れてくれてるけどさ、まとめて入れられんの？」⇒ 残りの小さいものはまとめて1つの版に
【測】arkhe 版87：1歩 10.0ms（実時間の 0.94倍）／ GPU 7.9ms ／ CPU が待つ 2回/歩
  コンテナで数えた命令（1歩）：網の1周目 70 ／ 脳の外側 85 ／ 下からの表の学び 35 ／ 恒常性の観測 10 ／ CL の学び 約15 …
まとめたもの（ANAPLEX_V88=0 で4つとも版87 と同じ形）
  1 1周目の恒常性（受け手の平均・駆動の平均・眠りで下げる／起きて上げる）を compile で（_homeo_core・_scale_down_core・_scale_up_core）
  2 恒常性の観測（この歩のどこかで撃ったか → 発火率の平均）を compile で（_rate_core）
  3 下からのシナプスの表の学び（三つの因子）を compile で（_bu_learn_core）
  4 CL → 皮質 の学び（三つの因子）を compile で（_cl_learn_core）
  いずれも式・並びは同じ。置き場所のある値（recv_ema・drive・rate_ema・rate_long）は今まで通り中身を書き換える
⚠️ 見つけたこと（直していない）：恒常性の観測で、誤差の体の行に表現の発火、表現の体の行に誤差の発火を入れている（版59 から）
   _any ＝ [表現, 誤差, 抑制] を fired[誤差], fired[表現], fired[抑制] の順に入れている ⇒ 直すかは【み】に聞く
【確】コンテナ（CPU）版78 と 200歩
"""
import ast, pathlib, re, sys

G2 = [
    ("def _flush_amp_core(",
     "def _homeo_core(cnt, exc, recv_ema, drive, a_: float):\n"
     "    \"\"\"版88：受け手の平均・駆動の平均（式は版87 の _pass と同じ）\"\"\"\n"
     "    return recv_ema + a_ * (cnt - recv_ema), drive + a_ * (exc.abs() - drive)\n"
     "\n"
     "\n"
     "def _scale_down_core(rate_ema, target, is_hippo, scale_skip, scale, a_: float, sleep: bool):\n"
     "    \"\"\"版88：眠りは下げるだけ（式は版87 と同じ）\"\"\"\n"
     "    f = (1 - 0.01 * (rate_ema / target - 1)).clamp(0.5, 1.0) ** a_\n"
     "    if sleep:\n"
     "        f = torch.where(is_hippo, torch.ones_like(f), f)\n"
     "    f = torch.where(scale_skip, torch.ones_like(f), f)\n"
     "    return scale * f\n"
     "\n"
     "\n"
     "def _scale_up_core(rate_long, target, scale_skip, scale_up):\n"
     "    \"\"\"版88：起きているあいだは上げるだけ（式は版87 と同じ）\"\"\"\n"
     "    up = torch.exp(K_WAKE_UP * (1.0 - rate_long / target).clamp(min=0.0))\n"
     "    return scale_up * torch.where(scale_skip, torch.ones_like(up), up)\n"
     "\n"
     "\n"
     "def _rate_core(a0, a1, a2, err_i, rep_i, inh_i, rate_ema, rate_long, n_recv: int, h: float, lt: float, awake: bool):\n"
     "    \"\"\"版88：恒常性の観測（式・入れる行は版87 の _rate_obs と同じ。⚠️ 誤差と表現の行が入れ違い。直すかは【み】に聞く）\"\"\"\n"
     "    fired = torch.zeros(n_recv, device=a0.device)\n"
     "    fired[err_i] = a0\n"
     "    fired[rep_i] = a1\n"
     "    fired[inh_i] = a2\n"
     "    re_ = rate_ema + h * (fired - rate_ema)\n"
     "    rl = rate_long + lt * (fired - rate_long) if awake else rate_long\n"
     "    return re_, rl\n"
     "\n"
     "\n"
     "def _flush_amp_core("),
    ("            a_ = 1.0 / self.homeo_tau\n"
     "            self.recv_ema += a_ * (cnt - self.recv_ema)\n"
     "            self.drive += a_ * (exc.abs() - self.drive)\n",
     "            a_ = 1.0 / self.homeo_tau\n"
     "            if V88:                                # 版88：compile で（式は同じ。中身を書き換える）\n"
     "                r_, d_ = _run(_homeo_core, cnt, exc, self.recv_ema, self.drive, a_)\n"
     "                self.recv_ema.copy_(r_)\n"
     "                self.drive.copy_(d_)\n"
     "            else:\n"
     "                self.recv_ema += a_ * (cnt - self.recv_ema)\n"
     "                self.drive += a_ * (exc.abs() - self.drive)\n"),
    ("            if getattr(self, \"scaling_on\", True):\n"
     "                # 版53：眠りは下げるだけ（⟲ 版52 まで両向き。上げる向きは人の根拠なしで【み】保留 10-02）\n",
     "            if V88 and getattr(self, \"scaling_on\", True):     # 版88：compile で（式は下と同じ）\n"
     "                self.scale = _run(_scale_down_core, self.rate_ema, self.target_recv, self.is_hippo, self.scale_skip,\n"
     "                                  self.scale, a_, bool(getattr(self, \"sleep_mode\", False)))\n"
     "            elif V88:\n"
     "                self.scale_up = _run(_scale_up_core, self.rate_long, self.target_recv, self.scale_skip, self.scale_up)\n"
     "            elif getattr(self, \"scaling_on\", True):\n"
     "                # 版53：眠りは下げるだけ（⟲ 版52 まで両向き。上げる向きは人の根拠なしで【み】保留 10-02）\n"),
    ("        fired = torch.zeros(self.n_recv_all, device=self.dev)\n"
     "        fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any\n"
     "        self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)\n",
     "        if V88:                                    # 版88：compile で（式・入れる行は下と同じ）\n"
     "            re_, rl = _run(_rate_core, *self._any, self.err_i, self.rep_i, self.inh_i, self.rate_ema, self.rate_long,\n"
     "                           self.n_recv_all, 1.0 / self.homeo_tau, 1.0 / LONG_TAU, not getattr(self, \"scaling_on\", True))\n"
     "            self.rate_ema.copy_(re_)\n"
     "            if not getattr(self, \"scaling_on\", True):\n"
     "                self.rate_long.copy_(rl)\n"
     "            return\n"
     "        fired = torch.zeros(self.n_recv_all, device=self.dev)\n"
     "        fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any\n"
     "        self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)\n"),
    ("class Group2:\n",
     "# 版88：学び・恒常性の小さな命令を compile で。ANAPLEX_V88=0 で版87 と同じ形\n"
     "V88 = _os.environ.get(\"ANAPLEX_V88\", \"1\") != \"0\"\n"
     "\n"
     "\n"
     "class Group2:\n"),
]
TB = [
    ("class Brain:\n",
     "def _bu_learn_core(src, pidx, pre_now, pre_prev, tgt, hippo, tab, post_age, d):\n"
     "    \"\"\"版88：_BUTable.learn の式（版87 と同じ）。返り値：重さに掛ける倍率・受け手が撃ってからの歩\"\"\"\n"
     "    post = src[pidx]\n"
     "    pre_e = pre_now | pre_prev\n"
     "    post_e = post[tgt] > 0\n"
     "    g_on = (2.0 * d - 1.0).clamp(0.0, 1.0)\n"
     "    ln_ctx = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS\n"
     "    ln_hip = LN_PRE_HIP * g_on\n"
     "    ln_pp = torch.where(hippo, ln_hip, ln_ctx)\n"
     "    k = post_age.clamp(max=WIN_MAXK + 1)[tgt]\n"
     "    pf = tab[hippo.long(), k] * (pre_now & ~post_e).float() * g_on\n"
     "    fac = torch.exp(ln_pp * (pre_e & post_e).float() + pf)\n"
     "    return fac, torch.where(post > 0, torch.ones_like(post_age), post_age + 1)\n"
     "\n"
     "\n"
     "def _cl_learn_core(any0, cl_ch, cl_spk, cl_pre_prev, cl_src, cl_recv, cl_post_prev, third, cl_w):\n"
     "    \"\"\"版88：CL → 皮質 の学び（式は版87 の Brain.step と同じ）。返り値：新しい重さ・この歩の受け手の発火\"\"\"\n"
     "    post = any0[cl_ch]\n"
     "    pre_e = (cl_spk + cl_pre_prev)[cl_src] > 0\n"
     "    post_e = post[cl_recv] > 0\n"
     "    d = (third / 2.0).clamp(0.0, 1.0)\n"
     "    ln_pp = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS\n"
     "    pf = (cl_post_prev[cl_recv] > 0) & ~post_e & (cl_spk[cl_src] > 0)\n"
     "    return cl_w * torch.exp(ln_pp * (pre_e & post_e).float() + _CL_LN_POST / CL_PAIRS * pf.float()), post\n"
     "\n"
     "\n"
     "class Brain:\n"),
    ("        post = src[self._pidx]\n"
     "        pre_now = self.pre_now\n"
     "        pre_e = pre_now | self.pre_prev\n",
     "        import torch_group2 as _G\n"
     "        if _G.V88 and torch.is_tensor(d) and d.dtype == self.w.dtype:    # 版88：compile で（式は下と同じ）\n"
     "            fac, pa = _G._run(_bu_learn_core, src, self._pidx, self.pre_now, self.pre_prev, self.tgt, self.hippo,\n"
     "                              self.tab, self.post_age, d)\n"
     "            self.w.mul_(fac)\n"
     "            self.pre_prev = self.pre_now.clone()\n"
     "            self.post_age = pa\n"
     "            return\n"
     "        post = src[self._pidx]\n"
     "        pre_now = self.pre_now\n"
     "        pre_e = pre_now | self.pre_prev\n"),
    ("            post = self.g._any[0][self.cl_ch]                      # この歩のどこかで撃ったか\n"
     "            pre_e = (self.cl_spk + self.cl_pre_prev)[self.cl_src] > 0\n",
     "          import torch_group2 as _G\n"
     "          if _G.V88 and torch.is_tensor(self.g.third):              # 版88：compile で（式は下と同じ）\n"
     "            self.cl_w, post = _G._run(_cl_learn_core, self.g._any[0], self.cl_ch, self.cl_spk, self.cl_pre_prev,\n"
     "                                      self.cl_src, self.cl_recv, self.cl_post_prev, self.g.third, self.cl_w)\n"
     "            self.cl_pre_prev = self.cl_spk\n"
     "            self.cl_post_prev = post\n"
     "          else:\n"
     "            post = self.g._any[0][self.cl_ch]                      # この歩のどこかで撃ったか\n"
     "            pre_e = (self.cl_spk + self.cl_pre_prev)[self.cl_src] > 0\n"),
]
BB = [("VERSION = 87          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86)",
       "VERSION = 88          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87)"),
      ("#  87  耳の入力はピン留めした置き場所から待たずに送る・誤差の和は GPU の上で足して最後に1回読む【測】版86 1歩 10.8ms\n",
       "#  87  耳の入力はピン留めした置き場所から待たずに送る・誤差の和は GPU の上で足して最後に1回読む【測】版86 1歩 10.8ms\n"
       "#  88  恒常性・恒常性の観測・下からの表の学び・CL の学びを compile で（まとめて4つ）【測】版87 1歩 10.0ms\n")]
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
    if babble.VERSION != 87:
        sys.exit(f"⚠️ babble.py が版87 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("torch_brain.py", TB), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版88 を当てた（torch_group2・torch_brain・babble）。学び・恒常性の小さな命令を compile で（4つ）", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "88" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['88'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版88 でも回るようにした", flush=True)
