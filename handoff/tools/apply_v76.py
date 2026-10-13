"""版76：外すスイッチを消す ＋ 網の辺の計算を torch.compile でまとめる（速さだけ。式は同じ）
（~/anaplex で。版75 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v76.py
【み】10-14「外していいよ」（ANAPLEX_BU・ANAPLEX_CL・ANAPLEX_SUM の 0）「HIP グラフと torch.compile も一緒に入れちゃってよかった」
A 外すスイッチを消した（既定のときの動きは版75 と同じ）
  ANAPLEX_BU=0（学ばない本数の和）・ANAPLEX_CL=0（CL を外す）・ANAPLEX_SUM=0（時間の足し合わせを外す）
  前の網の道（_band_sum・_proj・_premotor・K_PM 27・M_PC 4）も消した。bu_on・cl_on は道具のために True で残す
B 網（Group2）の辺の計算を torch.compile でまとめた（_sum：足し込み ／ _flush：まとめて足す ／ 学び）
  【測】probe_gpu2（10-14）：gfx1201 で torch.compile が動く（10-10 は ✗）。撃った辺の数は毎歩変わる ⇒ dynamic
  放出のくじ（binomial）は今までどおり外で先に引く（くじの順は変わらない）
  まとめられないときは1回だけ「⚠️ torch.compile が … で動かない」と出して、まとめない形で回す（値は同じ）
  ⚠️ GPU では足す順・丸めが少し変わりうる（10-10 の版54 と同じ「丸めの範囲」）
  初めの数歩は組み立て（compile）に時間がかかる（1回だけ）
C 道具：measure_check2 の「自分 0 で割る」を直した ／ measure_v67d は K_PM_SYN を見る
⚠️ HIP グラフはまだ（撃った辺の数が毎歩変わるので、決まった大きさの箱に詰める形にしてから。版77）
【測】コンテナ（CPU・相手200・作り物の声 300歩・放出のくじを期待値に）版75 と並べて 発火 違い 0 ／ 重さ・状態 差 0 ／ compile 3つとも動く・作り直し 0
"""
import ast, pathlib, re, sys

BRAIN = [('K_PC, M_PC = 8, 4                  # PPN → CL：8本受けて4本で撃つ ⇒ 約10.4Hz【計算】（CL 11Hz に合わせたノブ）\n', 'K_PC = 8                           # PPN → CL：1体が受ける本数（版76：重さ 1 で数えた M_PC 4 は消した。今は M_PC_SYN）\n'),
 ('K_PM = 27\n# 版71：シナプスのとき（ANAPLEX_BU が 0 でない）の値。短期の増減で1発の放出が下がるぶんを、それぞれの定義で出し直した\n', '#   版76：K_PM 27（重さ 1・二項分布の形）は消した。今は K_PM_SYN\n# 版71：シナプスの値。短期の増減で1発の放出が下がるぶんを、それぞれの定義で出し直した\n'),
 ('        # 版67：上行の覚醒の道（ANAPLEX_CL=0 で版66 と同じ）\n', '        # 版67：上行の覚醒の道（版76：外すスイッチ ANAPLEX_CL=0 を無くした。cl_on は道具のために残す）\n'),
 ('        self.cl_on = _os_cl.environ.get("ANAPLEX_CL", "1") != "0"\n', '        self.cl_on = True\n'),
 ('        # 版69：下から上への道を学ぶシナプスに（ANAPLEX_BU=0 で版68 と同じ）。本数・どこからどこへ は今の道のまま\n        self.bu_on = _os_cl.environ.get("ANAPLEX_BU", "1") != "0"\n',
  '        # 版69：下から上への道を学ぶシナプスに。本数・どこからどこへ は今の道のまま\n        #   版76：外すスイッチ ANAPLEX_BU=0（学ばない本数の和の網）を無くした。bu_on は道具のために残す\n        self.bu_on = True\n'),
 ('            if _os_cl.environ.get("ANAPLEX_SUM", "1") != "0":       # 版72：時間の足し合わせ\n'
  '                import math as _m_sum\n'
  '                dk = lambda tau: _m_sum.exp(-10.67 / tau)\n'
  '                vd = torch.full((self.g.n_ch_all,), dk(SUM_TAU_MS), device=dev)\n'
  '                for nm_h in ("歯状回", "CA3", "CA1"):\n'
  '                    o_h = int(self.off_ch[NAMES.index(nm_h)]); vd[o_h:o_h + self.n[nm_h]] = dk(SUM_TAU_HIPPO_MS)\n'
  '                self.g.v_decay = vd\n'
  '                self.g.v_carry = torch.zeros(self.g.n_ch_all, device=dev)\n'
  '                self.g.v_decay_inh = dk(SUM_TAU_INH_MS)\n'
  '                self.g.v_carry_inh = None                              # 1周目に大きさを合わせて作る\n',
  '            # 版72：時間の足し合わせ（版76：外すスイッチ ANAPLEX_SUM=0 を無くした）\n'
  '            import math as _m_sum\n'
  '            dk = lambda tau: _m_sum.exp(-10.67 / tau)\n'
  '            vd = torch.full((self.g.n_ch_all,), dk(SUM_TAU_MS), device=dev)\n'
  '            for nm_h in ("歯状回", "CA3", "CA1"):\n'
  '                o_h = int(self.off_ch[NAMES.index(nm_h)]); vd[o_h:o_h + self.n[nm_h]] = dk(SUM_TAU_HIPPO_MS)\n'
  '            self.g.v_decay = vd\n'
  '            self.g.v_carry = torch.zeros(self.g.n_ch_all, device=dev)\n'
  '            self.g.v_decay_inh = dk(SUM_TAU_INH_MS)\n'
  '            self.g.v_carry_inh = None                              # 1周目に大きさを合わせて作る\n'),
 ('        drive = self.but.drive(0, [ec2])[0] if self.bu_on else ec2[self.perforant].sum(1)   # 版70・75（表の1回目）\n', '        drive = self.but.drive(0, [ec2])[0]                    # 版70・75（表の1回目）\n'),
 ('            self._bo["下丘→視床"] if self.bu_on else ic,   # 下丘 → 視床（1本 → 1体、帯をまたがない。版70 シナプス・版75 表）\n',
  '            self._bo["下丘→視床"],                         # 下丘 → 視床（1本 → 1体、帯をまたがない。版70 シナプス・版75 表）\n'),
 ('            self._bo["苔状線維"] if self.bu_on else (dg[self.mossy].max(1).values > 0).float(),        # 苔状線維（起爆）\n', '            self._bo["苔状線維"],                          # 苔状線維（起爆）\n'),
 ('            self._bo["シャッファー"] if self.bu_on else (self.ca3_prev[self.schaffer].max(1).values > 0).float(),\n', '            self._bo["シャッファー"],\n'),
 ('            ppn = self._ppn if self.bu_on else (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()   # 版75\n'
  '            cl = ((self._bo["PPN→CL"] >= M_PC_SYN) if self.bu_on else (ppn[self.cl_ppn].sum(1) >= M_PC)).float()   # 版70・71・75\n',
  '            cl = (self._bo["PPN→CL"] >= M_PC_SYN).float()      # 版70・71・75（PPN のくじは表の前で引いた）\n'),
 ('    def _band_sum(self, thal):\n'
  '        """聴覚の細胞は、自分の帯の視床の本数（繊維の段）の合計を受ける\n'
  '        【ヒト】一次聴覚野は柱の中で周波数の同調がそろう（Bitterman 2008）⇒ 同じ帯の細胞は同じ入力を受ける\n'
  '        帯の並びは 帯 × 繊維（耳の encode と同じ）"""\n'
  '        import config as C\n'
  '        n_fib = len(thal) // C.N_MELS\n'
  '        return thal.view(C.N_MELS, n_fib).sum(1).repeat_interleave(n_fib)\n'
  '\n',
  ''),
 ('        """版69：下から上への道。シナプスなら重さ × 放出の和、ANAPLEX_BU=0 なら前の本数の和（proj が None なら視床 → 聴覚の帯の和）"""\n'
  '        if self.bu_on:\n'
  '            return self._bo[key]                                   # 版75：表で足した結果（入力 v は表の側で同じものを使った）\n'
  '        return self._band_sum(v) if proj is None else self._proj(v, proj)\n'
  '\n'
  '    def _proj(self, v, proj):\n'
  '        """届いた本数を足す（x_gain は「何本ぶんの入力か」）\n'
  '        🔴 本数で割らない。09-11 に「撃った数で割る」を周転円として落としている\n'
  '           （割り算は脳の形として人にも霊長類にも出てこない）。\n'
  '           広い窓の体に本数が多く届くのは、測ってから原因を見る"""\n'
  '        src, tgt, cnt = proj\n'
  '        return torch.zeros(len(cnt), device=self.dev).index_add_(0, tgt, v[src])\n',
  '        """版69：下から上への道（シナプス）の足した結果。版75：表で足した結果を返す（入力 v は表の側で同じものを使った）\n'
  '        版76：前の「学ばない本数の和」（_band_sum・_proj）は消した\n'
  '        🔴 本数で割らない。09-11 に「撃った数で割る」を周転円として落としている"""\n'
  '        return self._bo[key]\n'),
 ('        """版70：運動前野の送り手（1体に K_PM 体、体ごとに撃つ）→ 運動（シナプス）。ANAPLEX_BU=0 なら前の二項分布"""\n        if not self.bu_on:\n            return self._premotor(cmd)\n',
  '        """版70：運動前野の送り手（1体に K_PM_SYN 体、体ごとに撃つ）→ 運動（シナプス）"""\n'),
 ('    def _premotor(self, cmd):\n'
  '        """版49：運動前野の送り手 K_PM 本のうち、この歩に撃った数（指令が入っている体は R_PM_ON、ほかは R_PM_OFF）"""\n'
  '        p = torch.where(cmd > 0, torch.full_like(cmd, R_PM_ON), torch.full_like(cmd, R_PM_OFF))\n'
  '        return torch.binomial(torch.full_like(p, float(K_PM)), p, generator=self.gen)\n'
  '\n',
  '')]
G2 = [('import os as _os\n',
  'import os as _os\n'
  '\n'
  '# 版76：辺ごとの計算を torch.compile でまとめる（小さな命令を1つにして、呼ぶ回数と GPU の時間を減らす）\n'
  '#   【測】probe_gpu2（10-14、torch 2.14.1・ROCm 7.14）：gfx1201 で動く・小さい命令の列で 9.6倍・同じ値（丸めの範囲）\n'
  '#   撃った辺の数は毎歩変わる ⇒ dynamic（数が変わっても作り直さない）\n'
  '#   動かないとき（作るときの誤り）は1回だけ知らせて、まとめない形で回す（値は同じ）\n'
  '_CC = {}\n'
  '\n'
  '\n'
  'def _cc(fn):\n'
  '    c = _CC.get(fn)\n'
  '    if c is None:\n'
  '        c = _CC[fn] = torch.compile(fn, dynamic=True)\n'
  '    return c\n'
  '\n'
  '\n'
  'def _run(fn, *a):\n'
  '    try:\n'
  '        return _cc(fn)(*a)\n'
  '    except Exception as ex:                                   # まとめられないときは元の形で（1回だけ知らせる）\n'
  '        if _CC.get(fn) is not fn:\n'
  '            print(f"⚠️ torch.compile が {fn.__name__} で動かない（{type(ex).__name__}）。まとめずに回す", flush=True)\n'
  '            _CC[fn] = fn\n'
  '        return fn(*a)\n'
  '\n'
  '\n'
  'def _sum_core(e, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r, n_recv: int):\n'
  '    """版76：Group2._sum の辺の計算（式は版75 と同じ）"""\n'
  '    re_ = e_recv[e].long()\n'
  '    se_ = e_src[e].long()\n'
  '    ie = e_inh[e]\n'
  '    amp = torch.where(ie, 4.0, c_exc[re_]) * w[e]\n'
  '    if amp_s is not None:\n'
  '        amp = amp * amp_s[e]\n'
  '    amp = amp * send_gain[se_]\n'
  '    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)\n'
  '    both.scatter_add_(0, re_ + n_recv * ie.long(), amp)\n'
  '    mm = mod_send[se_] & ~ie\n'
  '    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])\n'
  '    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, re_, amp * mm)\n'
  '    return both, mod\n'
  '\n'
  '\n'
  'def _flush_core(idx, k_start, elig, A, a_now, w):\n'
  '    """版76：Group2._flush の辺の計算（a_now ＝ A[kcur]）"""\n'
  '    ks = k_start[idx].long()\n'
  '    pend = elig[idx] * (a_now - A[ks - 1]).float()\n'
  '    w[idx] = (w[idx] + pend).clamp(min=0.0)\n'
  '\n'
  '\n'
  'def _learn_core(e, e_recv, e_src, e_inh, is_inh_r, err_sig, g, pw, elig, hist_n, w, c_exc, n_recv: int):\n'
  '    """版76：Group2._pass の学びの辺の計算（式は版75 と同じ。g ＝ eta × eta_gain、pw ＝ 減り^(−k)）"""\n'
  '    r = e_recv[e].long()\n'
  '    gg = g * err_sig[r]\n'
  '    to_inh = is_inh_r[r]\n'
  '    elig[e] += torch.where(e_inh[e] | to_inh, torch.zeros_like(gg), gg) * pw\n'
  '    se = e_src[e].long()\n'
  '    ltd = to_inh & ~e_inh[e] & (hist_n[se] >= INH_LTD_N) & (w[e] * c_exc[r] > INH_LTD_TH)\n'
  '    w[e] = torch.where(ltd, w[e] * INH_LTD_F, w[e])\n'
  '    cnt = torch.zeros(n_recv, device=w.device, dtype=w.dtype)\n'
  '    cnt.scatter_add_(0, r, torch.ones(e.shape[0], device=w.device, dtype=w.dtype))\n'
  '    return cnt\n'),
 ('        ks = self.k_start[idx].long()\n'
  '        pend = self.elig[idx] * (self.A[self.kcur] - self.A[ks - 1]).float()\n'
  '        self.w[idx] = (self.w[idx] + pend).clamp(min=0.0)\n'
  '        self.k_start.index_fill_(0, idx, self.kcur + 1)     # 版54b\n',
  '        _run(_flush_core, idx, self.k_start, self.elig, self.A, self.A[self.kcur], self.w)   # 版76\n        self.k_start.index_fill_(0, idx, self.kcur + 1)     # 版54b\n'),
 ('        both = torch.zeros(self.n_recv_all * 2, device=self.dev)\n'
  '        mod = torch.zeros(self.n_recv_all, device=self.dev)\n'
  '        if e is None or len(e) == 0:\n'
  '            return both, mod\n'
  '        # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0\n'
  '        re_ = self.e_recv[e].long()\n'
  '        se_ = self.e_src[e].long()\n'
  '        amp = torch.where(self.e_inh[e], 4.0, self.c_exc[re_]) * self.w[e]\n'
  '        if self.stp is not None:\n',
  '        if e is None or len(e) == 0:\n'
  '            return (torch.zeros(self.n_recv_all * 2, device=self.dev), torch.zeros(self.n_recv_all, device=self.dev))\n'
  '        # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0\n'
  '        # 版76：放出（くじ）は今までどおり先に。辺の計算は _sum_core（torch.compile）で1回に\n'
  '        if self.stp is not None:\n'),
 ('            amp = amp * self._amp[e]\n'
  '        amp = amp * self.send_gain[se_]   # 脳幹：眠りで上からの戻りを落とす（版38）\n'
  '        # 足し込みは1回で済ませる（抑制を負にして繋ぎ、あとで分ける）\n'
  '        both.scatter_add_(0, re_ + self.n_recv_all * self.e_inh[e].long(), amp)\n'
  '        mm = self.mod_send[se_] & ~self.e_inh[e]\n'
  '        mm = mm & ~(self.mod_drive_inh[se_] & self.is_inh_r[re_])   # 版66：印の送り手 → 抑制の体 は撃たせる側\n'
  '        mod = mod.index_add(0, re_, amp * mm)     # 版54b：選ばない辺は 0 を足す（選んで足すのと同じ値）\n'
  '        return both, mod\n',
  '        # 脳幹：眠りで上からの戻りを落とす（版38）／ 足し込みは1回（抑制を負にして繋ぎ、あとで分ける）\n'
  '        # 版66：印の送り手 → 抑制の体 は撃たせる側 ／ 版54b：選ばない辺は 0 を足す（選んで足すのと同じ値）\n'
  '        return _run(_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.w, self.c_exc,\n'
  '                    self._amp if self.stp is not None else None, self.send_gain, self.mod_send,\n'
  '                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)\n'),
 ('            r = self.e_recv[e].long()\n            gg = self.eta * self.eta_gain * err_sig[r]   # eta_gain：驚きで上がる（青斑核。版42 で戻した）\n', '            # eta_gain：驚きで上がる（青斑核。版42 で戻した）\n'),
 ('            to_inh = self.is_inh_r[r]\n'
  '            self.elig[e] += torch.where(self.e_inh[e] | to_inh, torch.zeros_like(gg), gg) * (self.elig_decay ** (-k))\n'
  '            # 版47：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）\n'
  '            se = self.e_src[e].long()\n'
  '            ltd = to_inh & ~self.e_inh[e] & (self.hist_n[se] >= INH_LTD_N) & \\\n'
  '                (self.w[e] * self.c_exc[r] > INH_LTD_TH)\n'
  '            self.w[e] = torch.where(ltd, self.w[e] * INH_LTD_F, self.w[e])     # 版54b：判定で待たない\n',
  '            # 版47：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）\n'
  '            # 版76：辺の計算は _learn_core（torch.compile）で1回に（式は同じ。減り^(−k) と eta は値で渡す）\n'
  '            g_ = torch.as_tensor(self.eta * self.eta_gain, dtype=self.w.dtype, device=self.dev)\n'
  '            pw_ = torch.as_tensor(self.elig_decay ** (-k), dtype=self.w.dtype, device=self.dev)\n'
  '            cnt = _run(_learn_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig, g_, pw_,\n'
  '                       self.elig, self.hist_n, self.w, self.c_exc, self.n_recv_all)\n'),
 ('            cnt = torch.zeros(self.n_recv_all, device=self.dev)\n            cnt.scatter_add_(0, r, torch.ones(len(e), device=self.dev))\n', '')]
BB = [('VERSION = 75          # コードの版。網の作りを変えたら上げる\nSAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74)', 'VERSION = 76          # コードの版。網の作りを変えたら上げる\nSAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75)'),
 ('#      【測】版74 1歩 38.3ms ／ シナプスの分 8.6ms（17回/歩 CPU が待つ）【み】10-14「それならいいよ」\n',
  '#      【測】版74 1歩 38.3ms ／ シナプスの分 8.6ms（17回/歩 CPU が待つ）【み】10-14「それならいいよ」\n'
  '#  76  外すスイッチ（ANAPLEX_BU・CL・SUM の 0）と前の網の道を消した【み】10-14「外していいよ」\n'
  '#      網の辺の計算（足し込み・学び・まとめて足す）を torch.compile で1つに（式・足す順は同じ）【測】版75 1歩 33.0ms\n')]
TOOL_FIX = {'measure_check2.py': [('          f"  ⇒ 録音÷自分 {r[\'録音\']/r[\'自分\']:.2f}（＞1 が人の向き）  返し手÷自分 {r[\'返し手\']/r[\'自分\']:.2f}"\n',
                        '          f"  ⇒ 録音÷自分 {(r[\'録音\']/r[\'自分\'] if r[\'自分\'] else float(\'nan\')):.2f}（＞1 が人の向き）"   # 版76：自分 0 で割らない\n'
                        '          f"  返し手÷自分 {(r[\'返し手\']/r[\'自分\'] if r[\'自分\'] else float(\'nan\')):.2f}"\n')],
 'measure_v67d.py': [('TB.K_PM = int(args[2]) if len(args) > 2 else (20 if B.VERSION == 67 else TB.K_PM)   # 版68 は torch_brain の値（27）\n',
                      'TB.K_PM_SYN = int(args[2]) if len(args) > 2 else TB.K_PM_SYN   # 版76：K_PM（重さ 1 の形）は消した\n'),
                     ('print(f"K_PM {TB.K_PM}   的：', 'print(f"K_PM {TB.K_PM_SYN}   的：')]}
TOOLS = ('measure_check.py',
 'measure_check2.py',
 'measure_v59a.py',
 'measure_v59b.py',
 'measure_v59c.py',
 'measure_v59d.py',
 'measure_v62a.py',
 'measure_v62b.py',
 'measure_v65a.py',
 'measure_v66a.py',
 'measure_v67a.py',
 'measure_v67b.py',
 'measure_v67d.py',
 'measure_speed75.py')


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
    if babble.VERSION != 75:
        sys.exit(f"⚠️ babble.py が版75 ではない（版 {babble.VERSION}）。先に apply_v75.py")
    out = [edit("torch_brain.py", BRAIN), edit("torch_group2.py", G2), edit("babble.py", BB)]
    tools = [edit(f, prs) for f, prs in TOOL_FIX.items() if pathlib.Path(f).exists()]
    for p, s in out + tools:
        p.write_text(s)
    print("✅ 版76 を当てた（torch_brain・torch_group2・babble）。外すスイッチを消した・網の辺の計算を torch.compile で1つに")
    for p, s in tools:
        print(f"   {p.name}：直した")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "76" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['76'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版76 でも回るようにした")
