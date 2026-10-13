"""版75：下からのシナプスの道（17本）を1つの表にまとめて回す（速さだけ。線・重さ・式は変えない）
（~/anaplex で。版74 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v75.py
【み】10-14「結局さ、その時間の壁を超えないといけないんよね」「今の環境を測るところからじゃない？」／（1 を噛み砕いた説明に）「それならいいよ」
【測】measure_speed74（10-14、torch 2.14.1・ROCm 7.14）
  版74 1歩 38.3ms ／ ANAPLEX_BU=0（版68 と同じ網）28.0ms ⇒ シナプスの分 約10ms
  下からのシナプス 8.6ms（足し込み 4.6 ／ 学び 4.0）：17本の道を1本ずつ、足し込みのたびに torch.nonzero（CPU が GPU を待つ。17回/歩）
  17本の合計は 15万本（網の 7,200万本の 0.2%）⇒ 時間は量ではなく、呼ぶ回数と待ちの分
直すもの（意味は変えない）
  道ごとの _BUSyn は作るとき・保存・読み込みの形として残す。作ったあと1つの表（_BUTable）にまとめて回す
  足し込み：1歩に2回（貫通路 ／ 残りの16本。苔状線維は歯状回の発火を待つので分ける）
    撃った線を数えずに、表の全部の線で「撃った線だけ放出、撃っていない線は 0」を計算する（CPU が待たない）
    短期の増減の状態は、撃った線だけ進める（今までと同じ）
  学び：1歩に1回（17本ぶんを1回で）。式は版74 と同じ（窓・門・海馬の印は線ごとに持つ）
  乱数の順：歯状回のくじ → 運動前野の送り手のくじ → 蹴り → CA3 の成功、は今までと同じ順（self.gen）
            PPN のくじ（cl_gen）は同じ歩の中で前へ動かす（cl_gen を使うのは PPN と CL の放出だけ。CL の放出より前のまま）
⚠️ 変わること：短期の増減の放出（二項分布のくじ）を、撃った線だけ → 全部の線（撃っていない線は数 0）で引く
   ⇒ くじの数の並びが変わる。分布は同じ（撃っていない線は必ず 0）。同じ種でも発火の列は版74 と一致しない
   確かめ：くじを期待値に置き換えた形（check_v75.py）で、版74 と版75 の発火・重さが一致することを見る
保存：state.pt の形は版74 と同じ（道ごと）。保存の前に表から道へ写し、読み込みの後に表を作り直す
"""
import ast, pathlib, re, sys

BRAIN = [
    # 1 表のクラス（_BUSyn の後ろ）
    ("class Brain:\n",
     "class _BUTable:\n"
     "    \"\"\"版75：下からのシナプスの道を1つの表にまとめる。足し込み・学びの式は _BUSyn と同じ（撃った線を数えずに表の全部で計算）\"\"\"\n\n"
     "    def __init__(self, bu, groups):\n"
     "        self.groups = [list(g) for g in groups]\n"
     "        self.keys = [k for g in self.groups for k in g]\n"
     "        ss = [bu[k] for k in self.keys]\n"
     "        dev = ss[0].w.device\n"
     "        ne = [len(s.w) for s in ss]\n"
     "        nt = [s.n_tgt for s in ss]\n"
     "        eo = [0]\n"
     "        to = [0]\n"
     "        for a, b in zip(ne, nt):\n"
     "            eo.append(eo[-1] + a); to.append(to[-1] + b)\n"
     "        self.eo, self.to, self.nt = eo, to, nt\n"
     "        self.src = [s.src for s in ss]\n"
     "        self.tgt = torch.cat([s.tgt + to[i] for i, s in enumerate(ss)])\n"
     "        self.pid = torch.cat([torch.full((n,), i, dtype=torch.long, device=dev) for i, n in enumerate(ne)])\n"
     "        self.hippo = torch.cat([torch.full((n,), bool(s.hippo), dtype=torch.bool, device=dev) for n, s in zip(ne, ss)])\n"
     "        self.w = torch.cat([s.w for s in ss])\n"
     "        st = ss[0].stp\n"
     "        self.tbl, self.dec_d, self.dec_f, self.N, self.gen = st.tbl, st.dec_d, st.dec_f, st.N, st.gen\n"
     "        self.row = torch.cat([s.stp.row for s in ss])\n"
     "        self.state = torch.cat([s.stp.state for s in ss])\n"
     "        self.fac = torch.cat([s.stp.fac for s in ss])\n"
     "        self.last = torch.cat([s.stp.last for s in ss])\n"
     "        self.deplete = torch.cat([s.stp.deplete for s in ss])\n"
     "        self.t = torch.tensor([s.t for s in ss], dtype=torch.long, device=dev)   # 道ごとの歩の数（読み込んだ道は進んでいることがある）\n"
     "        self.post_age = torch.cat([s.post_age for s in ss])\n"
     "        self.pre_now = torch.zeros(len(self.w), dtype=torch.bool, device=dev)\n"
     "        self.pre_prev = torch.zeros(len(self.w), dtype=torch.bool, device=dev)\n"
     "        self.tab = torch.stack([ss[0]._tab[False], ss[0]._tab[True]])\n"
     "        self.ch0 = [s.ch0 for s in ss]\n"
     "        self._gsrc = {}\n"
     "        self._pidx = None\n\n"
     "    def _span(self, gi):\n"
     "        i0 = self.keys.index(self.groups[gi][0])\n"
     "        i1 = i0 + len(self.groups[gi])\n"
     "        return i0, i1\n\n"
     "    @torch.no_grad()\n"
     "    def drive(self, gi, vs):\n"
     "        \"\"\"gi 番目の組の道を1回で足す。vs：組の道の順の入力。返り値：道ごとの「届いた重さ × 放出 の和」\"\"\"\n"
     "        i0, i1 = self._span(gi)\n"
     "        a, b, ta, tb = self.eo[i0], self.eo[i1], self.to[i0], self.to[i1]\n"
     "        if gi not in self._gsrc:                           # 入力を繋いだ番号（入力の長さは毎歩同じ）\n"
     "            off, parts = 0, []\n"
     "            for i, v in zip(range(i0, i1), vs):\n"
     "                parts.append(self.src[i] + off); off += v.shape[0]\n"
     "            self._gsrc[gi] = torch.cat(parts)\n"
     "        x = torch.cat(vs)\n"
     "        fired = x[self._gsrc[gi]] > 0\n"
     "        self.pre_now[a:b] = fired\n"
     "        te = self.t[self.pid[a:b]]\n"
     "        # 短期の増減（torch_area.STP.release_at と同じ式。撃った線だけ状態を進める）\n"
     "        r = self.row[a:b].long()\n"
     "        n = (te - self.last[a:b]).float()\n"
     "        dep = self.deplete[a:b]\n"
     "        s0, f0 = self.state[a:b], self.fac[a:b]\n"
     "        st = s0 * torch.where(dep, torch.ones_like(n), self.dec_d[r] ** n)\n"
     "        fc = f0 * self.dec_f[r] ** n\n"
     "        Pr0 = self.tbl[r, 0]\n"
     "        P = (Pr0 + (1 - Pr0) * fc).clamp(0, 1)\n"
     "        N = self.N[r]\n"
     "        avail = torch.where(dep, st, 1 - st)\n"
     "        ff = fired.float()\n"
     "        cnt = torch.binomial((N * avail).round().clamp(min=0) * ff, P, generator=self.gen)   # 撃っていない線は数 0\n"
     "        rel = cnt / (N * Pr0).clamp(min=1e-6)\n"
     "        k = torch.where(dep, P, self.tbl[r, 1])\n"
     "        st2 = torch.where(dep, st * (1 - k), st + (1 - st) * k)\n"
     "        fc2 = fc + (1 - fc) * self.tbl[r, 3]\n"
     "        self.state[a:b] = torch.where(fired, st2, s0)\n"
     "        self.fac[a:b] = torch.where(fired, fc2, f0)\n"
     "        self.last[a:b] = torch.where(fired, te.to(self.last.dtype), self.last[a:b])\n"
     "        out = torch.zeros(tb - ta, device=x.device).index_add_(0, self.tgt[a:b] - ta, self.w[a:b] * rel * ff)\n"
     "        self.t[i0:i1] += 1\n"
     "        return list(torch.split(out, self.nt[i0:i1]))\n\n"
     "    @torch.no_grad()\n"
     "    def learn(self, anyr, cl_spk, d):\n"
     "        \"\"\"_BUSyn.learn と同じ式を、表の全部の線で1回に。anyr：網のこの歩のどこかで撃ったか ／ cl_spk：CL の発火（PPN → CL の受け手）\"\"\"\n"
     "        src = anyr if cl_spk is None else torch.cat([anyr, cl_spk])\n"
     "        if self._pidx is None:\n"
     "            na = anyr.shape[0]\n"
     "            dev = anyr.device\n"
     "            self._pidx = torch.cat([(torch.arange(n, device=dev) + (c if c >= 0 else na)) for c, n in zip(self.ch0, self.nt)])\n"
     "        post = src[self._pidx]\n"
     "        pre_now = self.pre_now\n"
     "        pre_e = pre_now | self.pre_prev\n"
     "        post_e = post[self.tgt] > 0\n"
     "        g_on = (2.0 * torch.as_tensor(d, dtype=self.w.dtype, device=self.w.device) - 1.0).clamp(0.0, 1.0)   # 版74：門の開き\n"
     "        ln_ctx = ((1.0 - d) * _CL_LN_NODA + d * _CL_LN_DA) / CL_PAIRS   # 入力が先（皮質）\n"
     "        ln_hip = LN_PRE_HIP * g_on                                     # 入力が先（海馬）\n"
     "        ln_pp = torch.where(self.hippo, ln_hip, ln_ctx)\n"
     "        k = self.post_age.clamp(max=WIN_MAXK + 1)[self.tgt]\n"
     "        pf = self.tab[self.hippo.long(), k] * (pre_now & ~post_e).float() * g_on   # 受け手が先\n"
     "        self.w.mul_(torch.exp(ln_pp * (pre_e & post_e).float() + pf))\n"
     "        self.pre_prev = pre_now.clone()\n"
     "        self.post_age = torch.where(post > 0, torch.ones_like(self.post_age), self.post_age + 1)\n\n"
     "    @torch.no_grad()\n"
     "    def unpack(self, bu):\n"
     "        \"\"\"表の今の値を道ごとの _BUSyn に写す（保存の前・中を見る道具のため）\"\"\"\n"
     "        tl = self.t.tolist()\n"
     "        for i, k in enumerate(self.keys):\n"
     "            s, a, b, ta, tb = bu[k], self.eo[i], self.eo[i + 1], self.to[i], self.to[i + 1]\n"
     "            s.w, s.t = self.w[a:b].clone(), int(tl[i])\n"
     "            for kk in (\"row\", \"state\", \"fac\", \"last\", \"deplete\"):\n"
     "                setattr(s.stp, kk, getattr(self, kk)[a:b].clone())\n"
     "            s.post_age = self.post_age[ta:tb].clone()\n\n\n"
     "class Brain:\n"),
    # 2 組み立ての最後に表を作る（PPN → CL の後ろ）
    ("                self.bu[\"PPN→CL\"] = _BUSyn(self.cl_ppn.reshape(-1), ar(N_CL).repeat_interleave(K_PC), N_CL, -1, stp_path, dev, gb)\n",
     "                self.bu[\"PPN→CL\"] = _BUSyn(self.cl_ppn.reshape(-1), ar(N_CL).repeat_interleave(K_PC), N_CL, -1, stp_path, dev, gb)\n"
     "            self.bu_pack()                                           # 版75：道をまとめた表\n"),
    # 3 貫通路（1回目の足し込み）
    ("        drive = self.bu[\"貫通路\"].drive(ec2) if self.bu_on else ec2[self.perforant].sum(1)   # 版70\n",
     "        drive = self.but.drive(0, [ec2])[0] if self.bu_on else ec2[self.perforant].sum(1)   # 版70・75（表の1回目）\n"),
    # 4 残りの道（2回目の足し込み）：入力は今までと同じもの
    ("        ic = self.ic_out                                       # 版43b：1歩前の下丘（耳 → 下丘 → 視床）\n"
     "        self.ic_out = ear\n",
     "        ic = self.ic_out                                       # 版43b：1歩前の下丘（耳 → 下丘 → 視床）\n"
     "        self.ic_out = ear\n"
     "        if self.bu_on:                                         # 版75：残りの道を1回で（入力は版74 と同じもの）\n"
     "            vin = {\"下丘→視床\": ic, \"視床→聴覚\": self.up[\"視床\"], \"運動前野→運動\": self._pm_in(cmd),\n"
     "                   \"聴覚→連合\": self.up[\"聴覚\"], \"連合→高次\": self.up[\"連合\"], \"苔状線維\": dg, \"シャッファー\": self.ca3_prev,\n"
     "                   \"高次→前頭前野\": self.up[\"高次\"], \"背内側視床→前頭前野\": self.md_out,\n"
     "                   \"高次→嗅内浅\": self.up[\"高次\"], \"前頭前野→嗅内浅\": self.up[\"前頭前野\"],\n"
     "                   \"CA1→嗅内深\": self.ca1_out, \"CA1→側坐核\": self.ca1_out, \"嗅内深→側坐核\": self.pred[\"嗅内深\"],\n"
     "                   \"下丘→上丘\": ic}\n"
     "            if \"PPN→CL\" in self.bu:                              # PPN のくじ（cl_gen）を CL の放出より前に引く\n"
     "                self._ppn = (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()\n"
     "                vin[\"PPN→CL\"] = self._ppn\n"
     "            g1 = self.but.groups[1]\n"
     "            self._bo = dict(zip(g1, self.but.drive(1, [vin[k] for k in g1])))\n"),
    ("            self.bu[\"下丘→視床\"].drive(ic) if self.bu_on else ic,   # 下丘 → 視床（1本 → 1体、帯をまたがない。版70 シナプス）\n",
     "            self._bo[\"下丘→視床\"] if self.bu_on else ic,   # 下丘 → 視床（1本 → 1体、帯をまたがない。版70 シナプス・版75 表）\n"),
    ("            self.bu[\"苔状線維\"].drive(dg) if self.bu_on else (dg[self.mossy].max(1).values > 0).float(),        # 苔状線維（起爆）\n",
     "            self._bo[\"苔状線維\"] if self.bu_on else (dg[self.mossy].max(1).values > 0).float(),        # 苔状線維（起爆）\n"),
    ("            self.bu[\"シャッファー\"].drive(self.ca3_prev) if self.bu_on else (self.ca3_prev[self.schaffer].max(1).values > 0).float(),\n",
     "            self._bo[\"シャッファー\"] if self.bu_on else (self.ca3_prev[self.schaffer].max(1).values > 0).float(),\n"),
    # 5 PPN → CL
    ("            ppn = (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()\n"
     "            cl = ((self.bu[\"PPN→CL\"].drive(ppn) >= M_PC_SYN) if self.bu_on else (ppn[self.cl_ppn].sum(1) >= M_PC)).float()   # 版70・71\n",
     "            ppn = self._ppn if self.bu_on else (torch.rand(N_PPN, device=self.dev, generator=self.cl_gen) < P_PPN).float()   # 版75\n"
     "            cl = ((self._bo[\"PPN→CL\"] >= M_PC_SYN) if self.bu_on else (ppn[self.cl_ppn].sum(1) >= M_PC)).float()   # 版70・71・75\n"),
    # 6 学び（17本を1回で）
    ("            for k_bu, s in self.bu.items():\n"
     "                s.learn(self.cl_spk if k_bu == \"PPN→CL\" else anyr[s.ch0:s.ch0 + s.n_tgt], d_bu)\n",
     "            self.but.learn(anyr, self.cl_spk if \"PPN→CL\" in self.bu else None, d_bu)   # 版75：表で1回\n"),
    # 7 入口の関数：表の足し込みの結果を返す
    ("        if self.bu_on:\n"
     "            return self.bu[key].drive(v)\n",
     "        if self.bu_on:\n"
     "            return self._bo[key]                                   # 版75：表で足した結果（入力 v は表の側で同じものを使った）\n"),
    ("        s = self.bu[\"運動前野→運動\"]\n"
     "        p = torch.where(cmd > 0, torch.full_like(cmd, R_PM_ON), torch.full_like(cmd, R_PM_OFF))[s.tgt]\n"
     "        return s.drive((torch.rand(len(p), device=self.dev, generator=self.gen) < p).float())\n",
     "        return self._bo[\"運動前野→運動\"]                           # 版75：表で足した結果\n\n"
     "    def _pm_in(self, cmd):\n"
     "        \"\"\"版75：運動前野の送り手が撃つか（版74 の _pm と同じくじ・同じ順）\"\"\"\n"
     "        s = self.bu[\"運動前野→運動\"]\n"
     "        p = torch.where(cmd > 0, torch.full_like(cmd, R_PM_ON), torch.full_like(cmd, R_PM_OFF))[s.tgt]\n"
     "        return (torch.rand(len(p), device=self.dev, generator=self.gen) < p).float()\n\n"
     "    def bu_pack(self):\n"
     "        \"\"\"版75：道ごとの _BUSyn（作った形・読み込んだ形）から、まとめた表を作り直す\"\"\"\n"
     "        self.but = _BUTable(self.bu, ([\"貫通路\"], [k for k in self.bu if k != \"貫通路\"]))\n"
     "        self._bo = {}\n\n"
     "    def bu_unpack(self):\n"
     "        \"\"\"版75：表の今の値を、道ごとの _BUSyn に写す（保存の前・重さを見る道具の前）\"\"\"\n"
     "        self.but.unpack(self.bu)\n"),
]
BB = [("VERSION = 74          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73)",
       "VERSION = 75          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74)"),
      ("#      【み】10-13「起きてる時にどうか調べて」「一旦暴走してるとこに入れよう」\n",
       "#      【み】10-13「起きてる時にどうか調べて」「一旦暴走してるとこに入れよう」\n"
       "#  75  下からのシナプス17本を1つの表で回す（速さだけ。線・重さ・式は同じ。放出のくじの並びだけ変わる）\n"
       "#      【測】版74 1歩 38.3ms ／ シナプスの分 8.6ms（17回/歩 CPU が待つ）【み】10-14「それならいいよ」\n"),
      ("    if getattr(b, \"bu_on\", False):     # 版69：下からのシナプス\n",
       "    if getattr(b, \"bu_on\", False):     # 版69：下からのシナプス\n"
       "        b.bu_unpack()                   # 版75：表から道へ写してから保存\n"),
      ("                for kk, vv in v[\"stp\"].items():\n"
       "                    setattr(s.stp, kk, vv)\n",
       "                for kk, vv in v[\"stp\"].items():\n"
       "                    setattr(s.stp, kk, vv)\n"
       "        b.bu_pack()                     # 版75：読み込んだ道から表を作り直す\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py")


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
    if babble.VERSION != 74:
        sys.exit(f"⚠️ babble.py が版74 ではない（版 {babble.VERSION}）。先に apply_v74.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版75 を当てた（torch_brain・babble）。下からのシナプス17本を1つの表で回す（足し込み 2回・学び 1回/歩）")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "75" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['75'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版75 でも回るようにした")
