"""版70：残りの道も学ぶシナプスに（海馬の中3本・下丘 → 視床・運動前野 → 運動・PPN → CL）。蹴りは保留のまま
（~/anaplex で。版69 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v70.py
【み】10-12「なんで全部シナプスに置き換えないの？」「1回ずつやると測る時間とトークンばっかかかって前に進まない」
       「全部適用してずれた所を直していくほうがよくないか」
組み方（版69 と同じ形：今の繋ぎ先・本数のまま、1本 ＝ 人の1本・対数正規・人の ex→ex の短期の増減・三つの因子）
  下丘 → 視床        1対1（400本）
  貫通路（嗅内浅 II → 歯状回） 1体 2本。歯状回は今までどおり「和の上位 2.5% だけ撃つ」（撃ち方は歯状回の側の規則のまま）
  苔状線維 → CA3     1体 4本（前は「1本でも撃てば起爆」⇒ 重さ × 放出の和 × gain（CA3 の 2 × 閾値）に）
  シャッファー → CA1  1体 4本（前は「1本でも撃てば 1」⇒ 重さ × 放出の和に）
  運動前野 → 運動     1体 K_PM 27本。送り手を体として置く（前は撃った数を二項分布で引くだけ）。率は今までどおり 指令あり 35Hz・なし 3.3Hz【ヒト】
  PPN → CL          1体 8本（前は重さ 1 の数）。CL は和が M_PC（4）以上で撃つ（今の撃ち方のまま）
  受け手の「撃ったか」：その領野の表現の体（PPN → CL は CL の体）【C】
⚠️ 人の値の無い所
  海馬の3本の短期の増減・学び：人の値を見つけていない ⇒ 人の皮質の ex→ex の表と皮質の学びを当てた【C】（ラットの値は使わない）
  下丘 → 視床：視床の閾値 0.3 は人の値でない（前の値のまま）⇒ 重さが 0.3 を割る1本（約4割）は、その視床の体を撃たせられない【計算】
  運動前野 → 運動：続けて撃つ送り手は短期の増減で弱まる ⇒ 指令の体の率が 35Hz から下がりうる【見】
ANAPLEX_BU=0 で版68 と同じ振る舞い（版69 の道も含めて全部前の形）
保存：Group2 の配線は同じ ⇒ 版66〜69 の state.pt は読める（増えた道の重さは初めから）
"""
import ast, pathlib, re, sys

BRAIN = [
    # 1 組み立て（版69 の道の組み立ての後に足す）
    ("                self.bu[key] = _BUSyn(p[0], p[1], n_tgt, self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n",
     "                self.bu[key] = _BUSyn(p[0], p[1], n_tgt, self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n"
     "            # 版70：残りの道（今の繋ぎ先・本数のまま）\n"
     "            ar = lambda n: torch.arange(n, device=dev)\n"
     "            nt, ndg, nc3, nc1, nm_ = self.n[\"視床\"], self.n[\"歯状回\"], self.n[\"CA3\"], self.n[\"CA1\"], self.n[\"運動\"]\n"
     "            more = {\"下丘→視床\": (ar(nt), ar(nt), nt, \"視床\"),\n"
     "                    \"貫通路\": (self.perforant.reshape(-1), ar(ndg).repeat_interleave(self.perforant.shape[1]), ndg, \"歯状回\"),\n"
     "                    \"苔状線維\": (self.mossy.reshape(-1), ar(nc3).repeat_interleave(self.mossy.shape[1]), nc3, \"CA3\"),\n"
     "                    \"シャッファー\": (self.schaffer.reshape(-1), ar(nc1).repeat_interleave(self.schaffer.shape[1]), nc1, \"CA1\"),\n"
     "                    \"運動前野→運動\": (ar(nm_ * K_PM), ar(nm_).repeat_interleave(K_PM), nm_, \"運動\")}\n"
     "            for key, (s_, t_, n_, nm) in more.items():\n"
     "                self.bu[key] = _BUSyn(s_, t_, n_, self.off_ch[NAMES.index(nm)], stp_path, dev, gb)\n"
     "            if self.cl_on:                                           # PPN → CL（受け手は CL の体。領野の外）\n"
     "                self.bu[\"PPN→CL\"] = _BUSyn(self.cl_ppn.reshape(-1), ar(N_CL).repeat_interleave(K_PC), N_CL, -1, stp_path, dev, gb)\n"),
    # 2 学び：PPN → CL の受け手は CL の体
    ("            for s in self.bu.values():\n"
     "                s.learn(anyr[s.ch0:s.ch0 + s.n_tgt], d_bu)\n",
     "            for k_bu, s in self.bu.items():\n"
     "                s.learn(self.cl_spk if k_bu == \"PPN→CL\" else anyr[s.ch0:s.ch0 + s.n_tgt], d_bu)\n"),
    # 3 毎歩
    ("        drive = ec2[self.perforant].sum(1)\n",
     "        drive = self.bu[\"貫通路\"].drive(ec2) if self.bu_on else ec2[self.perforant].sum(1)   # 版70\n"),
    ("            ic,                                                # 下丘 → 視床（1本 → 1体、帯をまたがない）",
     "            self.bu[\"下丘→視床\"].drive(ic) if self.bu_on else ic,   # 下丘 → 視床（1本 → 1体、帯をまたがない。版70 シナプス）"),
    ("            self._premotor(cmd),", "            self._pm(cmd),"),
    ("            (dg[self.mossy].max(1).values > 0).float(),",
     "            self.bu[\"苔状線維\"].drive(dg) if self.bu_on else (dg[self.mossy].max(1).values > 0).float(),"),
    ("            (self.ca3_prev[self.schaffer].max(1).values > 0).float(),",
     "            self.bu[\"シャッファー\"].drive(self.ca3_prev) if self.bu_on else (self.ca3_prev[self.schaffer].max(1).values > 0).float(),"),
    ("            cl = (ppn[self.cl_ppn].sum(1) >= M_PC).float()\n",
     "            cl = ((self.bu[\"PPN→CL\"].drive(ppn) if self.bu_on else ppn[self.cl_ppn].sum(1)) >= M_PC).float()   # 版70\n"),
    # 4 運動前野の送り手を体として
    ("    def _premotor(self, cmd):\n",
     "    def _pm(self, cmd):\n"
     "        \"\"\"版70：運動前野の送り手（1体に K_PM 体、体ごとに撃つ）→ 運動（シナプス）。ANAPLEX_BU=0 なら前の二項分布\"\"\"\n"
     "        if not self.bu_on:\n"
     "            return self._premotor(cmd)\n"
     "        s = self.bu[\"運動前野→運動\"]\n"
     "        p = torch.where(cmd > 0, torch.full_like(cmd, R_PM_ON), torch.full_like(cmd, R_PM_OFF))[s.tgt]\n"
     "        return s.drive((torch.rand(len(p), device=self.dev, generator=self.gen) < p).float())\n\n"
     "    def _premotor(self, cmd):\n"),
]
BB = [("VERSION = 69          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68)",
       "VERSION = 70          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69)"),
      ("#      gain は入口に残す（ハードの制約のノブ）【み】10-12「全体をシナプスにしよう」「1本は1本」「アでいいよ」\n",
       "#      gain は入口に残す（ハードの制約のノブ）【み】10-12「全体をシナプスにしよう」「1本は1本」「アでいいよ」\n"
       "#  70  残りの道（海馬の中3本・下丘 → 視床・運動前野 → 運動・PPN → CL）も学ぶシナプスに。蹴りは保留。\n"
       "#      海馬の短期の増減・学びは人の値なし ⇒ 皮質の値を当てた【C】【み】10-12「全部適用してずれた所を直していくほうがよくないか」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py")


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
    if babble.VERSION != 69:
        sys.exit(f"⚠️ babble.py が版69 ではない（版 {babble.VERSION}）。先に apply_v69.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版70 を当てた（torch_brain・babble）。海馬の中3本・下丘 → 視床・運動前野 → 運動・PPN → CL もシナプスに（蹴りは保留）")
    print("   Group2 の配線は同じ（版66〜69 の state.pt は読める。増えた道の重さは初めから）。ANAPLEX_BU=0 で版68 と同じ振る舞い")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "70" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['70'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版70 でも回るようにした")
