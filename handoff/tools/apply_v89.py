"""版89：恒常性の観測で誤差と表現の行が入れ違っていたのを直す（値が変わる）
（~/anaplex で。版88 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v89.py
【み】10-15「バグは直して」
直すもの（torch_group2）
  _any ＝ [表現, 誤差, 抑制]（_out_core の並び）を、fired[誤差], fired[表現], fired[抑制] の順に入れていた（版59 から）
    ⇒ 誤差の体の発火率に表現の発火、表現の体の発火率に誤差の発火が入っていた
    ⇒ 恒常性（眠りで下げる・起きて上げる）と刈り込みの静か／撃ちすぎの判定が、入れ違いの率で動いていた
  直したあと：fired[誤差] ＝ 誤差の発火、fired[表現] ＝ 表現の発火、fired[抑制] ＝ 抑制の発火（_rate_core と、版78 の形の _rate_obs の両方）
⚠️ 網の作りは同じ（state.pt は読める）。ただ、読んだ state の rate_ema・rate_long・scale・scale_up は入れ違いの率で育ったもの
"""
import ast, pathlib, re, sys

G2 = [
    ("    \"\"\"版88：恒常性の観測（式・入れる行は版87 の _rate_obs と同じ。⚠️ 誤差と表現の行が入れ違い。直すかは【み】に聞く）\"\"\"\n"
     "    fired = torch.zeros(n_recv, device=a0.device)\n"
     "    fired[err_i] = a0\n"
     "    fired[rep_i] = a1\n",
     "    \"\"\"版88：恒常性の観測。a0・a1・a2 ＝ _any ＝ [表現, 誤差, 抑制]\n"
     "    版89：誤差の行に誤差、表現の行に表現を入れる（⟲ 版59〜88 は入れ違い【み】10-15「バグは直して」）\"\"\"\n"
     "    fired = torch.zeros(n_recv, device=a0.device)\n"
     "    fired[err_i] = a1\n"
     "    fired[rep_i] = a0\n"),
    ("        fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any\n",
     "        # 版89：_any ＝ [表現, 誤差, 抑制]。誤差の行に誤差、表現の行に表現（⟲ 版59〜88 は入れ違い）\n"
     "        fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any[1], self._any[0], self._any[2]\n"),
]
BB = [("VERSION = 88          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87)",
       "VERSION = 89          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88)"),
      ("#  88  恒常性・恒常性の観測・下からの表の学び・CL の学びを compile で（まとめて4つ）【測】版87 1歩 10.0ms\n",
       "#  88  恒常性・恒常性の観測・下からの表の学び・CL の学びを compile で（まとめて4つ）【測】版87 1歩 10.0ms\n"
       "#  89  恒常性の観測で誤差と表現の行が入れ違っていた（版59 から）のを直す【み】10-15「バグは直して」⚠️ 値が変わる\n")]
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
    if babble.VERSION != 88:
        sys.exit(f"⚠️ babble.py が版88 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版89 を当てた（torch_group2・babble）。恒常性の観測の誤差と表現の入れ違いを直した（値が変わる）", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "89" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['89'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版89 でも回るようにした", flush=True)
