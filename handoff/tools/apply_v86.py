"""版86：焼いた網の見込みを1回に・グラフに渡す値のうち置き場所が変わらないものは写さずそのまま読む（速さだけ。値は同じ）
（~/anaplex で。版85 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v86.py
【み】10-15「詰めれるだけ詰めてみよう」
【測】arkhe 版85：1歩 10.9ms（実時間の 1.02倍）／ GPU の核 8.0ms
  焼いた網 1.35ms（8回/歩：型ごとの塊 8つ。1回に掛け算 2つ）／ aten::mm 17回/歩・CPU 58µs/回 ／ 写し copy_ 75回/歩
直すもの（torch_group2）
  焼いた網：8つの塊の見込みを torch.compile の1回に（式は Prior.logit と同じ。型ごとの重さは1回だけ積んでおく。焼いて凍結した値なので変わらない）
  2周目の頭：グラフに渡す値のうち、毎歩同じ置き場所のもの（入口の倍率・閾値・CL の駆動 など）は写さず、その置き場所をグラフが直に読む
    置き場所が一度でも変わった値は、今まで通り写す形に切り替えて撮り直す（中身は毎歩同じ）
【確】コンテナ（CPU）版78 と 200歩
"""
import ast, pathlib, re, sys

G2 = [
    ("def _flush_amp_core(",
     "def _prior_core(tb, W1, b1, W2, b2, segs):\n"
     "    \"\"\"版86：型ごとの塊の見込みを1回に（式は Prior.logit と同じ。W1・b1・W2・b2 は塊ごとに積んだもの）\"\"\"\n"
     "    out = []\n"
     "    for i, (lo, hi) in enumerate(segs):\n"
     "        out.append((torch.tanh(tb[lo:hi] @ W1[i] + b1[i]) @ W2[i] + b2[i]).squeeze(-1))\n"
     "    return torch.cat(out)\n"
     "\n"
     "\n"
     "def _flush_amp_core("),
    ("            self._prior_l = (self.prior.logit(self.tb) if len(self._prior_segs) == 1 else          # 版55\n"
     "                             torch.cat([pr.logit(self.tb[lo:hi]) for lo, hi, pr in self._prior_segs]))\n",
     "            if len(self._prior_segs) == 1:      # 版55\n"
     "                self._prior_l = self.prior.logit(self.tb)\n"
     "            elif type(self.areas[0].prior).logit is not _PRIOR_LOGIT0:   # 測る道具が差し替えているときは今まで通り\n"
     "                self._prior_l = torch.cat([pr.logit(self.tb[lo:hi]) for lo, hi, pr in self._prior_segs])\n"
     "            else:                               # 版86：8つの塊を1回に\n"
     "                if getattr(self, \"_pri_stack\", None) is None:\n"
     "                    ps = [pr for _, _, pr in self._prior_segs]\n"
     "                    self._pri_stack = tuple(torch.stack([getattr(p, k) for p in ps]) for k in (\"W1\", \"b1\", \"W2\", \"b2\"))\n"
     "                    self._pri_segs = tuple((lo, hi) for lo, hi, _ in self._prior_segs)\n"
     "                self._prior_l = _run(_prior_core, self.tb, *self._pri_stack, self._pri_segs)\n"),
    ("        X, key = self._gX, []\n"
     "        for k, v in vin.items():          # 歩ごとに変わる値は決まった置き場所へ写す\n"
     "            if v is None:\n"
     "                key.append((k, False))\n"
     "                continue\n",
     "        X, key = self._gX, []\n"
     "        AL = self.__dict__.setdefault(\"_galias\", {})      # 版86：置き場所の変わらない値（グラフが直に読む）\n"
     "        CP = self.__dict__.setdefault(\"_gcopy\", set())    # 版86：置き場所が変わったことのある値（写す）\n"
     "        for k, v in vin.items():          # 歩ごとに変わる値は決まった置き場所へ写す\n"
     "            if v is None:\n"
     "                key.append((k, False))\n"
     "                continue\n"
     "            if k in AL:\n"
     "                if v is AL[k]:                # 同じ置き場所 ⇒ 写さない（グラフは再生のときの中身を読む）\n"
     "                    key.append((k, True))\n"
     "                    continue\n"
     "                del AL[k]                     # 置き場所が変わった ⇒ これからは写す\n"
     "                X.pop(k, None)\n"
     "                CP.add(k)\n"
     "                self._graphs.clear()\n"
     "            elif torch.is_tensor(v) and k not in CP and k not in X:\n"
     "                AL[k] = X[k] = v              # 初めて見た値は、まず直に読む形で\n"
     "                self._graphs.clear()\n"
     "                key.append((k, True))\n"
     "                continue\n"),
    ("# 版79：差し替えられていない元の関数（測る道具が差し替えているときは、版78 と同じ周ごとで回す）\n",
     "# 版86：焼いた網の元の関数（測る道具が差し替えているときは塊ごとに呼ぶ）\n"
     "from torch_area import Prior as _Prior\n"
     "_PRIOR_LOGIT0 = _Prior.logit\n"
     "# 版79：差し替えられていない元の関数（測る道具が差し替えているときは、版78 と同じ周ごとで回す）\n"),
]
BB = [("VERSION = 85          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84)",
       "VERSION = 86          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85)"),
      ("#  85  下からのシナプスの表の drive を torch.compile で（くじの前と後。くじは同じ所）【測】版84 1歩 11.2ms\n",
       "#  85  下からのシナプスの表の drive を torch.compile で（くじの前と後。くじは同じ所）【測】版84 1歩 11.2ms\n"
       "#  86  焼いた網の見込みを1回に・グラフに渡す値のうち置き場所の変わらないものは写さない【測】版85 1歩 10.9ms\n")]
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
    if babble.VERSION != 85:
        sys.exit(f"⚠️ babble.py が版85 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版86 を当てた（torch_group2・babble）。焼いた網の見込みを1回に・置き場所の変わらない値は写さない", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "86" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['86'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版86 でも回るようにした", flush=True)
