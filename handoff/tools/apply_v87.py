"""版87：耳の入力を GPU へ送るとき・誤差の和を数えるときに CPU が GPU を待たない（速さだけ。値は同じ）
（~/anaplex で。版86 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v87.py
【測】arkhe 版86：1歩 10.8ms ／ GPU 8.0ms ／ hipMemcpyWithStream 3回/歩・1回 508µs（CPU が GPU の追いつくのを待っている）
  入力を torch.tensor(…, device="cuda") で送ると、GPU に積んである仕事が終わるまで CPU が待つ（ピン留めしていない置き場所からの写し）
  babble の回す所は毎歩 float(e.sum()) でも待っていた
直すもの
  torch_brain.to_dev：入力をピン留めした置き場所（8つを順に使い回す）に写してから、待たずに GPU へ送る（値は同じ）
  babble の run・own_voice：入力は to_dev で ／ 誤差の和は GPU の上で float64 に足し、最後に1回だけ読む（CPU の float で足すのと同じ値）
  measure_speed79：入力を to_dev で（道具の待ち 2回/歩 を外す）
【確】コンテナ（CPU）：to_dev は CPU では今まで通り torch.tensor。値の比べは版86 と同じ
"""
import ast, pathlib, re, sys

TB = [
    ("class Brain:\n",
     "_UP = {}\n"
     "\n"
     "\n"
     "def to_dev(a, dev):\n"
     "    \"\"\"版87：入力（numpy）を GPU へ。ピン留めした置き場所（8つを順に）から待たずに送る（値・型は torch.tensor と同じ）\n"
     "    【測】版86：torch.tensor(…, device=\"cuda\") は GPU に積んである仕事が終わるまで CPU が待っていた（1回 約0.5ms）\"\"\"\n"
     "    if not str(dev).startswith(\"cuda\"):\n"
     "        return torch.tensor(a, device=dev)\n"
     "    a = np.ascontiguousarray(a)\n"
     "    key = (a.shape, a.dtype.str)\n"
     "    r = _UP.get(key)\n"
     "    if r is None:\n"
     "        dt = torch.from_numpy(a).dtype\n"
     "        r = _UP[key] = {\"i\": 0, \"buf\": [torch.empty(a.shape, dtype=dt, pin_memory=True) for _ in range(8)],\n"
     "                        \"ev\": [None] * 8}\n"
     "    i = r[\"i\"]; r[\"i\"] = (i + 1) % 8\n"
     "    if r[\"ev\"][i] is not None:\n"
     "        r[\"ev\"][i].synchronize()          # 8つ前に送った分は、とっくに送り終わっている\n"
     "    b = r[\"buf\"][i]\n"
     "    b.numpy()[...] = a\n"
     "    out = b.to(dev, non_blocking=True)\n"
     "    ev = torch.cuda.Event()\n"
     "    ev.record()\n"
     "    r[\"ev\"][i] = ev\n"
     "    return out\n"
     "\n"
     "\n"
     "class Brain:\n"),
]
BB = [("from torch_brain import Brain, NAMES\n",
       "from torch_brain import Brain, NAMES, to_dev\n"),
      ("        c = torch.tensor(cmd[t], device=DEV) if own else zero\n"
       "        _, e, _, _ = b.step(torch.tensor(ear[t], device=DEV), c, learn=learn)\n"
       "        n += float(e.sum())\n"
       "    return n\n",
       "        c = to_dev(cmd[t], DEV) if own else zero            # 版87：待たずに送る\n"
       "        _, e, _, _ = b.step(to_dev(ear[t], DEV), c, learn=learn)\n"
       "        n = n + e.sum().double()                              # 版87：GPU の上で float64 に足す（CPU の float で足すのと同じ値）\n"
       "    return float(n)\n"),
      ("        _, e, _, _ = b.step(quiet, one if t >= w0 else zero, learn=learn)\n"
       "        n += float(e.sum())\n",
       "        _, e, _, _ = b.step(quiet, one if t >= w0 else zero, learn=learn)\n"
       "        n = n + e.sum().double()                              # 版87\n"),
      ("        _, e, _, _ = b.step(torch.tensor(snd[t], device=DEV), one, learn=learn)\n"
       "        n += float(e.sum())\n",
       "        _, e, _, _ = b.step(to_dev(snd[t], DEV), one, learn=learn)     # 版87\n"
       "        n = n + e.sum().double()\n"),
      ("    return ear, cmd, n, [m for m, *_ in morae], [m for m, *_ in said]\n",
       "    return ear, cmd, float(n), [m for m, *_ in morae], [m for m, *_ in said]\n"),
      ("VERSION = 86          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85)",
       "VERSION = 87          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86)"),
      ("#  86  焼いた網の見込みを1回に・グラフに渡す値のうち置き場所の変わらないものは写さない【測】版85 1歩 10.9ms\n",
       "#  86  焼いた網の見込みを1回に・グラフに渡す値のうち置き場所の変わらないものは写さない【測】版85 1歩 10.9ms\n"
       "#  87  耳の入力はピン留めした置き場所から待たずに送る・誤差の和は GPU の上で足して最後に1回読む【測】版86 1歩 10.8ms\n")]
MS = [("            yield torch.tensor(ear[t], device=dev), torch.tensor(cmd[t], device=dev)\n",
       "            yield (TB.to_dev(ear[t], dev), TB.to_dev(cmd[t], dev)) if hasattr(TB, \"to_dev\") else \\\n"
       "                (torch.tensor(ear[t], device=dev), torch.tensor(cmd[t], device=dev))   # 版87：待たずに送る\n")]
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
    if babble.VERSION != 86:
        sys.exit(f"⚠️ babble.py が版86 ではない（版 {babble.VERSION}）")
    out = [edit("torch_brain.py", TB), edit("babble.py", BB)]
    if pathlib.Path("measure_speed79.py").exists():
        out.append(edit("measure_speed79.py", MS))
    for p, s in out:
        p.write_text(s)
    print("✅ 版87 を当てた（torch_brain・babble・measure_speed79）。入力を待たずに送る・誤差の和は最後に1回読む", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "87" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['87'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版87 でも回るようにした", flush=True)
