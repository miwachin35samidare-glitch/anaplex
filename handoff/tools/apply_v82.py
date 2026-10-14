"""版82：1周目の足し込みの下書きを 64枚に分けて、最後に足し合わせる（速さだけ。届く周・相手・量は同じ）
（~/anaplex で。版81 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v82.py
【み】10-14「それって信号のくる順番大丈夫なの？」⇒ 1周の中で届いた信号は受け手ごとに1つの合計にまとめてから読む（周の中に順番は無い）。
  写しはその合計を作る途中の下書きで、受け手が読む前に足し合わせる ⇒「その仕組みならいいよ」
【測】probe_scatter_real（網の本当の番号）：1周目の学びの辺 292万本 → 受け手 9,736（1つあたり平均 166本・最大 1469）⇒ 同じ所への足し込みが順番待ち
【測】probe_scatter_real2：写し1つ 2.37ms ／ 4枚 1.29 ／ 16枚 1.15 ／ 64枚・束 256 1.10 ／ 256枚 1.13ms。値 差の最大（割合）1.7e-7〜2.0e-7（小数を足す順の丸め）
直すもの：_scatter3_k は束ごとに 64枚の下書きのどれかへ足し、_scatter3 で下書きを足し合わせる（辺の数は整数なので同じ値）
⚠️ 小数を足す順による最後の桁の丸めは変わる（GPU はもともと毎回足す順が違う。check_gpu で確かめる）。CPU は版80 と同じ
"""
import ast, pathlib, re, sys

G2 = [
    ("    def _scatter3_k(r_p, ie_p, amp_p, mm_p, both_p, mod_p, cnt_p, n, n_recv, DO_CNT: tl.constexpr, BLOCK: tl.constexpr):\n"
     "        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
     "        m = o < n\n"
     "        r = tl.load(r_p + o, mask=m, other=0)\n"
     "        ie = tl.load(ie_p + o, mask=m, other=0).to(tl.int32)\n"
     "        a = tl.load(amp_p + o, mask=m, other=0.0)\n"
     "        mm = tl.load(mm_p + o, mask=m, other=0).to(tl.int32)\n"
     "        tl.atomic_add(both_p + r + n_recv * ie, a, mask=m, sem=\"relaxed\")\n"
     "        tl.atomic_add(mod_p + r, a, mask=m & (mm != 0), sem=\"relaxed\")\n"
     "        if DO_CNT:\n"
     "            tl.atomic_add(cnt_p + r, tl.full([BLOCK], 1, tl.int32), mask=m, sem=\"relaxed\")\n",
     "    def _scatter3_k(r_p, ie_p, amp_p, mm_p, both_p, mod_p, cnt_p, n, n_recv, DO_CNT: tl.constexpr, BLOCK: tl.constexpr,\n"
     "                    K: tl.constexpr):\n"
     "        # 版82：束ごとに K 枚の下書きのどれか（束の番号 ÷ K の余り）へ足す。1か所に同時に来る数が 1/K に\n"
     "        pid = tl.program_id(0)\n"
     "        o = pid * BLOCK + tl.arange(0, BLOCK)\n"
     "        m = o < n\n"
     "        rep = pid % K\n"
     "        r = tl.load(r_p + o, mask=m, other=0)\n"
     "        ie = tl.load(ie_p + o, mask=m, other=0).to(tl.int32)\n"
     "        a = tl.load(amp_p + o, mask=m, other=0.0)\n"
     "        mm = tl.load(mm_p + o, mask=m, other=0).to(tl.int32)\n"
     "        tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie, a, mask=m, sem=\"relaxed\")\n"
     "        tl.atomic_add(mod_p + rep * n_recv + r, a, mask=m & (mm != 0), sem=\"relaxed\")\n"
     "        if DO_CNT:\n"
     "            tl.atomic_add(cnt_p + rep * n_recv + r, tl.full([BLOCK], 1, tl.int32), mask=m, sem=\"relaxed\")\n"),
    ("    n = r.shape[0]\n"
     "    both = torch.zeros(n_recv * 2, device=amp.device, dtype=amp.dtype)\n"
     "    mod = torch.zeros(n_recv, device=amp.device, dtype=amp.dtype)\n"
     "    cnt = torch.zeros(n_recv if do_cnt else 1, device=amp.device, dtype=torch.int32)\n"
     "    if n:\n"
     "        _scatter3_k[(triton.cdiv(n, 1024),)](r, ie, amp, mm, both, mod, cnt, n, n_recv, DO_CNT=do_cnt, BLOCK=1024)\n"
     "    return both, mod, (cnt.to(amp.dtype) if do_cnt else None)\n",
     "    # 版82：下書きを K 枚に分けて足し、受け手が読む前に足し合わせる（届く周・相手・量は同じ。小数の丸めの順だけ変わる）\n"
     "    #   【測】probe_scatter_real2：写し1つ 2.37ms ／ 64枚・束 256 1.10ms\n"
     "    n = r.shape[0]\n"
     "    K, BL = _SC_K, _SC_BLOCK\n"
     "    both = torch.zeros(K, n_recv * 2, device=amp.device, dtype=amp.dtype)\n"
     "    mod = torch.zeros(K, n_recv, device=amp.device, dtype=amp.dtype)\n"
     "    cnt = torch.zeros(K, n_recv if do_cnt else 1, device=amp.device, dtype=torch.int32)\n"
     "    if n:\n"
     "        _scatter3_k[(triton.cdiv(n, BL),)](r, ie, amp, mm, both, mod, cnt, n, n_recv, DO_CNT=do_cnt, BLOCK=BL, K=K)\n"
     "    return both.sum(0), mod.sum(0), (cnt.sum(0).to(amp.dtype) if do_cnt else None)\n"),
    ("SCATTER = _os.environ.get(\"ANAPLEX_SCATTER\", \"1\") != \"0\"\n",
     "SCATTER = _os.environ.get(\"ANAPLEX_SCATTER\", \"1\") != \"0\"\n"
     "_SC_K, _SC_BLOCK = int(_os.environ.get(\"ANAPLEX_SCATTER_K\", \"64\")), 256    # 版82：下書きの枚数・束の大きさ（K=1 で版81 と同じ足し方）\n"),
]
BB = [("VERSION = 81          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80)",
       "VERSION = 82          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81)"),
      ("#      【測】版80 1歩 15.8ms ／ probe_atomic 小数の足し込み 0.67ms・整数 0.158ms（230万本）\n",
       "#      【測】版80 1歩 15.8ms ／ probe_atomic 小数の足し込み 0.67ms・整数 0.158ms（230万本）\n"
       "#  82  1周目の足し込みの下書きを 64枚に分けて最後に足し合わせる（292万本が受け手 9,736 に集まり順番待ちだった）\n"
       "#      【み】10-14「その仕組みならいいよ」【測】probe_scatter_real2 2.37 → 1.10ms\n")]
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
    if babble.VERSION != 81:
        sys.exit(f"⚠️ babble.py が版81 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版82 を当てた（torch_group2・babble）。1周目の足し込みの下書きを 64枚に", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "82" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['82'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版82 でも回るようにした", flush=True)
