"""版85：下からのシナプスの表の drive を torch.compile で（速さだけ。式・くじの順は同じ）
（~/anaplex で。版84 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v85.py
【み】10-15「詰め切ってから」
【測】arkhe 版84：1歩 11.2ms ／ 下からの表 1.8ms（drive 1.5・2回/歩）／ GPU の index_add（下からの表・CL）200µs × 3回/歩
  コンテナで数えた命令：表の drive 約80/回（2回/歩）
直すもの（torch_brain の _BUTable.drive）
  放出のくじ（binomial）の前と後を torch.compile で1つずつに（版78 の網の放出と同じやり方。くじは外で、同じ所で引く）
  足し込み（index_add）も後ろの方に入れる（compile の足し込みは aten の index_add より速い【測】probe_atomic 0.71 vs 0.68ms・小さい表では呼び出しが減る）
【確】コンテナ（CPU）版78 と 200歩 一致（下からのシナプスの重さ・状態も）
"""
import ast, pathlib, re, sys

TB = [
    ("class _BUTable:\n",
     "def _bu_pre(x, gsrc, te, row, last, deplete, state, fac, dec_d, dec_f, tbl, Nt):\n"
     "    \"\"\"版85：_BUTable.drive の放出のくじの前（式は版84 と同じ）\"\"\"\n"
     "    fired = x[gsrc] > 0\n"
     "    r = row.long()\n"
     "    n = (te - last).float()\n"
     "    dep = deplete\n"
     "    st = state * torch.where(dep, torch.ones_like(n), dec_d[r] ** n)\n"
     "    fc = fac * dec_f[r] ** n\n"
     "    Pr0 = tbl[r, 0]\n"
     "    P = (Pr0 + (1 - Pr0) * fc).clamp(0, 1)\n"
     "    N = Nt[r]\n"
     "    avail = torch.where(dep, st, 1 - st)\n"
     "    ff = fired.float()\n"
     "    return fired, r, st, fc, Pr0, P, N, ff, (N * avail).round().clamp(min=0) * ff\n"
     "\n"
     "\n"
     "def _bu_post(cnt, fired, r, dep, st, fc, Pr0, P, N, ff, tbl, state, fac, last, te, w, tgt, n_out: int):\n"
     "    \"\"\"版85：_BUTable.drive の放出のくじの後（式は版84 と同じ）。返り値：新しい状態・増減・最後の歩・足し込み\"\"\"\n"
     "    rel = cnt / (N * Pr0).clamp(min=1e-6)\n"
     "    k = torch.where(dep, P, tbl[r, 1])\n"
     "    st2 = torch.where(dep, st * (1 - k), st + (1 - st) * k)\n"
     "    fc2 = fc + (1 - fc) * tbl[r, 3]\n"
     "    out = torch.zeros(n_out, device=w.device, dtype=w.dtype).index_add(0, tgt, w * rel * ff)\n"
     "    return (torch.where(fired, st2, state), torch.where(fired, fc2, fac),\n"
     "            torch.where(fired, te.to(last.dtype), last), out)\n"
     "\n"
     "\n"
     "class _BUTable:\n"),
    ("        x = torch.cat(vs)\n"
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
     "        out = torch.zeros(tb - ta, device=x.device).index_add_(0, self.tgt[a:b] - ta, self.w[a:b] * rel * ff)\n",
     "        x = torch.cat(vs)\n"
     "        te = self.t[self.pid[a:b]]\n"
     "        # 短期の増減（torch_area.STP.release_at と同じ式。撃った線だけ状態を進める）\n"
     "        # 版85：くじの前と後を torch.compile で1つずつに（くじ binomial は今と同じ所で。撃っていない線は数 0）\n"
     "        from torch_group2 import _run\n"
     "        if gi not in getattr(self, \"_tgt_rel\", {}):\n"
     "            self._tgt_rel = getattr(self, \"_tgt_rel\", {})\n"
     "            self._tgt_rel[gi] = self.tgt[a:b] - ta\n"
     "        fired, r, st, fc, Pr0, P, N, ff, cn = _run(_bu_pre, x, self._gsrc[gi], te, self.row[a:b], self.last[a:b],\n"
     "                                                   self.deplete[a:b], self.state[a:b], self.fac[a:b], self.dec_d,\n"
     "                                                   self.dec_f, self.tbl, self.N)\n"
     "        self.pre_now[a:b] = fired\n"
     "        cnt = torch.binomial(cn, P, generator=self.gen)\n"
     "        s2, f2, l2, out = _run(_bu_post, cnt, fired, r, self.deplete[a:b], st, fc, Pr0, P, N, ff, self.tbl,\n"
     "                               self.state[a:b], self.fac[a:b], self.last[a:b], te, self.w[a:b], self._tgt_rel[gi], tb - ta)\n"
     "        self.state[a:b] = s2\n"
     "        self.fac[a:b] = f2\n"
     "        self.last[a:b] = l2\n"),
]
BB = [("VERSION = 84          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83)",
       "VERSION = 85          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84)"),
      ("#      【測】版83 1歩 11.6ms ／ GPU 8.9ms\n",
       "#      【測】版83 1歩 11.6ms ／ GPU 8.9ms\n"
       "#  85  下からのシナプスの表の drive を torch.compile で（くじの前と後。くじは同じ所）【測】版84 1歩 11.2ms\n")]
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
    if babble.VERSION != 84:
        sys.exit(f"⚠️ babble.py が版84 ではない（版 {babble.VERSION}）")
    out = [edit("torch_brain.py", TB), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版85 を当てた（torch_brain・babble）。下からのシナプスの表の drive を torch.compile で", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "85" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['85'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版85 でも回るようにした", flush=True)
