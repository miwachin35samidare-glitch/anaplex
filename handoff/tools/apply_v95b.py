"""版95b：足し込みの渋滞を直す（意味は版95 と同じ。足す順だけ変わる）
（~/anaplex で。版95 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v95b.py
【測】arkhe 版95 measure_speed75（10-22）：1歩 256.5ms（実時間の 24倍）
  下からのシナプスの表の1回目 214.9ms：triton の足し込み（index_add）1回 104ms。版95 の 領野 → 線条体 約870万本が、
    受け手1体に 2,700本ずつ同じ所へ足し込む ⇒ GPU の足し込み（原子的な足し算）が渋滞
  網の中：核どうしの線（NucNet）の index_add 0.48ms × 約40回/歩
直すもの
  1 下からのシナプスの表：線はもともと受け手の順に並んでいる（コンテナで確かめた）⇒ 線ごとの値を出して、受け手ごとの合計を
     累積和の差で取る（float64 で足してから float32 に）。並んでいない組は今まで通り index_add
  2 NucNet：組んだときに線を受け手の順に並べ直しておき、同じく累積和の差で
⚠️ 足す順が変わるので、値は丸めの桁でだけ変わる（今の index_add も順は決まっていない）
"""
import ast, pathlib, sys

TB = [
    ('def _bu_post(cnt, fired, r, dep, st, fc, Pr0, P, N, ff, tbl, state, fac, last, te, w, tgt, n_out: int):\n',
     'def _segsum(v, ends):\n'
     '    """版95b：受け手の順に並んだ線の値 v を、受け手ごとに合計する（ends ＝ 受け手ごとの終わりの番号。累積和の差。float64 で足す）"""\n'
     '    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=v.device), v.double().cumsum(0)])\n'
     '    starts = torch.cat([torch.zeros(1, dtype=ends.dtype, device=ends.device), ends[:-1]])\n'
     '    return (cs[ends] - cs[starts]).to(v.dtype)\n'
     '\n'
     '\n'
     'def _bu_post_v(cnt, fired, r, dep, st, fc, Pr0, P, N, ff, tbl, state, fac, last, te, w):\n'
     '    """版95b：_bu_post と同じ式。足し込みの代わりに線ごとの値を返す（受け手ごとの合計は _segsum で）"""\n'
     '    rel = cnt / (N * Pr0).clamp(min=1e-6)\n'
     '    k = torch.where(dep, P, tbl[r, 1])\n'
     '    st2 = torch.where(dep, st * (1 - k), st + (1 - st) * k)\n'
     '    fc2 = fc + (1 - fc) * tbl[r, 3]\n'
     '    return (torch.where(fired, st2, state), torch.where(fired, fc2, fac),\n'
     '            torch.where(fired, te.to(last.dtype), last), w * rel * ff)\n'
     '\n'
     '\n'
     'def _bu_post(cnt, fired, r, dep, st, fc, Pr0, P, N, ff, tbl, state, fac, last, te, w, tgt, n_out: int):\n'),
    ('        s2, f2, l2, out = _run(_bu_post, cnt, fired, r, self.deplete[a:b], st, fc, Pr0, P, N, ff, self.tbl,\n'
     '                               self.state[a:b], self.fac[a:b], self.last[a:b], te, self.w[a:b], self._tgt_rel[gi], tb - ta)\n',
     '        ends = self._seg_ends(gi, tb - ta)\n'
     '        if ends is not None:                                # 版95b：受け手の順に並んだ線 ⇒ 累積和の差で（渋滞しない）\n'
     '            s2, f2, l2, vals = _run(_bu_post_v, cnt, fired, r, self.deplete[a:b], st, fc, Pr0, P, N, ff, self.tbl,\n'
     '                                    self.state[a:b], self.fac[a:b], self.last[a:b], te, self.w[a:b])\n'
     '            out = _segsum(vals, ends)\n'
     '        else:\n'
     '            s2, f2, l2, out = _run(_bu_post, cnt, fired, r, self.deplete[a:b], st, fc, Pr0, P, N, ff, self.tbl,\n'
     '                                   self.state[a:b], self.fac[a:b], self.last[a:b], te, self.w[a:b], self._tgt_rel[gi], tb - ta)\n'),
    ('    @torch.no_grad()\n'
     '    def learn(self, anyr, cl_spk, d):\n',
     '    def _seg_ends(self, gi, n_out):\n'
     '        """版95b：組 gi の線が受け手の順に並んでいれば、受け手ごとの終わりの番号（並んでいなければ None）"""\n'
     '        c = self.__dict__.setdefault("_ends", {})\n'
     '        if gi not in c:\n'
     '            t = self._tgt_rel[gi]\n'
     '            ok = bool((t[1:] >= t[:-1]).all()) if len(t) > 1 else True\n'
     '            c[gi] = torch.bincount(t, minlength=n_out).cumsum(0) if ok else None\n'
     '        return c[gi]\n'
     '\n'
     '    @torch.no_grad()\n'
     '    def learn(self, anyr, cl_spk, d):\n'),
]
TG = [
    ('        self.tail = tail\n'
     '        self.hist = torch.zeros(self.NUC_A_WIN, n_ch * 2, dtype=torch.uint8, device=dev)\n',
     '        self.tail = tail\n'
     '        # 版95b：線を受け手の順に並べ直して、受け手ごとの合計を累積和の差で（index_add の渋滞を避ける）\n'
     '        o = torch.argsort(self.dst[self.ie1], stable=True); self.ie1 = self.ie1[o]\n'
     '        o = torch.argsort(self.dst[self.ie2], stable=True); self.ie2 = self.ie2[o]\n'
     '        o = torch.argsort(self.ti1, stable=True); self.ii1, self.ti1 = self.ii1[o], self.ti1[o]\n'
     '        o = torch.argsort(self.ti2, stable=True); self.ii2, self.ti2 = self.ii2[o], self.ti2[o]\n'
     '        ends = lambda t, n: torch.bincount(t, minlength=n).cumsum(0)\n'
     '        self.end_e1, self.end_e2 = ends(self.dst[self.ie1], n_ch), ends(self.dst[self.ie2], n_ch)\n'
     '        self.end_i1, self.end_i2 = ends(self.ti1, n_recv), ends(self.ti2, n_recv)\n'
     '        self.hist = torch.zeros(self.NUC_A_WIN, n_ch * 2, dtype=torch.uint8, device=dev)\n', 1),
    ('        x1 = torch.zeros(n_ch, device=self.dev).index_add_(0, self.dst[self.ie1], c[self.ie1])\n'
     '        x2 = torch.zeros(n_ch, device=self.dev).index_add_(0, self.dst[self.ie2], c[self.ie2])\n'
     '        h1 = torch.zeros(n_recv, device=self.dev).index_add_(0, self.ti1, c[self.ii1])\n'
     '        h2 = torch.zeros(n_recv, device=self.dev).index_add_(0, self.ti2, c[self.ii2])\n',
     '        x1, x2 = _segsum_nuc(c[self.ie1], self.end_e1), _segsum_nuc(c[self.ie2], self.end_e2)   # 版95b\n'
     '        h1, h2 = _segsum_nuc(c[self.ii1], self.end_i1), _segsum_nuc(c[self.ii2], self.end_i2)\n', 1),
]
TG_FN = '''

def _segsum_nuc(v, ends):
    """版95b：受け手の順に並んだ値を受け手ごとに合計（累積和の差。float64 で足す）"""
    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=v.device), v.double().cumsum(0)])
    starts = torch.cat([torch.zeros(1, dtype=ends.dtype, device=ends.device), ends[:-1]])
    return (cs[ends] - cs[starts]).to(v.dtype)
'''
BB = [("#  95  基底核",
       "#  95b 足し込みの渋滞を直す（下からのシナプスの表・NucNet を受け手の順に並べて累積和の差で。意味は同じ）\n#  95  基底核")]


def edit(s, pairs, path):
    for pr in pairs:
        a, b = pr[0], pr[1]
        n = pr[2] if len(pr) > 2 else 1
        if s.count(a) != n:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（{n} 個のはず）：{a[:70]!r}。何も書いていない")
        s = s.replace(a, b)
    return s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 95 or "95b" in pathlib.Path("babble.py").read_text():
        sys.exit("⚠️ 版95 ではない（apply_v95.py のあとに1回だけ当てる）")
    s_tb = edit(pathlib.Path("torch_brain.py").read_text(), TB, "torch_brain.py")
    s_tg = edit(pathlib.Path("torch_group2.py").read_text(), TG, "torch_group2.py") + TG_FN
    s_bb = edit(pathlib.Path("babble.py").read_text(), BB, "babble.py")
    for s in (s_tb, s_tg, s_bb):
        ast.parse(s)
    pathlib.Path("torch_brain.py").write_text(s_tb)
    pathlib.Path("torch_group2.py").write_text(s_tg)
    pathlib.Path("babble.py").write_text(s_bb)
    print("✅ 版95b を当てた（下からのシナプスの表・NucNet の足し込みを累積和の差で。意味は版95 と同じ）", flush=True)
