"""版52：1歩を縮める（意味は変えない）／ 脳の中の乱数を、走らせるたびに変える（~/anaplex で。印が1つでも無ければ何も書かずに止まる）
  python3 apply_v52.py
網の作り（体・辺・重み・式）は変えない ⇒ 版51 の state.pt をそのまま続けて使える
"""
import ast, pathlib, sys

def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s

# ── torch_group2.py：速さ A・B・C ──
OLD_FE = '''    def _fired_edges(self):
        if self._e_cache is not None:
            return self._e_cache
        if self._base is None:                    # 時間細胞と buf は1歩に1回でいい
            self._base = torch.cat([self.ch.spikes()] + [b.reshape(-1) for b in self.bufs])
        s = self._base
        if self._within is not None:
            s = torch.maximum(s, self._within)    # 周回の中で撃った分も送り手になる
        fired = torch.nonzero(s).squeeze(1)
        empty = torch.empty(0, dtype=torch.long, device=self.dev)
        if len(fired) == 0:
            self._e_cache = empty
            return empty
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        fired, lens = fired[keep], lens[keep]
        if len(fired) == 0:
            self._e_cache = empty
            return empty
        base = self.start[fired].repeat_interleave(lens)
        off = torch.arange(int(lens.sum()), device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens)
        self._e_cache = base + off
        return self._e_cache
'''
NEW_FE = '''    def _expand(self, fired):
        """撃った送り手 → その辺の番号（送り手ごとに連ねて持っているので、start から切り出す）"""
        empty = torch.empty(0, dtype=torch.long, device=self.dev)
        if len(fired) == 0:
            return empty
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        fired, lens = fired[keep], lens[keep]
        if len(fired) == 0:
            return empty
        base = self.start[fired].repeat_interleave(lens)
        off = torch.arange(int(lens.sum()), device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens)
        return base + off

    def _fired_edges(self):
        # 版52 C：時間細胞と buf の辺は1歩に1回だけ作り、周回の中で新しく撃った送り手の辺だけ足す
        #   ⟲ 版51 まで周回ごとに全部作り直していた。足す辺の組は同じ（並びだけ違う）【測】measure_speed 10-07
        if self._e_cache is not None:
            return self._e_cache
        if self._base is None or self._base_of is not self._base:
            if self._base is None:                # 時間細胞と buf は1歩に1回でいい
                self._base = torch.cat([self.ch.spikes()] + [b.reshape(-1) for b in self.bufs])
            self._base_of = self._base
            self._base_on = self._base > 0
            self._base_e = self._expand(torch.nonzero(self._base_on).squeeze(1))
        if self._within is None:
            self._e_cache = self._base_e
        else:                                     # 周回の中で撃った分も送り手になる
            new = torch.nonzero((self._within > 0) & ~self._base_on).squeeze(1)
            self._e_cache = torch.cat([self._base_e, self._expand(new)]) if len(new) else self._base_e
        return self._e_cache
'''
G2 = [("        self._release_areas()\n\n",
       "        self._release_areas()\n"
       "        # 版52 A・B：領野の境目と complete の有無は変わらないので CPU の数で持つ\n"
       "        #   ⟲ 版51 まで GPU の配列。周回ごとに int() で読むたび GPU を待っていた（1歩 約360回）\n"
       "        self.off_recv, self.off_ch, self.off_src = (self.off_recv.cpu().numpy(), self.off_ch.cpu().numpy(),\n"
       "                                                    self.off_src.cpu().numpy())\n"
       "        self._complete_any = bool(self.complete.any())\n"
       "        self._base_of = None\n\n"),
      (OLD_FE, NEW_FE),
      ("        if self.complete.any():           # 入力が無いところは予測が足しになる（撃たせはしない）\n",
       "        if self._complete_any:           # 入力が無いところは予測が足しになる（撃たせはしない）。版52 B\n")]

# ── babble.py：版・種・読み込み ──
H_OLD = "#      （【マカク】核の中の側枝は記録した中で1本 Wilson 1989 ／【ネコ】側枝の約2割が中継の細胞へ Bickford 2008）\n"
RESEED = '''


# 版52：脳の中の乱数を引き直す
#   【み】10-07「とても致命的に見える　君たちでさえ毎回回答が変わるのに」
#   配線は組み立てのときの seed=0 のまま（保存した重みと合わせる）。走らせるたびに、脳の中の雑音の並びを変える
#   ⟲ 版51 まで、走らせ直すたびに脳の中の雑音が毎回同じ並びで始まっていた
_OURS = ("torch_brain", "torch_group2", "torch_area", "midbrain", "brainstem", "sleep")


def _generators(b):
    """脳が持っている torch の乱数（重なりなし。見つけた順は毎回同じ）"""
    gens, seen, todo = [], set(), [b]
    while todo:
        o = todo.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Generator):
            gens.append(o)
        elif isinstance(o, (list, tuple)):
            todo += list(o)
        elif isinstance(o, dict):
            todo += list(o.values())
        elif type(o).__module__ in _OURS:
            todo += list(vars(o).values())
    return gens


def reseed(b, argv=None):
    """--seed N があればその種（同じ1日の再生。GPU の足し算の順も決める）。無ければ OS の乱れから。返り値は種"""
    argv = sys.argv if argv is None else argv
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1])
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        seed = int.from_bytes(os.urandom(4), "little")
    gens = _generators(b)
    for k, g in enumerate(gens):
        g.manual_seed(seed + 7919 * k)
    print(f"脳の中の乱数の種 {seed}（乱数 {len(gens)} 個）{'  再生：足し算の順を決めた' if '--seed' in argv else ''}", flush=True)
    return seed
'''
BB = [("VERSION = 51          # コードの版。網の作りを変えたら上げる",
       "VERSION = 52          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51,)      # 網の作りが同じで、state.pt をそのまま読める版（版52 は速さと乱数の種だけ）"),
      (H_OLD, H_OLD + "#  52  網の作りは変えない（版51 の state.pt を読める）。1歩を縮める（Group2 の待ち合わせ A・B、撃った辺の一覧 C）【み】10-07 ／\n"
                      "#      脳の中の乱数を走らせるたびに OS の乱れから引き直す（種はログ。--seed N で再生）【み】10-07「脳ってそうだったか？」\n"),
      ("    if v != VERSION:\n", "    if v != VERSION and v not in SAME_NET:\n"),
      ("\n\ndef load(b, path=STATE):", RESEED + "\n\ndef load(b, path=STATE):"),
      ("    done, log = (0, []) if fresh else load(b)\n",
       "    done, log = (0, []) if fresh else load(b)\n    seed = reseed(b)                            # 版52\n"),
      ("        log.append(f\"{datetime.now():%m-%d %H:%M} 喃語 {done}回（+{n_add}）\"",
       "        log.append(f\"{datetime.now():%m-%d %H:%M} 喃語 {done}回（+{n_add}）種 {seed}\"")]

# ── day.py ──
DD = [("    done, log = (0, []) if \"--fresh\" in sys.argv else B.load(b)\n",
       "    done, log = (0, []) if \"--fresh\" in sys.argv else B.load(b)\n    seed = B.reseed(b)                        # 版52：脳の中の乱数を走らせるたびに変える\n"),
      ("  python3 day.py 0              回さずに測るだけ\n",
       "  python3 day.py 0              回さずに測るだけ\n"
       "  python3 day.py 1 --seed 123   脳の中の乱数をその種で（同じ state から同じ1日を再生する。版52）\n"),
      ("        log.append(f\"{time.strftime('%m-%d %H:%M')} 喃語 {done}回（+{n_wake}）\"",
       "        log.append(f\"{time.strftime('%m-%d %H:%M')} 喃語 {done}回（+{n_wake}）種 {seed}\"")]

out = [edit("torch_group2.py", G2), edit("babble.py", BB), edit("day.py", DD)]
for p, s in out:
    p.write_text(s)
print("✅ 版52 を当てた（torch_group2.py・babble.py・day.py）。網の作りは同じなので、版51 の state.pt をそのまま続けられる")
