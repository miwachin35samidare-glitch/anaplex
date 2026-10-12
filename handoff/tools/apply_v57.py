"""版57（コンテナの写し用。編集は claude/道具_0926/apply_v57.py と同じ）"""
import ast, pathlib, sys

AREA = [
    ("                 inh_n_exc=None, relay_exc=None):\n",
     "                 inh_n_exc=None, relay_exc=None, inh_exc_from_rep=False):\n"),
    ("        exc_pool = torch.nonzero(~is_inh_send).squeeze(1)\n"
     "        def pick(n_rows, k_all, k_inh=None):\n",
     "        exc_pool = torch.nonzero(~is_inh_send).squeeze(1)\n"
     "        # 版57：海馬の抑制の体は 194 の前提（相手は錐体）どおり、遅れの列の興奮の行（時間細胞ではない）から引く\n"
     "        self.inh_exc_from_rep = inh_exc_from_rep\n"
     "        exc_pool_rep = exc_pool[exc_pool >= self.ch.n_cells]\n"
     "        def pick(n_rows, k_all, k_inh=None, pool=None):\n"
     "            pool = exc_pool if pool is None else pool\n"),
    ("            s[:, k_inh:] = exc_pool[torch.randint(0, len(exc_pool), (n_rows, k_all - k_inh),\n",
     "            s[:, k_inh:] = pool[torch.randint(0, len(pool), (n_rows, k_all - k_inh),\n"),
    ("                         pick(self.n_inh, n_pi)])\n",
     "                         pick(self.n_inh, n_pi, pool=(exc_pool_rep if inh_exc_from_rep else None))])\n"),
]
BRAIN = [
    ("            relay_exc=(RELAY_EXC if nm == \"視床\" else None),   # 版51\n",
     "            relay_exc=(RELAY_EXC if nm == \"視床\" else None),   # 版51\n"
     "            inh_exc_from_rep=(nm in HIPPO_TC),               # 版57：抑制の体の興奮の相手は錐体（遅れの列）から\n"),
]
G2 = [
    ("        if getattr(self, \"_pool_ei\", None) is None:\n"
     "            self._pool_ei = [(p[~self.is_inh_all[p]], p[self.is_inh_all[p]]) for p in self.pool_send]\n",
     "        if getattr(self, \"_pool_ei\", None) is None:\n"
     "            self._pool_ei = [(p[~self.is_inh_all[p]], p[self.is_inh_all[p]]) for p in self.pool_send]\n"
     "            # 版57：海馬の抑制の体へ生やす興奮は、遅れの列の興奮の行（時間細胞ではない）から\n"
     "            nb = self.n_send_all - sum(a.n_src * a.D for a in self.areas)      # 遅れの列の始まり（通し番号）\n"
     "            self._pool_e_rep = [pe[pe >= nb] for pe, _ in self._pool_ei]\n"
     "            self._inh_rep_area = [bool(getattr(a, \"inh_exc_from_rep\", False)) for a in self.areas]\n"),
    ("                if k:\n"
     "                    pool = self._pool_ei[i][kind]\n"
     "                    new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev, generator=self.gen)]\n",
     "                if k:\n"
     "                    pool = self._pool_ei[i][kind]\n"
     "                    new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev, generator=self.gen)]\n"
     "                    if kind == 0 and self._inh_rep_area[i]:     # 版57：抑制の体の受け手だけ引き直す\n"
     "                        mi = m & self.is_inh_r[self.e_recv[cuts].long()]\n"
     "                        ki = int(mi.sum())\n"
     "                        if ki:\n"
     "                            pr_ = self._pool_e_rep[i]\n"
     "                            new[mi] = pr_[torch.randint(0, len(pr_), (ki,), device=self.dev, generator=self.gen)]\n"),
]
BB = [("VERSION = 56          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52, 53, 54, 55)   # 網の作りが同じで、state.pt をそのまま読める版（版52 速さと乱数の種、版53 刈り方と恒常性、版54 速さ、版55 焼いた網、版56 海馬の時間細胞を調整する側に）",
       "VERSION = 57          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()   # 版57 で海馬の抑制の体の配線を引き直した ⇒ 版56 までの state.pt は読めない"),
      ("#  56  海馬の時間細胞を調整する側に（予測に入り、撃たせない）。v55b で海馬の表現の 70〜99% を自分の時間細胞が撃たせていた【み】10-10\n",
       "#  56  海馬の時間細胞を調整する側に（予測に入り、撃たせない）。v55b で海馬の表現の 70〜99% を自分の時間細胞が撃たせていた【み】10-10\n"
       "#  57  海馬の抑制の体の興奮の相手 194 を錐体（遅れの列）から引く（194 の前提どおり）。版56 で抑制の体が止まった【み】10-10\n")]


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    out = [edit("torch_area.py", AREA), edit("torch_brain.py", BRAIN), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版57 を当てた")
