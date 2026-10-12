"""版55（コンテナの写し用。編集は claude/道具_0926/apply_v55.py と同じ）"""
import ast, os, pathlib, sys


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:70]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


BR = [("""from torch_group2 import Group2
""", """from torch_group2 import Group2
import os

# 版55：焼いた網を型ごとに（【み】10-10「型ごとに分けるでいいよ」「いいよ」）。bake_prior.py で焼いたもの
PRIOR_TYPE = {"聴覚": "皮質", "運動": "皮質", "視覚": "皮質", "連合": "皮質", "高次": "皮質", "前頭前野": "皮質",
              "視床": "視床", "背内側視床": "視床",
              "歯状回": "海馬", "CA3": "海馬", "CA1": "海馬", "嗅内浅": "海馬", "嗅内深": "海馬",
              "側坐核": "側坐核", "上丘": "上丘"}


def type_priors(prior, dev):
    \"\"\"型 → 焼いた網。ANAPLEX_PRIOR=v1 なら全部 prior（比べる用）\"\"\"
    if os.environ.get("ANAPLEX_PRIOR") == "v1":
        return {}
    out = {}
    for t in sorted(set(PRIOR_TYPE.values())):
        f = f"prior_v2_{t}.npz"
        if not os.path.exists(f):
            raise SystemExit(f"⚠️ {f} が無い。bake_prior.py で焼いてから（ANAPLEX_PRIOR=v1 で前の網）")
        out[t] = Prior.load(f, dev=dev)
    return out
"""),
("""        mk = lambda nm, i: Area(
            n_ch=self.n[nm], prior=prior, n_ext=self.n_ext[nm], eta=eta,
""", """        tp = type_priors(prior, dev)       # 版55
        mk = lambda nm, i: Area(
            n_ch=self.n[nm], prior=tp.get(PRIOR_TYPE.get(nm), prior), n_ext=self.n_ext[nm], eta=eta,
""")]
G2 = [("""        self._complete_any = bool(self.complete.any())
""", """        self._complete_any = bool(self.complete.any())
        # 版55：焼いた網は型ごと。受け手の並びで同じ網が続く塊にまとめる（全部同じなら1つ）
        segs = []
        for i, a in enumerate(self.areas):
            lo, hi = int(self.off_recv[i]), int(self.off_recv[i + 1])
            if segs and segs[-1][2] is a.prior and segs[-1][1] == lo:
                segs[-1][1] = hi
            else:
                segs.append([lo, hi, a.prior])
        self._prior_segs = segs
"""),
("""            self._prior_l = self.prior.logit(self.tb)
""", """            self._prior_l = (self.prior.logit(self.tb) if len(self._prior_segs) == 1 else          # 版55
                             torch.cat([pr.logit(self.tb[lo:hi]) for lo, hi, pr in self._prior_segs]))
""")]
BB = [("VERSION = 54          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52, 53)   # 網の作りが同じで、state.pt をそのまま読める版（版52 速さと乱数の種、版53 刈り方と恒常性、版54 速さ）",
       "VERSION = 55          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52, 53, 54)   # 網の作りが同じで、state.pt をそのまま読める版（版52 速さと乱数の種、版53 刈り方と恒常性、版54 速さ、版55 焼いた網）"),
      ("#  54b 意味を変えずに CPU の待ちを減らす（印で選ぶ → 番号で選ぶ ほか。measure_sync 10-10）\n",
       "#  54b 意味を変えずに CPU の待ちを減らす（印で選ぶ → 番号で選ぶ ほか。measure_sync 10-10）\n"
       "#  55  焼いた網を型ごとに焼き直したもの（皮質・視床・海馬・側坐核・上丘）に。床 0.121 → 0.003〜0.010【み】10-10\n")]

if __name__ == "__main__":
    out = [edit("torch_brain.py", BR), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版55 を当てた")
