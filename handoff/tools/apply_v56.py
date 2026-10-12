"""版56（コンテナの写し用。編集は claude/道具_0926/apply_v56.py と同じ）"""
import ast, pathlib, sys

A = """            self.g.mod_send[self._send_rows(area, first, cnt)] = True
"""
B = """            self.g.mod_send[self._send_rows(area, first, cnt)] = True
        # 版56：海馬の時間細胞は調整する側（予測には入り、撃たせない）【み】10-10「いいよ」
        #   【ヒト】時間細胞は細胞そのものの撃ち方（Umbach 2020）。自分の過去が自分を撃たせる入力の根拠は無い
        #   【測】v55b：版55 まで海馬の表現の 70〜99% を自分の時間細胞が撃たせ、CA1 に CA3 はほぼ届いていなかった
        #   ANAPLEX_HIPPO_TC=drive で版55 と同じ（比べる用）
        import os as _os
        if _os.environ.get("ANAPLEX_HIPPO_TC") != "drive":
            for nm in HIPPO_TC:
                a, i = self.areas[nm], NAMES.index(nm)
                per = a.ch.n_cell + a.ch.n_ramp
                if per:
                    lo = int(self.g.off_send[i])
                    self.g.mod_send[lo:lo + (a.n_ch + a.n_ext) * per] = True   # 興奮の行（自分・ext）だけ。抑制の行はそのまま
"""
C = """EC_II = 20 """
D = """HIPPO_TC = ("歯状回", "CA3", "CA1", "嗅内浅", "嗅内深")   # 版56：時間細胞を調整する側にする領野
EC_II = 20 """
BB = [("VERSION = 55          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52, 53, 54)   # 網の作りが同じで、state.pt をそのまま読める版（版52 速さと乱数の種、版53 刈り方と恒常性、版54 速さ、版55 焼いた網）",
       "VERSION = 56          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (51, 52, 53, 54, 55)   # 網の作りが同じで、state.pt をそのまま読める版（版52 速さと乱数の種、版53 刈り方と恒常性、版54 速さ、版55 焼いた網、版56 海馬の時間細胞を調整する側に）"),
      ("#  55  焼いた網を型ごとに焼き直したもの（皮質・視床・海馬・側坐核・上丘）に。床 0.121 → 0.003〜0.010【み】10-10\n",
       "#  55  焼いた網を型ごとに焼き直したもの（皮質・視床・海馬・側坐核・上丘）に。床 0.121 → 0.003〜0.010【み】10-10\n"
       "#  56  海馬の時間細胞を調整する側に（予測に入り、撃たせない）。v55b で海馬の表現の 70〜99% を自分の時間細胞が撃たせていた【み】10-10\n")]


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    out = [edit("torch_brain.py", [(A, B), (C, D)]), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版56 を当てた")
