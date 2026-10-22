"""版95c：CL の体数を人の決め（片側 100万体あたり 20体・下限 64）に（4,096 → 64）
（~/anaplex で。版95b の上に当てる。印が合わなければ何も書かずに止まる）
  python3 apply_v95c.py
【み】10-22 20:39「ア」（CL を人・霊長類の値で決め直す）・21:02「いいよ」
⟲ 4,096 は版67 の「ハードの側のノブ」（CL がまだ素子でなかった頃）のまま残っていた
【ヒト】CL の体積 40.5mm³ ／ 視床 8,671mm³（0.47%、FreeSurfer の視床の核の地図・1人の例 ⚠️）× 視床 片側 4,929万（Wegiel 2014）
  ≈ 23万体 → 20/100万で 約5体 → 下限 64【計算】（決め2【み】10-15・基底核の核と同じ決まり）
⚠️ 皮質の表現の体は CL から 40本ずつ受ける（K_CL）⇒ 64体から取ると送り手がほぼ重なり、覚醒の入力が皮質全体でそろって届く形になる（見張る）
"""
import ast, pathlib, sys

TB = [('N_PPN, N_CL = 512, 4096            # ⚠️ ハードの側のノブ\n',
       'N_PPN, N_CL = 512, 64              # ⚠️ PPN はハードの側のノブ ／ 版95c：CL は人の決め（20/100万・下限 64）【計算】⟲ 版67〜95b 4,096\n')]
BB = [("#  95b 足し込みの渋滞を直す",
       "#  95c CL 4,096 → 64（人の決め 20/100万・下限 64）【み】10-22\n#  95b 足し込みの渋滞を直す")]


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
    if babble.VERSION != 95 or "95b" not in pathlib.Path("babble.py").read_text() or "95c" in pathlib.Path("babble.py").read_text():
        sys.exit("⚠️ 版95b ではない（apply_v95b.py のあとに1回だけ当てる）")
    out = [edit("torch_brain.py", TB), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版95c を当てた（CL 4,096 → 64）。⚠️ 網の形が変わるので state.pt は 0 から", flush=True)
