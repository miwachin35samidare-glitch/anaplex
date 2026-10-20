"""版94b：中脳の核を人と照らして、違うと分かった所を直す（版94 の上に当てる。印が合わなければ何も書かずに止まる）
  python3 apply_v94b.py
【み】10-20 18:10「人の値になってない状態で焼いて平気？」「人の値を的にして焼いた？」「人と照らし合わせてないなら照らし合わせて」
直すもの
  1 素子の型を核ごとに（版94 は5核で1つの型「中脳」）。細胞の型が違う（黒質のドーパミン細胞・視床下核のグルタミン酸の細胞・
    吻内側被蓋核の GABA の細胞）のに同じ素子を共有していた ⇒ 版55 の「型ごと」【み】10-10 に合わせる
  2 視床下核の抑制の体の割合 16%（皮質の値）→ 7.5%【ヒト】Lévesque & Parent 2005 Mov Disord（人の視床下核の GABA の介在細胞 約7.5%）
  3 焼かない：prior_v2_中脳.npz（版94 の網の列で焼いたもの）は使わない。核ごとの型は、率を人に合わせてから録って焼くまで prior_v1
    ⟲ 10-20 に「rec_streams → bake_prior」を勧めたのは誤り：焼く的は網の列（下から越えた歩）で人の値ではない。
       人から外れた列（黒質の的 48.7%/歩 ≒ 46Hz、人 4.56Hz）で焼くと、外れた率をそのまま素子に凍らせる
⚠️ 配線は版94 と同じ・数値は合わせない（10-18 22:08【み】）
"""
import ast, pathlib, sys

TB = [
    ('              "側坐核": "側坐核", "上丘": "上丘", **{k: "中脳" for k in NUC}}   # 版94\n'
     'UNBAKED_OK = ("中脳",)   # 版94：焼く前は prior_v1（録って焼くまで）\n',
     '              "側坐核": "側坐核", "上丘": "上丘", **{k: k for k in NUC}}   # 版94b：核ごとの型（⟲ 版94 は5核で「中脳」1つ）\n'
     'UNBAKED_OK = NUC   # 版94b：率を人に合わせてから録って焼くまで prior_v1（⟲ 版94 の列で焼いた prior_v2_中脳 は使わない）\n'),
    ('            inh_frac=(0.275 if nm in ("視床", "背内側視床") else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n',
     '            inh_frac=(0.275 if nm in ("視床", "背内側視床") else 0.075 if nm == "視床下核" else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n'
     '            #   版94b：視床下核の GABA の介在細胞 約7.5%【ヒト】Lévesque & Parent 2005\n'),
    ('                print(f"⚠️ {f} が無い ⇒ 型「{t}」は焼く前の素子（prior_v1）で。rec_streams → bake_prior で焼く", flush=True)\n',
     '                print(f"⚠️ {f} が無い ⇒ 型「{t}」は焼く前の素子（prior_v1）で。率を人に合わせてから録って焼く", flush=True)\n'),
]
BB = [
    ("#  94  中脳の5核を素子に",
     "#  94b 核ごとの素子の型・視床下核の抑制の体 7.5%【ヒト】・人から外れた列で焼かない【み】10-20\n#  94  中脳の5核を素子に"),
]


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
    if babble.VERSION != 94:
        sys.exit(f"⚠️ babble.py が版94 ではない（版 {babble.VERSION}）")
    out = [edit("torch_brain.py", TB), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版94b を当てた（核ごとの素子の型・視床下核の抑制の体 7.5%）", flush=True)
    q = pathlib.Path("prior_v2_中脳.npz")
    if q.exists():
        q.rename("prior_v2_中脳.npz.使わない")
        print("   prior_v2_中脳.npz は使わない（人から外れた列で焼いた）⇒ prior_v2_中脳.npz.使わない に名前を変えた", flush=True)
