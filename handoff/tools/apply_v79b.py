"""版79b：箱の余りの辺を1か所に集めない（速さだけ。値・網は版79 のまま）
（~/anaplex で。版79 の上に当てる。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v79b.py
【測】arkhe 版79（10-14）：1歩 21.5ms（版78 23.2）／ 2周目から 0.97ms/周（版78 1.19）しか縮まなかった
  GPU の足し込みの核：11回/歩で 9.3ms（版78 4.7ms）。グラフの中の1周ぶん 約0.55ms 増えていた
読み【C】：箱の余り（数千〜2万本）を全部「先頭の辺」の写しにしていた ⇒ 同じ受け手の1か所へ 0 を足す（ぶつかって順番待ち）・同じ所へ書く が何千回
直すもの：余りの j 番目は j ÷ 本当の数 の余りの辺の写しに（写しは1本の辺につき多くて1つ ⇒ ぶつからない）
  足すのは 0（valid）・書くのは同じ値（値は先に全部読んでから書く）なので、値は版79 と同じ
"""
import ast, pathlib, sys

G2 = [
    ("    \"\"\"版79：辺の番号を大きさ C の箱に詰める。余りは先頭の辺の写し（足すときは valid で 0 にする）\"\"\"\n",
     "    \"\"\"版79：辺の番号を大きさ C の箱に詰める。余りは本当の辺の写し（足すときは valid で 0 にする）\n"
     "    版79b：余りの j 番目は j ÷ 本当の数 の余りの辺（⟲ 版79 は全部先頭の辺。1か所にぶつかって遅かった）\"\"\"\n"),
    ("    jj = torch.where(valid, j, torch.zeros_like(j))\n"
     "    s = torch.searchsorted(csum, jj, right=True)\n",
     "    jj = torch.where(valid, j, j % csum[-1])\n"
     "    s = torch.searchsorted(csum, jj, right=True)\n"),
    ("    \"\"\"版79：_flush_core の値だけ（書くのは外で。箱の余りは先頭の辺と同じ値を同じ所へ書く）\"\"\"\n",
     "    \"\"\"版79：_flush_core の値だけ（書くのは外で。箱の余りは写した辺と同じ値を同じ所へ書く）\"\"\"\n"),
]
BB = [("#      【測】版78 1歩 23.2ms（2周目から 1.19ms/周）ANAPLEX_GRAPH=0 で版78 と同じ周ごと\n",
       "#      【測】版78 1歩 23.2ms（2周目から 1.19ms/周）ANAPLEX_GRAPH=0 で版78 と同じ周ごと\n"
       "#  79b 箱の余りの辺を1か所に集めない（版79 1歩 21.5ms。余りが先頭の辺の受け手にぶつかっていた）。値は同じ\n")]


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
    if babble.VERSION != 79:
        sys.exit(f"⚠️ babble.py が版79 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版79b を当てた（torch_group2・babble）。箱の余りの辺を1か所に集めない（値は同じ）")
