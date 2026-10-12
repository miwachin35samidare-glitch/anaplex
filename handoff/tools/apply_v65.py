"""版65：聴覚の表現・誤差の体の抑制の相手も、版64 の運動と同じ「抑制の体そのものを 23% で1回ずつ」にする（~/anaplex で。印が合わなければ何も書かずに止まる）
  python3 apply_v65.py      （版64 の上に当てる）
【み】10-11 23時28分「まぁいいよ、やってみて」（A：人の形の抑制を時間細胞を持つ皮質の領野へ1つずつ）
【測】版64（10-11）：運動は人の形の抑制で ihn 1.4〜1.6 → 約0.1。ほかの領野・言えた・ゴール1 に害なし
【コード】時間細胞を持つ皮質の領野の表現の抑制は、93% が抑制の行の時間細胞（抑制の体が撃ったあと 341ms まで写す）・6% が傾き・抑制の体そのもの 0.4〜0.9%
人：【ヒト】Jiang 2012（側頭・前頭の新皮質の切片）速い発火の細胞 → 錐体 23%・減り 8.5ms ／ 聴覚（ヘシュル回）は側頭の新皮質
  皮質の遅れの列は D＝1（1歩前の列だけ）⇒ 抑制は歩の中（版63）と1歩前で届く。運動（D＝8）と同じ届き方
なぜ聴覚から：皮質のいちばん下の段。ここの抑制が人でないと、上の段（連合・高次・前頭前野）への下からの入力がそのまま歪む
  ⚠️ ゴール1（聴覚の誤差で自分・録音・返し手を比べる）が聴覚で測られる ⇒ ゴール1 の動きも見る
変えるもの：torch_brain.py の INH_SLICE ＝ ("運動",) → ("運動", "聴覚")（版64 の仕組みのまま）
  arkhe の大きさで：聴覚の抑制の体 76体 × 0.23 ⇒ 表現1体あたり 17本（前は 1,456本の写し）
変えないもの：聴覚の抑制の体が受ける相手（抑制の行の時間細胞は抑制の体の受け手にはまだ届く）・重さ・ほかの領野
⚠️ 配線が変わる ⇒ 前の state.pt は読めない（SAME_NET を空に）。健診は0から
"""
import ast, pathlib, re, sys

BRAIN = [("INH_SLICE = (\"運動\",)\n",
          "INH_SLICE = (\"運動\", \"聴覚\")   # 版65：聴覚も【み】10-11「まぁいいよ、やってみて」\n")]
BB = [("VERSION = 64          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()",
       "VERSION = 65          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()"),
      ("#  64  運動の表現・誤差の抑制の相手を、抑制の体そのもの（1歩前の列）から 体の数 × 23% を1回ずつ【ヒト】Jiang 2012【み】10-11「Aで様子を見よう」\n",
       "#  64  運動の表現・誤差の抑制の相手を、抑制の体そのもの（1歩前の列）から 体の数 × 23% を1回ずつ【ヒト】Jiang 2012【み】10-11「Aで様子を見よう」\n"
       "#  65  聴覚も版64 と同じ抑制の相手に（INH_SLICE に聴覚）【み】10-11「まぁいいよ、やってみて」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py")


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
    if babble.VERSION != 64:
        sys.exit(f"⚠️ babble.py が版64 ではない（版 {babble.VERSION}）。先に apply_v64.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版65 を当てた（torch_brain・babble）。聴覚の表現・誤差の抑制も抑制の体そのものから 23% を1回ずつ")
    print("   ⚠️ 前の state.pt は読めない。健診は0から")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "65" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['65'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版65 でも回るようにした")
