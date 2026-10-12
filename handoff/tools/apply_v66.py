"""版66：指令の写し（運動 → 聴覚）が、聴覚の抑制の体を撃たせる（表現・誤差の体には今までどおり調整する側）
（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v66.py      （版65 の上に当てる）
【み】10-12 0時53分「まぁアでいいよ」「ただし人じゃなくて霊長類だからねそれ」
【測】measure_v65a（版65）：聴覚の抑制の体 音あり 0.84%・無音 0.15%。撃ち方に入る興奮は閾値の 1/13。相手 180 のうち 131本が調整する側（約8割が指令の写し【計算】）
根拠（霊長類。人ではない）
  【マーモセット】Eliades & Wang 2003：自分の発声で聴覚野の大半の細胞が抑えられ、抑えは声の数百 ms 前から始まり、自発も下がる。「皮質下だけでは説明できない」
  【マーモセット】2024 bioRxiv：声の約200ms 前から聴覚が抑えられ、前頭 → 聴覚 の結びつきが声の前に上がる
  ⚠️ 抑えが「抑制の細胞を撃たせて」起きるのかは、霊長類でも確かめられていない（細胞の型は分けていない）
  【ヒト】話すと上側頭回が撃たなくなる（Chan 2014）は働きの段。声の前から始まるかは人で同じ形では見ていない
変えるもの
  torch_brain.py   聴覚の指令の写しの送り手（版39 で調整する側にした行。時間細胞・傾き・遅れの列）に印 mod_drive_inh
  torch_group2.py  印の送り手から抑制の体へ届く分は、抑制の体の撃ち方から引かない（＝撃たせる側）。表現・誤差の体には今までどおり調整する側
  ANAPLEX_COPY_INH=0 で版65 と同じ振る舞い（比べる用）
変えないもの：連合からの戻り（調整する側のまま）・配線・重み・抑制の相手（版64・65）・ほかの領野
  ⇒ 配線は同じ ⇒ 版65 の state.pt は読める（SAME_NET に 65）
⚠️ 写しの送り手には写しの時間細胞（運動の発火の 341ms までの遅れ）も入る（版39 で調整する側にした組と同じ組を使う）
"""
import ast, pathlib, re, sys

BRAIN = [("        # 版56：海馬の時間細胞は調整する側（予測には入り、撃たせない）【み】10-10「いいよ」\n",
          "        # 版66：指令の写し（運動 → 聴覚）は聴覚の抑制の体を撃たせる（表現・誤差には調整する側のまま）\n"
          "        #   【マーモセット】Eliades & Wang 2003：自分の発声の数百 ms 前から聴覚野が抑えられる【み】10-12「まぁアでいいよ」\n"
          "        #   ⚠️ 人ではなく霊長類。抑えが抑制の細胞を通るかは霊長類でも未確認\n"
          "        if _os_copy.environ.get(\"ANAPLEX_COPY_INH\", \"1\") != \"0\":\n"
          "            self.g.mod_drive_inh[self._send_rows(\"聴覚\", n[\"聴覚\"], n[\"運動\"])] = True\n"
          "        # 版56：海馬の時間細胞は調整する側（予測には入り、撃たせない）【み】10-10「いいよ」\n"),
         ("        n = self.n\n"
          "        for area, first, cnt in ((\"聴覚\", n[\"聴覚\"], n[\"運動\"]),) + self.topdown_rows + (\n",
          "        n = self.n\n"
          "        import os as _os_copy\n"
          "        for area, first, cnt in ((\"聴覚\", n[\"聴覚\"], n[\"運動\"]),) + self.topdown_rows + (\n")]
G2 = [("        self.mod_send = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 調整する側の送り手（版39。torch_brain が印を付ける）\n",
       "        self.mod_send = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 調整する側の送り手（版39。torch_brain が印を付ける）\n"
       "        self.mod_drive_inh = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 版66：調整する側のうち抑制の体は撃たせる送り手\n"),
      ("        mm = self.mod_send[se_] & ~self.e_inh[e]\n",
       "        mm = self.mod_send[se_] & ~self.e_inh[e]\n"
       "        mm = mm & ~(self.mod_drive_inh[se_] & self.is_inh_r[re_])   # 版66：印の送り手 → 抑制の体 は撃たせる側\n")]
BB = [("VERSION = 65          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()",
       "VERSION = 66          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (65,)"),
      ("#  65  聴覚も版64 と同じ抑制の相手に（INH_SLICE に聴覚）【み】10-11「まぁいいよ、やってみて」\n",
       "#  65  聴覚も版64 と同じ抑制の相手に（INH_SLICE に聴覚）【み】10-11「まぁいいよ、やってみて」\n"
       "#  66  指令の写しが聴覚の抑制の体を撃たせる（配線は版65 のまま）【マーモセット】Eliades & Wang 2003【み】10-12「まぁアでいいよ」「人じゃなくて霊長類」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py")


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
    if babble.VERSION != 65:
        sys.exit(f"⚠️ babble.py が版65 ではない（版 {babble.VERSION}）。先に apply_v65.py")
    out = [edit("torch_brain.py", BRAIN), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版66 を当てた（torch_brain・torch_group2・babble）。指令の写しが聴覚の抑制の体を撃たせる【マーモセット】")
    print("   配線は版65 のまま（版65 の state.pt は読める）。ANAPLEX_COPY_INH=0 で版65 と同じ振る舞い")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "66" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['66'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版66 でも回るようにした")
