"""版63：周回の中で撃った抑制の体の発火も、次の周で相手へ届ける（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v63.py      （版62 の上に当てる）
【み】10-11 21時41分「周転円でないならいいよ」（21時36分「他が下がりすぎる懸念があるんじゃないの？」）
【測】measure_v62b（10-11、版62）：運動の指令の入った体の 1＋ihn は 10周とも同じ（2.4〜2.6）。撃つ体は 3周ごと（不応期の上限）に1歩 3〜4回
【コード】_pass：周回の中で次の周へ届けていたのは表現の発火だけ（w_new[within_index] ＝ rep）。抑制の体の発火は次の歩まで届かない
人：【ヒト】Szegedi 2017（人の皮質の切片 L2/3）：錐体1発 → 速い籠細胞が 0.78 の確率で撃つ（遅れ 平均 2.67ms）→ 錐体へ抑制は錐体の発火の 4.1ms 後
  ⇒ 人では抑制は 10.67ms の歩の中で返る。模型の周（1.102ms ＝ シナプス1本の遅れ）でも 錐体 → 籠細胞 → 錐体 は 2周（2.2ms）で返せる
  周転円ではない：足すのは補正の項ではなく、表現の発火と同じ届け方を抑制の体の発火にも当てるだけ（人で欠けていた道を埋める）
変えるもの
  torch_group2.py  周回の中の抑制の体の発火を、短い遅れの「1歩前」の枠（buf の抑制の行）に重ねる（表現の within_index と同じ形）
  ANAPLEX_WITHIN_INH=0 で版62 と同じ振る舞い（比べる用）
変えないもの：配線・重み・抑制の強さ・閾値・不応期・抑制の行の時間細胞（歩ごとに進むまま）
  ⇒ 配線は同じ ⇒ 版62 の state.pt は読める（SAME_NET に 62）
効くところ【コード・コンテナで数えた】表現の体の抑制の送り手のうち抑制の体そのもの：視床・運動・背内側視床 50% ／ ほかの領野 0.4〜0.9%
  ⇒ 動くのは主に視床・運動・背内側視床。視床が下がりすぎないか（聴覚への下からも減る）を健診で見る
⚠️ 皮質・海馬の抑制の 93% は抑制の行の時間細胞の写し（人の形ではない。この版では触らない）
"""
import ast, pathlib, re, sys

G2 = [("REFRACT_PASSES = 3\n",
       "REFRACT_PASSES = 3\n"
       "# 版63：周回の中で撃った抑制の体の発火も次の周で届ける【ヒト】Szegedi 2017：籠細胞の抑制は錐体の発火の 4.1ms 後【み】10-11\n"
       "#   ANAPLEX_WITHIN_INH=0 で版62 と同じ（比べる用）\n"
       "import os as _os\n"
       "WITHIN_INH = _os.environ.get(\"ANAPLEX_WITHIN_INH\", \"1\") != \"0\"\n"),
      ("        self.within_index = torch.cat(\n"
       "            [torch.arange(a.n_ch, device=dev) * a.D + buf_base + int(buf_off[i])\n"
       "             for i, a in enumerate(self.areas)])\n",
       "        self.within_index = torch.cat(\n"
       "            [torch.arange(a.n_ch, device=dev) * a.D + buf_base + int(buf_off[i])\n"
       "             for i, a in enumerate(self.areas)])\n"
       "        # 版63：抑制の体の行（送り元の並び 表現・ext・抑制 の最後）。並びは inh_spk と同じ（領野ごとに n_inh）\n"
       "        self.within_inh_index = torch.cat(\n"
       "            [(a.n_ch + a.n_ext + torch.arange(a.n_inh, device=dev)) * a.D + buf_base + int(buf_off[i])\n"
       "             for i, a in enumerate(self.areas)])\n"),
      ("        w_new[self.within_index] = rep\n",
       "        w_new[self.within_index] = rep\n"
       "        if WITHIN_INH:                            # 版63：抑制の体の発火も次の周で届く\n"
       "            w_new[self.within_inh_index] = inh_spk\n")]
BB = [("VERSION = 62          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()",
       "VERSION = 63          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (62,)"),
      ("#  62  上丘の時間細胞を調整する側へ・抑制の体の相手を錐体から（海馬の版56・57 と同じ）。v59d で上丘の横の 98% が時間細胞【み】10-11\n",
       "#  62  上丘の時間細胞を調整する側へ・抑制の体の相手を錐体から（海馬の版56・57 と同じ）。v59d で上丘の横の 98% が時間細胞【み】10-11\n"
       "#  63  周回の中で撃った抑制の体の発火も次の周で届ける（配線は版62 のまま）【ヒト】Szegedi 2017。v62b で運動の抑制が10周とも同じ【み】10-11\n")]
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
    if babble.VERSION != 62:
        sys.exit(f"⚠️ babble.py が版62 ではない（版 {babble.VERSION}）。先に apply_v62.py")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版63 を当てた（torch_group2・babble）。周回の中の抑制の体の発火も次の周で届く")
    print("   配線は版62 のまま（版62 の state.pt は読める）。ANAPLEX_WITHIN_INH=0 で版62 と同じ振る舞い")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "63" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['63'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版63 でも回るようにした")
