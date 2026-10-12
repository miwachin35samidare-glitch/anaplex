"""版61：視床（感覚の中継）の中継細胞どうしの興奮を外す（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v61.py      （版59 の上に当てる。revert_v60 のあと）
【み】10-11 18時07分「いいよ」（18時03分「A」「正しい人の形を調べる」「最終目標とマイルストーンを意識してね」）
【測】measure_v59c・v59d（10-11、版59）：無音で耳 0本なのに視床 22%（99% が横）。横はほぼ全部「自分の遅れ」（中継どうし）で、
  届いた1本の重さ 0.7〜1.2（閾値 0.3 を1本で越える）・横の合計は閾値の 15〜40倍 ⇒ 中継細胞が入力なしで撃ち合って回る
  その発火が無音でも聴覚の下からの入力（音ありの約8割）になり、聴覚が音と無音を区別しない
人（無い）→ 霊長類
  【マカク】Wilson 1989 J Neurosci（外側膝状体の中継細胞 9体、電子顕微鏡）：核の中に側枝を出したのは 9本のうち 1本（層1 に入って消える）
            中継細胞のシナプス：網膜（駆動）30〜34%・皮質の戻り 約3分の1・抑制の終末
  【マカク】Carandini, Horton & Sincich 2007 J Vis：網膜1本の入力の大きさはほぼ一定。外側膝状体の発火は網膜の発火の足し合わせで決まる
  ⇒ 中継細胞を撃たせるのは駆動する入力（耳 → 下丘 → 中継）。中継細胞どうしの興奮はほぼ無い
  （09-11 の設計「視床は感覚を中継して皮質を駆動する。予測の輪に入らない」と同じ向き）
変えるもの
  torch_brain.py  RELAY_EXC 64 → 0（中継の表現・誤差の行は、自分の領野の中から興奮を受けない。抑制性の相手の数は変えない）
  torch_area.py   relay_exc が 0 の領野に印（no_lat_exc）
  torch_group2.py 刈り込みで生やすとき、印の領野の表現・誤差の行には興奮を生やさない（抑制を切ったら抑制で生やす）
                  ⟲ 版53 の「静かなら抑制を切って興奮で生やす」をそのままにすると、日をかけて中継どうしの興奮がまた生える
変えないもの：視床の抑制の体（介在ニューロン）の相手（興奮 180本は中継の発火から）・閾値 0.3・耳 → 中継 1本 ・ほかの領野
⚠️ 配線が変わる ⇒ 前の state.pt は読めない（SAME_NET を空に）。健診は0から
⚠️ 霊長類の形のうち、網様核（中継 → 網様核 → 中継の抑制の戻り）と皮質 L6 からの戻り（調整）は、まだ模型に無い
"""
import ast, pathlib, re, sys

BRAIN = [("RELAY_EXC = 64\n",
          "# ⟲ 版61：64 → 0【み】10-11「いいよ」。【マカク】Wilson 1989：外側膝状体の中継細胞 9体のうち核の中に側枝を出したのは1本\n"
          "#   【マカク】Carandini 2007：中継の発火は網膜の駆動の足し合わせで決まる ⇒ 中継どうしの興奮はほぼ無い\n"
          "#   【測】measure_v59d（版59）：64本が学びで育ち、1本で閾値を越えて入力なしで撃ち合っていた（無音で視床 22%）\n"
          "RELAY_EXC = 0\n")]
AREA = [("        if relay_exc is not None:\n"
         "            n_partner = self.n_inh_partner_rows + relay_exc\n",
         "        if relay_exc is not None:\n"
         "            n_partner = self.n_inh_partner_rows + relay_exc\n"
         "        self.no_lat_exc = relay_exc == 0     # 版61：中継の表現・誤差の行は自分の領野の中から興奮を受けない\n")]
G2 = [("        to_inh = torch.cat([(over & do_i)[cut_e], ~quiet[cut_i]])   # 生やす種類（True ＝ 抑制）\n",
       "        to_inh = torch.cat([(over & do_i)[cut_e], ~quiet[cut_i]])   # 生やす種類（True ＝ 抑制）\n"
       "        # 版61：中継（no_lat_exc の領野）の表現・誤差の行には興奮を生やさない（【マカク】中継どうしの興奮はほぼ無い）\n"
       "        if getattr(self, \"_no_lat_exc_r\", None) is None:\n"
       "            fl = torch.tensor([bool(getattr(a, \"no_lat_exc\", False)) for a in self.areas], device=self.dev)\n"
       "            self._no_lat_exc_r = fl[self.area_of] & ~self.is_inh_r\n"
       "        to_inh = to_inh | self._no_lat_exc_r[self.e_recv[cuts].long()]\n")]
BB = [("VERSION = 59          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (57, 58, 60)",
       "VERSION = 61          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()"),
      ("#  ⟲ 10-11 版60 の f-I を外して版59 に戻した（切片に電流を流した値の当てはめで、人の自然な反応の根拠が無い【み】）\n",
       "#  ⟲ 10-11 版60 の f-I を外して版59 に戻した（切片に電流を流した値の当てはめで、人の自然な反応の根拠が無い【み】）\n"
       "#  61  視床の中継細胞どうしの興奮を外す（RELAY_EXC 64 → 0、刈り込みでも生やさない）【マカク】Wilson 1989・Carandini 2007【み】10-11\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py")


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
    if babble.VERSION != 59:
        sys.exit(f"⚠️ babble.py が版59 ではない（版 {babble.VERSION}）。版60 なら先に python3 revert_v60.py")
    out = [edit("torch_brain.py", BRAIN), edit("torch_area.py", AREA), edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版61 を当てた（torch_brain・torch_area・torch_group2・babble）。視床の中継どうしの興奮を外した")
    print("   ⚠️ 前の state.pt は読めない。健診は0から")
    for f in TOOLS:            # 測る道具の版の確かめに 61 を足す（無ければ飛ばす）
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "61" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['61'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版61 でも回るようにした")
