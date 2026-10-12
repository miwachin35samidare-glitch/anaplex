"""版62：上丘の時間細胞（と傾きの細胞）を調整する側へ。上丘の抑制の体の興奮の相手を錐体（遅れの列）から引く（~/anaplex で。印が1つでも合わなければ何も書かずに止まる）
  python3 apply_v62.py      （版61 の上に当てる）
【み】10-11 19時40分「c」（A：上丘の時間細胞を調整する側へ を当てて測ってから、B：黒質網様部 → 上丘の抑制 を決める）
【測】measure_v59d（10-11、版59）：上丘の横の 98% が自分の時間細胞（1歩 570〜1,060本・閾値の 8〜23倍）。無音でも 14% 撃つ（版61 の measure_v59c）
人：時間細胞は【ヒト】Umbach 2020 の海馬・嗅内の細胞の撃ち方。上丘に置く根拠は無い（人の上丘の単一細胞の記録は見つからない）
  海馬は版56 で同じく調整する側へ移した（予測 p には入り、表現・抑制の体を撃たせない）
変えるもの（海馬の版56・版57 と同じ扱いを上丘にも）
  1 上丘の興奮の行（自分の表現）の時間細胞・傾きの細胞を mod_send に（予測には入る・撃たせない）
  2 上丘の抑制の体の興奮の相手を、遅れの列の興奮の行から引く（版57。⟲ 時間細胞が撃たせなくなると抑制の体が止まった形を避ける）
     組み立てと刈り込みで生やすときの両方（版57 の inh_exc_from_rep の印を上丘にも付ける）
  ANAPLEX_HIPPO_TC=drive で 1 だけ版61 と同じ（比べる用。2 は戻らない）
変えないもの：下丘 → 上丘（1体 8本・3本で閾値）・閾値・上丘の体数 64・黒質への線
⚠️ 抑制の体の相手を引き直すので配線が変わる ⇒ 前の state.pt は読めない。健診は0から
⚠️ 黒質網様部 → 上丘の常にかかる抑制（【マカク】Hikosaka & Wurtz 1983）は、まだ無い（この版を測ってから決める）
"""
import ast, pathlib, re, sys

BRAIN = [
    ("HIPPO_TC = (\"歯状回\", \"CA3\", \"CA1\", \"嗅内浅\", \"嗅内深\")   # 版56：時間細胞を調整する側にする領野\n",
     "HIPPO_TC = (\"歯状回\", \"CA3\", \"CA1\", \"嗅内浅\", \"嗅内深\")   # 版56：時間細胞を調整する側にする領野\n"
     "# 版62：上丘も時間細胞を調整する側に（時間細胞は【ヒト】海馬・嗅内の撃ち方。上丘に置く根拠なし）【み】10-11「c」\n"
     "TC_MOD = HIPPO_TC + (\"上丘\",)\n"),
    ("            inh_exc_from_rep=(nm in HIPPO_TC),               # 版57：抑制の体の興奮の相手は錐体（遅れの列）から\n",
     "            inh_exc_from_rep=(nm in TC_MOD),                 # 版57：抑制の体の興奮の相手は錐体（遅れの列）から。版62 で上丘も\n"),
    ("            for nm in HIPPO_TC:\n"
     "                a, i = self.areas[nm], NAMES.index(nm)\n",
     "            for nm in TC_MOD:                               # 版62：上丘も\n"
     "                a, i = self.areas[nm], NAMES.index(nm)\n"),
]
BB = [("VERSION = 61          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()",
       "VERSION = 62          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = ()"),
      ("#  61  視床の中継細胞どうしの興奮を外す（RELAY_EXC 64 → 0、刈り込みでも生やさない）【マカク】Wilson 1989・Carandini 2007【み】10-11\n",
       "#  61  視床の中継細胞どうしの興奮を外す（RELAY_EXC 64 → 0、刈り込みでも生やさない）【マカク】Wilson 1989・Carandini 2007【み】10-11\n"
       "#  62  上丘の時間細胞を調整する側へ・抑制の体の相手を錐体から（海馬の版56・57 と同じ）。v59d で上丘の横の 98% が時間細胞【み】10-11\n")]
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
    if babble.VERSION != 61:
        sys.exit(f"⚠️ babble.py が版61 ではない（版 {babble.VERSION}）。先に apply_v61.py")
    out = [edit("torch_brain.py", BRAIN), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版62 を当てた（torch_brain・babble）。上丘の時間細胞は調整する側、抑制の体の相手は錐体から")
    print("   ⚠️ 前の state.pt は読めない。健診は0から")
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "62" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['62'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版62 でも回るようにした")
