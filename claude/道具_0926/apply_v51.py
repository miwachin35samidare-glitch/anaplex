"""版51：視床（感覚の視床）の中継の細胞が、自分の領野の中から受ける興奮の本数を RELAY_EXC 本にする（~/anaplex で。印が1つでも無ければ何も書かずに止まる）
  python3 apply_v51.py
【み選】10-05 B「疎にして残す」
  【ヒト】分からない（AV の Golgi は中継の細胞らしい I 型の軸索が始まりしか染まらない。Al-Hussain 2007）
  【マカク】核の中に出る側枝は記録した中で1本（Wilson 1989）
  【ネコ】側枝のシナプスの約2割が中継の細胞へ、層と層の間だけ（Bickford 2008）
  ⇒ あるが少ない。本数は人・霊長類に値が無いのでノブ【C】
版50 まで：中継の細胞1体が自分の領野の中から 約6,400本（遅れの送り手 3,200 ＋ 傾きの細胞 3,200）
【測】sim_thal（コンテナ・CPU、視床1領野、1,600歩、学びながら）40% を越える歩：全部 3.8〜4.5% ／ 640 0.2〜1.0% ／ 64 0%
抑制性の相手の数（約 2,430）と、中継 → 抑制の体の繋がりは変えない。背内側視床は変えない
"""
import ast, pathlib, sys

def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s

A = [("                 inh_n_exc=None):", "                 inh_n_exc=None, relay_exc=None):"),
("""        n_partner = min(n_partner, self.n_send)
        self.n_partner = n_partner""", """        n_partner = min(n_partner, self.n_send)
        # 版51：視床の中継の細胞が自分の領野の中から受ける興奮の本数（relay_exc。ノブ【C】）。抑制性の相手の数は変えない
        self.n_inh_partner_rows = int(round(n_partner * inh_frac))
        if relay_exc is not None:
            n_partner = self.n_inh_partner_rows + relay_exc
        self.n_partner = n_partner"""),
("""        def pick(n_rows, k_all):
            k_inh = int(round(k_all * inh_frac))""", """        def pick(n_rows, k_all, k_inh=None):
            k_inh = int(round(k_all * inh_frac)) if k_inh is None else k_inh"""),
("        src = torch.cat([pick(n_ch * 2, n_partner), pick(self.n_inh, n_pi)])",
 "        src = torch.cat([pick(n_ch * 2, n_partner, self.n_inh_partner_rows if relay_exc is not None else None),\n                         pick(self.n_inh, n_pi)])")]
BR = [("N_PARTNER = {\"CA3\": 16100}\n", """N_PARTNER = {"CA3": 16100}
# 版51：視床の中継の細胞が自分の領野の中から受ける興奮の本数 ⚠️ ノブ【C】【み選】10-05 B「疎にして残す」
#   【ヒト】分からない ／【マカク】核の中に出る側枝は記録した中で1本（Wilson 1989）／
#   【ネコ】側枝のシナプスの約2割が中継の細胞へ、層と層の間だけ（Bickford 2008）⇒ あるが少ない
#   ⟲ 版50 まで 約6,400本（遅れの送り手 3,200 ＋ 傾きの細胞 3,200）。視床が中で自分を押し上げ、一斉発火の大元になっていた【測】v50k
#   64：sim_thal（コンテナ）で 40% を越える歩が消えた本数
RELAY_EXC = 64
"""),
("            inh_n_exc=(INH_N_EXC_HIPPO if nm in (\"歯状回\", \"CA3\", \"CA1\", \"嗅内浅\", \"嗅内深\") else INH_N_EXC),",
 "            inh_n_exc=(INH_N_EXC_HIPPO if nm in (\"歯状回\", \"CA3\", \"CA1\", \"嗅内浅\", \"嗅内深\") else INH_N_EXC),\n            relay_exc=(RELAY_EXC if nm == \"視床\" else None),   # 版51")]
H_OLD = "#      （Group2 にまとめたときのずれ【C】。1領野の Area は合っていた。【測】chk_tb 10-04）\n"
BB = [("VERSION = 50          # コードの版。網の作りを変えたら上げる", "VERSION = 51          # コードの版。網の作りを変えたら上げる"),
      (H_OLD, H_OLD + "#  51  視床の中継の細胞が自分の領野の中から受ける興奮を 約6,400本 → 64本（ノブ【C】）【み選】10-05 B「疎にして残す」\n"
                      "#      （【マカク】核の中の側枝は記録した中で1本 Wilson 1989 ／【ネコ】側枝の約2割が中継の細胞へ Bickford 2008）\n")]
out = [edit("torch_area.py", A), edit("torch_brain.py", BR), edit("babble.py", BB)]
for p, s in out:
    p.write_text(s)
print("✅ 版51 を当てた（torch_area.py・torch_brain.py・babble.py）。state.pt は版50 のものなので使えない（0から）")
