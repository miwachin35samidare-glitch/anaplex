"""比べる用の切り替え：型ごとに前の焼いた素子（prior_v1）へ戻す（~/anaplex で。版55 のまま・既定の動きは変えない）
  python3 apply_prior_mix.py      ← 1回だけ当てる
  使い方  ANAPLEX_PRIOR_V1=海馬,側坐核 python3 measure_check2.py   ← 海馬・側坐核だけ前の網、ほかは焼き直した網
          ANAPLEX_PRIOR=v1 は今までどおり全部前の網
【測】10-10 14時34分：前の網に戻すと CA1・嗅内深・側坐核は版53 並み ⇒ 約2倍にしたのは焼き直し
  海馬・側坐核自身の焼いた素子のせいか、上流（皮質・視床の焼き直し）から多く届くせいかを分ける【み】10-10 15時「いいよ」
型の名前：皮質・視床・海馬・側坐核・上丘
"""
import ast, pathlib, sys

A = '''    out = {}
    for t in sorted(set(PRIOR_TYPE.values())):
'''
B = '''    out = {}
    old = {s for s in os.environ.get("ANAPLEX_PRIOR_V1", "").split(",") if s}   # 比べる用：この型は前の網
    bad = old - set(PRIOR_TYPE.values())
    if bad:
        raise SystemExit(f"⚠️ ANAPLEX_PRIOR_V1 に無い型：{sorted(bad)}（皮質・視床・海馬・側坐核・上丘）")
    if old:
        print(f"焼いた素子：{'・'.join(sorted(old))} は前の素子（prior_v1）、ほかは焼き直した素子", flush=True)
    for t in sorted(set(PRIOR_TYPE.values()) - old):
'''
p = pathlib.Path("torch_brain.py"); s = p.read_text()
if "ANAPLEX_PRIOR_V1" in s:
    sys.exit("もう当ててある")
if s.count(A) != 1:
    sys.exit(f"⚠️ torch_brain.py に印が {s.count(A)} 個（1 個のはず）。何も書いていない")
s = s.replace(A, B); ast.parse(s); p.write_text(s)
print("✅ 当てた。ANAPLEX_PRIOR_V1=海馬,側坐核 python3 measure_check2.py")
