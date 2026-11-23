"""健診（measure_check.py）から、暮らしを人に合わせた健診 measure_check2.py を作る（~/anaplex で。measure_check.py は書き換えない）
  python3 make_check2.py      ⇒ measure_check2.py を書く
  python3 measure_check2.py   （約30分。0から・学びながら・保存しない）
【み】10-09「赤ちゃんが録音を延々と聞かされる状況って現実には無い」「学ばずに聞かせるだけって反復横飛びじゃないの？」→ 10-09「やってみましょう」
  暮らし：喃語ごとに 自分で言う → 返し手が返せば聞く ／ 試し：10回ごとに録音を3つ（2回以上前の自分の声）。学びは止めない（measure_v53f と同じ）
⚠️ ゴール1 の録音の行は 10回ごとに3発話ぶん（揺れる）。全部の行で読む
"""
import pathlib, sys
src = pathlib.Path("measure_check.py").read_text()
old = '''    def rec():
        if len(past) > DELAY:
            c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
            st["tag"] = "録音"; B.run(b, c0, e0, False, True)

    def rep():
        if rp is not None:
            st["tag"] = "返し手"; B.reply(b, rp, rng_r)

    for f in ((rec, rep) if rng_p.random() < 0.5 else (rep, rec)):
        f()
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
'''
new = '''    if rp is not None:                           # 暮らし：返し手が返せば聞く（babble.py と同じ）
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        # 試し：10回ごとに録音を3つ（2回以上前の自分の声からでたらめ）。学びは止めない
        for _ in range(3):
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
'''
if src.count(old) != 1:
    sys.exit(f"⚠️ measure_check.py の暮らしの所が {src.count(old)} 個（1 個のはず）。何も書いていない")
src = src.replace(old, new)
head = '"""健診：目標値をまとめて測る'
if src.count(head) != 1:
    sys.exit("⚠️ measure_check.py の頭書きが見つからない。何も書いていない")
src = src.replace(head, '"""健診2（暮らし＝自分の声＋返し手、録音は10回ごとに3つの試し。make_check2.py が measure_check.py から作った）\n健診：目標値をまとめて測る', 1)
src = src.replace("python3 measure_check.py", "python3 measure_check2.py")
pathlib.Path("measure_check2.py").write_text(src)
print("✅ measure_check2.py を書いた（measure_check.py はそのまま）")
