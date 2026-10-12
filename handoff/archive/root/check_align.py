"""写しと音の時刻が合っているかを、直に確かめる（arkhe 用）

いまの問題
  写しが「ここで音が来る」と言う歩に、実際の音が来ていないと、
  聴覚は間違った時刻を予測するように学ぶ ⇒ 学ぶほど悪くなる

見るもの
  写しが届いた歩から、実際に耳が立ち上がるまでの差。0 に近ければ合っている
  子音の有無で分けて見る（子音の無いモーラでずれていた）

使い方  python3 check_align.py
"""
import numpy as np
from voice import Voice
from mouth import Mouth

vo, mo = Voice(), Mouth()
diffs, n_cons, n_vowel = [], 0, 0
for i in range(40):
    morae, vol = vo.random_morae(np.random.default_rng(6000 + i))
    if not morae:
        continue
    for m, _, _, _ in morae:
        if vo.inventory[m][0] is None:
            n_vowel += 1
        else:
            n_cons += 1
    mel = vo.say(morae, vol)
    cmd = vo.cmd_stream(morae, vol, len(mel), mo)
    # 耳が「床から立ち上がる」歩
    lit = (mel > vo.floor + 0.5).any(1)
    on_c = np.nonzero(cmd.sum(1))[0]
    on_e = np.nonzero(lit[1:] & ~lit[:-1])[0] + 1
    for t in on_c:
        near = on_e[np.abs(on_e - t) < 30]
        if len(near):
            diffs.append(int(near[np.argmin(np.abs(near - t))] - t))

d = np.array(diffs)
print(f"\n写しが届いた歩 → 最も近い 音の立ち上がり までの差（歩）")
print(f"  件数 {len(d)}  子音ありのモーラ {n_cons} ／ 子音なし {n_vowel}")
print(f"  中央 {np.median(d):+.0f}歩  平均 {d.mean():+.1f}歩  ばらつき {d.std():.1f}歩")
print(f"  |差| <= 1歩 の割合 {float((np.abs(d) <= 1).mean()):.0%}")
print(f"  |差| <= 3歩 の割合 {float((np.abs(d) <= 3).mean()):.0%}")
print(f"\n  ⇒ 合っていれば、中央が 0 付近・ばらつきが小さい")
print(f"     ずれていれば、モーラが進むほど差が広がる（子音なしのモーラで手前にずれる）")
