"""耳のメルの分布を見る（arkhe 用）—— 固定の閾値を決める材料

いまの問題
  閾値を「その発話の最大値の40%」で置いていた ＝ 発話ごとに耳の感度が変わる
  ⇒ 大きい声でも小さい声でも同じに見える（強さの情報が消える）。人の耳はそうではない

前の作りから分かること
  ear.hear は 対数メル（床 ＝ log(1e-6) ＝ -13.8）を、T_MAX=160 歩の固定長で返す
  ⇒ 短い発話は床で埋められる。撃つ歩が少なく見えるのはこのため
  発火率の動作点は 0.45%/歩（前の作りが合わせていた値）

使い方  python3 check_mel.py
"""
import numpy as np
from voice import Voice

vo = Voice()
vals, maxs, per_utt = [], [], []
n_sil = 0
for i in range(30):
    morae, vol = vo.random_morae(np.random.default_rng(5000 + i))
    if not morae:
        n_sil += 1
        continue
    mel = vo.say(morae, vol)
    vals.append(mel.ravel())
    maxs.append(mel.max())
    per_utt.append((len(mel), float((mel > vo.floor + 0.5).mean())))
v = np.concatenate(vals)
print(f"\n[対数メルの分布] 発話 {len(maxs)}回（沈黙 {n_sil}回）  床 {vo.floor:.2f}")
for q in (50, 75, 90, 95, 99, 99.9):
    print(f"  上位 {100-q:5.1f}%  {np.percentile(v, q):7.2f}")
print(f"  最大 {v.max():.2f}  平均 {v.mean():.2f}")
print(f"  床より上の割合 {float((v > vo.floor + 0.5).mean()):.1%}")
print(f"  発話の最大値  中央 {np.median(maxs):.2f}  幅 {np.min(maxs):.2f}〜{np.max(maxs):.2f}")
print(f"    ⇒ 発話ごとに最大値が {np.max(maxs)-np.min(maxs):.1f} 違う。"
      f"その40%を閾値にすると、感度が発話ごとに変わる")

print(f"\n[固定の閾値だと、何%の帯域が越えるか]")
print(f"  （【ヒト】聴覚の皮質はまばら。前の作りの動作点は 0.45%/歩）")
for th in np.arange(-12, 2, 1.0):
    rate = float((v > th).mean())
    print(f"  閾値 {th:6.1f}  越える割合 {rate:7.2%}"
          + ("   ← 0.45% に近い" if 0.002 < rate < 0.01 else ""))

print(f"\n[1回の発話の長さ]")
ls = [p[0] for p in per_utt]
print(f"  歩数 中央 {np.median(ls):.0f}（ear.py が T_MAX=160 歩の固定長で返すため）")
print(f"  そのうち 床より上の歩の割合 中央 {np.median([p[1] for p in per_utt]):.1%}")
print(f"  ⚠️ 短い発話は床で埋められている。実際に音が鳴っているのはその一部")
