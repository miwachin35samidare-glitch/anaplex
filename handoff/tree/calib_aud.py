"""聴覚の1本の重さ（ノブ）を決めるために、本物の声で「帯の10本のうち同時に何本撃つか」を数える

聴覚の細胞は自分の帯の視床10本の合計を受け、合計 × 重さ ＞ 閾値 14.3（【ヒト】）で的が立つ。
「m本以上そろったら撃つ」とすると、的の発火率は下の表の値になる。
人の皮質の細胞は 1歩 0.53〜1.80%（中央値 0.97%、【ヒト】Chung 2022）。これに近い m を選ぶ。
重さは 14.3 ÷ (m − 0.5)（m本で越え、m−1本では越えない）。

⚠️ 学ぶ前の視床 ≈ 耳（1本→1体）として、耳で数える。学んだあとの視床は横の繋がりでも撃つので多めになる
使い方: python3 calib_aud.py [発話数=40]
"""
import sys
import numpy as np
import config as C


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    from voice import Voice
    vo = Voice()
    sums = []
    for i in range(n):
        r = np.random.default_rng(50000 + i)
        morae, vol = vo.random_morae(r)
        mel = vo.say(morae, vol)
        ear = vo.encode(mel)                      # [歩, 400]（帯 × 繊維の順）
        n_fib = ear.shape[1] // C.N_MELS
        band = ear.reshape(len(ear), C.N_MELS, n_fib).sum(2)      # [歩, 40]
        sums.append(band)
    s = np.concatenate(sums)
    print(f"{n}発話・{len(s)}歩・{s.shape[1]}帯  耳の1本の発火 {s.mean() / n_fib:.2%}/歩", flush=True)
    print("  m本以上  的の発火率   重さ（14.3 ÷ (m − 0.5)）", flush=True)
    th = 16.0 / 1.12
    for m in range(1, n_fib + 1):
        rate = float((s >= m).mean())
        mark = "  ← 人の範囲" if 0.0053 <= rate <= 0.018 else ""
        print(f"  {m:3d}      {rate:7.2%}     {th / (m - 0.5):6.2f}{mark}", flush=True)


if __name__ == "__main__":
    main()
