"""耳と写しの中身を、作り物と VOICEVOX で直に比べる（arkhe 用）—— 回さずに見る

見るもの
  1 耳の発火    1歩あたり何体。撃つ歩の割合。同じ音を2回聞いたとき同じか
  2 写し       何歩に届くか。音との時間差
  3 予測できるか  同じ写しが来たとき、同じ音が返るか（違えば学びようがない）
  4 揃い方     写しが届いた歩から何歩後に音が来るか。そのばらつき
              ⚠️ ばらつくと、随伴放電が当てられない

使い方  python3 check_pair.py
"""
import numpy as np
import babble as B
from voice import Voice
from mouth import Mouth

vo, mo = Voice(), Mouth()
inv = B.make_inventory(np.random.default_rng(0))


def pairs(kind, n=30):
    out = []
    for i in range(n):
        r = np.random.default_rng(3000 + i)
        if kind == "作り物":
            mel, cmd, _ = B.utter(inv, r, 20)
            ear = B.encode(mel)
        else:
            morae, vol = vo.random_morae(r)
            mel = vo.say(morae, vol)
            cmd = vo.cmd_stream(morae, vol, len(mel), mo)
            ear = vo.encode(mel)
        out.append((ear, cmd))
    return out


for kind in ("作り物", "VOICEVOX"):
    ps = pairs(kind)
    ear_rate = np.mean([e.sum() / len(e) for e, _ in ps])
    ear_steps = np.mean([(e.sum(1) > 0).mean() for e, _ in ps])
    cmd_steps = np.mean([(c.sum(1) > 0).mean() for _, c in ps])
    lags = []
    for e, c in ps:
        on_c = np.nonzero(c.sum(1))[0]
        on_e = np.nonzero(e.sum(1))[0]
        for t in on_c:
            after = on_e[on_e >= t]
            if len(after):
                lags.append(after[0] - t)
    print(f"\n[{kind}]")
    print(f"  耳の発火 1歩あたり {ear_rate:5.2f}体  撃つ歩 {ear_steps:.1%}")
    print(f"  写しが届く歩 {cmd_steps:.1%}")
    if lags:
        print(f"  写し → 音 の遅れ  中央 {np.median(lags):.0f}歩  "
              f"幅 {np.percentile(lags,10):.0f}〜{np.percentile(lags,90):.0f}歩  "
              f"ばらつき {np.std(lags):.1f}歩")
        print(f"    ⚠️ ばらつきが大きいと、写しから『いつ来るか』を当てられない")

# 3 予測できるか：同じ指令で同じ耳の発火になるか
print("\n[同じ指令で同じ耳の発火になるか]")
for kind in ("作り物", "VOICEVOX"):
    same = []
    for i in range(10):
        r1 = np.random.default_rng(4000 + i)
        r2 = np.random.default_rng(4000 + i)
        if kind == "作り物":
            m1, _, _ = B.utter(inv, r1, 20); m2, _, _ = B.utter(inv, r2, 20)
            e1, e2 = B.encode(m1), B.encode(m2)
        else:
            mo1, v1 = vo.random_morae(r1); mo2, v2 = vo.random_morae(r2)
            e1, e2 = vo.encode(vo.say(mo1, v1)), vo.encode(vo.say(mo2, v2))
        same.append(e1.shape == e2.shape and np.array_equal(e1, e2))
    print(f"  {kind:10s} 10回中 {sum(same)}回 完全に一致")
