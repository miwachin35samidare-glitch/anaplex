"""1日を回す（arkhe 用）—— 起きて喃語、眠る、をくり返す

人に合わせた量（【ヒト】6〜9か月。喃語が出るころ）
  発声        4〜5回／分（起きているあいだ。類人猿の3〜5倍）
  起きている長さ 2.5〜3.5時間 ⇒ 1回の覚醒で 675〜945回。ここでは 800回
  眠る        1回 1〜1.5時間。2〜3回の昼寝 ＋ 夜
  ⚠️ 眠る長さは人の1.5時間を移さない。海馬から皮質へ渡り終えるまでが何歩かを測って決める
     いまは長めに置いて、前後の測りで変化を見る

使い方
  python3 day.py 3              起きて喃語800回 → 眠る、を3回くり返す
  python3 day.py 3 --wake 400   1回の覚醒を400回にする
  python3 day.py 1 --sleep 8000 眠りを8000歩にする
  python3 day.py 0              回さずに測るだけ
  ⚠️ 途中で止めても state.pt から続けられる
"""
import sys
import time
import numpy as np
import torch

import torch_area as T
import babble as B
import sleep as S
import brainstem as BS
from torch_brain import Brain

DEV = "cuda"
WAKE = 800          # 1回の覚醒で言う回数（【ヒト】2.5〜3.5時間 × 4.5回/分）
SLEEP = BS.SLEEP_STEPS   # 1回の眠り（版38）。起きる:眠る を【ヒト】乳児の約2.5:1 に。N3 と REM をくり返す


def arg(name, default):
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    n_cycle = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
    wake = arg("--wake", None)                # 付けなければ、眠りの圧が 1 に届くまで起きている（版39c）
    n_sleep = arg("--sleep", None)            # 付けなければ、脳幹の年齢から（版39c）
    fake = "--fake" in sys.argv

    prior = T.Prior.load("prior_v1.npz", dev=DEV)
    b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
    done, log = (0, []) if "--fresh" in sys.argv else B.load(b)
    print(f"体 {b.n_cells:,}  辺 {len(b.g.w):,}  これまで喃語 {done} 回（版 {B.VERSION}）", flush=True)
    for e in log[-3:]:
        print(f"  これまで: {e}", flush=True)

    rng = np.random.default_rng(0)
    draw, say, rp = B.voices(b, fake)        # 版43d：口が運動の領野を読む（babble.own_voice）
    for _ in range(done):                     # 声の乱数を進めて、同じ列を繰り返さない
        draw(rng)

    b.bs.set_state(BS.WAKE)                   # 起きている（版38：恒常性・上からの戻りは脳幹が決める）
    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（版40）

    for cyc in range(n_cycle):
        # ── 起きている ──
        t0 = time.time()
        i = 0
        while (i < wake) if wake is not None else not b.bs.sleep_due:
            say(rng, True)
            if rp is not None:
                B.reply(b, rp, rng_r)
            done += 1
            i += 1
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  [{cyc+1}/{n_cycle}] 起きて {i}回  門 {b.bs.gate_mean():.3f}  経過 {el/60:.1f}分  "
                      f"眠りの圧 {b.bs.pressure:.2f}  覚醒 {float(b.bs.arousal):.2f}  疲れ {b.bs.fatigue:.2f}  "
                      f"年齢 {b.bs.age_days/30.44:.2f}か月", flush=True)
        n_wake = i
        ns = n_sleep if n_sleep is not None else b.bs.sleep_steps()
        o, r, s, ok, tr = B.measure(b, say)
        print(f"  [{cyc+1}] 眠る前  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  言えた {ok}/{tr}", flush=True)
        B.save(b, done, log=log)

        # ── 眠る ──
        fat = b.bs.fatigue                        # 起きると 0 に戻るので、眠る前に控える
        print(f"  [{cyc+1}] 眠る（{ns}歩 ＝ {ns*10.67/1000/60:.1f}分ぶん）  起きていた {n_wake}回  "
              f"眠りの圧 {b.bs.pressure:.2f}  疲れ {b.bs.fatigue:.2f}", flush=True)
        S.sleep(b, n_step=ns, report_every=max(1, ns // 4))
        o2, r2, s2, ok2, tr2 = B.measure(b, say)
        print(f"  [{cyc+1}] 起きたあと 自分 {o2:.1%} ／ 録音 {r2:.1%}  supp {s2:.2f}  言えた {ok2}/{tr2}"
              f"  （眠る前 {s:.2f} → {s2:.2f}）", flush=True)
        log.append(f"{time.strftime('%m-%d %H:%M')} 喃語 {done}回（+{n_wake}）"
                   f" 眠り {b.bs.last_slept}/{ns}歩  疲れ {fat:.2f}  supp {s:.2f} → {s2:.2f}"
                   f" 声={'作り物' if fake else 'VOICEVOX'}")
        B.save(b, done, log=log)

    if n_cycle == 0:
        o, r, s, ok, tr = B.measure(b, say)
        print(f"喃語 {done} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  言えた {ok}/{tr}", flush=True)


if __name__ == "__main__":
    main()
