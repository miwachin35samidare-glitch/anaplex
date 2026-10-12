"""発火を「下からの入力だけで越えた歩」と「横の繋がり（興奮）が要った歩」に分ける（学ばずに測る）

出すもの（領野ごと）
  学んだあいだの発火率（誤差・表現・抑制の種類別、恒常性が見ていた移動平均）
  自分で話す発話を流したときの
    下から越え   x × 重さ ＞ 閾値 になった歩の割合（下からの入力だけで撃てる歩）
    表現の発火   表現のニューロンが撃った歩の割合
    うち下から   表現の発火のうち、下から越えの歩だった割合（残りは横の繋がり・自発・補完で撃った）
    誤差の発火   誤差のニューロンが撃った歩の割合
使い方: python3 measure_fire.py [回数=10]
"""
import sys
import numpy as np
import torch
import torch_area as T
import babble as B
from torch_brain import Brain, NAMES


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
    b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
    done, _ = B.load(b)
    g = b.g
    print(f"学んだ網 喃語 {done} 回（版 {B.VERSION}）", flush=True)

    # ── 学んだあいだの発火率（種類別） ──
    print("\n[学んだあいだの発火率（移動平均）]  人は 0.45%", flush=True)
    print(f"  {'領野':6s} {'誤差':>8s} {'表現':>8s} {'抑制':>8s}", flush=True)
    for i, nm in enumerate(NAMES):
        m = g.area_of == i
        row = []
        for mask in (g.is_err, g.is_rep, g.is_inh_r):
            sel = m & mask
            row.append(float(g.rate_ema[sel].mean()) if bool(sel.any()) else float("nan"))
        print(f"  {nm:6s} {row[0]:8.2%} {row[1]:8.2%} {row[2]:8.2%}", flush=True)

    # ── 発話を流して、下から越えた歩と表現・誤差の発火を数える ──
    cap = {}
    orig = g.step

    def hooked(x_all, ext_all, learn=True):
        out = orig(x_all, ext_all, learn)
        cap["xa0"] = (x_all * g.x_gain > g.fire_th).float()          # 抑制なしなら越え
        cap["xa"] = g.last_xt if hasattr(g, "last_xt") else cap["xa0"]  # 的（抑制込み）
        cap["rep"], cap["err"] = out[0], out[1]
        return out
    g.step = hooked

    from voice import Voice
    from mouth import Mouth
    vo, mo = Voice(), Mouth()
    N = len(NAMES)
    s_xa, s_rep, s_rx, s_err, s_xa0 = (np.zeros(N) for _ in range(5))
    steps = 0
    for k in range(n):
        r = np.random.default_rng(40000 + k)
        morae, vol = vo.random_morae(r)
        mel = vo.say(morae, vol)
        cmd = vo.cmd_stream(morae, vol, len(mel), mo)
        ear = vo.encode(mel)
        b.new_utterance(bool(np.any(cmd)))
        for t in range(len(ear)):
            b.step(torch.tensor(ear[t], device=B.DEV), torch.tensor(cmd[t], device=B.DEV), learn=False)
            xa, rep, err, xa0 = cap["xa"], cap["rep"], cap["err"], cap["xa0"]
            for i, nm in enumerate(NAMES):
                lo, hi = int(b.off_ch[i]), int(b.off_ch[i]) + b.n[nm]
                s_xa[i] += float(xa[lo:hi].sum())
                s_rep[i] += float(rep[lo:hi].sum())
                s_rx[i] += float((rep[lo:hi] * xa[lo:hi]).sum())
                s_err[i] += float(err[lo:hi].sum())
                s_xa0[i] += float(xa0[lo:hi].sum())
            steps += 1

    print(f"\n[自分で話す {n}発話・{steps}歩を流したとき]（体あたり・歩あたりの割合）", flush=True)
    print(f"  {'領野':6s} {'抑制なし越え':>10s} {'的（抑制込）':>10s} {'表現の発火':>10s} {'うち的':>10s} {'誤差の発火':>10s}", flush=True)
    for i, nm in enumerate(NAMES):
        d = steps * b.n[nm]
        frac = s_rx[i] / s_rep[i] if s_rep[i] else float("nan")
        print(f"  {nm:6s} {s_xa0[i] / d:10.2%} {s_xa[i] / d:10.2%} {s_rep[i] / d:10.2%} {frac:10.1%} {s_err[i] / d:10.2%}", flush=True)


if __name__ == "__main__":
    main()
