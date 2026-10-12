"""起きているときと眠っているときで、1歩の重さ（撃った数・触った辺の数）がどう違うかを測る
（arkhe の ~/anaplex で  python3 measure_load.py ）

中身は何も変えない。0から作った網で、作り物の声（VOICEVOX を使わない）で喃語 → 眠り N3 → REM → 喃語 と回し、
1歩ごとに 時間・周回ごとに触った辺の数・表現の発火の数 を控える（state.pt は読まない・書かない）
  ⚠️ 版38 の state.pt はまだ無い（版37 以前の state は版の番号で弾かれる）ので、0からの網だけで測る
回す時間の目安：1歩 0.3秒として 約15分（喃語5回は約1000歩）
  --k0      驚きを切る（SURPRISE_K ＝ 0。学ぶ速さは常に1倍）
  --wake    喃語だけ（眠りを飛ばす。約5分）
"""
import sys
import time
import numpy as np
import torch

import torch_area as T
import babble as B
import sleep as S
import brainstem as BS
from torch_brain import Brain, NAMES

DEV = "cuda"
if "--k0" in sys.argv:
    BS.SURPRISE_K = 0.0
    print("驚きを切った（SURPRISE_K ＝ 0）", flush=True)
N_UTT, N_SLEEP = 5, 300

prior = T.Prior.load("prior_v1.npz", dev=DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
g = b.g
print(f"体 {b.n_cells:,}  辺 {len(g.w):,}  周回 {g.recur}  送り手 {g.n_send_all:,}  表現 {g.n_ch_all:,}", flush=True)

rec = {"edges": [], "rep": []}
orig_fe, orig_pass = g._fired_edges, g._pass


def fe():
    e = orig_fe()
    rec["edges"].append(len(e))
    return e


def pas(*a, **k):
    out = orig_pass(*a, **k)
    rec["rep"].append(float(out[0].sum()))
    rec["last"] = out[0]
    return out


g._fired_edges, g._pass = fe, pas
# _fired_edges は1周に1回以上呼ばれうる（キャッシュ）ので、周回ごとの辺は _pass の直前の値で数える
steps = []
area_acc = []
etas = []
orig_step = b.step


def step(*a, **k):
    rec["edges"].clear(); rec["rep"].clear()
    torch.cuda.synchronize(); t0 = time.time()
    out = orig_step(*a, **k)
    torch.cuda.synchronize()
    steps.append((time.time() - t0, list(rec["edges"]), list(rec["rep"])))
    eg = g.eta_gain
    etas.append(float(eg) if torch.is_tensor(eg) else eg)
    area_acc.append(torch.stack([rec["last"][int(g.off_ch[i]):int(g.off_ch[i + 1])].mean() for i in range(len(NAMES))]))
    return out


b.step = step
inv = B.make_inventory(np.random.default_rng(0))
rng = np.random.default_rng(0)


def report(name):
    ms = np.array([s[0] for s in steps]) * 1000
    e_first = np.array([s[1][0] if s[1] else 0 for s in steps])
    e_last = np.array([s[1][-1] if s[1] else 0 for s in steps])
    e_sum = np.array([sum(s[1]) for s in steps])
    r = np.array([np.mean(s[2]) if s[2] else 0 for s in steps])
    print(f"{name:10s} {len(steps):5d}歩  1歩 {ms.mean():7.1f} ms（中央 {np.median(ms):6.1f}）  "
          f"触った辺 1周目 {e_first.mean()/1e6:6.2f}百万 ／ 最後の周 {e_last.mean()/1e6:6.2f}百万 ／ 1歩の合計 {e_sum.mean()/1e6:6.1f}百万  "
          f"表現の発火 {r.mean():8.0f}体（{r.mean()/g.n_ch_all:.1%}）", flush=True)
    et = np.array(etas)
    print(f"           学ぶ速さの倍率（驚き）  平均 {et.mean():.1f}  最大 {et.max():.1f}  最初の10歩 {np.round(et[:10], 1).tolist()}", flush=True)
    a = torch.stack(area_acc).mean(0)
    print("           領野ごとの表現の発火（最後の周）  " + "  ".join(f"{nm} {float(v):.1%}" for nm, v in zip(NAMES, a)), flush=True)
    steps.clear(); area_acc.clear(); etas.clear()


def wake(name):
    b.bs.set_state(BS.WAKE)
    for _ in range(N_UTT):
        mel, cmd, _ = B.utter(inv, rng, b.n["運動"])
        B.run(b, cmd, B.encode(mel), True, True)
    report(name)


def sleep_part(name, st):
    cyc = BS.CYCLE_STEPS
    BS.CYCLE_STEPS = 2 * N_SLEEP
    k0 = 0 if st == BS.N3 else N_SLEEP
    orig = b.bs.sleep_state
    b.bs.sleep_state = lambda k: orig(k + k0)
    S.sleep(b, n_step=N_SLEEP, verbose=False)
    b.bs.sleep_state = orig
    BS.CYCLE_STEPS = cyc
    report(name)


wake("喃語（前）")
if "--wake" not in sys.argv:
    sleep_part("眠り N3", BS.N3)
    sleep_part("眠り REM", BS.REM)
    wake("喃語（後）")

print("領野ごとの scale の平均（恒常性）  " + "  ".join(
    f"{nm} {float(g.scale[int(g.off_recv[i]):int(g.off_recv[i + 1])].mean()):.3f}" for i, nm in enumerate(NAMES)), flush=True)
