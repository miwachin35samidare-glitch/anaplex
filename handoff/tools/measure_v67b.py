"""版67：K_PM（運動前野の送り手の本数）を振って、運動の表現の体の率を測る（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v67b.py [喃語の回数=10] [K_PM の並び=115,80,60,40,30,20,10,0]      （コンテナ：python3 measure_v67b.py 2 40,0 --test）
【み】10-12「a」：的はそのまま（指令の体 35Hz・指令の無い体 3.3Hz）、K_PM を今の抑制（ihn 約0.12）と CL を入れた形で決め直す
  K_PM の定義（torch_brain の注）：指令の体が【ヒト】話すあいだの 35Hz で撃つ本数 ⇒ 指令の体の率が 1歩 37%（35 × 10.67ms）になる K を探す
並べるもの（K ごとに同じ種から網を作り直す。1歩に1回でも撃てば撃った ＝ b.motor_rep）
  指令の体（語の指令の窓で cmd > 0 の体）／ 指令の窓の指令の無い体 ／ 指令の無い歩の全部の体   の 1歩の率と Hz
  言えた（選んだとおりに言えた ／ 声を出そうとした）
⚠️ 網は学びながら回るので、K ごとの差には学びの違いも入る（同じ種・同じ声の順）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (67,):
    sys.exit(f"⚠️ babble.py が版67 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 10
KS = [int(k) for k in args[1].split(",")] if len(args) > 1 else [115, 80, 60, 40, 30, 20, 10, 0]
if TEST:
    B.DEV = "cpu"
STEP_MS = 10.67
ON_P = TB.R_PM_ON            # 的：指令の体 35Hz の 1歩の確率
OFF_P = TB.R_PM_OFF          # 的：指令の無い体 3.3Hz
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)


def hz(p):
    return p / STEP_MS * 1000.0      # 模型の換算（35Hz × 10.67ms ＝ 37%/歩）に合わせる


print(f"的：指令の体 {ON_P:.1%}/歩（35Hz）・指令の無い体 {OFF_P:.1%}/歩（3.3Hz）")
print("  K_PM   指令の体          窓の指令の無い体   指令の無い歩       言えた")
for K in KS:
    TB.K_PM = K
    torch.manual_seed(0)
    b = Brain(prior, n_partner=200 if TEST else 9100, hippo_scale=4.0 if TEST else 20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    _, say, rp = B.voices(b, TEST)
    acc = dict(on=[0.0, 0.0], offw=[0.0, 0.0], quiet=[0.0, 0.0])
    ostep = b.step

    def step(ear, cmd, *a, **k):
        out = ostep(ear, cmd, *a, **k)
        c = torch.as_tensor(cmd).to(b.motor_rep.device).reshape(-1)
        f = (b.motor_rep > 0).float()
        if float(c.sum()) > 0:
            on = c > 0
            acc["on"][0] += float(f[on].sum()); acc["on"][1] += float(on.sum())
            acc["offw"][0] += float(f[~on].sum()); acc["offw"][1] += float((~on).sum())
        else:
            acc["quiet"][0] += float(f.sum()); acc["quiet"][1] += f.numel()
        return out

    b.step = step
    rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
    ok = tried = 0
    for i in range(N):
        ear, cmd, n, want, got = say(rng, True)
        if want:
            tried += 1; ok += int(want == got)
        if rp is not None:
            B.reply(b, rp, rng_r)
    r = {k: v[0] / max(v[1], 1) for k, v in acc.items()}
    print(f"  {K:4d}   {r['on']:6.1%} {hz(r['on']):6.1f}Hz   {r['offw']:6.1%} {hz(r['offw']):6.1f}Hz   "
          f"{r['quiet']:6.1%} {hz(r['quiet']):6.1f}Hz   {ok:2d}/{tried:<3d}", flush=True)
    del b
    if not TEST:
        torch.cuda.empty_cache()
print("読み：指令の体の率が 37% 前後になる K が「定義どおりの K_PM」。指令の無い体は CL（K_CL）と運動前野の 3.3Hz の送り手で決まる")
