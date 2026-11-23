"""版67：K_CL（CL → 皮質の表現の体の本数）を振って、皮質の率を測る（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v67d.py [喃語の回数=10] [K_CL の並び=78,60,50,40,30] [K_PM=20]      （コンテナ：python3 measure_v67d.py 1 40 20 --test）
【測】measure_v67c（K_PM 20）：運動の指令の無い体は 黙り・聞く 10.3%・窓 10.0%（的 3.5%）。撃った体の 85% は CL だけで閾値を越えた
【計算】K_CL 78 は「CL だけ・抑制 0 で 3.3Hz」の本数として出した。重さ 1 のまま数えた値で、版67 で入れた対数正規の重さ（中央 0.41・平均 1）を入れ直していない
  対数正規の重さで数え直すと（CL 1歩 10.4%・短期の増減と学びは入れない）：78本 12%／60本 7%／40本 3.3%／30本 2.1%
⇒ 定義どおりの K_CL を網の中で探す（短期の増減・学び・抑制込み）。K ごとに同じ種から網を作り直す
並べるもの（1歩に1回でも撃てば撃った）
  運動：指令の体 ／ 窓の指令の無い体 ／ 黙り・聞く（指令も蹴りも無い歩）／ 準備・蹴り
  聴覚・連合・高次・前頭前野：音のある歩 ／ 無音の歩（耳の入力の和が無音の床より大きいか）
  言えた
"""
import sys, gc, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (67, 68, 69, 70, 71, 72, 73):
    sys.exit(f"⚠️ babble.py が版67〜73 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 10
KS = [int(k) for k in args[1].split(",")] if len(args) > 1 else [78, 60, 50, 40, 30]
TB.K_PM = int(args[2]) if len(args) > 2 else (20 if B.VERSION == 67 else TB.K_PM)   # 版68 は torch_brain の値（27）
if TEST:
    B.DEV = "cpu"
STEP_MS = 10.67
AREAS = ("聴覚", "連合", "高次", "前頭前野")
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)


def hz(p):
    return p / STEP_MS * 1000.0      # 模型の換算（35Hz × 10.67ms ＝ 37%/歩）


print(f"K_PM {TB.K_PM}   的：運動の指令の体 37.3%（35Hz）・指令の無い体 3.5%（3.3Hz）／ 皮質の起きている錐体 3.5%（3.3Hz）")
for K in KS:
    TB.K_CL = K
    torch.manual_seed(0)
    b = Brain(prior, n_partner=200 if TEST else 9100, hippo_scale=4.0 if TEST else 20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    _, say, rp = B.voices(b, TEST)
    g = b.g
    sl = {nm: slice(int(g.off_ch[NAMES.index(nm)]), int(g.off_ch[NAMES.index(nm) + 1])) for nm in AREAS}
    acc = {}
    floor = [float("inf")]
    ostep = b.step

    def add(key, f, m=None):
        a = acc.setdefault(key, [0.0, 0.0])
        if m is None:
            a[0] += float(f.sum()); a[1] += f.numel()
        else:
            a[0] += float(f[m].sum()); a[1] += float(m.sum())

    def step(ear, cmd, *a, **k):
        kick = b.kick_left > 0
        out = ostep(ear, cmd, *a, **k)
        c = torch.as_tensor(cmd).to(b.motor_rep.device).reshape(-1)
        on = c > 0
        f = (b.motor_rep > 0).float()
        if bool(on.any()):
            add("運動・指令の体", f, on); add("運動・窓の無い体", f, ~on)
        elif kick:
            add("運動・準備・蹴り", f)
        else:
            add("運動・黙り・聞く", f)
        es = float(torch.as_tensor(ear).float().sum())       # 音のある歩：耳の和が無音の床（これまでの最小）より大きい（measure_check2 と同じ見方）
        floor[0] = min(floor[0], es)
        snd = "音" if es > floor[0] + 1e-6 else "無音"
        rep = g._any[0]
        for nm, s in sl.items():
            add(f"{nm}・{snd}", (rep[s] > 0).float())
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
    print(f"── K_CL {K}   言えた {ok}/{tried}   CL → 皮質 の重さ 平均 {float(b.cl_w.mean()):.3f}", flush=True)
    for key in ("運動・指令の体", "運動・窓の無い体", "運動・黙り・聞く", "運動・準備・蹴り") + \
               tuple(f"{nm}・{x}" for nm in AREAS for x in ("音", "無音")):
        if key in acc:
            v = acc[key]; r = v[0] / max(v[1], 1)
            print(f"   {key:14s} {r:6.2%}  {hz(r):5.1f}Hz", flush=True)
    del b, g, say, rp, ostep, step, add
    gc.collect()
    if not TEST:
        torch.cuda.empty_cache()
print("読み：運動の指令の無い体（黙り・聞く）が 3.5% 前後になる K が「定義どおりの K_CL」。ほかの領野が人の率からどれだけ離れるかも見る")
