"""版43b の働きの試し その2（arkhe）：本物の喃語と返し手で見る（覚えは残さない。state.pt は読まない・書かない）
  python3 measure_v43c.py [喃語の回数=100]
見るもの（区間ごとの平均）
  自分の声のあいだ ／ 返事を待つあいだ（返事あり・なし）／ 返事の声のあいだ
  連合・高次・前頭前野・側坐核・上丘の誤差の率、前頭前野の的（下からの入力で撃つ体の数）、来なかった誤差の数
  中脳の核の率、門
⟲ measure_v43b.py は乱数の音型だったので、連合・高次の誤差が1度も出なかった（09-29 arkhe）"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES, N_OMIT
from voice import Voice, Replier
from mouth import Mouth

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
vo, mo = Voice(), Mouth()
rp = Replier(vo)
lo = int(b.off_ch[NAMES.index("前頭前野")]); npf = b.n["前頭前野"]
KEYS = ("連合", "高次", "前頭前野", "側坐核", "上丘")
MBK = ("黒質", "視床下核", "境", "外側手綱核", "吻内側被蓋核")
COLS = [f"{k}誤差" for k in KEYS] + ["前頭前野の的", "来なかった", "門"] + list(MBK)
seg = {}          # 区間の名前 → [合計の列, 歩数]
cur = ["?"]
orig = b.g.step


def hook(x, e, learn):
    out = orig(x, e, learn)
    hook.xt = b.g.last_xt
    return out


b.g.step = hook
ostep = b.step


def step(*a, **k):
    r = ostep(*a, **k)
    v = torch.stack([*(b.up[k].mean() for k in KEYS), hook.xt[lo:lo + npf].sum(), b.omit.sum(), b.gate.reshape(()),
                     *(b.mb.spk[k].mean() for k in MBK)])
    s = seg.setdefault(cur[0], [torch.zeros_like(v), 0])
    s[0] += v; s[1] += 1
    return r


b.step = step
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    morae, vol = vo.random_morae(rng)
    mel = vo.say(morae, vol)
    cmd = vo.cmd_stream(morae, vol, len(mel), mo)
    cur[0] = "自分の声"
    B.run(b, cmd, vo.encode(mel), True, True)
    ans = rp.answer(rng_r)
    if ans is None:
        cur[0] = "返事なし（待つ 94歩）"      # 約1秒。返事が来る時の遅れは REPLY_CUM（秒単位）
        B.run(b, None, vo.encode(np.full((94, mel.shape[1]), vo.floor, np.float32)), False, True)
    else:
        loud = (ans > vo.floor + 1e-6).any(1)
        first = int(np.argmax(loud)) if loud.any() else len(ans)
        cur[0] = "返事あり（待つ）"
        if first:
            B.run(b, None, vo.encode(ans[:first]), False, True)
        cur[0] = "返事の声"
        B.run(b, None, vo.encode(ans[first:]), False, True)
    if (i + 1) % 20 == 0:
        print(f"  喃語 {i+1} 回  {(time.time()-t0)/60:.1f}分", flush=True)
HZ = 1000 / 10.67
print("区間ごとの平均（誤差は率、的・来なかったは体の数、核は Hz）")
print("  " + " ".join(f"{c:>7s}" for c in COLS))
for name, (s, n) in seg.items():
    v = (s / max(n, 1)).cpu().numpy()
    v[-len(MBK):] *= HZ
    print(f"{name}（{n}歩）\n  " + " ".join(f"{x:7.3f}" if j < len(KEYS) else f"{x:7.2f}" for j, x in enumerate(v)))
print(f"VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB")
