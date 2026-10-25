"""版98 の1歩の内訳（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_speed98.py [歩=300]
  （コンテナ：python3 measure_speed98.py 20 --test ⇒ CPU・相手 200）
部分ごとに GPU を待って時間を測る（待つぶん全体は少し遅く出る）。どこが版97 から増えたかを見る
"""
import sys, time, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB
from torch_brain import Brain

if B.VERSION != 98:
    sys.exit(f"⚠️ babble.py が版98 ではない（版 {B.VERSION}）")
TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 300
if TEST:
    B.DEV = "cpu"
dev = B.DEV
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=200 if TEST else 9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(50):
    life.step()

acc = collections.defaultdict(float)


def timed(name, f):
    def w(*a, **k):
        sync(); t0 = time.time()
        r = f(*a, **k)
        sync(); acc[name] += time.time() - t0
        return r
    return w


b.g.step = timed("網の領野（Group2）", b.g.step)
b.but.drive = timed("下からの線の表（足す）", b.but.drive)
b.but.learn = timed("下からの線の表（学ぶ）", b.but.learn)
b.bu[TB.PF_KEY].drive = timed("平行線維（足す）", b.bu[TB.PF_KEY].drive)
b._pf_learn = timed("平行線維（学ぶ）", b._pf_learn)
b.but_inh.drive = timed("抑制の線", b.but_inh.drive)
if getattr(b.g, "nucnet", None) is not None:
    b.g.nucnet.learn = timed("核どうしの学び", b.g.nucnet.learn)
step0 = b.step
b.step = timed("1歩ぜんぶ", step0)
sync(); t0 = time.time()
for _ in range(NS):
    life.step()
sync(); tot = (time.time() - t0) / NS * 1000
print(f"暮らしの1歩 {tot:.1f}ms（部分ごとに待つので普段より少し遅い）", flush=True)
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<14s} {v / NS * 1000:7.2f}ms", flush=True)
a = b.areas["顆粒"]
print(f"（顆粒の領野：体 {a.n_cells:,}・平行線維 {len(b.bu[TB.PF_KEY].w):,}本 ／ 版96 の内訳：網 12.56・足す 3.58・学ぶ 0.59・核どうし 0.20・外 1.53ms）", flush=True)
