"""版97 の速さ：続けて流す暮らし（Life.step）の1歩と、網だけ（b.step）の1歩を比べる（arkhe の ~/anaplex で。保存しない）
  python3 measure_life97.py [歩=600]
  （コンテナで道具を確かめるとき：python3 measure_life97.py 40 --test ⇒ CPU・相手 200。時間は読まない）
声は作り物（FakeVoice・FakeReplier。VOICEVOX を待たない）⇒ 網の外の重さ（音の重ね・耳の符号・口の筋と読み）だけを見る
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, life as L
from torch_brain import Brain

if B.VERSION not in (97, 98, 99, 100, 101, 102):
    sys.exit(f"⚠️ babble.py が版97〜100 ではない（版 {B.VERSION}）")
TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 600
if TEST:
    B.DEV = "cpu"
dev = B.DEV
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=200 if TEST else 9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(50):                      # 温める
    life.step()
sync(); t0 = time.time()
for _ in range(NS):
    life.step()
sync(); t_life = (time.time() - t0) / NS * 1000
ear = torch.zeros(B.N_EAR, device=dev); cmd = torch.zeros(life.mo.n_cmd, device=dev)
sync(); t0 = time.time()
for _ in range(NS):
    b.step(ear, cmd, learn=True)
sync(); t_net = (time.time() - t0) / NS * 1000
st = life.stats
print(f"暮らしの1歩 {t_life:.1f}ms ／ 網だけ（無音）の1歩 {t_net:.1f}ms ／ 差 {t_life - t_net:+.1f}ms", flush=True)
print(f"（{life.t}歩・合図 {st['plans']}・声 {st['voices']}・自発 {st['spont']}・返し手 {st['replies']}）", flush=True)
print(f"人の速さ（1歩 {T.DT_MS}ms）に対して {t_life / T.DT_MS:.2f}倍の時間", flush=True)
