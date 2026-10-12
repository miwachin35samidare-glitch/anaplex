"""版51：自分の声・録音・返し手の声を聞き分けられているか（喃語の段のゴール1）を、2回目に聞く偏りを減らして測る
  （arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51w.py [喃語の回数=50] [録音は少なくとも何回前か=2]
  （コンテナで道具を確かめるとき：python3 measure_v51w.py 4 1 --test ⇒ CPU・相手 200・作り物の声・返し手なし）
ゴール1（論点台帳の追記 10-05）：随伴放電が学べて、自分の声・録音・返し手の声を聞き分けられる【み】
⟲ measure_v51h・v51u は言った直後に同じ音を録音で聞いた（学びながら）⇒ 録音が2回目に聞く分だけ有利（supp を下げる向き）
回し方：言う（写しあり・学ぶ）→ 返し手が返せば聞く ／ 録音を聞く（写しなし・学ぶ）。返し手と録音の順は回ごとにでたらめ
  録音は、2回以上前に言った自分の声から、でたらめに1つ選ぶ（何回前かを決めない）
  ⟲ 初版は「5回前」に決めていた【み】10-07「5回前固定にすると予想できるようになる」
  ⇒ 録音は「少し前の自分の声の録音」。人が自分の声の録音を後で聞くのに近い。⚠️ 一度言った音なので、まったくの初めてではない
⚠️ 模型の耳は録音と全く同じ音を聞く（人は骨からも聞く）⇒ 自分と録音の差は写しと蹴りからしか出ない（【み】10-07）
並べるもの（10回ごと・全部）：聴覚の誤差（音のある歩 1歩あたりの体数）自分 ／ 録音 ／ 返し手、録音÷自分（supp）・返し手÷自分
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
DELAY = int(args[1]) if len(args) > 1 else 2
if TEST:
    B.DEV = "cpu"
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
TAGS = ("自分", "録音", "返し手")
st = {"tag": "他"}; floor = {"v": None}
blk = {k: [0.0, 0] for k in TAGS}; tot = {k: [0.0, 0] for k in TAGS}
orig_step = Brain.step


def step(self, ear, cmd, eye=None, learn=True):
    out = orig_step(self, ear, cmd, eye, learn)
    if st["tag"] in blk and floor["v"] is not None and float(torch.as_tensor(ear).float().sum()) > floor["v"]:
        n = float((out[1] > 0).sum())
        blk[st["tag"]][0] += n; blk[st["tag"]][1] += 1
        tot[st["tag"]][0] += n; tot[st["tag"]][1] += 1
    return out


Brain.step = step


def line(d):
    r = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in d.items()}
    return (f"自分 {r['自分']:.2f} ／ 録音 {r['録音']:.2f} ／ 返し手 {r['返し手']:.2f}"
            f"  ⇒ 録音÷自分 {r['録音']/r['自分']:.2f}  返し手÷自分 {r['返し手']/r['自分']:.2f}"
            f"  （歩 {d['自分'][1]} ／ {d['録音'][1]} ／ {d['返し手'][1]}）")


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if floor["v"] is None:
        floor["v"] = float(ear.sum(1).min()) + 1e-6      # 無音の耳の強さ（measure_v51h と同じ）
    if ear.sum() > 0:
        past.append((cmd, ear))
    def rec():
        if len(past) > DELAY:                              # 2回以上前に言った声から、でたらめに1つ
            c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
            st["tag"] = "録音"; B.run(b, c0, e0, False, True)

    def rep():
        if rp is not None:
            st["tag"] = "返し手"; B.reply(b, rp, rng_r)

    for f in ((rec, rep) if rng_p.random() < 0.5 else (rep, rec)):
        f()
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        print(f"── {i+1}回目まで（この10回）聴覚の誤差（音のある歩 1歩あたり）  " + line(blk), flush=True)
        for v in blk.values():
            v[0] = 0.0; v[1] = 0
print(f"全部（録音は {DELAY}回以上前の自分の声からでたらめ）  " + line(tot), flush=True)
