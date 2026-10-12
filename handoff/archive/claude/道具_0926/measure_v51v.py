"""版51：写し（運動の領野の発火 → 聴覚）が、聴覚の予測に効いているかを分ける（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51v.py [喃語の回数=30]
  （コンテナで道具を確かめるとき：python3 measure_v51v.py 2 --test ⇒ CPU・相手 200・作り物の声）
【測】measure_v51u（10-07）：相手 9100／3000／1000 のどれでも supp 1 前後。自分と録音の誤差が一緒に下がる
過去：版27（09-25）から test_copy で 本物 3.2 ／ でたらめ 3.3 ／ ゼロ 3.1%（写しの中身による予測が消えた。原因は分けていない）
⚠️ measure_v51u・v51h の supp は「言った直後に同じ音を録音として聞く（学びながら）」ので、録音の側が2回目に聞く分だけ有利（supp を下げる向き）
回し方（1回の喃語ごと）
  1 ふつうに言う（口・VOICEVOX・学びながら）⇒ 耳の入力 ear と指令 cmd を得る（この回の誤差は数えない）
  2 同じ ear を、次の4つの条件で聞き直す（どれも学びながら。並びは回ごとにずらして、2回目に聞く有利さを打ち消す）
      本物     蹴りあり・運動の領野に本物の指令（写しは本物）
      でたらめ  蹴りあり・指令の体の並びを入れ替えた指令（時刻は同じ・中身が違う）
      ゼロ     蹴りあり・指令なし
      録音     蹴りなし・指令なし（measure_v51u の録音と同じ）
  3 返し手が返せば聞く
並べるもの（10回ごと・全部）：条件ごとの聴覚の誤差（耳の発火 1本あたり）と、本物に対する比
  写しが効いていれば  本物 ＜ でたらめ ≒ ゼロ。蹴りが効いていれば ゼロ ＜ 録音
⚠️ 人との違い（【み】10-07）：人は自分の声を骨からも聞くので、録音とは音が違う。模型は録音と全く同じ音を聞く
   ⇒ 模型が自分と録音を分けられる手がかりは写しだけ
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 30
if TEST:
    B.DEV = "cpu"
CONDS = ("本物", "でたらめ", "ゼロ", "録音")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
n_cmd = b.n["運動"]
perm = np.random.default_rng(7).permutation(n_cmd)        # でたらめ：指令の体の並びを入れ替える（毎回同じ入れ替え）


def listen(cmd, ear, kick):
    """B.run と同じ回し方。蹴りの有無を別に決められるようにした"""
    n = 0.0
    b.new_utterance(kick)
    for t in range(len(ear)):
        c = torch.tensor(cmd[t], device=B.DEV)
        _, e, _, _ = b.step(torch.tensor(ear[t], device=B.DEV), c, learn=True)
        n += float(e.sum())
    return n


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
tot = np.zeros(len(CONDS)); blk = np.zeros(len(CONDS)); sp_t = sp_b = 0.0
t0 = time.time()
for i in range(N):
    out = say(rng, True)
    ear, cmd = out[0], np.asarray(out[1], np.float32)
    if ear.sum() > 0 and np.any(cmd):
        zero = np.zeros_like(cmd)
        variants = {"本物": (cmd, True), "でたらめ": (cmd[:, perm], True), "ゼロ": (zero, True), "録音": (zero, False)}
        order = [CONDS[(j + i) % 4] for j in range(4)]   # 並びを回ごとにずらす
        for c in order:
            v = listen(variants[c][0], ear, variants[c][1])
            k = CONDS.index(c); tot[k] += v; blk[k] += v
        sp = float(ear.sum()); sp_t += sp; sp_b += sp
    if rp is not None:
        B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        r = blk / max(sp_b, 1)
        print(f"  {i+1:4d}回（{(time.time()-t0)/60:.1f}分）  " + "  ".join(f"{c} {r[k]:.4f}" for k, c in enumerate(CONDS))
              + "  ｜ 本物に対して " + "  ".join(f"{c} {blk[k]/max(blk[0],1e-9):.2f}" for k, c in enumerate(CONDS) if k), flush=True)
        blk[:] = 0; sp_b = 0.0
r = tot / max(sp_t, 1)
print("全部  " + "  ".join(f"{c} {r[k]:.4f}" for k, c in enumerate(CONDS))
      + "  ｜ 本物に対して " + "  ".join(f"{c} {tot[k]/max(tot[0],1e-9):.2f}" for k, c in enumerate(CONDS) if k), flush=True)
print("写しが効いていれば 本物 ＜ でたらめ ≒ ゼロ ／ 版27 の test_copy：本物 3.2 ／ でたらめ 3.3 ／ ゼロ 3.1%", flush=True)
