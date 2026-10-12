"""焼き直しの材料を録る：各細胞の 撃った（表現 rep）と 細胞体に入力が来た（的）の列（arkhe の ~/anaplex で。
0から・学びながら・保存しない・網は書き換えない。暮らしは 自分の声 ＋ 返し手（録音は流さない）。声と乱数の並びは measure_v53f と同じ）
  python3 rec_streams.py [喃語の回数=50]      ⇒ streams_v53.npz（数MB〜20MB の見込み）を ~/anaplex に書く
  （コンテナで道具を確かめるとき：python3 rec_streams.py 3 --test）
【み】10-10「Aやね」「焼くのは回数の縛りじゃなくて運用する前に焼くってだけ」「焼きはニューロンにする手段であって目的じゃない」
素子の役目（役割3・決まったこと 09-11 のイ）：樹状突起の焼いた網が、自分の細胞体の少し先を当てる
  今の網での入力と出力（torch_group2）：見込み ＝ 焼いた網(tb)、tb ＝ 同じ口の表現の発火の減衰の和（τ 1〜128歩）、当てる相手 ＝ その歩の的
  ⇒ 録るのは 口ごとに 表現の発火 rep_t と 的 tgt_t（1歩ずつ、最後の周の rep・その歩の的）
  焼くとき：rep の t−1 までの減衰の和 → tgt_t を当てるように焼く（コンテナで）
⚠️ 今の焼いた網（prior_v1）で育った網の列。焼き直したら、もう一度録って焼いてよい（【み】10-10）
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (53,):
    sys.exit(f"⚠️ babble.py が版53 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
off_ch = [int(v) for v in g.off_ch]
rec = {"rep": [], "tgt": [], "tag": []}
st = {"tag": 0}                    # 0 自分 ／ 1 返し手
orig_pass = G.Group2._pass


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    if advance:
        rec["rep"].append(np.packbits((out[0] > 0).cpu().numpy()))
        rec["tgt"].append(np.packbits((self.last_xt > 0.5).cpu().numpy()))
        rec["tag"].append(st["tag"])
    return out


G.Group2._pass = hooked_pass
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    st["tag"] = 0
    say(rng, True)
    if rp is not None:
        st["tag"] = 1; B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        print(f"── {i+1}回目まで（{(time.time()-t0)/60:.1f}分）歩 {len(rec['tag'])}", flush=True)
n_ch = int(g.n_ch_all)
R = np.stack(rec["rep"]); Tg = np.stack(rec["tgt"])
out = "streams_v53.npz"
np.savez_compressed(out, rep=R, tgt=Tg, tag=np.array(rec["tag"], np.uint8), n_ch=n_ch,
                    names=np.array(NAMES), off_ch=np.array(off_ch + [n_ch]),
                    taus=T.TAUS.cpu().numpy(), dt_ms=T.DT_MS)
rep = np.unpackbits(R, axis=1)[:, :n_ch]; tg = np.unpackbits(Tg, axis=1)[:, :n_ch]
print(f"書いた：{out}（歩 {R.shape[0]}・口 {n_ch}）", flush=True)
print("   領野      表現の率   的の率", flush=True)
for j, nm in enumerate(NAMES):
    s = slice(off_ch[j], off_ch[j + 1] if j + 1 < len(off_ch) else n_ch)
    print(f"   {nm:6s}  {rep[:, s].mean():7.2%}  {tg[:, s].mean():7.2%}", flush=True)
