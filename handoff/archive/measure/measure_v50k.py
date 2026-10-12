"""版50：一斉発火の大元（−2 の聴覚の誤差）が音の立ち上がりで立つかを測る（arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v50k.py [喃語の回数=10]
【測】measure_v50j（10-04）：高次の誤差の一斉発火は 聴覚（−2）→ 連合（−1）→ 高次（0）→ 前頭前野（+1）と1歩ずつ移る
道のり（torch_brain.step）：耳（t）→ 下丘（1歩おく）→ 視床（t＋1）→ 聴覚の入口（t＋2）⇒ 聴覚の誤差 −2 に効くのは 耳 −4・視床 −3
並べるもの
  1 一斉発火の頭（0）の前後 −8〜0 歩の 耳の強さ（ふだんの何倍）・耳が無音か・視床の発火・聴覚の的・聴覚の誤差
  2 音の立ち上がり（無音が 3歩続いたあと音が来た歩）のあと 2〜6 歩に、一斉発火の頭が来る割合（ほかの歩と比べる）
  3 一斉発火の頭のうち、4歩前までに音の立ち上がりがあったものの割合
無音 ＝ 耳の強さがその回で一番小さい値（v49i で 81% の歩が同じ一番下の値だった）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION != 50:
    sys.exit(f"⚠️ babble.py が版50 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BURST = 0.10
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
ia = NAMES.index("聴覚"); c_a = int(g.off_ch[ia]); n_a = g.areas[ia].n_ch
n_th, n_hi = b.n["視床"], b.n["高次"]
st = {"phase": ""}
rec = []      # (場面, 耳の強さ, 視床, 聴覚の的, 聴覚の誤差, 高次の誤差)
orig_step = Brain.step


def step(self, ear, cmd, eye=None, learn=True):
    out = orig_step(self, ear, cmd, eye, learn)
    rec.append((st["phase"], float(torch.as_tensor(ear).float().sum()), int((self.up["視床"] > 0).sum()),
                int((g.last_xt[c_a:c_a + n_a] > 0.5).sum()), int((self.up["聴覚"] > 0).sum()),
                int((self.up["高次"] > 0).sum())))
    return out


Brain.step = step
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
for i in range(N):
    st["phase"] = "話す"; say(rng, True)
    st["phase"] = "聞く"; B.reply(b, rp, rng_r)

L = len(rec)
en = np.array([r[1] for r in rec]); th = np.array([r[2] for r in rec]) / n_th
tg = np.array([r[3] for r in rec]) / n_a; ae = np.array([r[4] for r in rec]) / n_a
hi = np.array([r[5] for r in rec]) / n_hi
silent = en <= en.min() + 1e-6
burst = hi >= BURST
onset = np.where(burst & ~np.r_[False, burst[:-1]])[0]
snd_on = np.array([t >= 3 and not silent[t] and silent[t - 3:t].all() for t in range(L)])
base_en = en[~silent].mean() if (~silent).any() else 1.0
print(f"歩 {L}  無音の歩 {silent.mean():.1%}  一斉発火の頭 {len(onset)}回  音の立ち上がり {snd_on.sum()}回")
print("1 一斉発火の頭（0）の前の歩（頭ごとの平均）")
print("      耳の強さ（音のある歩の平均を 1）  無音の割合  音の立ち上がり  視床    聴覚の的  聴覚の誤差")
for lag in range(-8, 1):
    k = onset + lag; k = k[(k >= 0) & (k < L)]
    if len(k) == 0:
        continue
    print(f"  {lag:+d}   {en[k].mean()/base_en:6.2f}                    {silent[k].mean():6.1%}     {snd_on[k].mean():6.1%}      "
          f"{th[k].mean():6.1%}  {tg[k].mean():6.1%}   {ae[k].mean():6.1%}")
print(f"  ふだん（全歩） 耳 {en.mean()/base_en:.2f}  無音 {silent.mean():.1%}  立ち上がり {snd_on.mean():.1%}  視床 {th.mean():.1%}  的 {tg.mean():.1%}  誤差 {ae.mean():.1%}")
is_head = np.zeros(L, bool); is_head[onset] = True
after = np.zeros(L, bool)
for t in np.where(snd_on)[0]:
    after[t + 2:t + 7] = True
after &= np.arange(L) < L
print("2 頭が来る割合（1歩あたり）")
print(f"   音の立ち上がりの 2〜6歩あと {is_head[after].mean():.2%}（{after.sum()}歩）  ほかの歩 {is_head[~after].mean():.2%}（{(~after).sum()}歩）")
near = np.array([snd_on[max(0, o - 6):o - 1].any() for o in onset]) if len(onset) else np.array([])
print(f"3 頭のうち 2〜6歩前に音の立ち上がりがあったもの {near.mean():.1%}（{int(near.sum())}/{len(onset)}）")
for p_ in ("話す", "聞く"):
    m = np.array([r[0] == p_ for r in rec])
    if m.any():
        print(f"   {p_}  無音 {silent[m].mean():.1%}  頭 {is_head[m].mean():.2%}/歩")
