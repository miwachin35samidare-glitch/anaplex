"""版50：高次の誤差の一斉発火がどこから来るかを測る（arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v50j.py [喃語の回数=10]
【測】measure_v49i（10-04）：嗅内浅 II のまとまった発火は、1歩前に 高次・前頭前野 の誤差が 2割近く一斉に撃つ歩に重なる
  （ふだんは 0.3〜0.6%）。音にも場面にも揃っていない
誤差の体が撃つ ＝ 的 − 予測 p ＞ 0.5（torch_group2）。的 ＝ 下からの入力 × x_gain ＞ 閾値（TGT_BY_FIRE の領野）
  ⇒ 誤差が一斉に撃つのは「的が一斉に立った（下からの入力が増えた）」か「的はいつもどおりで予測 p が落ちた」のどちらか
下からの入力の道（torch_brain.step）：耳 → 視床 → 聴覚 →（聴覚の誤差）→ 連合 →（連合の誤差）→ 高次 →（高次の誤差）→ 前頭前野
並べるもの（領野 聴覚・連合・高次・前頭前野・運動）
  1 高次の誤差の一斉発火（頭の歩）の前後 −6〜+2 歩の 誤差・的・表現・抑制の体 の割合（どこが何歩先に立つか）
      誤差 ÷ 的 ＝ 立った的のうち予測が当たらなかった割合（1 に近いほど予測が効いていない）
  2 ふだん（一斉発火から 10歩以上離れた歩）の同じ値
  3 一斉発火の頭どうしの間隔（規則的か）・場面ごとの起きた割合
読み方：表現と抑制の体は最後の周の値（torch_group2 が tb に足す発火を、tau 1 の欄から戻す）。誤差は送り出した値（b.up）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain

if B.VERSION != 50:
    sys.exit(f"⚠️ babble.py が版50 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BURST = 0.10        # 高次の誤差がこの割合以上撃った歩を「一斉発火」とする（v49i：撃った歩の1歩前 18.2% ／ ふだん 0.6%）
LAGS = range(-6, 3)
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
AREAS = ("聴覚", "連合", "高次", "前頭前野", "運動")
from torch_brain import NAMES
idx = {nm: NAMES.index(nm) for nm in AREAS}
# ⚠️ torch_group2 の tb の足し込みは cat([rep, rep, inh_spk])（種類ごとに全領野を並べた順）で、
#    tb の行（領野ごとに 誤差・表現・抑制）とは並びが違う。戻した値はその cat の順で読む
NC = int(g.off_ch[-1])
off_inh = np.cumsum([0] + [a.n_inh for a in g.areas])
sl = {}
for nm, i in idx.items():
    a = g.areas[i]
    c, h = int(g.off_ch[i]), NC * 2 + int(off_inh[i])
    sl[nm] = dict(rep=slice(NC + c, NC + c + a.n_ch), inh=slice(h, h + a.n_inh),
                  ch=slice(c, c + a.n_ch), n=a.n_ch, n_inh=a.n_inh)
d0 = g.decay[:, 0].clone()          # tau 1 の欄の減り（受け手ごと）
st = {"phase": "", "t": 0}
rec = []      # 歩ごと {領野: (誤差, 的, 表現, 抑制)}, 場面, 歩
orig_step, orig_new = Brain.step, Brain.new_utterance


def new_utt(self, speaking=True):
    st["t"] = 0
    return orig_new(self, speaking)


def step(self, ear, cmd, eye=None, learn=True):
    tb0 = g.tb[:, 0].clone()
    out = orig_step(self, ear, cmd, eye, learn)
    spk = (g.tb[:, 0] - tb0 * d0) / (1 - d0)       # 最後の周に足された値（cat([rep, rep, inh_spk]) の順）
    xt = g.last_xt
    row = {}
    for nm, s in sl.items():
        row[nm] = (int((self.up[nm] > 0).sum()), int((xt[s["ch"]] > 0.5).sum()),
                   int((spk[s["rep"]] > 0.5).sum()), int((spk[s["inh"]] > 0.5).sum()))
    rec.append((row, st["phase"], st["t"]))
    st["t"] += 1
    return out


Brain.step = step
Brain.new_utterance = new_utt
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
for i in range(N):
    st["phase"] = "話す"; st["t"] = 0; say(rng, True)
    st["phase"] = "聞く"; st["t"] = 0; B.reply(b, rp, rng_r)

L = len(rec)
arr = {nm: np.array([[r[0][nm][k] for k in range(4)] for r in rec], float) for nm in AREAS}
ph = np.array([r[1] for r in rec])
hi = arr["高次"][:, 0] / sl["高次"]["n"]
burst = hi >= BURST
onset = np.where(burst & ~np.r_[False, burst[:-1]])[0]
near = np.zeros(L, bool)
for o in np.where(burst)[0]:
    near[max(0, o - 10):o + 11] = True
calm = ~near
print(f"歩 {L}  高次の誤差 {BURST:.0%} 以上の歩 {burst.mean():.1%}（{burst.sum()}歩）・頭 {len(onset)}回  ふだんの歩 {calm.sum()}")


def fmt(nm, m):
    a = arr[nm][m]
    n, ni = sl[nm]["n"], sl[nm]["n_inh"]
    if len(a) == 0:
        return "―"
    e, t, r, h = a.mean(0)
    ratio = f"{e / t:4.2f}" if t > 0 else " ― "
    return f"誤 {e / n:5.1%} 的 {t / n:5.1%} 誤÷的 {ratio} 表 {r / n:5.1%} 抑 {h / ni:5.1%}"


print("1 一斉発火の頭（0）の前後（頭ごとの平均）")
for nm in AREAS:
    print(f"  {nm}")
    for lag in LAGS:
        m = np.zeros(L, bool)
        k = onset + lag
        m[k[(k >= 0) & (k < L)]] = True
        print(f"    {lag:+d}  {fmt(nm, m)}")
print("2 ふだん（一斉発火から 10歩以上離れた歩）")
for nm in AREAS:
    print(f"  {nm:5s} {fmt(nm, calm)}")
gap = np.diff(onset)
if len(gap):
    cv = gap.std() / gap.mean() if gap.mean() > 0 else float("nan")
    print(f"3 頭どうしの間隔  平均 {gap.mean():.1f}歩  中央 {np.median(gap):.0f}  最短 {gap.min()}  最長 {gap.max()}  変動係数 {cv:.2f}（1 ≒ でたらめ、0 に近いほど規則的）")
    h, e = np.histogram(gap, bins=[1, 5, 10, 20, 30, 40, 60, 100, 10 ** 6])
    print("   " + "  ".join(f"{int(e[j])}〜{int(e[j + 1]) - 1 if e[j + 1] < 10 ** 6 else ''}: {h[j]}" for j in range(len(h))))
for p_ in ("話す", "聞く"):
    m = ph == p_
    if m.any():
        print(f"   {p_}  一斉発火の歩 {burst[m].mean():.1%}（{m.sum()}歩）")
