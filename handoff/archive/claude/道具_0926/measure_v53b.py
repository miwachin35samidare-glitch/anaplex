"""聴覚の誤差の体が、下から越えた音を予測で消しているとき、その予測がどこから来ているか（arkhe の ~/anaplex で。
0から・学びながら・返し手と録音あり・保存しない・網は書き換えない。声と乱数の並びは measure_check と同じ）
  python3 measure_v53b.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v53b.py 4 --test）
【み】10-09「送らずに消しちゃってるってこと？」「調べるのは書いて」
【測】measure_v53a（版53）：聴覚は下から越えた体の約8割で誤差が撃たない（上へ送らない）。自分・録音・返し手で同じくらい
誤差の体（torch_group2._pass）
  p ＝ sigmoid(見込み(履歴 tb) ＋ exc ÷ (1 ＋ ihn))、誤差が撃つ ＝ 的 − p ＞ 0.5（的 ＝ 下から越えたら 1）
  ⇒ 越えた歩に p ≥ 0.5 なら誤差は撃たず、上へは何も送らない（＝ 予測で消した）
  ロジット L ＝ 見込み ＋ Σ 出どころごとの exc ÷ (1 ＋ ihn)。L ≥ 0 が「消した」
聴覚の送り手の出どころ（その行の時間細胞と短い遅れ）
  自分の時間  聴覚の表現の行の時間細胞（自分の過去）
  自分の遅れ  聴覚の表現の行の短い遅れ（1〜2歩前の横）
  写し        運動の領野の表現（1歩前）＝ 指令の写し（随伴放電）
  上から      連合の表現（上からの予測）
並べるもの（10回ごと、自分 ／ 録音 ／ 返し手 に分けて。音のある歩、聴覚の誤差の体が下から越えた歩だけ）
  越えた体の数 ／ 消した割合（L ≥ 0）／ L の内訳の平均（見込み ・ 自分の時間 ・ 自分の遅れ ・ 写し ・ 上から）
  その出どころを抜いたら消えなくなる（L < 0 になる）割合 ＝ その出どころが消しているぶん
読み：人では、自分の声だけを写しで消す（他人の声は消さない）。録音・返し手を消しているのが写し以外なら、そこが外れ
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
DELAY = 2
CATS = ("自分の時間", "自分の遅れ", "写し", "上から")
NC = len(CATS)
TAGS = ("自分", "録音", "返し手")

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev
ia = NAMES.index("聴覚"); a = g.areas[ia]
lo = int(g.off_recv[ia]); nc = a.n_ch
ERR = slice(lo, lo + nc)                       # 聴覚の誤差の体（受け手の行）
CH = slice(int(g.off_ch[ia]), int(g.off_ch[ia]) + nc)
n_cmd, n_top = b.n["運動"], b.n["連合"]
assert a.n_ext == n_cmd + n_top, f"聴覚の ext が {a.n_ext}（写し {n_cmd} ＋ 上から {n_top} のはず）"

# 送り手 → 出どころ（聴覚の送り手だけ。ほかは −1）
send_cat = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
ch_off = np.cumsum([0] + [x.ch.n_cells for x in g.areas])
buf_base = int(ch_off[-1])
buf_off = buf_base + np.cumsum([0] + [x.n_src * x.D for x in g.areas])
per = a.ch.n_cell + a.ch.n_ramp


def row_cat(row, own):
    return torch.where(row < nc, own, torch.where(row < nc + n_cmd, 2, torch.where(row < nc + n_cmd + n_top, 3, -1)))


if per:
    row = torch.arange(a.ch.n_cells, device=dev) // per
    send_cat[ch_off[ia]:ch_off[ia + 1]] = row_cat(row, 0)
row = torch.arange(a.n_src * a.D, device=dev) // a.D
send_cat[buf_off[ia]:buf_off[ia + 1]] = row_cat(row, 1)

st = {"tag": "他"}; floor = {"v": None}
chk = [0, 0]
acc = {t: np.zeros(4 + NC + NC) for t in TAGS}    # 越えた ・ 消した ・ 見込みの和 ・ ihn の和 ・ 出どころごとの和 ・ 抜くと消えなくなる数
acc_all = {t: np.zeros_like(v) for t, v in acc.items()}
orig_pass = G.Group2._pass


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    if not (first and st["tag"] in TAGS and st["sound"]):
        return orig_pass(self, x_all, ext_all, learn, advance, first)
    e = self._fired_edges()
    w0, sc0 = self.w[e].clone(), (self.scale * self.scale_up).clone()
    sci = self.scale.clone()
    prior_l = self.prior.logit(self.tb[ERR])
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    tgt = (self.last_xt[CH] > 0.5)               # 誤差の体の的（下から越えた）
    k = int(tgt.sum())
    if k == 0 or len(e) == 0:
        return out
    r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
    m = (r >= ERR.start) & (r < ERR.stop)
    r, se, inh, w = r[m] - ERR.start, se[m], inh[m], w0[m]
    amp = torch.where(inh, 4.0, self.c_exc[r + ERR.start]) * w
    if self.stp is not None:
        amp = amp * self._amp[e][m]
    amp = amp * self.send_gain[se]
    ihn = torch.zeros(nc, device=self.dev).index_add_(0, r[inh], amp[inh] * sci[r[inh] + ERR.start])
    cat = send_cat[se]
    ok = ~inh & (cat >= 0)
    part = torch.zeros(NC * nc, device=self.dev).index_add_(0, cat[ok] * nc + r[ok], amp[ok] * sc0[r[ok] + ERR.start])
    part = part.reshape(NC, nc) / (1.0 + ihn)
    L = prior_l + part.sum(0)
    killed = (L >= 0) & tgt
    if TEST:   # 確かめ：L < 0 の越えた体 ＝ 実際に誤差が撃った体
        fired = (out[1][CH] > 0) & tgt
        chk[0] += int((fired != (tgt & (L < 0))).sum()); chk[1] += k
    v = acc[st["tag"]]; va = acc_all[st["tag"]]
    add = np.zeros_like(v)
    add[0] = k; add[1] = float(killed.sum()); add[2] = float(prior_l[tgt].sum()); add[3] = float(ihn[tgt].sum())
    add[4:4 + NC] = part[:, tgt].sum(1).cpu().numpy()
    add[4 + NC:] = [float((killed & ((L - part[j]) < 0)).sum()) for j in range(NC)]
    v += add; va += add
    return out


G.Group2._pass = hooked_pass
orig_step = Brain.step


def step(self, ear, cmd, eye=None, learn=True):
    st["sound"] = floor["v"] is not None and float(torch.as_tensor(ear).float().sum()) > floor["v"]
    return orig_step(self, ear, cmd, eye, learn)


Brain.step = step


def show(d, head):
    print(f"   {head}", flush=True)
    for t in TAGS:
        v = d[t]; k = max(v[0], 1)
        print(f"     {t:3s} 越えた {int(v[0]):6d}  消した {v[1]/k:5.1%}  ｜ L の内訳（越えた体の平均）見込み {v[2]/k:+.2f}  ihn {v[3]/k:.2f}  "
              + "  ".join(f"{c} {v[4+j]/k:+.3f}" for j, c in enumerate(CATS))
              + "  ｜ 抜くと消えなくなる（消した体のうち）" + "  ".join(f"{c} {v[4+NC+j]/max(v[1],1):5.1%}" for j, c in enumerate(CATS)), flush=True)


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t_start = time.time()
st["sound"] = False
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if floor["v"] is None:
        floor["v"] = float(ear.sum(1).min()) + 1e-6
    if ear.sum() > 0:
        past.append((cmd, ear))

    def rec():
        if len(past) > DELAY:
            c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
            st["tag"] = "録音"; B.run(b, c0, e0, False, True)

    def rep():
        if rp is not None:
            st["tag"] = "返し手"; B.reply(b, rp, rng_r)

    for f in ((rec, rep) if rng_p.random() < 0.5 else (rep, rec)):
        f()
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        print(f"── {i+1}回目まで（{(time.time()-t_start)/60:.1f}分）", flush=True)
        show(acc, "この10回")
        for v in acc.values():
            v[:] = 0
show(acc_all, "全部")
if TEST:
    print(f"   （確かめ）L から決めた誤差と実際の誤差の食い違い {chk[0]} ／ 越えた {chk[1]}", flush=True)
print("読み：人では自分の声だけを写しで消す。録音・返し手を消しているのが写し以外（自分の時間・遅れ・上から・見込み）なら、そこが外れ", flush=True)
