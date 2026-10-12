"""measure_v53d と同じ測り。ただし暮らしを人に合わせる（録音は毎回ではなく、区切りごとの短い試し）
【み】10-09「録音が自分と全く同じ声」「赤ちゃんが録音を延々と聞かされる状況って現実には無い」「学ばずに聞かせるだけって反復横飛びじゃないの？」
⟲ measure_v51h（10-05）〜 measure_check・v51u・v51w・v53a〜d は、喃語ごとに録音を1つ聞かせていた（自分の声と同じだけ）
⟲ v53e（録音を学ばない）は取り下げ：10-03【み】「リアルタイム学習前提のモデルを学びなしで見る意味は何？」⇒ 人の可塑性は止まらない
暮らし  喃語ごとに 自分で言う → 返し手が返せば聞く（babble.py・day.py と同じ）
試し    10回ごとに、2回以上前の自分の声の録音を 3つ流す。学びは止めない（人が実験の場で録音を聞くのと同じ）
  python3 measure_v53f.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v53f.py 10 --test）
⚠️ 録音の行は 10回ごとに 3発話ぶんだけ（数が少ないので揺れる）

（以下は measure_v53d の説明）
写しが自分の声を当てられない理由を分ける（arkhe の ~/anaplex で。0から・学びながら・返し手と録音あり・保存しない・網は書き換えない。
声と乱数の並びは measure_v53b・c と同じ）
  python3 measure_v53d.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v53d.py 3 --test）
【測】measure_v53c（版53）：写しだけ残すと なお消えるのは 自分 21.7% ／ 録音 28.2%。写しが足すのは 自分 +6.4 ／ 録音 +4.9 点で、
  自分の声を選んでいない。越えた数で分けても、どの区分でも 録音の方が多く消える
⇒ 写しに「何が聞こえるか」の中身が来ていないのか（中身）、来ているのに重みが育たないのか（学び）を分ける
並べるもの（10回ごと、自分 ／ 録音 ／ 返し手。音のある歩だけ。聴覚の誤差の体について）
  1 見分け：出どころごとの寄与（exc ÷ (1＋ihn)）を、その歩に越えた体 と 越えなかった体 で平均し、差をとる
      差が大きい ＝ その出どころは「どの体が越えるか」を当てている
      人の形なら：自分の声のとき 写しの差が大きく、録音・返し手のとき 写しの差は 0 近く
  2 重み：聴覚の誤差の体へ入る興奮の辺の重みの平均（出どころごと、10回ごとの終わりに）。写しの重みが育っているか
読み：写しの差が 自分の声でも 0 近く ⇒ 中身が来ていない（運動の領野の発火が音と揃っていない）。
      差はあるのに重みが小さい ⇒ 学びで育たない（ほかの出どころに先に当てられている）
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
N_PROBE = 3          # 10回ごとの録音の試しの数【C の置き方。み 10-09「やってみましょう」】
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
ERR = slice(lo, lo + nc)
CH = slice(int(g.off_ch[ia]), int(g.off_ch[ia]) + nc)
n_cmd, n_top = b.n["運動"], b.n["連合"]
assert a.n_ext == n_cmd + n_top, f"聴覚の ext が {a.n_ext}（写し {n_cmd} ＋ 上から {n_top} のはず）"

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
mk = lambda: np.zeros(3 + 2 * NC)        # 越えた体の数 ・ 越えなかった体の数 ・ 出どころごとの寄与の和（越えた）・（越えなかった）・ 消した数
acc = {t: mk() for t in TAGS}
acc_all = {t: mk() for t in TAGS}
orig_pass = G.Group2._pass


def weights():
    """聴覚の誤差の体へ入る興奮の辺の重みの平均（出どころごと）"""
    r = g.e_recv.long(); m = (r >= ERR.start) & (r < ERR.stop) & ~g.e_inh
    c = send_cat[g.e_src[m].long()]; w = g.w[m]
    return [float(w[c == j].mean()) if int((c == j).sum()) else float("nan") for j in range(NC)], \
           [int((c == j).sum()) for j in range(NC)]


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    if not (first and st["tag"] in TAGS and st["sound"]):
        return orig_pass(self, x_all, ext_all, learn, advance, first)
    e = self._fired_edges()
    w0, sc0 = self.w[e].clone(), (self.scale * self.scale_up).clone()
    sci = self.scale.clone()
    prior_l = self.prior.logit(self.tb[ERR])
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    tgt = (self.last_xt[CH] > 0.5)
    k = int(tgt.sum())
    if k == 0 or k == nc or len(e) == 0:
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
    add = mk()
    add[0] = k; add[1] = nc - k
    add[2:2 + NC] = part[:, tgt].sum(1).cpu().numpy()
    add[2 + NC:2 + 2 * NC] = part[:, ~tgt].sum(1).cpu().numpy()
    add[2 + 2 * NC] = float(((prior_l + part.sum(0) >= 0) & tgt).sum())   # 予測で消した（v53b と同じ L）
    acc[st["tag"]] += add; acc_all[st["tag"]] += add
    return out


G.Group2._pass = hooked_pass
orig_step = Brain.step


def step(self, ear, cmd, eye=None, learn=True):
    st["sound"] = floor["v"] is not None and float(torch.as_tensor(ear).float().sum()) > floor["v"]
    return orig_step(self, ear, cmd, eye, learn)


Brain.step = step


def show(d, head):
    print(f"   {head}  見分け ＝ 越えた体の平均 − 越えなかった体の平均（寄与 exc÷(1＋ihn)）", flush=True)
    for t in TAGS:
        v = d[t]; k1 = max(v[0], 1); k0 = max(v[1], 1)
        s = "  ".join(f"{c} {v[2+j]/k1 - v[2+NC+j]/k0:+.3f}（{v[2+j]/k1:.2f}／{v[2+NC+j]/k0:.2f}）" for j, c in enumerate(CATS))
        print(f"     {t:3s} 越えた {int(v[0]):7d}  消した {v[2+2*NC]/k1:5.1%}  ｜ {s}", flush=True)


def show_w():
    mw, nw = weights()
    print("   重みの平均（辺の数）" + "  ".join(f"{c} {mw[j]:.4f}（{nw[j]}）" for j, c in enumerate(CATS)), flush=True)


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t_start = time.time()
st["sound"] = False
print("始め", flush=True); show_w()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if floor["v"] is None:
        floor["v"] = float(ear.sum(1).min()) + 1e-6
    if ear.sum() > 0:
        past.append((cmd, ear))

    if rp is not None:                           # 暮らし：返し手が返せば聞く（babble.py と同じ）
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        # 試し：10回ごとに録音を N_PROBE 個（2回以上前の自分の声からでたらめ）。学びは止めない
        for _ in range(N_PROBE):
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
        print(f"── {i+1}回目まで（{(time.time()-t_start)/60:.1f}分）", flush=True)
        show(acc, "この10回")
        show_w()
        for v in acc.values():
            v[:] = 0
show(acc_all, "全部")
print("読み：消した割合が 自分 ＞ 録音・返し手 なら人の向き（自分の声を選んで消す）。写しの見分けが 自分の声でも 0 近く ⇒ 写しに中身が来ていない。見分けはあるのに重みが育たない ⇒ 学びでほかに当てられている", flush=True)
