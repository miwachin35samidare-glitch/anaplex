"""聴覚の誤差の体の「見込み」の床：焼いた素子が出す p と、実際に的が立つ率を比べる（arkhe の ~/anaplex で。
0から・学びながら・保存しない・網は書き換えない。暮らしと試しは measure_v53f・g と同じ）
  python3 measure_v53h.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v53h.py 10 --test）
【み】10-10「測るのと、人がどうか調べるのをやって」
【測】measure_v53g：学ぶ辺が撃った歩の p は 0.20〜0.26（逆算）、的が立つのは 1〜4%。どの出どころも押しは −
式（torch_group2）：p ＝ sigmoid(見込み ＋ exc ÷ (1＋ihn))、exc ≥ 0 ⇒ p は sigmoid(見込み) より下がれない
並べるもの（聴覚の誤差の体 × 歩。声ごとに 音のある歩 ／ 音の無い歩 で分ける）
  的の率         的が立った割合（体×歩あたり）
  見込みの p     sigmoid(見込み) の平均と 10・50・90% 点
  全体の p       sigmoid(見込み ＋ exc ÷ (1＋ihn)) の平均
  床が上         見込みの p が その組の的の率より上だった体×歩の割合
  的あり・なし   的が立った体と立たなかった体で、見込みの p の平均
読み：音の無い歩でも見込みの p が的の率より大きく上なら、焼いた素子の床がいつも的より上（学ぶ辺はどれも下げられる）
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
BINS = np.linspace(0.0, 1.0, 101)
KEYS = [(t, snd) for t in TAGS for snd in (True, False)]
mk = lambda: {k: {"n": 0, "tgt": 0.0, "pp": 0.0, "pa": 0.0, "above": 0.0, "pp1": 0.0, "pp0": 0.0,
                  "hist": np.zeros(len(BINS) - 1)} for k in KEYS}
acc = mk(); acc_all = mk()
orig_pass = G.Group2._pass


def weights():
    """聴覚の誤差の体へ入る興奮の辺の重みの平均（出どころごと）"""
    r = g.e_recv.long(); m = (r >= ERR.start) & (r < ERR.stop) & ~g.e_inh
    c = send_cat[g.e_src[m].long()]; w = g.w[m]
    return [float(w[c == j].mean()) if int((c == j).sum()) else float("nan") for j in range(NC)], \
           [int((c == j).sum()) for j in range(NC)]


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    if not (first and st["tag"] in TAGS):
        return orig_pass(self, x_all, ext_all, learn, advance, first)
    e = self._fired_edges()
    w0, sc0 = self.w[e].clone(), (self.scale * self.scale_up).clone()
    sci = self.scale.clone()
    prior_l = self.prior.logit(self.tb[ERR])
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    tgt = (self.last_xt[CH] > 0.5).float()
    exc = torch.zeros(nc, device=self.dev)
    ihn = torch.zeros(nc, device=self.dev)
    if len(e):
        r = self.e_recv[e].long(); m = (r >= ERR.start) & (r < ERR.stop)
        if bool(m.any()):
            r, se, inh, w = r[m] - ERR.start, self.e_src[e][m].long(), self.e_inh[e][m], w0[m]
            amp = torch.where(inh, 4.0, self.c_exc[r + ERR.start]) * w
            if self.stp is not None:
                amp = amp * self._amp[e][m]
            amp = amp * self.send_gain[se]
            ihn.index_add_(0, r[inh], amp[inh] * sci[r[inh] + ERR.start])
            exc.index_add_(0, r[~inh], amp[~inh] * sc0[r[~inh] + ERR.start])
    pp = torch.sigmoid(prior_l); pa = torch.sigmoid(prior_l + exc / (1.0 + ihn))
    rate = float(tgt.mean())
    for d in (acc, acc_all):
        v = d[(st["tag"], st["sound"])]
        v["n"] += nc; v["tgt"] += float(tgt.sum()); v["pp"] += float(pp.sum()); v["pa"] += float(pa.sum())
        v["above"] += float((pp > max(rate, 1e-9)).sum())
        v["pp1"] += float((pp * tgt).sum()); v["pp0"] += float((pp * (1 - tgt)).sum())
        v["hist"] += np.histogram(pp.cpu().numpy(), BINS)[0]
    return out


def edges_per_cat():
    r = g.e_recv.long(); m = (r >= ERR.start) & (r < ERR.stop) & ~g.e_inh
    c = send_cat[g.e_src[m].long()]
    return [max(int((c == j).sum()), 1) for j in range(NC)]


G.Group2._pass = hooked_pass
orig_step = Brain.step


def step(self, ear, cmd, eye=None, learn=True):
    st["sound"] = floor["v"] is not None and float(torch.as_tensor(ear).float().sum()) > floor["v"]
    return orig_step(self, ear, cmd, eye, learn)


Brain.step = step


def q(h, f):
    c = np.cumsum(h); return BINS[int(np.searchsorted(c, f * c[-1]))] if c[-1] else float("nan")


def show(d, head, w=None):
    print(f"   {head}  的の率 ／ 見込みの p 平均（10・50・90%点）／ 全体の p ／ 床が上 ／ 見込みの p 的あり・なし", flush=True)
    for t in TAGS:
        for snd in (True, False):
            v = d[(t, snd)]; n = max(v["n"], 1); k1 = max(v["tgt"], 1); k0 = max(v["n"] - v["tgt"], 1)
            print(f"     {t:3s} {'音あり' if snd else '音なし'} 体×歩 {v['n']:9d} ｜ 的 {v['tgt']/n:6.2%} ／ 見込み {v['pp']/n:.3f}"
                  f"（{q(v['hist'],.1):.2f}・{q(v['hist'],.5):.2f}・{q(v['hist'],.9):.2f}）／ 全体 {v['pa']/n:.3f} ／ 床が上 {v['above']/n:5.1%}"
                  f" ／ {v['pp1']/k1:.3f}・{v['pp0']/k0:.3f}", flush=True)


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
        acc.update(mk())
show(acc_all, "全部")
print("読み：音の無い歩でも見込みの p が的の率より大きく上なら、焼いた素子の床がいつも的より上", flush=True)
