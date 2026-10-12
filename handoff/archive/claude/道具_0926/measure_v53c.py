"""聴覚の誤差の体が越えた音を消すとき、出どころを組で抜く ／ 1歩に何体越えたかで分ける（arkhe の ~/anaplex で。
0から・学びながら・返し手と録音あり・保存しない・網は書き換えない。声と乱数の並びは measure_v53b と同じ）
  python3 measure_v53c.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v53c.py 3 --test）
【み】10-09「両方」（A：もう1つ分ける ／ B：09-11 の決めに戻る）
【測】measure_v53b（版53）：写しを抜いても消えなくなるのは 自分 1.5%（録音 1.9%）。消しているのは 自分の時間・上から。
  1つ抜いても消えたままが約6割（重ねて消している）。消した割合は 10回ごとの越えた数と順位相関 0.94
並べるもの（10回ごと、自分 ／ 録音 ／ 返し手。音のある歩、聴覚の誤差の体が下から越えた歩だけ）
  1 組で抜く：越えた体のうち、なお消える割合
      全部（今のまま）／ 自分の時間と上からを抜く ／ 写し以外を全部抜く（見込み＋写しだけ）／ 見込みだけ
     ⇒ 人の形なら：写しだけ残したとき 自分は消え、録音・返し手は消えない
  2 1歩に越えた体の数で分ける（1〜3 ／ 4〜15 ／ 16〜63 ／ 64〜）：越えた体の数と消した割合
     ⇒ 誰の声かではなく、その歩に何体越えたか（一斉に越えたか）で消すかが決まっているかを見る
v53b と同じ式：L ＝ 見込み ＋ Σ 出どころごとの exc ÷ (1 ＋ ihn)。L ≥ 0 が「消した」
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
# 組で抜いたあと残すもの（True ＝ 残す）。順は CATS
KEEPS = (("全部", (1, 1, 1, 1)), ("時間・上からを抜く", (0, 1, 1, 0)), ("写しだけ", (0, 0, 1, 0)), ("見込みだけ", (0, 0, 0, 0)))
BINS = (1, 4, 16, 64)
BIN_NAME = ("1〜3", "4〜15", "16〜63", "64〜")

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

KM = torch.tensor([k for _, k in KEEPS], dtype=torch.float32, device=dev)     # (組, NC)
st = {"tag": "他"}; floor = {"v": None}
chk = [0, 0]
NK, NB = len(KEEPS), len(BINS)
# 越えた ・ 組ごとに消えた数 ・ 越えた数の区分ごとの（越えた・消した）
mk = lambda: np.zeros(1 + NK + 2 * NB)
acc = {t: mk() for t in TAGS}
acc_all = {t: mk() for t in TAGS}
orig_pass = G.Group2._pass


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
    L_kept = prior_l.unsqueeze(0) + KM @ part                                    # (組, nc)
    killed = (L_kept >= 0) & tgt.unsqueeze(0)
    if TEST:
        fired = (out[1][CH] > 0) & tgt
        chk[0] += int((fired != (tgt & (L_kept[0] < 0))).sum()); chk[1] += k
    add = mk()
    add[0] = k
    add[1:1 + NK] = killed.sum(1).float().cpu().numpy()
    bi = int(np.searchsorted(BINS, k, side="right")) - 1
    add[1 + NK + bi] = k
    add[1 + NK + NB + bi] = float(killed[0].sum())
    acc[st["tag"]] += add; acc_all[st["tag"]] += add
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
        s1 = "  ".join(f"{name} {v[1+j]/k:5.1%}" for j, (name, _) in enumerate(KEEPS))
        s2 = "  ".join(f"{BIN_NAME[j]} {int(v[1+NK+j]):6d}／{v[1+NK+NB+j]/max(v[1+NK+j],1):5.1%}" for j in range(NB))
        print(f"     {t:3s} 越えた {int(v[0]):6d}  ｜ なお消える {s1}  ｜ 1歩に越えた数ごと（越えた／消した）{s2}", flush=True)


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
print("読み：人の形なら「写しだけ」で 自分は消え・録音と返し手は消えない。区分ごとの消した割合が越えた数とともに上がるなら、一斉に越えた歩を消している", flush=True)
