"""写しの重みが育たない理由：送り手が撃った歩のうち、的が立った割合と、そのとき重みが押される向き（arkhe の ~/anaplex で。
0から・学びながら・保存しない・網は書き換えない。暮らしと試しは measure_v53f と同じ）
  python3 measure_v53g.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v53g.py 10 --test）
【み】10-10「書いて」
【測】measure_v53f：写しの見分けは自分の声で録音の約3倍（中身は来ている）が、重みは育っては下がる。どの出どころも同じ形で、写しがいつも一番小さい
学びの規則（torch_group2._pass）：送り手が撃った辺ごとに、跡 ＋＝ η × 驚き × (的 − p)。重みは跡から動く。抑制・錐体 → 抑制の体は学ばない
  ⇒ 送り手が撃っても受け手の的が立たない歩では、(0 − p) で下がる。的が立てば (1 − p) で上がる
並べるもの（聴覚の誤差の体へ入る興奮の辺、出どころごと。喃語・返し手・録音のすべての歩。音の無い歩も入れる）
  撃った ＝ 送り手が撃った辺の数（1歩あたり）
  的あり ＝ そのうち受け手の的が立っていた割合（P(的 | 送り手が撃った)）
  押し   ＝ 1回あたりの (的 − p) の平均（＋なら育つ向き、−なら痩せる向き）。上げ ／ 下げ の内訳も
  辺あたり ＝ 押しの和 ÷ その出どころの辺の数（その間に1本がどれだけ押されたか）
読み：写しだけ 的あり が低く 押し が − なら、「撃っても的が立たない歩が多く、痩せる」見立てが当たり
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
mk = lambda: np.zeros((NC, 5))      # 撃った ・ 的あり ・ 押しの和 ・ 上げの和 ・ 下げの和
acc = {t: mk() for t in TAGS}
acc_all = {t: mk() for t in TAGS}
steps = {t: [0, 0] for t in TAGS}    # この10回 ・ 全部 の歩数
orig_pass = G.Group2._pass


def weights():
    """聴覚の誤差の体へ入る興奮の辺の重みの平均（出どころごと）"""
    r = g.e_recv.long(); m = (r >= ERR.start) & (r < ERR.stop) & ~g.e_inh
    c = send_cat[g.e_src[m].long()]; w = g.w[m]
    return [float(w[c == j].mean()) if int((c == j).sum()) else float("nan") for j in range(NC)], \
           [int((c == j).sum()) for j in range(NC)]


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    if not (learn and st["tag"] in TAGS):
        return orig_pass(self, x_all, ext_all, learn, advance, first)
    e = self._fired_edges()
    el0 = self.elig[e].clone() if len(e) else None
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    if first:
        steps[st["tag"]][0] += 1; steps[st["tag"]][1] += 1
    if el0 is None:
        return out
    r = self.e_recv[e].long()
    m = (r >= ERR.start) & (r < ERR.stop) & ~self.e_inh[e]
    if not bool(m.any()):
        return out
    em = e[m]; rr = r[m] - ERR.start
    push = (self.elig[em] - el0[m] * self.elig_decay) / (self.eta * self.eta_gain)   # ＝ 的 − p
    tg = (self.last_xt[CH] > 0.5)[rr].float()
    cat = send_cat[self.e_src[em].long()]
    add = mk()
    for j in range(NC):
        k = cat == j
        if bool(k.any()):
            pk = push[k]
            add[j] = [float(k.sum()), float(tg[k].sum()), float(pk.sum()), float(pk.clamp(min=0).sum()), float(pk.clamp(max=0).sum())]
    acc[st["tag"]] += add; acc_all[st["tag"]] += add
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


def show(d, head, w):
    ne = edges_per_cat()
    print(f"   {head}  出どころ 撃った（1歩あたり）／ 的あり ／ 押し 1回あたり 的−p（上げ 下げ）／ 辺あたりの押しの和", flush=True)
    for t in TAGS:
        v = d[t]; n = max(steps[t][w], 1)
        cells = []
        for j, c in enumerate(CATS):
            k = max(v[j, 0], 1)
            cells.append(f"{c} {v[j,0]/n:8.1f}／{v[j,1]/k:5.1%}／{v[j,2]/k:+.3f}（{v[j,3]/k:+.3f} {v[j,4]/k:+.3f}）／{v[j,2]/ne[j]:+.2f}")
        print(f"     {t:3s} 歩 {steps[t][w]:6d}  ｜ " + "  ".join(cells), flush=True)


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
        show(acc, "この10回", 0)
        show_w()
        for v in acc.values():
            v[:] = 0
        for v in steps.values():
            v[0] = 0
show(acc_all, "全部", 1)
print("読み：写しだけ 的あり が低く 押し が − なら、撃っても的が立たない歩が多く痩せる（学びの規則の側）", flush=True)
