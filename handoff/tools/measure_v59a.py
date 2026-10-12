"""版59 用（measure_v57b と同じ。差し込む所を不応期のあとにし、1歩に2回以上撃った体の割合を足した）
1歩の10周のうち、最後の周だけで数えた率と、どこかの周で撃った率を、領野ごとに並べる（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は書き換えない。
暮らしと試しは measure_check2 と同じ）
  python3 measure_v57b.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v57b.py 3 --test）
【測】measure_v57a（10-10 22時42分）：海馬の抑制の体が1歩に受ける撃った錐体の辺（自分・遅れの列）
  CA1 9.4本 ／ 194本のうち CA1 の錐体 約141本【計算 194 × 320 ÷ 441】⇒ 送り手の率 約6.7%。健診の CA1 の表現（最後の周）1.1〜1.9%
  歯状回 約5.2%（健診 0.5〜0.7%）／ 嗅内深 約3.2%（0.33%）／ 嗅内浅 約4.6%（1.0〜1.5%）⇒ 3〜10倍
【コード】torch_group2：表現は周ごとに作り直す。どの周で撃っても、その発火は周回の中の送り手（遅れの列）として相手へ届く（_within を周ごとに重ねる）
  健診（measure_check・check2）・時間細胞・履歴 tb に入るのは最後の周の発火だけ
  ⇒ 相手に届く発火と、率として数える発火が違う。人の率と比べるべきなのはどちらかを決めるために、両方を測る
並べるもの（領野ごと・10回ごと）：表現 ／ 誤差 ／ 抑制の体 それぞれ 最後の周の率 ／ どこかの周で撃った率 ／ 撃った体が撃った周の数の平均
"""
import sys, time, inspect, textwrap, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (59,):
    sys.exit(f"⚠️ babble.py が版57 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
N_PROBE = 3

# ── _pass の抑制の体の発火のあとに、読むだけの行を1つ差し込む ──
src = textwrap.dedent(inspect.getsource(G.Group2._pass))
MARK = "    self.pred_rep = pr\n"   # 版59：不応期のあと
if src.count(MARK) != 1:
    sys.exit("⚠️ _pass に差し込む印が見つからない")
src_i = src.replace(MARK, "    _PROBE(self, rep, err, inh_spk, advance)\n" + MARK, 1)
ns = dict(G.__dict__)
box = {"f": None}
ns["_PROBE"] = lambda self, rep, err, inh, adv: box["f"] and box["f"](self, rep, err, inh, adv)
exec(src_i, ns)
G.Group2._pass = ns["_pass"]

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev
off_ch = [int(v) for v in g.off_ch]
inh_off = np.cumsum([0] + [a.n_inh for a in g.areas])
KINDS = ("表現", "誤差", "抑制")
st = {"tag": "他"}
cur = {"n": None}                    # この歩の周ごとの数（表現・誤差・抑制）
acc = None


def new_acc():
    return {k: np.zeros((len(NAMES), 4)) for k in KINDS} | {"steps": 0}      # 最後の周 ・ どこか ・ 撃った周の数の和 ・ 2回以上撃った体


def probe(self, rep, err, inh, adv):
    global acc
    if st["tag"] == "他":
        return
    if cur["n"] is None:
        cur["n"] = [torch.zeros_like(rep), torch.zeros_like(err), torch.zeros_like(inh)]
    for j, v in enumerate((rep, err, inh)):
        cur["n"][j] += (v > 0).float()
    if not adv:
        return
    acc["steps"] += 1
    for j, (k, last) in enumerate(zip(KINDS, (rep, err, inh))):
        cnt = cur["n"][j]
        for i, a in enumerate(g.areas):
            sl = slice(int(inh_off[i]), int(inh_off[i + 1])) if k == "抑制" else slice(off_ch[i], off_ch[i] + a.n_ch)
            c = cnt[sl]
            acc[k][i] += [float((last[sl] > 0).float().mean()), float((c > 0).float().mean()), float(c.sum()), float((c >= 2).float().sum())]
    cur["n"] = None


box["f"] = probe


def show(head):
    s = max(acc["steps"], 1)
    print(f"── {head}（{acc['steps']}歩）  最後の周 → どこかの周（撃った体が撃った周の数）", flush=True)
    for i, nm in enumerate(NAMES):
        cells = []
        for k in KINDS:
            v = acc[k][i]
            n_any = v[1] / s
            per = v[2] / max(v[1] / s * s * (g.areas[i].n_inh if k == "抑制" else g.areas[i].n_ch), 1e-9)
            two = v[3] / max(v[1] / s * s * (g.areas[i].n_inh if k == "抑制" else g.areas[i].n_ch), 1e-9)
            cells.append(f"{k} {v[0]/s:6.2%} → {n_any:6.2%}（{per:4.1f}周・2発以上 {two:4.0%}）")
        print(f"   {nm:5s} " + "  ".join(cells), flush=True)


acc = new_acc()
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t_start = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if ear.sum() > 0:
        past.append((cmd, ear))
    if rp is not None:                           # 暮らし：返し手が返せば聞く（babble.py と同じ）
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        for _ in range(N_PROBE):                 # 試し：10回ごとに録音を3つ。学びは止めない
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
        show(f"{i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回")
        acc = new_acc()
print("読み：2発以上 ＝ 撃った体のうち1歩に2回以上撃った割合 ⇔【ヒト】10ms 未満の間隔の割合 錐体 0.17〜0.25・嗅内 0.07・抑制 0.10〜0.14（Ison 2011）", flush=True)
