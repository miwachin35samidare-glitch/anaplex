"""運動の領野の指令の入った体が、なぜ 1歩の 81〜88% で撃つのか：周ごとの抑制・下からの駆動・撃った周を並べる
（arkhe の ~/anaplex で。版61・62。0から・学びながら・保存しない・網は書き換えない。暮らしと試しは measure_check2 と同じ）
  python3 measure_v62b.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v62b.py 3 --test）
【み】10-11 21時02分「いいよ」
【測】measure_v62a（版62）：指令の入った体は 1歩の 81〜88% で撃つ（≒ 76〜82Hz）⇔【ヒト】話すあいだの山 30〜40Hz（Stavisky 2019）
       下から届く数 x 42.9。K_PM 115 は「版48 の指令の歩の抑制 ihn 約2 で 35Hz」の本数（ノブ【C】）⇒ 42.9 ÷ 3 ≒ 閾値 14.3 の設計
分けたい2つ
  (a) 抑制 ihn が版48 の約2 より弱い         ⇒ 周ごとの 1＋ihn の平均・3 を割る体の割合を見る
  (b) 版59 で「歩のどこかの周で撃った」を数える（版48 は最後の周だけ）⇒ 最後の周だけで数えた率と並べる
並べるもの（指令の入った歩の、指令の入った体。比べに同じ歩の指令の無い体）
  周ごと：1＋ihn の平均 ／ 下からだけの駆動 x÷(1＋ihn) の平均 ／ 閾値を越えた割合（全部の駆動・下からだけ）／ 不応期のあと撃った割合
  歩ごと：どこかの周で撃った割合（＝ v62a・健診の数え方）／ 最後の周で閾値を越えた割合（版48 の数え方・不応期なし）／
          1歩に撃った周の数の分け方 ／ 歩の中で 1＋ihn の最小が 3 を割った体の割合
"""
import sys, time, inspect, textwrap, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (61, 62):
    sys.exit(f"⚠️ babble.py が版61・62 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
N_PROBE = 3

src = textwrap.dedent(inspect.getsource(G.Group2._pass))
M2 = ("self._any = [rep, err, inh_spk] if first else [torch.maximum(a, v) for a, v in zip(self._any, (rep, err, inh_spk))]\n")
if src.count(M2) != 1:
    sys.exit(f"⚠️ _pass に差し込む印が {src.count(M2)} 個（1 個のはず）")
ind2 = src.split(M2)[0].split("\n")[-1]
src = src.replace(ind2 + M2, ind2 + M2 + ind2 + "_PROBE2(self, locals())\n", 1)
ns = dict(G.__dict__)
box = {"p2": None}
ns["_PROBE2"] = lambda self, L: box["p2"] and box["p2"](self, L)
exec(src, ns)
G.Group2._pass = ns["_pass"]

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
im = NAMES.index("運動")
lo, nM = int(g.off_ch[im]), b.n["運動"]
R = g.recur
st = {"tag": "他", "cmd": None}
cur = {}
# 周ごとの和（指令あり体 c ／ なし体 n）：1+ihn・x÷(1+ihn)・全部で越えた・下からで越えた・撃った・1+ihn<3
acc = {k: np.zeros((R, 6)) for k in ("c", "n")}
cnt = {"c": 0.0, "n": 0.0, "steps": 0}
stp = {k: np.zeros(4) for k in ("c", "n")}     # どこかの周で撃った・最後の周で越えた・最小(1+ihn)<3・体×歩
npass = np.zeros(R + 1)                          # 指令あり体の、1歩に撃った周の数
_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["cmd"] = (cmd > 0) if cmd is not None and len(cmd) == nM else None
    return _step(ear, cmd, *a, **k)


b.step = step_wrap


def probe2(self, L):
    c = st["cmd"]
    if st["tag"] != "自分" or c is None or not bool(c.any()):
        return
    th = (self.fire_th * self.th_gain)
    th = th[lo:lo + nM] if torch.is_tensor(th) and th.dim() else th
    ihn1 = 1.0 + L["ihn"][self.rep_i][lo:lo + nM]
    xd = (L["x_all"] * self.x_gain)[lo:lo + nM] / ihn1
    dr = L["drive"][lo:lo + nM]
    rep = L["rep"][lo:lo + nM] > 0
    if L["first"]:
        cur.update(i=0, fired=torch.zeros(nM, device=self.dev), minih=ihn1.clone())
    i = cur["i"]
    cur["fired"] += rep.float(); cur["minih"] = torch.minimum(cur["minih"], ihn1)
    for k, m in (("c", c), ("n", ~c)):
        if not bool(m.any()):
            continue
        acc[k][i] += [float(ihn1[m].sum()), float(xd[m].sum()), float((dr > th)[m].sum()),
                      float((xd > th)[m].sum()), float(rep[m].sum()), float((ihn1 < 3)[m].sum())]
    if i == 0:
        for k, m in (("c", c), ("n", ~c)):
            cnt[k] += float(m.sum())
        cnt["steps"] += 1
    if L["advance"]:
        for k, m in (("c", c), ("n", ~c)):
            stp[k] += [float((cur["fired"] > 0)[m].sum()), float((dr > th)[m].sum()),
                       float((cur["minih"] < 3)[m].sum()), float(m.sum())]
        npass[:] += np.bincount(cur["fired"][c].long().cpu().numpy(), minlength=R + 1)[:R + 1]
    cur["i"] = i + 1


box["p2"] = probe2


def show(head):
    if cnt["steps"] == 0:
        return
    print(f"── {head}  指令の入った歩 {cnt['steps']}", flush=True)
    for k, lab in (("c", "指令の入った体"), ("n", "指令の無い体（同じ歩）")):
        n = max(cnt[k], 1)
        print(f"   {lab}（体×歩 {cnt[k]:.0f}）", flush=True)
        print("     周   1+ihn   x÷(1+ihn)   越えた（全部）  越えた（下から）  撃った（不応期のあと）  1+ihn<3", flush=True)
        for i in range(R):
            a = acc[k][i] / n
            print(f"     {i+1:2d}   {a[0]:5.2f}   {a[1]:7.2f}      {a[2]:6.1%}         {a[3]:6.1%}            {a[4]:6.1%}            {a[5]:6.1%}", flush=True)
        s = stp[k]; m = max(s[3], 1)
        print(f"     歩：どこかの周で撃った {s[0] / m:6.1%}（健診の数え方）／ 最後の周で越えた {s[1] / m:6.1%}（版48 の数え方）／ "
              f"歩の中の最小の 1+ihn が 3 未満 {s[2] / m:6.1%}", flush=True)
    t = max(npass.sum(), 1)
    print("   指令の入った体の、1歩に撃った周の数：" + "  ".join(f"{j}周 {npass[j] / t:5.1%}" for j in range(R + 1) if npass[j]), flush=True)
    th = g.fire_th * g.th_gain
    th = float(th[lo:lo + nM].mean()) if torch.is_tensor(th) and th.dim() else float(th)
    print(f"   運動の閾値（平均）{th:.2f}", flush=True)
    for d in (acc["c"], acc["n"], stp["c"], stp["n"], npass):
        d[:] = 0
    cnt.update(c=0.0, n=0.0, steps=0)


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t0 = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if ear.sum() > 0:
        past.append((cmd, ear))
    if rp is not None:
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        for _ in range(N_PROBE):
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
        show(f"{i+1}回目まで（{(time.time()-t0)/60:.1f}分）この10回")
print("読み：x は下からの入力（運動前野の送り手の届いた数 × x_gain。蹴りは指令の入った歩には来ない）。閾値 14.3", flush=True)
print("     越えた ＝ 不応期の前。撃った ＝ 不応期のあと＋自発。1+ihn ＝ 抑制（短絡）の割り算の分母。版48 の設計は 約3", flush=True)
