"""無音の歩で、耳・視床・皮質を撃たせているものを音のある歩と並べる（arkhe の ~/anaplex で。版59。0から・学びながら・保存しない・網は書き換えない。
暮らしと試しは measure_check2 と同じ）
  python3 measure_v59c.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v59c.py 3 --test）
【み】10-11 16時10分「戻して無音を調べる」
【測】measure_v60a --fi-off（版59 の振る舞い）：聴覚の表現 音あり 3.90% ／ 無音 2.78%（1.40倍）・視床 26.7 ／ 21.1%。無音でも視床・聴覚が撃っている
【コード】耳（voice.encode）：メル → 繊維ごとの閾値で 0／1（1チャネル10本、閾値 −11〜−2）。乱数は無い ⇒ 無音（床のメル）でも閾値の低い繊維は毎歩撃つ
並べるもの（音あり ／ 無音）
  1 耳：撃った繊維の割合 ／ 撃った繊維のうち前の歩も撃っていた割合（同じ繊維が撃ち続けているか）
  2 領野ごと（表現の体）：率 ／ 撃った体×歩の訳（下からだけで越えた ／ 横が要った ／ 自発だけ）／ 撃った体のうち前の歩も撃っていた割合
     下からだけ ＝ 下からの入力 x だけでも 駆動 ＞ 閾値（抑制で割った後）。横 ＝ 同じ領野の時間細胞・遅れの列など（調整する側は除く）
  3 視床の下からの入力（下丘 ＝ 1歩前の耳）が来た体の割合
音のある歩：耳の和 ＞ 静かな耳の和（measure_check と同じ）
"""
import sys, time, inspect, textwrap, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (59,):
    sys.exit(f"⚠️ babble.py が版59 ではない（版 {B.VERSION}）。版60 なら先に python3 revert_v60.py")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
N_PROBE = 3

src = textwrap.dedent(inspect.getsource(G.Group2._pass))
M1 = "rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,\n"
M2 = ("self._any = [rep, err, inh_spk] if first else [torch.maximum(a, v) for a, v in zip(self._any, (rep, err, inh_spk))]\n")
for m in (M1, M2):
    if src.count(m) != 1:
        sys.exit(f"⚠️ _pass に差し込む印が {src.count(m)} 個（1 個のはず）：{m[:50]!r}")
ind1 = src.split(M1)[0].split("\n")[-1]
src = src.replace(ind1 + M1, ind1 + "_PROBE1(self, locals())\n" + ind1 + M1, 1)
ind2 = src.split(M2)[0].split("\n")[-1]
src = src.replace(ind2 + M2, ind2 + M2 + ind2 + "_PROBE2(self, locals())\n", 1)
ns = dict(G.__dict__)
box = {"p1": None, "p2": None}
ns["_PROBE1"] = lambda self, L: box["p1"] and box["p1"](self, L)
ns["_PROBE2"] = lambda self, L: box["p2"] and box["p2"](self, L)
exec(src, ns)
G.Group2._pass = ns["_pass"]

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
NA = len(NAMES)
area_of_ch = torch.cat([torch.full((a.n_ch,), i, dtype=torch.long) for i, a in enumerate(g.areas)]).to(g.dev)
n_cells = np.array([a.n_ch for a in g.areas], float)
off_th = int(b.off_ch[NAMES.index("視床")]) if hasattr(b, "off_ch") else int(g.off_ch[NAMES.index("視床")])
st = {"tag": "他", "ear": None}
cur = {}
prev = {"ear": None, "rep": None}
rows = []   # (声, 耳の和, 耳[撃った, 前も撃った], 領野×[撃った, 下からだけ, 横が要った, 自発だけ, 前も撃った], 視床に下から来た体)

_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["ear"] = ear
    return _step(ear, cmd, *a, **k)


b.step = step_wrap


def per_area(v):
    return torch.zeros(NA, device=v.device).index_add_(0, area_of_ch, v.float())


def probe1(self, L):
    if st["tag"] == "他":
        return
    th = self.fire_th * self.th_gain
    rep, drive, x_all, ihn = L["rep"], L["drive"], L["x_all"], L["ihn"]
    x_only = (x_all * self.x_gain) / (1.0 + ihn[self.rep_i]) > th
    if L["first"]:
        n = rep.shape[0]
        cur["over"] = torch.zeros(n, dtype=torch.bool, device=self.dev)
        cur["xonly"] = torch.zeros(n, dtype=torch.bool, device=self.dev)
        cur["x_in"] = x_all > 0
    over = rep > 0                                   # 駆動（と見込みで足す分）で越えた。自発・不応期の前
    cur["over"] |= over
    cur["xonly"] |= over & x_only


def probe2(self, L):
    if st["tag"] == "他" or not L["advance"] or "over" not in cur:
        return
    fired = self._any[0] > 0
    drv = fired & cur["over"]
    xo = drv & cur["xonly"]
    lat = drv & ~cur["xonly"]
    spo = fired & ~cur["over"]
    pr = prev["rep"] if prev["rep"] is not None and prev["rep"].shape == fired.shape else torch.zeros_like(fired)
    keep = fired & pr
    m = torch.stack([per_area(fired), per_area(xo), per_area(lat), per_area(spo), per_area(keep)], 1)
    prev["rep"] = fired
    ear = (st["ear"] > 0)
    pe = prev["ear"] if prev["ear"] is not None else torch.zeros_like(ear)
    e2 = np.array([float(ear.sum()), float((ear & pe).sum())])
    prev["ear"] = ear
    th_x = float(cur["x_in"][off_th:off_th + b.n["視床"]].float().mean())
    rows.append((st["tag"], float(st["ear"].sum()), e2, m.cpu().numpy(), th_x))


box["p1"], box["p2"] = probe1, probe2
N_EAR_ALL = None


def show(rs, head):
    if not rs:
        return
    ears = np.array([r[1] for r in rs])
    snd = ears > floor["v"]
    E = np.stack([r[2] for r in rs]); M = np.stack([r[3] for r in rs]); TX = np.array([r[4] for r in rs])
    ne = float(B.N_EAR)
    print(f"── {head}  歩 {len(rs)}（音のある歩 {snd.mean():.1%}）", flush=True)
    for lab, sel in (("音あり", snd), ("無音", ~snd)):
        if sel.sum() == 0:
            continue
        e = E[sel].sum(0)
        print(f"   耳（{lab}）撃った繊維 {e[0] / (sel.sum() * ne):6.2%} ／ そのうち前の歩も撃った {e[1] / max(e[0], 1):5.1%}"
              f"   視床に下丘から来た体 {TX[sel].mean():6.2%}", flush=True)
    print("   領野      表現の率 音あり／無音     撃った訳 下からだけ・横が要った・自発だけ（音あり ｜ 無音）   前の歩も撃った（音あり／無音）", flush=True)
    for i, nm in enumerate(NAMES):
        out = []
        for sel in (snd, ~snd):
            k = sel.sum()
            if k == 0:
                out.append(None); continue
            s = M[sel, i].sum(0)
            out.append((s[0] / (k * n_cells[i]), s[1] / max(s[0], 1), s[2] / max(s[0], 1), s[3] / max(s[0], 1),
                        s[4] / max(s[0], 1)))

        def r_(o):
            return "   ―  " if o is None else f"{o[0]:6.2%}"

        def w_(o):
            return "      ―      " if o is None else f"{o[1]:4.0%}・{o[2]:4.0%}・{o[3]:4.0%}"

        def k_(o):
            return "  ― " if o is None else f"{o[4]:4.0%}"
        print(f"   {nm:6s}  {r_(out[0])} ／ {r_(out[1])}     {w_(out[0])} ｜ {w_(out[1])}     {k_(out[0])} ／ {k_(out[1])}", flush=True)


floor = {"v": None}
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
blk = 0
t_start = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if floor["v"] is None:
        floor["v"] = float(ear.sum(1).min()) + 1e-6      # 静かな耳の和（measure_check と同じ）
        print(f"静かな耳の和 {floor['v']:.0f}（耳 {B.N_EAR} 本のうち）", flush=True)
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
        show(rows[blk:], f"{i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回")
        blk = len(rows)
print("", flush=True)
show(rows, "全部")
print("読み：撃った訳の3つの和 ＝ 100%（撃った体×歩）。「前の歩も撃った」が高い ⇒ 同じ体が撃ち続けている（無音でそれが起きていれば、変わらない入力が撃たせている）", flush=True)
