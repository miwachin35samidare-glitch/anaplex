"""運動の領野の率を、喃語の段ごと（自分：準備・声・指令なし ／ 返し手 ／ 録音）に分けて、指令が入った体と入らない体で比べる
（arkhe の ~/anaplex で。版61・62。0から・学びながら・保存しない・網は書き換えない。暮らしと試しは measure_check2 と同じ）
  python3 measure_v62a.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v62a.py 3 --test）
【み】10-11 20時45分「いいよ」
【測】版62 健診：運動 9.5〜11.4% ▲（人 1.8〜7.0%）。measure_v59c：音あり 10.5 ／ 無音 9.9%、無音でも 72% が下からだけ・63% が前の歩も撃った
【計算】指令の無い体：運動前野の送り手 115本 × 3.3Hz（1歩 3.5%）⇒ 1歩に届く 平均 4.1本（ばらつき 2.0）。閾値 14.3 は届く数だけではほぼ越えない
  指令の入った体：115本 × 35Hz（1歩 37%）⇒ 平均 43本 ⇒ 毎歩越える
  ⇒ 無音で下から越えているのは、声の前の準備（耳は静か・指令は入っている）か、蹴り（運動視床）の見込み【計算】
並べるもの（段ごと）
  歩の数・全部の歩に占める割合 ／ 運動の表現の率 ／ 指令が入った体の割合 ／ 指令の入った体の率・入らない体の率 ／
  撃った訳（下からだけ・横が要った・自発だけ）／ 蹴りが来ていた歩の割合 ／ 下から x の平均（指令の入った体・入らない体）
段：自分・準備（指令あり・耳は静か）／ 自分・声（指令あり・音あり）／ 自分・指令なし（黙る回・声のあと）／ 返し手 ／ 録音
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
im = NAMES.index("運動")
lo, nM = int(g.off_ch[im]), b.n["運動"]
st = {"tag": "他", "ear": 0.0, "cmd": None, "kick": False}
cur = {}
rows = []    # (段, 指令の入った体の数, [指令あり体: 撃った, 下からだけ, 横, 自発, x の和] [指令なし体: 同じ], 蹴り)
PHASES = ("自分・準備", "自分・声", "自分・指令なし", "返し手", "録音")

_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["ear"] = float(ear.sum())
    st["cmd"] = (cmd > 0) if cmd is not None and len(cmd) == nM else torch.zeros(nM, dtype=torch.bool, device=g.dev)
    st["kick"] = b.kick_left > 0
    return _step(ear, cmd, *a, **k)


b.step = step_wrap


def probe1(self, L):
    if st["tag"] == "他":
        return
    th = self.fire_th * self.th_gain
    rep, x_all, ihn = L["rep"], L["x_all"], L["ihn"]
    xo = (x_all * self.x_gain) / (1.0 + ihn[self.rep_i]) > th
    if L["first"]:
        cur["over"] = torch.zeros(nM, dtype=torch.bool, device=self.dev)
        cur["xo"] = torch.zeros(nM, dtype=torch.bool, device=self.dev)
        cur["x"] = (x_all * self.x_gain)[lo:lo + nM].clone()
    over = (rep > 0)[lo:lo + nM]
    cur["over"] |= over
    cur["xo"] |= over & xo[lo:lo + nM]


def probe2(self, L):
    if st["tag"] == "他" or not L["advance"] or "over" not in cur:
        return
    fired = (self._any[0] > 0)[lo:lo + nM]
    drv = fired & cur["over"]
    xo = drv & cur["xo"]; lat = drv & ~cur["xo"]; spo = fired & ~cur["over"]
    c = st["cmd"]
    if st["tag"] == "自分":
        ph = ("自分・声" if st["ear"] > 0 else "自分・準備") if bool(c.any()) else "自分・指令なし"
    else:
        ph = st["tag"]
    out = []
    for m in (c, ~c):
        out.append([float((v & m).sum()) for v in (fired, xo, lat, spo)] + [float(cur["x"][m].sum())])
    rows.append((ph, int(c.sum()), np.array(out), st["kick"]))


box["p1"], box["p2"] = probe1, probe2


def show(rs, head):
    if not rs:
        return
    print(f"── {head}  歩 {len(rs)}", flush=True)
    print("   段            歩（割合）    運動の率   指令の体の割合   率：指令あり ／ なし   撃った訳（指令あり ｜ なし）下から・横・自発   蹴り   x：指令あり ／ なし", flush=True)
    for ph in PHASES:
        sel = [r for r in rs if r[0] == ph]
        if not sel:
            continue
        k = len(sel)
        nc = np.array([r[1] for r in sel], float)
        A = np.stack([r[2] for r in sel])          # 歩 × 2 × 5
        s = A.sum(0)
        n_c, n_n = nc.sum(), (nM - nc).sum()
        rate = (s[0, 0] + s[1, 0]) / (k * nM)

        def why(i):
            f = max(s[i, 0], 1)
            return f"{s[i, 1] / f:4.0%}・{s[i, 2] / f:4.0%}・{s[i, 3] / f:4.0%}" if s[i, 0] else "      ―      "
        kick = np.mean([r[3] for r in sel])
        rc = f"{s[0, 0] / n_c:6.2%}" if n_c else "   ―  "
        xc = f"{s[0, 4] / n_c:5.1f}" if n_c else "  ― "
        print(f"   {ph:8s}  {k:6d}（{k / len(rs):5.1%}）  {rate:6.2%}     {n_c / (k * nM):6.1%}       {rc} ／ {s[1, 0] / max(n_n, 1):6.2%}"
              f"      {why(0)} ｜ {why(1)}    {kick:5.1%}   {xc} ／ {s[1, 4] / max(n_n, 1):5.2f}", flush=True)
    tot = sum(r[2][:, 0].sum() for r in rs) / (len(rs) * nM)
    print(f"   全部の歩の運動の率 {tot:6.2%}（健診と同じ量）", flush=True)


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
blk = 0
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
        show(rows[blk:], f"{i+1}回目まで（{(time.time()-t0)/60:.1f}分）この10回")
        blk = len(rows)
print("", flush=True)
show(rows, "全部")
print("読み：x は下からの入力（運動前野の送り手の届いた数 ＋ 蹴り）。閾値 14.3。撃った訳の3つの和 ＝ 100%", flush=True)
print("     ⚠️ 自分・声 と 自分・準備 は耳の和で分けた（耳が静かな歩 ＝ 準備）。録音は指令を入れない（写しなし）", flush=True)
