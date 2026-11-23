"""表現の体の率を「音のある歩 ／ 無音の歩」に分け、閾値を越えた体のうち 撃った ／ f-I で止まった ／ 不応期だけで止まった を数える
（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は書き換えない。暮らしと試しは measure_check2・measure_v59b と同じ）
  python3 measure_v60a.py [喃語の回数=50]            版60 のまま
  python3 measure_v60a.py [喃語の回数=50] --fi-off   f-I を道具の側で効かなくして回す（版59 と同じ振る舞い。越えた数を比べる用）
  （コンテナで道具を確かめるとき：python3 measure_v60a.py 3 --test）
【み】10-11 14時18分「aでもいいけど、AnaPlexには聴覚しか入力がないことを考慮に入れるべきかもしれないね」→ 14時22分「それでいいよ」
ねらい：版60 で 聴覚 0.74〜1.06%・高次 0.9〜1.7% が人の範囲の下へ出た。どこから来るかを分ける
  1 音のある歩の率 と 無音の歩の率（暮らしの音の少なさが効いているか）
  2 閾値を越えた体×歩（駆動 ＞ 閾値 の周が1つでもある）のうち
      撃った（駆動で）／ f-I で止まった（不応期の外で越えたのに f-I で待った）／ 不応期だけで止まった
    ⇒ f-I の止めが主か、越えた数そのものが少ないか（上流から届く分）
  3 --fi-off（版59 の振る舞い）と並べて、越えた数が減ったかを見る
音のある歩：耳の入力の和が、静かなときの値（耳の和の最小 ＋ 1e−6）を上回る歩（measure_check と同じ決め方）
⚠️ f-I・不応期・自発の式は torch_group2._pass のものをそのまま読む（道具は数えるだけ。--fi-off だけ FI_SLOPE を大きくして f-I を効かなくする）
"""
import sys, time, inspect, textwrap, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
FI_OFF = "--fi-off" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (60,):
    sys.exit(f"⚠️ babble.py が版60 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
if FI_OFF:
    G.FI_SLOPE = 1e12          # f が十分大きい ⇒ 越えれば不応期明けごと（版59 と同じ）
DELAY = 2
N_PROBE = 3

# ── _pass に読むだけの行を2つ差し込む ──
src = textwrap.dedent(inspect.getsource(G.Group2._pass))
M1 = ("rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,\n")
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
st = {"tag": "他", "ear": 0.0}
cur = {}
rows = []          # 歩ごと：(声の種類, 耳の和, 領野×[越えた, 撃った(駆動), f-Iで止まった, 不応期だけ, どこかで撃った])

# 耳の和を歩ごとに覚える（Brain.step の入口）
_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["ear"] = float(ear.sum())
    return _step(ear, cmd, *a, **k)


b.step = step_wrap


def per_area(v):
    return torch.zeros(NA, device=v.device).index_add_(0, area_of_ch, v.float())


def probe1(self, L):
    """f-I・自発の前。この周で 越えたか ／ 不応期の内か ／ f-I で待つか"""
    if st["tag"] == "他":
        return
    th = self.fire_th * self.th_gain
    d_fi = L["d_fi"]
    over = d_fi > th
    if L["first"]:
        n = over.shape[0]
        for k in ("over", "drv", "fib", "refb"):
            cur[k] = torch.zeros(n, dtype=torch.bool, device=self.dev)
    lf = getattr(self, "_last_fire", None)
    if lf is None:
        refr = torch.zeros_like(over); fi_ok = torch.ones_like(over)
    else:
        gap_p = self._pc + 1 - lf[0]                       # この周の番号は _pc＋1（_pc はこのあとで進む）
        refr = gap_p < G.REFRACT_PASSES
        f_hz = G.FI_SLOPE * G.FI_RHEO * (d_fi / th - 1.0).clamp(min=0.0)
        fi_ok = gap_p.float() * G.PASS_MS * f_hz >= 1000.0
    cur["over"] |= over
    cur["drv"] |= over & ~refr & fi_ok
    cur["fib"] |= over & ~refr & ~fi_ok
    cur["refb"] |= over & refr


def probe2(self, L):
    """歩の終わり（最後の周）。この歩のどこかで撃ったか と合わせて歩ごとに数える"""
    if st["tag"] == "他" or not L["advance"] or "over" not in cur:
        return
    fired_any = self._any[0] > 0
    drv = cur["drv"]
    fib = cur["fib"] & ~drv                                  # 撃たずに終わった体のうち f-I で待った
    refb = cur["refb"] & ~drv & ~cur["fib"]                  # 撃たずに終わり、越えた周がどれも不応期の内
    m = torch.stack([per_area(cur["over"]), per_area(drv), per_area(fib), per_area(refb), per_area(fired_any)], 1)
    rows.append((st["tag"], st["ear"], m.cpu().numpy()))


box["p1"], box["p2"] = probe1, probe2


def show(rs, head):
    if not rs:
        return
    ears = np.array([r[1] for r in rs]); floor = ears.min() + 1e-6 if all_floor["v"] is None else all_floor["v"]
    snd = ears > floor
    M = np.stack([r[2] for r in rs])                 # 歩 × 領野 × 5
    tags = np.array([r[0] for r in rs])
    frac = {t: (snd[tags == t].mean() if (tags == t).any() else float("nan")) for t in ("自分", "返し手", "録音")}
    print(f"── {head}  歩 {len(rs)}（音のある歩 {snd.mean():.1%}：自分 {frac['自分']:.1%}・返し手 {frac['返し手']:.1%}・録音 {frac['録音']:.1%}）"
          f"{'  ⚠️ f-I を道具で切った（版59 の振る舞い）' if FI_OFF else ''}", flush=True)
    print("   領野      表現の率  音あり ／ 無音    越えた体の割合 音あり ／ 無音   越えた体のうち 撃った・f-Iで止まった・不応期だけ（音あり ｜ 無音）", flush=True)
    for i, nm in enumerate(NAMES):
        out = []
        for sel in (snd, ~snd):
            k = sel.sum()
            if k == 0:
                out.append(None); continue
            s = M[sel, i].sum(0)
            out.append((s[4] / (k * n_cells[i]), s[0] / (k * n_cells[i]),
                        s[1] / max(s[0], 1), s[2] / max(s[0], 1), s[3] / max(s[0], 1), s[0]))
        allr = M[:, i, 4].sum() / (len(rs) * n_cells[i])

        def f(o, j):
            return "   ―  " if o is None else f"{o[j]:6.2%}"

        def g3(o):
            return "    ―" if o is None or o[5] == 0 else f"{o[2]:4.0%}・{o[3]:4.0%}・{o[4]:4.0%}"
        print(f"   {nm:6s}  {allr:6.2%}   {f(out[0], 0)} ／ {f(out[1], 0)}    {f(out[0], 1)} ／ {f(out[1], 1)}     "
              f"{g3(out[0])} ｜ {g3(out[1])}", flush=True)


all_floor = {"v": None}
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
blk_start = 0
t_start = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if all_floor["v"] is None:
        all_floor["v"] = float(ear.sum(1).min()) + 1e-6      # 静かな耳の和（measure_check と同じ）
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
        show(rows[blk_start:], f"{i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回")
        blk_start = len(rows)
print("", flush=True)
show(rows, "全部")
print("読み：音ありの率が人の範囲で無音が低い ⇒ 暮らしの音の少なさ ／ 音ありでも低く「f-Iで止まった」が多い ⇒ f-I の当てはめ ／"
      " 越えた体の割合そのものが低い（--fi-off と比べて減った）⇒ 上流から届く分", flush=True)
print("     撃った ＋ f-Iで止まった ＋ 不応期だけ ＝ 100%（越えた体×歩）。表現の率には自発で撃った分も入る", flush=True)
