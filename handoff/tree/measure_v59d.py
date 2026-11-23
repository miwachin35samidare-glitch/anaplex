"""版59：視床・聴覚などの表現の体を撃たせている「横の興奮」を、送り手の種類と重さで分ける。音あり／無音で（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は書き換えない。
暮らしと試しは measure_check2 と同じ）
  python3 measure_v59d.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v59d.py 3 --test）
【み】10-11 17時04分「いいよ」
【測】measure_v59c（10-11）：無音で耳 0 本・視床に下から 0.10% なのに、視床 22%（99% が横・68% が前の歩も撃った）。音ありでも視床の 85%・聴覚の 85% が横
【測】measure_v49d（10-03、版49）：視床の横はほぼ全部「自分の遅れ」（1体 3,202本）⇒ 版51 で中継の横の相手を 64本（RELAY_EXC、ノブ【C】）にした
横（表現の体へ届いた興奮。抑制・調整する側（上からの戻り・写し）は除く）を送り手の種類で4つに分ける（measure_v49d と同じ割り方）
  自分の遅れ   自分の領野の表現の発火（数歩の遅れの写し。同じ歩の周回の中の分も含む）
  自分の時間   自分の表現の発火のあと決まった間で撃つ時間細胞
  傾き        傾きの細胞（自分の発火からの経過で率が変わる乱数。1体 平均 15%/歩【計算】）
  横の入口     樹状突起の入口のうち撃たせる側のもの・誤差の行の遅れ（自分の表現ではない送り手）
並べるもの（音あり／無音、表現の体1体・1歩の平均。最後の周に届いた分）
  下から x（閾値と同じ目盛り）／ 横の合計 ／ 種類ごとに 量・届いた本数・重さ（× 接触 × 短期 × 送り手の倍率 × 恒常性）／ 閾値
最初と最後に、表現の体の興奮の繋がり（種類ごとの 1体あたりの本数・重さの平均。届いたかに関係なく）
"""
import sys, time, torch, numpy as np
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
AREAS = ("視床", "聴覚", "連合", "前頭前野", "上丘")
KINDS = ("自分の遅れ", "自分の時間", "傾き", "横の入口")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev

# ── 送り手の種類の表（-1 ＝ 数えない：抑制・調整する側）──
kind = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
buf_base = g.n_send_all - sum(a.n_src * a.D for a in g.areas)
buf_off = 0
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    lo = int(g.off_send[i])
    nc = a.n_ch + a.n_ext                      # 抑制の行は数えない
    for r_lo, r_hi, own in ((0, a.n_ch, True), (a.n_ch, nc, False)):
        if r_hi <= r_lo:
            continue
        rows = torch.arange(r_lo, r_hi, device=dev)
        if per:
            tc = lo + rows.unsqueeze(1) * per + torch.arange(per, device=dev)
            if a.ch.n_cell:
                kind[tc[:, :a.ch.n_cell].reshape(-1)] = 1 if own else 3
            kind[tc[:, a.ch.n_cell:].reshape(-1)] = 2
        bf = buf_base + buf_off + rows.unsqueeze(1) * a.D + torch.arange(a.D, device=dev)
        kind[bf.reshape(-1)] = 0 if own else 3
    buf_off += a.n_src * a.D
kind[g.mod_send] = -1
kind[g.is_inh_all] = -1

rep_recv = torch.nonzero(g.is_rep).squeeze(1)
area_of_recv = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=dev)
for nm in AREAS:
    lo = int(g.off_ch[NAMES.index(nm)])
    area_of_recv[rep_recv[lo:lo + b.n[nm]]] = AREAS.index(nm)
NA, NK = len(AREAS), len(KINDS)
n_area = torch.tensor([b.n[nm] for nm in AREAS], device=dev, dtype=torch.float)


def w_static():
    g.flush_all()
    r = g.e_recv.long(); se = g.e_src.long()
    k = kind[se]; ar = area_of_recv[r]
    m = (k >= 0) & (ar >= 0)
    idx = ar[m] * NK + k[m]
    s = torch.zeros(NA * NK, device=dev).index_add_(0, idx, g.w[m])
    c = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(g.w[m]))
    return (s / c.clamp(min=1)).view(NA, NK).cpu().numpy(), (c.view(NA, NK) / n_area.unsqueeze(1)).cpu().numpy()


st = {"tag": "他", "snd": False}
floor = {"v": None}
_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["snd"] = floor["v"] is not None and float(ear.sum()) > floor["v"]
    return _step(ear, cmd, *a, **k)


b.step = step_wrap
# acc[音あり/無音]：量・本数・量（体で割らない）・下から x・閾値・歩の数
acc = {s: {"q": np.zeros((NA, NK)), "n": np.zeros((NA, NK)), "qa": np.zeros((NA, NK)),
           "x": np.zeros(NA), "th": np.zeros(NA), "k": 0} for s in (True, False)}
orig = G.Group2._pass
area_of_ch = torch.cat([torch.full((a.n_ch,), i, dtype=torch.long) for i, a in enumerate(g.areas)]).to(dev)
sel_ch = torch.full_like(area_of_ch, -1)
for nm in AREAS:
    sel_ch[area_of_ch == NAMES.index(nm)] = AREAS.index(nm)


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if (advance and st["tag"] != "他") else None
    out = orig(self, x_all, ext_all, learn, advance, first)
    if e is None or len(e) == 0:
        return out
    r = self.e_recv[e].long(); se = self.e_src[e].long()
    k = kind[se]; ar = area_of_recv[r]
    m = (k >= 0) & (ar >= 0)
    r, se, k, ar, ee = r[m], se[m], k[m], ar[m], e[m]
    amp = self.c_exc[r] * self.w[ee] * self._amp[ee] * self.send_gain[se] * (self.scale[r] * self.scale_up[r])
    idx = ar * NK + k
    q = torch.zeros(NA * NK, device=dev).index_add_(0, idx, amp).view(NA, NK)
    n = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(amp)).view(NA, NK)
    s = acc[st["snd"]]
    s["q"] += (q / n_area.unsqueeze(1)).cpu().numpy(); s["n"] += (n / n_area.unsqueeze(1)).cpu().numpy()
    s["qa"] += q.cpu().numpy()
    mm = sel_ch >= 0
    xs = torch.zeros(NA, device=dev).index_add_(0, sel_ch[mm], (x_all * self.x_gain)[mm])
    th = torch.zeros(NA, device=dev).index_add_(0, sel_ch[mm], (self.fire_th * self.th_gain).expand_as(x_all)[mm])
    s["x"] += (xs / n_area).cpu().numpy(); s["th"] += (th / n_area).cpu().numpy()
    s["k"] += 1
    return out


G.Group2._pass = hooked


def show(head):
    print(f"── {head}", flush=True)
    for snd, lab in ((True, "音あり"), (False, "無音")):
        s = acc[snd]; k = max(s["k"], 1)
        if s["k"] == 0:
            continue
        print(f"   {lab}（{s['k']}歩）", flush=True)
        for a, nm in enumerate(AREAS):
            tot = s["q"][a].sum() / k
            parts = "  ".join(f"{KINDS[j]} {s['q'][a, j] / k:7.2f}（{s['n'][a, j] / k:7.1f}本・重さ "
                              f"{s['qa'][a, j] / max(s['n'][a, j] * b.n[nm], 1e-9):6.3f}）" for j in range(NK))
            print(f"     {nm:5s} 閾値 {s['th'][a] / k:6.2f}  下から {s['x'][a] / k:6.2f}  横 {tot:7.2f}  {parts}", flush=True)
        for v in s.values():
            if isinstance(v, np.ndarray):
                v[:] = 0
        s["k"] = 0


w0, c0 = w_static()
print("0から：表現の体の興奮の繋がり（1体あたりの本数・重さの平均）", flush=True)
for a, nm in enumerate(AREAS):
    print("    " + f"{nm:5s} " + "  ".join(f"{KINDS[j]} {c0[a, j]:6.0f}本 {w0[a, j]:6.3f}" for j in range(NK)), flush=True)
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t0 = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if floor["v"] is None:
        floor["v"] = float(ear.sum(1).min()) + 1e-6
    if ear.sum() > 0:
        past.append((cmd, ear))
    if rp is not None:
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        for _ in range(N_PROBE):
            if len(past) > DELAY:
                c0_, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0_, e0, False, True)
        st["tag"] = "他"
        show(f"{i+1}回目まで（{(time.time()-t0)/60:.1f}分）この10回")
w1, c1 = w_static()
print("終わり：表現の体の興奮の繋がり（1体あたりの本数・重さの平均。0からとの比）", flush=True)
for a, nm in enumerate(AREAS):
    print("    " + f"{nm:5s} " + "  ".join(
        f"{KINDS[j]} {c1[a, j]:6.0f}本 {w1[a, j]:6.3f}（×{w1[a, j] / max(w0[a, j], 1e-9):5.1f}）" for j in range(NK)), flush=True)
print("読み：量は ÷(1＋抑制) の前。閾値・下から と同じ目盛り。⚠️ 最初の歩は1回目の喃語の頭（音あり／無音の決めが付く前）を数えない", flush=True)
