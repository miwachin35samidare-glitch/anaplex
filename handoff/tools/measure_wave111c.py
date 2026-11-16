"""版110・111：揺れを「波」として見る —— 全部の領野・核の表現の発火を1歩ごとに取り、速い揺れ・領野どうしの先後を出す（arkhe の ~/anaplex で。網は変えない）
  python3 measure_wave111c.py [歩=15000] [CLの体数]
    CLの体数を書くと、版111 の上で CL の体数と CL → 皮質 の線（受け手1体の 40本を別々の体から）だけ差し替える（measure_cl112 と同じ差し替え）。表は wave111c_cl体数.npz
  何のため：【み】11-16 22:37「波が出ているのかも見逃さないようにしてね」（10 §1：波・渦は外せない／波は伝達とシナプスの経路の変化が立ち上げるもの）
    wave111・111b は刻み 200歩（2.1秒）の平均なので、見えたのは約40秒の揺れだけ。その中の速い揺れ（歩の刻み 10.67ms ⇒ 約47Hz まで）と、
    領野から領野へ伝わる先後（領野どうしは1歩に1回伝わる、10 §4）は見えていない ⇒ 揺れを直す前に、波かどうか・何を崩すかを見る
  取るもの：毎歩、領野・核ごとの「撃った体の割合」（歩 × 領野の表）⇒ 終わりに wave111c.npz に書く（測りの表だけ。網は保存しない）
  出すもの（終わりに1回。始めの 3000歩は捨てる）
    ① 皮質（運動の外）の平均の揺れの帯ごとの強さ（全体に対する割合）と、帯の中で一番強い周期
    ② 遅い揺れの先後：100歩でならした率を皮質の平均と比べ、一番合うずれ（歩。負＝皮質の平均より先）と合い方。領野・核ごと
    ③ 速い先後：1歩ごとの率（100歩のならしを引いたもの）を皮質の平均と比べ、ずれ −3〜＋3歩の合い方。領野・核ごと
    ④ 立ち上がりの順：皮質の平均が谷から山へ上がるたび、各領野が「自分の山の半分」を越えた歩（皮質の平均が越えた歩との差）
  毎歩に足す仕事：表現の発火（体の数 110,406）の累積和 1回・領野の端の読み（領野の数）・表の1行への書き込み。
    線・辺には触らない・少ない箱への足し込み（index_add・scatter）なし・網と同じ大きさの写しは累積和の1本だけ（体の数）
  歩は life.step（見張り・休み・止めはそのまま働く）。グラフで回す（ふだんと同じ）。終わりの計算は CPU（numpy）で、区切りごとに beat
  ⚠️ 位置（空間の並び）は領野の中に無いので、見えるのは領野・核どうしの先後だけ（領野の中を進む波・渦はこの道具では見えない）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 15000
NCL = int(args[1]) if len(args) > 1 else None
OUT = f"wave111c_cl{NCL}.npz" if NCL else "wave111c.npz"
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION not in (110, 111):
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版110・111 で回す")

if NCL:
    assert B.VERSION == 111 and TB.N_CL == 5 and TB.N_NEW["CL"] == 5, "CL の差し替えは版111（N_CL 5）の上で"
    TB.N_CL = NCL
    TB.N_NEW["CL"] = NCL
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
if NCL:
    nr_ = len(b.cl_ch)
    K = len(b.cl_w) // nr_
    assert b.n["CL"] == NCL and NCL >= K
    gc_ = torch.Generator(device="cpu"); gc_.manual_seed(112)
    src = torch.empty(nr_, K, dtype=torch.long)
    for a0 in range(0, nr_, 512):
        src[a0:a0 + 512] = torch.rand(min(512, nr_ - a0), NCL, generator=gc_).argsort(1)[:, :K]
    b.cl_src = src.reshape(-1).to(device=b.cl_src.device, dtype=b.cl_src.dtype)
    print(f"CL {NCL} 体・CL → 皮質 は受け手1体 {K} 本を別々の体から", flush=True)
n_all = g.n_ch_all
DT = 10.67 / 1000.0
off = [int(x) for x in b.off_ch]
assert off[-1] <= n_all, "領野の端が体の数を越えている"
lo = torch.tensor(off[:-1], device=dev)
hi = torch.tensor(off[1:], device=dev)
size = np.array([off[i + 1] - off[i] for i in range(len(NAMES))], dtype=np.float64)
CXI = [i for i, n in enumerate(NAMES) if TB.PRIOR_TYPE.get(n) == "皮質" and n != "運動"]
R = torch.zeros(NS, len(NAMES), device=dev)
zero = torch.zeros(1, device=dev)

print(f"版 {B.VERSION} ／ {NS} 歩 ／ 領野・核 {len(NAMES)}・皮質（運動の外）{len(CXI)} ／ 体 {n_all:,}", flush=True)
for i in range(NS):
    life.step(learn=True)
    c = torch.cat([zero, g._any[0][:n_all].float().cumsum(0)])
    R[i] = c[hi] - c[lo]
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1} 歩", flush=True)
L.beat()
X = R.cpu().numpy().astype(np.float64) / np.maximum(size, 1.0)[None, :]     # 歩 × 領野：撃った体の割合
np.savez_compressed(OUT, rate=X.astype(np.float32), names=np.array(NAMES), dt=DT, version=B.VERSION)
L.beat()

SKIP = min(3000, NS // 5)
X = X[SKIP:]
n = len(X)
hz = X / DT
cx = hz[:, CXI].mean(1)
print(f"\n（始めの {SKIP} 歩を捨てた。残り {n} 歩 ＝ {n * DT:.0f} 秒）皮質の平均 {cx.mean():.2f}Hz・1発も撃たない歩 {float((X[:, CXI].sum(1) == 0).mean()):.0%}", flush=True)

# ① 帯ごとの強さ
x0 = cx - cx.mean()
P = np.abs(np.fft.rfft(x0 * np.hanning(n))) ** 2
F = np.fft.rfftfreq(n, DT)
tot = P[1:].sum()
print("\n① 皮質の平均の揺れ：帯ごとの強さ（全体に対する割合）と帯の中で一番強い周期", flush=True)
for a_, b_, nm in ((0, 0.1, "0.1Hz より遅い"), (0.1, 1, "0.1〜1Hz"), (1, 4, "1〜4Hz（δ）"), (4, 8, "4〜8Hz（θ）"), (8, 13, "8〜13Hz（α）"),
                   (13, 30, "13〜30Hz（β）"), (30, 47, "30〜47Hz（γ の下）")):
    m = (F > a_) & (F <= b_) if a_ > 0 else (F > 0) & (F <= b_)
    if not m.any():
        continue
    k = np.argmax(np.where(m, P, -1))
    print(f"  {nm:<16s}{P[m].sum() / tot:7.1%}   一番強い {F[k]:7.3f}Hz（周期 {1 / F[k]:7.2f} 秒）", flush=True)
L.beat()


def smooth(v, w=100):
    w = max(1, min(w, n // 4))
    k = np.ones(w) / w
    return np.convolve(v, k, mode="same")


def corr(a, b_, lag):
    """a（領野）を lag 歩 遅らせて b_（皮質の平均）と合わせる ⇒ 正の lag で合う ＝ 領野が皮質の平均より遅れている"""
    if lag > 0:
        u, v = a[lag:], b_[:-lag]
    elif lag < 0:
        u, v = a[:lag], b_[-lag:]
    else:
        u, v = a, b_
    su, sv = u.std(), v.std()
    return float(((u - u.mean()) * (v - v.mean())).mean() / (su * sv)) if su > 0 and sv > 0 else float("nan")


cs = smooth(cx)
LAGS = [k for k in range(-600, 601, 20) if abs(k) < n // 4]
print("\n② 遅い揺れの先後（100歩のならし。ずれ：負＝皮質の平均より先・正＝あと。1歩 10.67ms）／ ③ 速い先後（ならしを引いた1歩ごと。ずれ −3〜＋3歩の合い方）", flush=True)
print(f"  {'領野':<8s}{'平均Hz':>7s} ｜ {'ずれ(歩)':>7s}{'合い方':>7s} ｜ " + "".join(f"{k:+6d}" for k in range(-3, 4)), flush=True)
rows = []
for j, nm in enumerate(NAMES):
    v = hz[:, j]
    if v.std() == 0:
        rows.append((9999, f"  {nm:<8s}{v.mean():7.2f} ｜ （撃たない）"))
        continue
    vs = smooth(v)
    cc = [corr(vs, cs, k) for k in LAGS]
    kb = int(np.nanargmax(cc)) if not np.all(np.isnan(cc)) else 0
    fast = [corr(v - vs, cx - cs, k) for k in range(-3, 4)]
    rows.append((LAGS[kb], f"  {nm:<8s}{v.mean():7.2f} ｜ {LAGS[kb]:+7d}{cc[kb]:7.2f} ｜ " + "".join(f"{x:6.2f}" for x in fast)))
    if j % 10 == 0:
        L.beat()
for _, s in sorted(rows, key=lambda r: r[0]):
    print(s, flush=True)
L.beat()

# ④ 立ち上がりの順
half = (cs.max() + cs.min()) / 2
up = [t for t in range(1, n) if cs[t - 1] < half <= cs[t]]
ups = [t for k, t in enumerate(up) if k == 0 or t - up[k - 1] > min(1000, n // 4)]
print(f"\n④ 立ち上がりの順：皮質の平均（100歩のならし）が {half:.2f}Hz を下から越えた歩（捨てたあとの番号）{ups}", flush=True)
if ups:
    d = {}
    for j, nm in enumerate(NAMES):
        vs = smooth(hz[:, j])
        h = (vs.max() + vs.min()) / 2
        if vs.max() - vs.min() <= 0:
            continue
        out = []
        for t in ups:
            a0, a1 = max(1, t - 800), min(n, t + 800)
            seg = [u for u in range(a0, a1) if vs[u - 1] < h <= vs[u]]
            out.append(min(seg, key=lambda u: abs(u - t)) - t if seg else None)
        d[nm] = out
    order = sorted(d, key=lambda k: np.mean([x for x in d[k] if x is not None]) if any(x is not None for x in d[k]) else 9999)
    print("  （皮質の平均との差。負＝先に上がる。— は前後 800歩に越えた歩なし）", flush=True)
    for nm in order:
        print(f"  {nm:<8s}" + "".join(f"{'—':>7s}" if x is None else f"{x:+7d}" for x in d[nm]), flush=True)
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print(f"\n（終わり）{OUT} に 歩 × 領野 の表を書いた", flush=True)
