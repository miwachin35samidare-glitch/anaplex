"""版114：1歩の時間の内訳と、撃った辺（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_speed114.py [温め=6000] [歩=1000]
  （コンテナ：python3 measure_speed114.py 30 20 --test ⇒ CPU・相手 200）
  何のため：今の段は「実時間に合わせる ⇒ 遅さの元が暴走（撃った辺）っぽい ⇒ 速さの側面から暴走を抑える」（【み】11-22 08:08・11-01 17:55）。
    11-01（版103、measure_fire103）は撃った辺 1歩 1,770万本（辺の 18.7%）・撃った辺を読む核 1歩 12ms。率が人の近くに戻った今、
    1歩 33〜34ms のうち撃った辺に比例する分がどれだけ残っているかを見る（暴走を抑え続けるか、固定の手間を速くする段へ移るか）
  手順（温めのあと、同じ網で続けて）
    A 暮らしの1歩：何も差し替えずに 歩 回（1歩ごとに GPU を待って時間を取る）⇒ 中央・平均・最大
    B 撃った辺：歩 回。1歩の終わりに残る「その歩の始めに撃っていた送り手（時間細胞・遅れの列）」の印から、領野ごとの撃った辺と表現の体の率
       ⚠️ 周回の中で新しく撃った送り手の辺は入らない（下限）。合わせた数は網の持つ撃った辺の並びの長さと一致するかを出す
    C 部分ごとの時間：measure_speed98 と同じく部分ごとに GPU を待つ（待つぶん全体は少し遅く出る）
  毎歩に足す仕事：A は待ちだけ ／ B は撃った送り手の番号の取り出し 1回（数は撃った送り手の数）・その辺の数の累積和・領野の境の読み・
    表現の体の累積和 1回（体の数）。少ない箱への足し込み（index_add・scatter）なし・網と同じ大きさの写しなし ／ C は待ちだけ
    B・C は1歩ごとに GPU を待つ（測る仕事が次の歩の 0.5秒の守りに入らないように）
"""
import sys, time, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB, life as L
from torch_brain import Brain, NAMES

if B.VERSION not in (114,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版114 の網のため")
TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 6000
NS = int(args[1]) if len(args) > 1 else 1000
if TEST:
    B.DEV = "cpu"
dev = B.DEV
cuda = str(dev).startswith("cuda")
sync = torch.cuda.synchronize if cuda else (lambda: None)
DT = 10.67 / 1000.0
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
print(f"版 {B.VERSION} ／ 温め {WARM} 歩・測り {NS} 歩 × 3 ／ 網の中の辺 {len(g.w):,}・送り手 {g.n_send_all:,}・体 {g.n_ch_all:,}", flush=True)
t0 = time.time()
for i in range(WARM):
    life.step()
    if (i + 1) % 1000 == 0:
        L.beat()
        print(f"  温め {i + 1} 歩（{(time.time() - t0) / (i + 1) * 1000:.1f}ms/歩）", flush=True)

# ── A 暮らしの1歩 ──
sync()
ts = []
for i in range(NS):
    a0 = time.perf_counter()
    life.step()
    sync()
    ts.append((time.perf_counter() - a0) * 1000)
    if (i + 1) % 200 == 0:
        L.beat()
ts = np.array(ts)
rw = ts > 500
print(f"\nA 暮らしの1歩（{NS} 歩・1歩ごとに待つ）：中央 {np.median(ts):.1f}ms・平均 {ts.mean():.1f}ms・"
      f"0.5秒を越えた歩を除いた平均 {(f'{ts[~rw].mean():.1f}ms' if (~rw).any() else '—')}・最大 {ts.max():.0f}ms（0.5秒越え {int(rw.sum())} 歩 ＝ 刈り込みの歩）・実時間の {np.median(ts) / 10.67:.2f}倍（中央）", flush=True)

# ── B 撃った辺 ──
A_ = len(g.areas)
ch_off = [0]
for a in g.areas:
    ch_off.append(ch_off[-1] + a.ch.n_cells)
buf_off = [0]
for a in g.areas:
    buf_off.append(buf_off[-1] + a.n_src * a.D)
bb = ch_off[-1]
assert bb + buf_off[-1] == g.n_send_all, f"送り手の並びが思っていた形でない：{bb + buf_off[-1]} ≠ {g.n_send_all}"
segs = []                                                   # (領野の番号, lo, hi)
for i in range(A_):
    segs.append((i, ch_off[i], ch_off[i + 1]))
    segs.append((i, bb + buf_off[i], bb + buf_off[i + 1]))
seg_lo = torch.tensor([s[1] for s in segs], device=dev)
seg_hi = torch.tensor([s[2] for s in segs], device=dev)
seg_a = torch.tensor([s[0] for s in segs], device=dev)
start = g.start
deg_all = [int(start[hi] - start[lo]) for _, lo, hi in segs]
edges_by = np.zeros(A_)
for (i, _, _), d in zip(segs, deg_all):
    edges_by[i] += d
fe_seg = torch.zeros(len(segs), dtype=torch.float64, device=dev)
fs_seg = torch.zeros(len(segs), dtype=torch.float64, device=dev)
tot_list, chk = 0.0, []
off_ch = [int(x) for x in b.off_ch]
lo_c = torch.tensor(off_ch[:-1], device=dev); hi_c = torch.tensor(off_ch[1:], device=dev)
rep_acc = torch.zeros(len(NAMES), dtype=torch.float64, device=dev)
z1 = torch.zeros(1, dtype=torch.float64, device=dev)
for i in range(NS):
    life.step()
    on = getattr(g, "_base_on", None)
    assert on is not None and len(on) == g.n_send_all, "撃った送り手の印が思っていた形でない"
    idx = torch.nonzero(on).squeeze(1)                     # 撃った送り手（並びは小さい順）
    d = (start[idx + 1] - start[idx]).to(torch.float64)
    cd = torch.cat([z1, d.cumsum(0)])
    p_lo = torch.searchsorted(idx, seg_lo); p_hi = torch.searchsorted(idx, seg_hi)
    fe_seg += cd[p_hi] - cd[p_lo]
    fs_seg += (p_hi - p_lo).to(torch.float64)
    be = getattr(g, "_base_e", None)
    if be is not None and i < 50:
        chk.append((float(cd[-1]), len(be)))
    c_ = torch.cat([z1, g._any[0][:g.n_ch_all].to(torch.float64).cumsum(0)])
    rep_acc += c_[hi_c] - c_[lo_c]
    sync()
    if (i + 1) % 200 == 0:
        L.beat()
fe = np.zeros(A_); fs = np.zeros(A_); ns_ = np.zeros(A_)
fe_s, fs_s = fe_seg.cpu().numpy(), fs_seg.cpu().numpy()
for k, (ia, lo, hi) in enumerate(segs):
    fe[ia] += fe_s[k]; fs[ia] += fs_s[k]; ns_[ia] += hi - lo
tot = fe.sum() / NS
ok_chk = all(abs(a - b_) < 0.5 for a, b_ in chk) if chk else None
print(f"\nB 撃った辺（歩の始めに撃っていた送り手の分・{NS} 歩の平均）：1歩 {tot:,.0f}本（網の中の辺 {len(g.w):,} の {tot / len(g.w):.1%}）"
      f" ／ 網の持つ撃った辺の並びと一致：{ok_chk}（初めの {len(chk)} 歩）", flush=True)
print(f"  （版103 11-01：1歩 1,770万本・辺の 18.7%）", flush=True)
rep_hz = rep_acc.cpu().numpy() / NS / DT
rows = []
for i, nm in enumerate(NAMES):
    n_ = off_ch[i + 1] - off_ch[i]
    rows.append((fe[i] / NS, nm, fs[i] / NS / max(1.0, ns_[i]), rep_hz[i] / max(1, n_), edges_by[i]))
print(f"  {'領野':<8s} {'撃った辺/歩':>12s} {'割合':>6s} {'送り手の撃つ割合':>10s} {'表現 Hz':>8s} {'辺':>12s}", flush=True)
for fe_, nm, fsr, rh, ed in sorted(rows, reverse=True)[:25]:
    print(f"  {nm:<8s} {fe_:12,.0f} {fe_ / max(1.0, tot):6.1%} {fsr:10.1%} {rh:8.2f} {ed:12,.0f}", flush=True)
L.beat()

# ── C 部分ごとの時間 ──
acc = collections.defaultdict(float)


def timed(name, f):
    def w(*a, **k):
        sync(); t1 = time.perf_counter()
        r = f(*a, **k)
        sync(); acc[name] += time.perf_counter() - t1
        return r
    return w


parts = [("網の領野（Group2）", b.g, "step"), ("下からの線の表（足す）", b.but, "drive"), ("下からの線の表（学ぶ）", b.but, "learn"),
         ("抑制の線", getattr(b, "but_inh", None), "drive"), ("平行線維（学ぶ）", b, "_pf_learn"), ("rr の学び（版113）", b, "_rr_learn")]
if TB.PF_KEY in b.bu:
    parts.append(("平行線維（足す）", b.bu[TB.PF_KEY], "drive"))
if getattr(b.g, "nucnet", None) is not None:
    parts.append(("核どうしの学び", b.g.nucnet, "learn"))
for name, obj, attr in parts:
    if obj is not None and hasattr(obj, attr):
        setattr(obj, attr, timed(name, getattr(obj, attr)))
b.step = timed("1歩ぜんぶ（網の外の暮らしは入らない）", b.step)
sync(); t0 = time.perf_counter()
for i in range(NS):
    life.step()
    if (i + 1) % 200 == 0:
        L.beat()
sync(); tc = (time.perf_counter() - t0) / NS * 1000
print(f"\nC 部分ごとの時間（{NS} 歩・部分ごとに待つので A より少し遅い）：暮らしの1歩 {tc:.1f}ms", flush=True)
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<26s} {v / NS * 1000:7.2f}ms", flush=True)
print("  （版99 の内訳：網の領野 34.49・下からの表 足す 8.16ms ほか ／ 版96：網 12.56・足す 3.58・学ぶ 0.59・核どうし 0.20・外 1.53ms）", flush=True)
if cuda:
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
