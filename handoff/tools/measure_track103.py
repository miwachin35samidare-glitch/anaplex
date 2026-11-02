"""版103：暴走が残る所と、網が黙る所の原因を分ける（arkhe の ~/anaplex で。0から・学びながら・保存しない）
  python3 measure_track103.py [番=300] [区切り=25]
  measure_babble103（11-02 7:31）：運動・44・中側頭後・被殻は 103g でも暴走 ／ 300〜400番で網が黙る（皮質 0Hz）
  区切りごとに出すもの（受け手は表現の体の行）
    ① 表現の率（Hz）
    ② 脳幹：眠りの圧・覚醒・閾値の倍率 th_gain（覚醒が下がると閾値が上がる）・眠る時か
    ③ 恒常性の倍率 scale（眠りで下げる）・scale_up（起きている間に上げる）の平均
    ④ 興奮の重さの和（表現の行1つあたり。学びで動く）
    ⑤ 網の辺の出どころ（区切りの頭 20歩。自分の時間細胞・自分の表現の遅れの列・ext の遅れの列・ほかの領野。調整する側は除く）
  ⚠️ 0 からの網・眠りなし。見るだけ（ノブはこれを見てから）
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NT = int(args[0]) if len(args) > 0 else 300
W = int(args[1]) if len(args) > 1 else 25
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
WATCH = ("運動", "補足", "44", "中側頭後", "高次", "前頭前野", "眼窩前頭", "連合", "聴覚", "被殻")
SRC_N = 20
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g, bs = b.g, b.bs
A = len(g.areas)
ch_off, buf_off = [0], [0]
for a in g.areas:
    ch_off.append(ch_off[-1] + a.ch.n_cells); buf_off.append(buf_off[-1] + a.n_src * a.D)
bb = ch_off[-1]
lab = torch.empty(g.n_send_all, dtype=torch.long, device=dev)
for i in range(A):
    lab[ch_off[i]:ch_off[i + 1]] = i
    lab[bb + buf_off[i]:bb + buf_off[i + 1]] = i
is_buf = torch.arange(g.n_send_all, device=dev) >= bb
# 遅れの列の中の送り元：領野の送り元は [表現 n_ch ・ ext（ほかの領野から写した入口）n_ext ・ 抑制]、列は 送り元 × D の順
is_ext = torch.zeros(g.n_send_all, dtype=torch.bool, device=dev)
for i, a in enumerate(g.areas):
    o = bb + buf_off[i]
    is_ext[o + a.n_ch * a.D:o + (a.n_ch + a.n_ext) * a.D] = True
W_ = [nm for nm in WATCH if nm in NAMES]
rows = {}
for nm in W_:
    i = NAMES.index(nm); lo = int(g.off_recv[i]); nc = g.areas[i].n_ch
    rows[nm] = (i, lo + nc, lo + 2 * nc, nc, int(b.off_ch[i]))
r_all = g.e_recv.long()
sel = {}
for nm, (i, a0, a1, nc, oc) in rows.items():
    m = (r_all >= a0) & (r_all < a1) & ~g.e_inh
    sel[nm] = torch.nonzero(m).squeeze(1)
del r_all
sync = (lambda: torch.cuda.synchronize()) if dev != "cpu" else (lambda: None)
f = lambda x: float(x) if not torch.is_tensor(x) else float(x.float().mean())
out = []


def window(n):
    st0 = dict(life.stats); t0s = life.t
    rep_n = torch.zeros(g.n_ch_all, device=dev)
    src = {nm: torch.zeros(4, device=dev, dtype=torch.float64) for nm in W_}   # 自分の時間細胞・自分の表現の遅れの列・ext の遅れの列・ほか
    k = [0]
    P0 = life.step

    def st(*a, **kw):
        P0(*a, **kw)
        rep_n.add_(g._any[0][:g.n_ch_all].float())
        on = getattr(g, "_base_on", None)
        if k[0] < SRC_N and on is not None:
            for nm, e in sel.items():
                s = g.e_src[e].long()
                live = (on[s] > 0) & ~g.mod_send[s]
                amp = (g.w[e] * g.c_exc[g.e_recv[e].long()] * g.send_gain[s] * live.float()).double()
                own = lab[s] == rows[nm][0]
                c = torch.where(own & ~is_buf[s], 0, torch.where(own & ~is_ext[s], 1, torch.where(own, 2, 3)))
                src[nm].index_add_(0, c, amp)
        k[0] += 1

    life.step = st
    sync(); t0 = time.time()
    life.run_plans(n, learn=True)
    sync()
    life.step = P0
    ns = max(1, life.t - t0s)
    d = {kk: life.stats[kk] - st0[kk] for kk in life.stats}
    wsum = torch.zeros(g.n_recv_all, device=dev).index_add_(0, g.e_recv.long(), g.w * (~g.e_inh).float())
    rec = {"ms": (time.time() - t0) / ns * 1000.0, "歩": ns, "声": d["voices"], "言えた": f"{d['ok']}/{d['tried']}",
           "圧": f(bs.pressure), "覚醒": f(bs.arousal), "th": f(g.th_gain), "眠る時": bool(bs.sleep_due), "a": {}}
    for nm, (i, a0, a1, nc, oc) in rows.items():
        sc = g.scale[a0:a1] if g.scale.numel() == g.n_recv_all else torch.full((1,), float("nan"))
        su = g.scale_up[a0:a1] if g.scale_up.numel() == g.n_recv_all else torch.full((1,), float("nan"))
        v = src[nm]; tot = float(v.sum())
        rec["a"][nm] = (float(rep_n[oc:oc + nc].mean()) / ns / (10.67 / 1000.0), f(sc), f(su), float(wsum[a0:a1].mean()),
                        tuple(float(x) / tot for x in v) if tot > 0 else (float("nan"),) * 4)
    out.append(rec)
    print(f"  {sum(1 for _ in out) * W:4d}番  {rec['歩']:5d}歩  {rec['ms']:5.1f}ms/歩  声 {rec['声']:3d}  言えた {rec['言えた']:>6s}  "
          f"圧 {rec['圧']:.2f}  覚醒 {rec['覚醒']:.2f}  th_gain {rec['th']:.3f}  眠る時 {'はい' if rec['眠る時'] else 'いいえ'}", flush=True)


print(f"版 {B.VERSION} ／ 0から {NT}番 ／ 区切り {W}番（眠りなし）", flush=True)
done = 0
while done < NT:
    n = min(W, NT - done)
    window(n); done += n
tags = [f"{(j + 1) * W}" for j in range(len(out))]
for title, fn, fmt in (("① 表現の率 Hz", lambda r: r[0], "{:8.1f}"),
                       ("③ 恒常性 scale", lambda r: r[1], "{:8.3f}"),
                       ("③ 恒常性 scale_up", lambda r: r[2], "{:8.3f}"),
                       ("④ 興奮の重さの和（行1つあたり）", lambda r: r[3], "{:8.2f}"),
                       ("⑤ 網の出どころ：自分の時間細胞", lambda r: r[4][0], "{:8.0%}"),
                       ("⑤ 網の出どころ：自分の表現の遅れの列（自分の発火が戻る）", lambda r: r[4][1], "{:8.0%}"),
                       ("⑤ 網の出どころ：ext の遅れの列（ほかの領野から写した入口・上からの戻り）", lambda r: r[4][2], "{:8.0%}"),
                       ("⑤ 網の出どころ：ほかの領野の送り手", lambda r: r[4][3], "{:8.0%}")):
    print(f"\n{title}（番）", flush=True)
    print(f"  {'領野':<8s}" + "".join(f"{t:>8s}" for t in tags), flush=True)
    for nm in W_:
        print(f"  {nm:<8s}" + "".join((fmt.format(fn(r['a'][nm])) if fn(r['a'][nm]) == fn(r['a'][nm]) else f"{'—':>8s}") for r in out), flush=True)
