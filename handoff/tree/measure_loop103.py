"""版103：自分の発火が自分に戻る輪の作りと重さ（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は変えない）
  python3 measure_loop103.py [番の区切り=0,50,200]
  measure_track103（11-02 20:47）：暴走は 自分の表現の発火 → 自分の遅れの列 → 自分の表現 の輪（98〜100%）
  ⇒ 表現の行1つあたり、撃たせる側の興奮の相手のうち「自分の領野の表現の遅れの列」を数え、人と並べる
  【ヒト】L2-3 錐体どうし：繋がる確率 15.8%（1,094／6,911組、中央 0.46mV、3mV を越えるのはごく少ない）Planert ほか（Peng・Geiger）2025／
          1.12mV（Hunt 2023）／ 2.01 ± 1.10mV（16組、Szegedi 2016）・0.1〜6mV（Geiger 2026 講演）
  出すもの（区切りごと・領野ごと）
    相手 ＝ その行の撃たせる側の興奮の辺のうち自分の表現の列から来る本数 ／ 送り手の体 ＝ そのうち別々の体の数
    繋がる割合 ＝ 送り手の体 ÷ 領野の表現の体（人 15.8%）／ 自分自身（同じ体の列）から来る辺
    重さ w（中央・90%・最大）と 1本の量 ＝ 接触 × w（閾値 14.3 と同じ単位）と 1本の量 ×（相手の数）＝ 全部が撃ったときの和
    表現の率（Hz）
  ⚠️ 見るだけ。0 からの網・眠りなし
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MARKS = [int(x) for x in (args[0] if args else "0,50,200").split(",")]
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
WATCH = ("運動", "補足", "44", "中側頭後", "高次", "前頭前野", "眼窩前頭", "連合", "聴覚")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
ch_off, buf_off = [0], [0]
for a in g.areas:
    ch_off.append(ch_off[-1] + a.ch.n_cells); buf_off.append(buf_off[-1] + a.n_src * a.D)
bb = ch_off[-1]
CH = 1 << 23
W_ = [nm for nm in WATCH if nm in NAMES]
# 辺の番号は網の中で変わらない（刈り込みで並べ直すまで）⇒ 区切りごとに選び直す
TH = float(g.fire_th[int(b.off_ch[NAMES.index("44")])]) if "44" in NAMES else 14.3


def pick(nm):
    i = NAMES.index(nm); a = g.areas[i]
    lo = int(g.off_recv[i]); nc = a.n_ch
    s0 = bb + buf_off[i]; s1 = s0 + nc * a.D          # 自分の表現の遅れの列
    E = []
    for c0 in range(0, len(g.e_recv), CH):
        r_ = g.e_recv[c0:c0 + CH]; s_ = g.e_src[c0:c0 + CH]
        m = (r_ >= lo + nc) & (r_ < lo + 2 * nc) & (s_ >= s0) & (s_ < s1) & ~g.e_inh[c0:c0 + CH]
        E.append((torch.nonzero(m).squeeze(1) + c0))
    e = torch.cat(E)
    r = g.e_recv[e].long() - (lo + nc)                   # 受け手の体 0..nc-1
    j = (g.e_src[e].long() - s0) // a.D                 # 送り手の体 0..nc-1
    return e, r, j, nc


def report(tag, rep_hz):
    print(f"\n[{tag}]  閾値 {TH:.1f}（1 ＝ 人の1本の平均 1.12mV。⇒ {TH*1.12:.0f}mV）", flush=True)
    print(f"  {'領野':<8s} {'体':>5s} {'相手':>7s} {'送り手':>7s} {'繋がる':>7s} {'自分':>5s} {'w中央':>7s} {'w90%':>7s} {'w最大':>7s} {'1本最大':>8s}"
          f" {'1本中央':>8s} {'全部の和':>9s} {'率Hz':>6s}", flush=True)
    for nm in W_:
        e, r, j, nc = pick(nm)
        if len(e) == 0:
            print(f"  {nm:<8s} {nc:5d}  （自分の表現の列から来る辺なし）", flush=True); continue
        w = g.w[e].float()
        ce = g.c_exc[g.e_recv[e].long()].float()
        amp = ce * w
        per_row = torch.zeros(nc, device=dev).index_add_(0, r, torch.ones_like(w))
        uniq = torch.unique(r * nc + j)
        senders = torch.zeros(nc, device=dev).index_add_(0, uniq // nc, torch.ones(len(uniq), device=dev))
        self_e = int((r == j).sum())
        amp_row = torch.zeros(nc, device=dev).index_add_(0, r, amp)
        q = torch.quantile(w[:min(len(w), 1 << 24)], torch.tensor([0.5, 0.9], device=dev))
        print(f"  {nm:<8s} {nc:5d} {float(per_row.mean()):7.1f} {float(senders.mean()):7.1f} {float(senders.mean()) / nc:7.1%}"
              f" {self_e / nc:5.2f} {float(q[0]):7.3f} {float(q[1]):7.3f} {float(w.max()):7.2f} {float(amp.max()):8.1f}"
              f" {float(amp.median()):8.3f} {float(amp_row.mean()):9.1f} {rep_hz.get(nm, float('nan')):6.1f}", flush=True)
        del e, r, j, w, ce, amp, uniq
    torch.cuda.empty_cache() if dev != "cpu" else None


done = 0
for mk in MARKS:
    rep_n = torch.zeros(g.n_ch_all, device=dev); t0 = life.t
    if mk > done:
        P0 = life.step

        def st(*a, **kw):
            P0(*a, **kw)
            rep_n.add_(g._any[0][:g.n_ch_all].float())

        life.step = st
        life.run_plans(mk - done, learn=True)
        life.step = P0
        done = mk
    ns = max(1, life.t - t0)
    hz = {}
    for nm in W_:
        i = NAMES.index(nm); nc, oc = g.areas[i].n_ch, int(b.off_ch[i])
        hz[nm] = float(rep_n[oc:oc + nc].mean()) / ns / (10.67 / 1000.0) if life.t > t0 else float("nan")
    report(f"{mk}番（{life.t}歩）", hz)
print("\n繋がる：素子1体 ＝ ニューロン1個（10 §7）なので人の 15.8%（側頭葉の切片の L2-3）と並べられる。⚠️ 運動野の人の値は引いていない", flush=True)
print("1本の量：1 ＝ 人の1本の平均 1.12mV（Hunt 2023）。人の錐体どうしは 0.1〜6mV ⇒ 0.09〜5.4", flush=True)
