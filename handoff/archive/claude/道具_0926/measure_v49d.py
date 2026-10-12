"""版49：視床・CA1・上丘の表現の体を撃たせている「横の興奮」が、どの送り手から来るかを分ける（arkhe の ~/anaplex で）
  python3 measure_v49d.py [喃語の回数=30]
measure_v49c と同じ回し方（0から・学びながら・返し手あり・保存しない）。比べるために聴覚・連合も並べる
【測】measure_v49c 50（10-03）：視床・CA1・上丘は撃った歩の 90〜100% が「下＋横」。横は 13〜240（聴覚・連合 2〜7）
横（表現の体へ届いた興奮。抑制と上からの戻り・写しは除く）を送り手の種類で4つに分ける
  自分の遅れ   自分の領野の表現の発火そのもの（数歩の遅れの写し。同じ歩の分も含む）
  自分の時間   自分の領野の表現の発火のあとに決まった間で撃つ時間細胞
  傾き        傾きの細胞（入力からの経過で率が変わる。0.3Hz の乱数）
  横の入口     樹状突起の入口のうち、撃たせる側のまま（未決）のもの（CA1 は 嗅内浅 層 III。視床・上丘は無い）
並べるもの（10回ごと、表現の体1体・1歩の平均）
  量    その種類から届いた興奮（v49c の「横」をこの4つに割ったもの）
  本    その種類から届いた繋がりの本数
  重さ   届いた繋がりの重さの平均（× 接触の数 × 短期の変化 × 送り手の倍率 × 恒常性の倍率）
最後に、表現の体の興奮の重さ（届いたかに関係なく全部）の平均を、0から（1回目の前）と終わりで並べる
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
AREAS = ("視床", "CA1", "上丘", "聴覚", "連合")
KINDS = ("自分の遅れ", "自分の時間", "傾き", "横の入口")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
dev = g.dev

# ── 送り手の種類の表（-1 ＝ 数えない：抑制・上からの戻り・写し）──
kind = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
buf_base = int(g.off_send[-1])
buf_off = 0
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    lo = int(g.off_send[i])
    nc = a.n_ch + a.n_ext                      # 抑制の行は数えない
    for r_lo, r_hi, own in ((0, a.n_ch, True), (a.n_ch, nc, False)):
        if r_hi <= r_lo:
            continue
        rows = torch.arange(r_lo, r_hi, device=dev)
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
is_rep_recv = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev); is_rep_recv[rep_recv] = True
# 受け手（表現の体）→ 領野の番号
area_of_recv = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=dev)
sl = {}
for nm in AREAS:
    lo = int(b.off_ch[NAMES.index(nm)]); sl[nm] = (lo, lo + b.n[nm])
    area_of_recv[rep_recv[lo:lo + b.n[nm]]] = AREAS.index(nm)
NA, NK = len(AREAS), len(KINDS)


def w_static():
    """表現の体の興奮の重さの平均（種類ごと。届いたかに関係なく）"""
    r = g.e_recv.long(); se = g.e_src.long()
    k = kind[se]; ar = area_of_recv[r]
    m = (k >= 0) & (ar >= 0)
    idx = ar[m] * NK + k[m]
    s = torch.zeros(NA * NK, device=dev).index_add_(0, idx, g.w[m])
    c = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(g.w[m]))
    return (s / c.clamp(min=1)).view(NA, NK).cpu().numpy(), (c.view(NA, NK) / torch.tensor(
        [b.n[nm] for nm in AREAS], device=dev).unsqueeze(1)).cpu().numpy()


w0, c0 = w_static()
acc = np.zeros((3, NA, NK)); nstep = [0]
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges()
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not advance or len(e) == 0:
        return out
    r = self.e_recv[e].long(); se = self.e_src[e].long()
    k = kind[se]; ar = area_of_recv[r]
    m = (k >= 0) & (ar >= 0)
    r, se, k, ar, ee = r[m], se[m], k[m], ar[m], e[m]
    amp = self.c_exc[r] * self.w[ee] * self._amp[ee] * self.send_gain[se] * self.scale[r]
    idx = ar * NK + k
    q = torch.zeros(NA * NK, device=dev).index_add_(0, idx, amp).view(NA, NK)
    n = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(amp)).view(NA, NK)
    per = torch.tensor([b.n[nm] for nm in AREAS], device=dev, dtype=torch.float).unsqueeze(1)
    acc[0] += (q / per).cpu().numpy(); acc[1] += (n / per).cpu().numpy(); acc[2] += q.cpu().numpy()
    nstep[0] += 1
    return out


G.Group2._pass = hooked


def show():
    k = max(nstep[0], 1)
    for a, nm in enumerate(AREAS):
        tot = acc[0, a].sum() / k
        parts = "  ".join(
            f"{KINDS[j]} {acc[0, a, j] / k:6.1f}（{acc[1, a, j] / k:6.1f}本・重さ "
            f"{(acc[2, a, j] / max(acc[1, a, j] * b.n[nm], 1e-9)):5.2f}）" for j in range(NK))
        print(f"    {nm} 横 {tot:6.1f}  {parts}", flush=True)


print("0から：表現の体の興奮の繋がり（1体あたりの本数・重さの平均）", flush=True)
for a, nm in enumerate(AREAS):
    print("    " + nm + "  " + "  ".join(f"{KINDS[j]} {c0[a, j]:6.0f}本 {w0[a, j]:5.3f}" for j in range(NK)), flush=True)
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    say(rng, True)
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        print(f"{i+1:4d}回  （{(time.time()-t0)/60:.1f}分）", flush=True)
        show()
        acc[:] = 0; nstep[0] = 0
w1, c1 = w_static()
print("終わり：表現の体の興奮の繋がり（1体あたりの本数・重さの平均。0からとの比）", flush=True)
for a, nm in enumerate(AREAS):
    print("    " + nm + "  " + "  ".join(
        f"{KINDS[j]} {c1[a, j]:6.0f}本 {w1[a, j]:5.3f}（×{w1[a, j] / max(w0[a, j], 1e-9):4.1f}）" for j in range(NK)), flush=True)
