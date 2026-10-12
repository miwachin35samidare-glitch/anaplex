"""版51：表現の体が受ける抑制が、どの送り手から来るかを分ける（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51t.py [喃語の回数=30]
  （コンテナで道具を確かめるとき：python3 measure_v51t.py 2 --test ⇒ CPU・相手 200・作り物の声）
【測】measure_v51r（10-05）：的が立った表現の体に届いた抑制 連合 58〜81本・前頭前野 57〜75本／歩、ihn 3〜5
【測】measure_v51s（10-06）：抑制の体そのものは 聴覚 0.3〜0.5% ／ 連合・高次・前頭前野 1.1〜3.0%
見立て【C の算数】：表現の体の抑制の相手（相手の 16%）は、抑制の行の送り手から一様に引く（torch_area の inh_pool）
  抑制の行の送り手 ＝ 抑制の体そのもの（buf、D 本）＋ 抑制の行の時間細胞（128）＋ 傾きの細胞（8）
  ⇒ 抑制の体そのものは 約0.7%。届く抑制の大半は「抑制の体が撃ったあと、遅れて撃つ時間細胞」の見込み
送り手の種類（抑制の側）
  抑制の体    抑制の体の発火そのもの（buf）
  抑制の時間   抑制の体が撃ったあと決まった間で撃つ時間細胞（341ms まで）
  抑制の傾き   抑制の行の傾きの細胞（RAMP_P 0.3Hz）
並べるもの（10回ごと。歩を進める周＝最後の周の値。表現の体1体・1歩の平均）
  種類ごと    届いた本数 ／ 抑制の量（ihn への寄与。接触 4.0 × 重さ × 短期の変化 × 恒常性の倍率）
  ihn       表現の体の抑制の合計の平均
最初に、表現の体1体の抑制の相手の内訳（本数）を種類ごとに出す
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 30
if TEST:
    B.DEV = "cpu"
AREAS = ("聴覚", "連合", "高次", "前頭前野", "運動", "視床", "CA1", "嗅内深", "上丘")
KINDS = ("抑制の体", "抑制の時間", "抑制の傾き")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev
AREAS = tuple(nm for nm in AREAS if nm in NAMES)

# ── 抑制の送り手の種類の表（-1 ＝ 興奮の送り手）──
kind = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
buf_base = int(g.off_send[-1])
buf_off = 0
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    lo = int(g.off_send[i])
    rows = torch.arange(a.n_ch + a.n_ext, a.n_src, device=dev)
    tc = lo + rows.unsqueeze(1) * per + torch.arange(per, device=dev)
    if a.ch.n_cell:
        kind[tc[:, :a.ch.n_cell].reshape(-1)] = 1
    kind[tc[:, a.ch.n_cell:].reshape(-1)] = 2
    bf = buf_base + buf_off + rows.unsqueeze(1) * a.D + torch.arange(a.D, device=dev)
    kind[bf.reshape(-1)] = 0
    buf_off += a.n_src * a.D
bad = int(((kind >= 0) & ~g.is_inh_all).sum()) + int(((kind < 0) & g.is_inh_all).sum())
if bad:
    sys.exit(f"⚠️ 抑制の送り手の表が is_inh_all と {bad} 本食い違う（並びを読み違えている）")

# ── 受け手（表現の体）→ 領野の番号 ──
area_of_recv = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=dev)
n_rep = {}
for nm in AREAS:
    i = NAMES.index(nm); a = g.areas[i]
    lo = int(g.off_recv[i]) + a.n_ch            # 受け手の並び：誤差 n_ch → 表現 n_ch → 抑制
    area_of_recv[lo:lo + a.n_ch] = AREAS.index(nm)
    n_rep[nm] = a.n_ch
NA, NK = len(AREAS), len(KINDS)
per_area = torch.tensor([n_rep[nm] for nm in AREAS], device=dev, dtype=torch.float)

r_all = g.e_recv.long(); k_all = kind[g.e_src.long()]; ar_all = area_of_recv[r_all]
m = (k_all >= 0) & (ar_all >= 0)
c0 = (torch.zeros(NA * NK, device=dev).index_add_(0, ar_all[m] * NK + k_all[m], torch.ones(int(m.sum()), device=dev))
      .view(NA, NK) / per_area.unsqueeze(1)).cpu().numpy()
del r_all, k_all, ar_all, m
print("表現の体1体の抑制の相手（本数）", flush=True)
for a_, nm in enumerate(AREAS):
    tot = c0[a_].sum()
    print(f"    {nm:5s} " + "  ".join(f"{KINDS[j]} {c0[a_, j]:7.1f}（{c0[a_, j]/max(tot,1e-9):5.1%}）" for j in range(NK)), flush=True)

acc_n = np.zeros((NA, NK)); acc_q = np.zeros((NA, NK)); nstep = [0]
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges()
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not advance:
        return out
    inh = self.e_inh[e]
    ee = e[inh]
    r = self.e_recv[ee].long(); se = self.e_src[ee].long()
    ar = area_of_recv[r]; k = kind[se]
    mm = ar >= 0
    r, se, ar, k, ee = r[mm], se[mm], ar[mm], k[mm], ee[mm]
    amp = 4.0 * self.w[ee] * self._amp[ee] * self.send_gain[se] * self.scale[r]
    idx = ar * NK + k
    q = torch.zeros(NA * NK, device=dev).index_add_(0, idx, amp).view(NA, NK)
    n = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(amp)).view(NA, NK)
    acc_q[:] += (q / per_area.unsqueeze(1)).cpu().numpy(); acc_n[:] += (n / per_area.unsqueeze(1)).cpu().numpy()
    nstep[0] += 1
    return out


G.Group2._pass = hooked
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    say(rng, True)
    if rp is not None:                     # 作り物の声（--test）には返し手が無い
        B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        k = max(nstep[0], 1)
        print(f"── {i+1}回目まで（{(time.time()-t0)/60:.1f}分）  表現の体1体・1歩の平均（本数 ／ 抑制の量）", flush=True)
        for a_, nm in enumerate(AREAS):
            tot = acc_q[a_].sum() / k
            parts = "  ".join(f"{KINDS[j]} {acc_n[a_, j]/k:6.2f}本 {acc_q[a_, j]/k:6.3f}（{acc_q[a_, j]/max(acc_q[a_].sum(),1e-12):5.1%}）"
                              for j in range(NK))
            print(f"   {nm:5s} ihn {tot:6.3f}  {parts}", flush=True)
        acc_n[:] = 0; acc_q[:] = 0; nstep[0] = 0
print("【ヒト】速い籠細胞の抑制は錐体の発火の 4.1ms 後（Szegedi 2017）", flush=True)
