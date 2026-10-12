"""版51：抑制の体を撃たせている興奮が、どの送り手から来るかを分ける（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51s.py [喃語の回数=30]
  （コンテナで道具を確かめるとき：python3 measure_v51s.py 2 --test ⇒ CPU・相手 200・作り物の声）
【測】measure_v51h（10-05）：抑制の体 聴覚 0.2% ／ 連合・高次・前頭前野 1.1〜3.0%（【ヒト】覚醒 4.70Hz ＝ 5.0%/歩、Verma 2024）
      CA1・嗅内深 約20%（人より高い）
抑制の体の撃ち方（torch_group2、版48）：(届いた興奮 − 上からの戻り) ÷ (1 ＋ 抑制) ＞ INH_TH（20.9mV ＝ 1.98）
  興奮性の相手 180（海馬・嗅内 194）。版47 で「錐体が【ヒト】3.30Hz で撃つとき、抑制の体が 4.70Hz になる数」として出した【C】
  ⚠️ その見積もりは、抑制の体が受ける抑制を入れていない
送り手の種類（領野の中の口。版49d と同じ分け方に、上からの戻りを足した）
  自分の遅れ   自分の領野の表現の発火そのもの（buf）
  自分の時間   自分の領野の表現の発火のあとに決まった間で撃つ時間細胞
  傾き        傾きの細胞（入力からの経過で率が変わる。RAMP_P 0.3Hz）
  横の入口     ext の行のうち撃たせる側（buf と時間細胞）
  上からの戻り  ext の行のうち調整する側（mod_send）。抑制の体の撃ち方からは引かれる
並べるもの（10回ごと。歩を進める周＝最後の周の値。抑制の体1体・1歩の平均）
  率        抑制の体が撃った割合 ／ 同じ領野の表現が撃った割合
  種類ごと    届いた本数 ／ 届いた興奮の量
  届いた興奮（上からの戻りを除く）・抑制 ihn・(興奮 ÷ (1＋ihn)) の平均 ／ 閾値 1.98
  抑制が無ければ撃てた割合（興奮 ＞ 閾値）
最初に、抑制の体1体の興奮性の相手の内訳（本数）を種類ごとに出す
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_area import INH_TH
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 30
if TEST:
    B.DEV = "cpu"
AREAS = ("聴覚", "連合", "高次", "前頭前野", "運動", "視床", "CA1", "嗅内深", "上丘")
KINDS = ("自分の遅れ", "自分の時間", "傾き", "横の入口", "上からの戻り")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev
AREAS = tuple(nm for nm in AREAS if nm in NAMES)

# ── 送り手の種類の表（-1 ＝ 数えない：抑制の送り手）──
kind = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
buf_base = int(g.off_send[-1])
buf_off = 0
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    lo = int(g.off_send[i])
    for r_lo, r_hi, own in ((0, a.n_ch, True), (a.n_ch, a.n_ch + a.n_ext, False)):
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
kind[g.mod_send & (kind >= 0)] = 4
kind[g.is_inh_all] = -1

# ── 受け手（抑制の体）→ 領野の番号 ──
area_of_recv = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=dev)
inh_rows, rep_sl, n_inh = {}, {}, {}
for nm in AREAS:
    i = NAMES.index(nm); a = g.areas[i]
    lo = int(g.off_recv[i]) + 2 * a.n_ch
    area_of_recv[lo:lo + a.n_inh] = AREAS.index(nm)
    inh_rows[nm] = slice(a.n_ch + a.n_ext, a.n_src)
    c = int(g.off_ch[i]); rep_sl[nm] = slice(c, c + a.n_ch)
    n_inh[nm] = a.n_inh
NA, NK = len(AREAS), len(KINDS)
per_area = torch.tensor([n_inh[nm] for nm in AREAS], device=dev, dtype=torch.float)


def static_counts():
    r = g.e_recv.long(); se = g.e_src.long()
    k = kind[se]; ar = area_of_recv[r]
    m = (k >= 0) & (ar >= 0)
    c = torch.zeros(NA * NK, device=dev).index_add_(0, ar[m] * NK + k[m], torch.ones(int(m.sum()), device=dev))
    ci = torch.zeros(NA, device=dev).index_add_(0, ar[(ar >= 0) & g.e_inh], torch.ones(int(((ar >= 0) & g.e_inh).sum()), device=dev))
    return (c.view(NA, NK) / per_area.unsqueeze(1)).cpu().numpy(), (ci / per_area).cpu().numpy()


c0, ci0 = static_counts()
print("抑制の体1体の相手（本数）", flush=True)
for a_, nm in enumerate(AREAS):
    print(f"    {nm:5s} 抑制の体 {n_inh[nm]:4d}体  興奮 " + "  ".join(f"{KINDS[j]} {c0[a_, j]:6.1f}" for j in range(NK))
          + f"  ／ 抑制 {ci0[a_]:6.1f}", flush=True)

K = ("n", "inh", "rep", "exc", "ihn", "d", "noinh")
acc = {k: np.zeros(NA) for k in K}
acc_n = np.zeros((NA, NK)); acc_q = np.zeros((NA, NK))
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges()
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not advance:
        return out
    r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
    ar = area_of_recv[r]
    m = ar >= 0
    r, se, inh, ar, ee = r[m], se[m], inh[m], ar[m], e[m]
    amp = torch.where(inh, 4.0, self.c_exc[r]) * self.w[ee] * self._amp[ee] * self.send_gain[se] * self.scale[r]
    if getattr(self, "exc_gain", None) is not None:
        amp = torch.where(inh, amp, amp * self.exc_gain)
    k = kind[se]
    mk = (~inh) & (k >= 0)
    idx = ar[mk] * NK + k[mk]
    q = torch.zeros(NA * NK, device=dev).index_add_(0, idx, amp[mk]).view(NA, NK)
    n = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(amp[mk])).view(NA, NK)
    acc_q[:] += (q / per_area.unsqueeze(1)).cpu().numpy(); acc_n[:] += (n / per_area.unsqueeze(1)).cpu().numpy()
    # 受け手ごとの興奮（上からの戻りを除く）と抑制
    ex = torch.zeros(self.n_recv_all, device=dev)
    keep = mk & (k != 4)
    ex.index_add_(0, r[keep], amp[keep])
    ih = torch.zeros(self.n_recv_all, device=dev)
    ih.index_add_(0, r[inh], amp[inh])
    for a_, nm in enumerate(AREAS):
        i = NAMES.index(nm); a = self.areas[i]
        lo = int(self.off_recv[i]) + 2 * a.n_ch
        exa, iha = ex[lo:lo + a.n_inh], ih[lo:lo + a.n_inh]
        d = exa / (1.0 + iha)
        acc["n"][a_] += 1
        acc["inh"][a_] += float(self.bufs[i][inh_rows[nm], 0].mean())
        acc["rep"][a_] += float((out[0][rep_sl[nm]] > 0).float().mean())
        acc["exc"][a_] += float(exa.mean()); acc["ihn"][a_] += float(iha.mean()); acc["d"][a_] += float(d.mean())
        acc["noinh"][a_] += float((exa > INH_TH).float().mean())
    return out


G.Group2._pass = hooked
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    say(rng, True)
    if rp is not None:                     # 作り物の声（--test）には返し手が無い
        B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        print(f"── {i+1}回目まで（{(time.time()-t0)/60:.1f}分）  抑制の体1体・1歩の平均  閾値 {INH_TH:.2f}", flush=True)
        for a_, nm in enumerate(AREAS):
            k = max(acc["n"][a_], 1)
            parts = "  ".join(f"{KINDS[j]} {acc_n[a_, j]/k:5.1f}本 {acc_q[a_, j]/k:6.3f}" for j in range(NK))
            print(f"   {nm:5s} 率 抑制 {acc['inh'][a_]/k:6.2%} 表現 {acc['rep'][a_]/k:6.2%}  興奮 {acc['exc'][a_]/k:6.3f}"
                  f"  ihn {acc['ihn'][a_]/k:6.3f}  ÷後 {acc['d'][a_]/k:6.3f}  抑制なしなら {acc['noinh'][a_]/k:6.2%}", flush=True)
            print(f"          {parts}", flush=True)
        for x in acc.values():
            x[:] = 0
        acc_n[:] = 0; acc_q[:] = 0
print("人：抑制の体 5.0%/歩（4.70Hz）・錐体 3.5%/歩（3.30Hz）Verma 2024 ／ 海馬 5.4%・3.3%（Le Van Quyen 2008）", flush=True)
