"""学ぶにつれ視床から皮質まで発火が下がるのは、表現の体の駆動のどの項が変わるからか（arkhe の ~/anaplex で。
0から・学びながら・返し手と録音あり・保存しない・網は書き換えない。声と乱数の並びは measure_check と同じ）
  python3 measure_v52a.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v52a.py 4 --test ⇒ CPU・相手 200・作り物の声・返し手なし）
【み】10-08「よい」（直す順 1「皮質の表現が下から撃てない」の入口）
【測】measure_check（版51・52）：表現 10回目 → 50回目 視床 18 → 4.7% ／ 聴覚 2.4 → 0.34% ／ 連合 1.05 → 0.36%
表現の体が撃つ式（torch_group2._pass）
  ((exc − exc_mod) ＋ x·x_gain) ÷ (1 ＋ ihn) ＞ fire_th · th_gain
    x         下からの入力（視床なら耳、聴覚なら視床の表現）。学ばない
    exc       学ぶシナプス（自分の領野の送り手：時間細胞・buf の遅れ）× 接触の数 × 短期可塑性 × scale（恒常性）
    exc_mod   そのうち調整する側（上からの戻り・指令の写し）。撃たせない
    ihn       抑制（学ばない。人の強さ W_INH）× scale
    th_gain   覚醒で下がる閾値の倍率（脳幹）
並べるもの（領野ごと、10回ごと。1周目の値＝その歩の初めに届いた分。周回の中で撃った分は入っていない）
  1 発火：表現（最後の周）／ 下から越える割合（x·x_gain ＞ fire_th·th_gain）／ 越えた体のうち撃てた ／ 越えない体のうち撃った
  2 表現の体1体・1歩あたりの平均：x·x_gain ／ exc を送り手の種類ごと ／ exc_mod ／ ihn ／ 閾値 fire_th·th_gain
      送り手の種類  時間細胞（自分の表現の行・ext の行）／ buf（自分の表現の行・ext の行）
                    ext は他の領野から入る行。調整する側の印（mod_send）が付いた送り手は「調整」に分ける
  3 10回目の終わりの重み：表現の体へ入る興奮の辺、種類ごとに 1体あたりの本数 ／ 重み w の平均 ／ w＝0 の割合 ／
      接触×w×scale の平均 ／ 表現の体の scale の平均
読み：下がるのが x なら下の段から ／ exc なら学び（論点台帳 §2）か刈り込み ／ scale なら恒常性 ／ ihn なら抑制 ／ th_gain なら覚醒
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (51, 52):
    sys.exit(f"⚠️ babble.py が版51・52 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
AREAS = ("視床", "聴覚", "連合", "高次")
CATS = ("時・表", "時・ext", "buf・表", "buf・ext", "調整")
NC = len(CATS)

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev
IDX = {nm: NAMES.index(nm) for nm in NAMES}
K = len(AREAS)

# 受け手：選んだ領野の表現の行 → 領野の番号（ほかは −1）
recv_key = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=dev)
n_rep = np.zeros(K)
ch_sl = {}
for k, nm in enumerate(AREAS):
    i = IDX[nm]; a = g.areas[i]; lo = int(g.off_recv[i])
    recv_key[lo + a.n_ch:lo + 2 * a.n_ch] = k
    n_rep[k] = a.n_ch
    ch_sl[nm] = slice(int(g.off_ch[i]), int(g.off_ch[i]) + a.n_ch)

# 送り手の種類（時間細胞・buf × 表現の行・ext の行。抑制の行は抑制の送り手なので興奮には出てこない）
send_cat = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
send_area = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
ch_off = np.cumsum([0] + [a.ch.n_cells for a in g.areas])
buf_base = int(ch_off[-1])
buf_off = buf_base + np.cumsum([0] + [a.n_src * a.D for a in g.areas])
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    if per and a.ch.n_cells:
        row = torch.arange(a.ch.n_cells, device=dev) // per
        send_cat[ch_off[i]:ch_off[i + 1]] = torch.where(row < a.n_ch, 0, torch.where(row < a.n_ch + a.n_ext, 1, -1))
        send_area[ch_off[i]:ch_off[i + 1]] = i
    row = torch.arange(a.n_src * a.D, device=dev) // a.D
    send_cat[buf_off[i]:buf_off[i + 1]] = torch.where(row < a.n_ch, 2, torch.where(row < a.n_ch + a.n_ext, 3, -1))
    send_area[buf_off[i]:buf_off[i + 1]] = i

acc = torch.zeros(K * NC, device=dev)      # 興奮の和（種類ごと）
acc_ihn = torch.zeros(K, device=dev)
fr = {nm: np.zeros(6) for nm in AREAS}      # 下から越える・越えて撃てた・越えずに撃った・x·x_gain の和・閾値の和・歩
rate = {nm: [0.0, 0] for nm in AREAS}       # 表現（最後の周）
thg = [0.0, 0]
drop = [0.0]
orig_pass = G.Group2._pass


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    if first:
        e = self._fired_edges()
        w0, sc0 = self.w[e].clone(), self.scale.clone()   # 1周目の学びで変わる前
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    if first and len(e):
        r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
        amp = torch.where(inh, 4.0, self.c_exc[r]) * w0
        if self.stp is not None:
            amp = amp * self._amp[e]
        amp = amp * self.send_gain[se] * sc0[r]
        kk = recv_key[r]
        m = (kk >= 0) & ~inh
        cat = torch.where(self.mod_send[se], NC - 1, send_cat[se])
        drop[0] += float(amp[m & (cat < 0)].sum()) if TEST else 0.0   # 種類に入らない興奮（0 のはず）
        m = m & (cat >= 0)
        acc.index_add_(0, kk[m] * NC + cat[m], amp[m])
        mi = (kk >= 0) & inh
        acc_ihn.index_add_(0, kk[mi], amp[mi])
    if first:
        th = self.fire_th * self.th_gain
        thg[0] += float(self.th_gain); thg[1] += 1
        for nm, s in ch_sl.items():
            xd = x_all[s] * self.x_gain[s]
            tgt = xd > th[s]
            f = fr[nm]; rep = out[0][s] > 0
            f[0] += float(tgt.sum()); f[1] += float((rep & tgt).sum()); f[2] += float((rep & ~tgt).sum())
            f[3] += float(xd.mean()); f[4] += float(th[s].mean()); f[5] += 1
    if advance:
        for nm, s in ch_sl.items():
            rate[nm][0] += float((out[0][s] > 0).float().mean()); rate[nm][1] += 1
    return out


G.Group2._pass = hooked_pass


def weights():
    """表現の体へ入る興奮の辺を、送り手の種類ごとに"""
    r = g.e_recv.long(); kk = recv_key[r]
    m = (kk >= 0) & ~g.e_inh
    r, se, w = r[m], g.e_src[m].long(), g.w[m]
    kk = kk[m]
    cat = torch.where(g.mod_send[se], NC - 1, send_cat[se])
    ok = cat >= 0
    r, w, kk, cat = r[ok], w[ok], kk[ok], cat[ok]
    key = kk * NC + cat
    eff = w * g.c_exc[r] * g.scale[r]
    cnt = torch.bincount(key, minlength=K * NC).float()
    sw = torch.bincount(key, weights=w, minlength=K * NC)
    zero = torch.bincount(key, weights=(w == 0).float(), minlength=K * NC)
    se_ = torch.bincount(key, weights=eff, minlength=K * NC)
    sc = torch.zeros(K, device=dev).index_add_(0, recv_key[recv_key >= 0], g.scale[recv_key >= 0])
    return (cnt.cpu().numpy(), sw.cpu().numpy(), zero.cpu().numpy(), se_.cpu().numpy(), sc.cpu().numpy())


def show():
    a = acc.cpu().numpy().reshape(K, NC); ih = acc_ihn.cpu().numpy()
    cnt, sw, zero, se_, sc = weights()
    print(f"   閾値の倍率 th_gain 平均 {thg[0]/max(thg[1],1):.3f}", flush=True)
    for k, nm in enumerate(AREAS):
        f = fr[nm]; st = max(f[5], 1)
        cells = n_rep[k] * st
        print(f"   {nm}  表現 {rate[nm][0]/max(rate[nm][1],1):6.2%}  下から越える {f[0]/cells:6.2%}"
              f"  越えて撃てた {f[1]/max(f[0],1):5.1%}  越えずに撃った {f[2]/max(cells-f[0],1):6.3%}", flush=True)
        print(f"      1体1歩  x·x_gain {f[3]/st:7.3f}  閾値 {f[4]/st:6.2f}  ihn {ih[k]/cells:6.3f}  exc "
              + "  ".join(f"{c} {a[k, j]/cells:7.3f}" for j, c in enumerate(CATS)), flush=True)
        print(f"      重み  scale {sc[k]/n_rep[k]:.3f}  " + "  ".join(
            f"{c} 本{cnt[k*NC+j]/n_rep[k]:.1f} w{sw[k*NC+j]/max(cnt[k*NC+j],1):.4f} 0={zero[k*NC+j]/max(cnt[k*NC+j],1):.0%}"
            f" 実{se_[k*NC+j]/max(cnt[k*NC+j],1):.4f}" for j, c in enumerate(CATS)), flush=True)
    acc.zero_(); acc_ihn.zero_()
    for nm in AREAS:
        fr[nm][:] = 0; rate[nm] = [0.0, 0]
    thg[0] = 0.0; thg[1] = 0


print("── 0回目（学ぶ前の重み）", flush=True)
cnt, sw, zero, se_, sc = weights()
for k, nm in enumerate(AREAS):
    print(f"   {nm}  重み  scale {sc[k]/n_rep[k]:.3f}  " + "  ".join(
        f"{c} 本{cnt[k*NC+j]/n_rep[k]:.1f} w{sw[k*NC+j]/max(cnt[k*NC+j],1):.4f} 0={zero[k*NC+j]/max(cnt[k*NC+j],1):.0%}"
        f" 実{se_[k*NC+j]/max(cnt[k*NC+j],1):.4f}" for j, c in enumerate(CATS)), flush=True)

rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t_start = time.time()
for i in range(N):
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if ear.sum() > 0:
        past.append((cmd, ear))

    def rec():
        if len(past) > DELAY:
            c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
            B.run(b, c0, e0, False, True)

    def rep():
        if rp is not None:
            B.reply(b, rp, rng_r)

    for f_ in ((rec, rep) if rng_p.random() < 0.5 else (rep, rec)):
        f_()
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        print(f"── {i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回", flush=True)
        show()
if TEST:
    print(f"   （確かめ）種類に入らなかった興奮の和 {drop[0]:.4f}", flush=True)
print("読み：下がるのが x なら下の段 ／ exc（w か本数）なら学びか刈り込み ／ scale なら恒常性 ／ ihn なら抑制 ／ th_gain なら覚醒", flush=True)
