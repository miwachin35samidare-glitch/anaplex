"""版65：抑制の体を撃たせている興奮が、どの送り手から来るかを分ける（measure_v51s を版65 に直したもの。音あり／無音で）
（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v65a.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v65a.py 3 --test）
【み】10-12 0時13分「すすめて」（0時06分「本当に人に沿うんだよね？流れで言ってない？」⇒ 版51 の古い測り（上からの戻り 93%）を版65 で測り直す）
【測】版65 健診：聴覚の抑制の体 0.25〜0.30%（【ヒト】覚醒 4.70Hz ＝ 1歩 5.0%、Verma 2024）。聴覚の表現の ihn 0.00
抑制の体の撃ち方【コード】(届いた興奮 − 上からの戻り) ÷ (1 ＋ 抑制) ＞ INH_TH（1.98）。興奮性の相手 180（【C】版47）
送り手の種類（measure_v51s と同じ分け方）
  自分の遅れ（自分の領野の表現の発火）／ 自分の時間（その時間細胞）／ 傾き ／ 横の入口（ext のうち撃たせる側）／ 上からの戻り（ext のうち調整する側。撃ち方から引かれる）
  写し（撃たせる）：版66 から。指令の写しのうち抑制の体を撃たせる送り手（版65 では「上からの戻り」に入る）
並べるもの（10回ごと・最後に全部。音あり／無音。歩を進める周＝最後の周に届いた分。抑制の体1体・1歩の平均）
  率：抑制の体（1歩のどこかで撃った）・同じ領野の表現 ／ 届いた興奮（上からの戻りを除く）・抑制・÷後 ／ 閾値 ／ 抑制が無ければ越えた割合
  種類ごと：届いた本数・届いた興奮の量
最初に、抑制の体1体の興奮性の相手の内訳（本数）を種類ごとに出す（配線そのもの。届いたかに関係なく）
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_area import INH_TH
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (65, 66):
    sys.exit(f"⚠️ babble.py が版65・66 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
AREAS = ("聴覚", "連合", "運動", "上丘")
KINDS = ("自分の遅れ", "自分の時間", "傾き", "横の入口", "上からの戻り", "写し（撃たせる）")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev

kind = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
buf_base = g.n_send_all - sum(a.n_src * a.D for a in g.areas)
buf_off = 0
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    lo = int(g.off_send[i])
    for r_lo, r_hi, own in ((0, a.n_ch, True), (a.n_ch, a.n_ch + a.n_ext, False)):
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
kind[g.mod_send & (kind >= 0)] = 4
if hasattr(g, "mod_drive_inh"):          # 版66：指令の写しのうち抑制の体を撃たせる側
    kind[g.mod_drive_inh & (kind == 4)] = 5
kind[g.is_inh_all] = -1

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
    g.flush_all()
    r = g.e_recv.long(); se = g.e_src.long()
    k = kind[se]; ar = area_of_recv[r]
    m = (k >= 0) & (ar >= 0)
    c = torch.zeros(NA * NK, device=dev).index_add_(0, ar[m] * NK + k[m], torch.ones(int(m.sum()), device=dev))
    mi = (ar >= 0) & g.e_inh
    ci = torch.zeros(NA, device=dev).index_add_(0, ar[mi], torch.ones(int(mi.sum()), device=dev))
    return (c.view(NA, NK) / per_area.unsqueeze(1)).cpu().numpy(), (ci / per_area).cpu().numpy()


c0, ci0 = static_counts()
print("抑制の体1体の相手（本数。0から）", flush=True)
for a_, nm in enumerate(AREAS):
    print(f"    {nm:5s} 抑制の体 {n_inh[nm]:4d}体  興奮 " + "  ".join(f"{KINDS[j]} {c0[a_, j]:6.1f}" for j in range(NK))
          + f"  ／ 抑制 {ci0[a_]:6.1f}", flush=True)

st = {"snd": False, "on": False}
_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["snd"] = float(ear.sum()) > 0
    return _step(ear, cmd, *a, **k)


b.step = step_wrap
K = ("n", "inh", "rep", "exc", "ihn", "d", "noinh")
mk_acc = lambda: {"s": {k: np.zeros(NA) for k in K}, "kn": np.zeros((NA, NK)), "kq": np.zeros((NA, NK))}
blk = {True: mk_acc(), False: mk_acc()}
tot = {True: mk_acc(), False: mk_acc()}
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if (advance and st["on"]) else None
    out = orig(self, x_all, ext_all, learn, advance, first)
    if e is None or len(e) == 0:
        return out
    r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
    ar = area_of_recv[r]
    m = ar >= 0
    r, se, inh, ar, ee = r[m], se[m], inh[m], ar[m], e[m]
    amp = torch.where(inh, 4.0, self.c_exc[r]) * self.w[ee] * self._amp[ee] * self.send_gain[se]
    amp = amp * torch.where(inh, self.scale[r], self.scale[r] * self.scale_up[r])    # 版53：起きて上げた分は興奮だけ
    if getattr(self, "exc_gain", None) is not None:
        amp = torch.where(inh, amp, amp * self.exc_gain)
    k = kind[se]
    mk = (~inh) & (k >= 0)
    idx = ar[mk] * NK + k[mk]
    q = torch.zeros(NA * NK, device=dev).index_add_(0, idx, amp[mk]).view(NA, NK)
    n = torch.zeros(NA * NK, device=dev).index_add_(0, idx, torch.ones_like(amp[mk])).view(NA, NK)
    ex = torch.zeros(self.n_recv_all, device=dev)
    keep = mk & (k != 4)
    ex.index_add_(0, r[keep], amp[keep])
    ih = torch.zeros(self.n_recv_all, device=dev)
    ih.index_add_(0, r[inh], amp[inh])
    for D in (blk[st["snd"]], tot[st["snd"]]):
        D["kq"] += (q / per_area.unsqueeze(1)).cpu().numpy(); D["kn"] += (n / per_area.unsqueeze(1)).cpu().numpy()
    for a_, nm in enumerate(AREAS):
        i = NAMES.index(nm); a = self.areas[i]
        lo = int(self.off_recv[i]) + 2 * a.n_ch
        exa, iha = ex[lo:lo + a.n_inh], ih[lo:lo + a.n_inh]
        vals = {"n": 1, "inh": float(self.bufs[i][inh_rows[nm], 0].mean()),
                "rep": float((out[0][rep_sl[nm]] > 0).float().mean()),
                "exc": float(exa.mean()), "ihn": float(iha.mean()), "d": float((exa / (1.0 + iha)).mean()),
                "noinh": float((exa > INH_TH).float().mean())}
        for D in (blk[st["snd"]], tot[st["snd"]]):
            for kk, v in vals.items():
                D["s"][kk][a_] += v
    return out


G.Group2._pass = hooked


def show(acc, head):
    print(f"── {head}  抑制の体1体・1歩の平均  閾値 {INH_TH:.2f}  ⇔【ヒト】抑制の体 5.0%/歩（4.70Hz、Verma 2024）", flush=True)
    for snd, lab in ((True, "音あり"), (False, "無音")):
        D = acc[snd]
        if D["s"]["n"].max() == 0:
            continue
        print(f"   {lab}（{int(D['s']['n'][0])}歩）", flush=True)
        for a_, nm in enumerate(AREAS):
            s = D["s"]; k = max(s["n"][a_], 1)
            print(f"     {nm:5s} 率 抑制の体 {s['inh'][a_]/k:6.2%} 表現 {s['rep'][a_]/k:6.2%}  興奮 {s['exc'][a_]/k:7.3f}"
                  f"  抑制 {s['ihn'][a_]/k:6.3f}  ÷後 {s['d'][a_]/k:7.3f}  抑制が無ければ越えた {s['noinh'][a_]/k:6.2%}", flush=True)
            print("            " + "  ".join(f"{KINDS[j]} {D['kn'][a_, j]/k:6.1f}本 {D['kq'][a_, j]/k:7.3f}" for j in range(NK)), flush=True)


def reset(acc):
    for D in acc.values():
        for x in D["s"].values():
            x[:] = 0
        D["kn"][:] = 0; D["kq"][:] = 0


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
st["on"] = True
for i in range(N):
    say(rng, True)
    if rp is not None:
        B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        show(blk, f"{i+1}回目まで（{(time.time()-t0)/60:.1f}分）この10回")
        reset(blk)
st["on"] = False
print("", flush=True)
show(tot, "全部")
c1, ci1 = static_counts()
print("終わりの抑制の体1体の相手（本数）", flush=True)
for a_, nm in enumerate(AREAS):
    print(f"    {nm:5s} 興奮 " + "  ".join(f"{KINDS[j]} {c1[a_, j]:6.1f}" for j in range(NK)) + f"  ／ 抑制 {ci1[a_]:6.1f}", flush=True)
print("読み：興奮は上からの戻りを除いた量（撃ち方に入る分）。上からの戻りの列は撃ち方から引かれる量。最後の周に届いた分", flush=True)
