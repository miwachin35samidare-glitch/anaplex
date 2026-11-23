"""版111 の上で、CL の体数と CL → 皮質 の線の引き方だけを変えて測る（arkhe の ~/anaplex で。保存しない・網のファイルは変えない。版112 を作る前の確かめ）
  python3 measure_cl112.py CLの体数 [歩=15000] [刻み=200]      （CLの体数 5 ＝ 今の網そのまま）
  python3 measure_cl112.py CLの体数 [歩] [刻み] --ppn      （PPN の自分で撃つ分 i0 を体ごとにばらす。下の「--ppn」）
  python3 measure_cl112.py CLの体数 --count                    （網を組んで辺・線の数を出して終わる。VRAM の見込み用）
  何のため：analyze111c（data/arkhe_analyze111c_1117.txt）：皮質が同じ歩にそろって撃つのは、CL が 5体で、皮質の表現の体がどれも同じ 5体から
    40本（重なりあり ＝ 1体から約8本）受けるから。10 §7【み】10-23「（CL 5 ほか）ちゃんと動かなかったら増やす」
  変えるもの（組む前に torch_brain の N_CL を差し替え、組んだあと cl_src を引き直す。ほかは版111 のまま）
    ・CL の体数 N_CL（5 → 指定の数）
    ・CL → 皮質 の線：受け手1体の 40本を、別々の CL の体から引く（重なりなし。体数が 40 より少ないときは今の引き方のまま）
    K_CL 40・重さ・閾値・PPN → CL 8本・CL の目標 11Hz は変えない（K_CL のコードの注は「CL だけ・抑制 0 で 3.3Hz【計算】」）
  --ppn（11-17）：analyze ⑥（data/arkhe_analyze6_1117.txt）で PPN 512体が同じ歩にそろって撃っていた（体の 50% 以上が撃つ歩に発火の 68〜71%）。
    【コード】核の i0 は核の中の全部の体で同じ値（torch.full）。【ヒト】PPN の細胞ごとの率は 23.2 ± 15.6Hz（32体、JNNP 2010。台帳_核の値）
    ⇒ 体ごとの率を、平均 23.2・SD 15.6 の対数正規【C】から引き、i0 に「その率で自分で撃つ行き着く先 ÷ 23.2Hz の行き着く先」を掛ける
      （行き着く先 ＝ 1 ÷ (1 − e^(−間隔/40ms))、torch_brain の NUC_VINF["PPN"] と同じ式。ノブの倍率はそのまま掛かる）
    ⚠️ 抑制（内節・網様部）の下での実際の率は式どおりにならない ⇒ PPN の率の列で見る。変えるのは PPN だけ
  --ppn-free（11-17）：--ppn では PPN のそろいが取れなかった（data/arkhe_cl112_256_ppn_1117.txt：PPN の一度に撃つ分 60〜76% のまま）。
    【コード】PPN へ入るのは 内節 → PPN・網様部 → PPN の抑制 各64本だけ（torch_brain 802行）⇒ PPN の体はどれもほぼ同じ抑制の波を受ける
    ⇒ 測るためだけに PPN を入力から切り離す：内節・網様部 → PPN の線の重さを 0・--ppn のばらしを掛ける（i0 のノブの倍率 2.81 はそのまま。--ppn-k=値 で変えられる）
       ＝ PPN の体がそれぞれ人の率（23.2 ± 15.6Hz）でばらばらに自分で撃つだけの形
    見ること：PPN がばらばらに撃てば、CL と皮質の「一度に撃つ」は消えるか（消えれば元は 基底核 → PPN の波・消えなければ別にもある）
    ⚠️ 人から外れた形（内節・網様部 → PPN は人・霊長類にある線）。測るためだけで、版にはしない
  --ppn-noi（11-17、--ppn-free と一緒に使う）：--ppn-free でも PPN の一度に撃つ分は 89〜96%（data/arkhe_cl112_256_ppnfree_1117.txt）⇒ PPN の中でそろう。
    【コード 版111、容器で数えた】PPN の領野は 体 512・中の抑制の体 98。辺は 体 → 抑制の体 の興奮 17,640本（98 × 180【C】30 §4）と
    抑制の体 → 体 の抑制 36,100本。体どうしの興奮は無い（10 §10・20_今の網「中の興奮の横なし」のとおり）。
    ⇒ 512体がみな同じ 98体の抑制を受ける輪（体が撃つ → 抑制の体 → 全体を抑える）がそろえている疑い
    ⇒ 測るためだけに、PPN の中の 抑制の体 → 体 の辺の重さを 0（刈り込みで 1000歩ごとに少し生え直る）
    ⚠️ 人から外れた形。版にはしない
  --table（11-17）：毎歩、領野・核ごとの「撃った体の割合」を表にして cl112_table.npz に書く（measure_wave111c と同じ形 ⇒ analyze_wave111c で読める）
    毎歩の仕事が1つ増える（体の数の累積和 1回・領野の端の読み）
  出すもの：刻みごとに wave111b の列 ＋ そろい（その刻みで皮質が黙った歩の割合・皮質の発火のうち「皮質の 20% 以上が同じ歩に撃った歩」に入る割合）
    ＋ CL の率 ＋ 声・自発・言えた
  毎歩に足す仕事：表現の発火（体の数）を1本の数えへ足す 1回・皮質の発火の平均 1回（数 1つを読む）。線・辺には触らない・少ない箱への足し込みなし
  刻みごと（200歩に1回）：下からの線の道ごとの切り出しの和・CL の和（wave111b と同じ）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    sys.exit("CL の体数を1つ（5 ＝ 今のまま）")
NCL = int(args[0])
NS = int(args[1]) if len(args) > 1 else 15000
EV = int(args[2]) if len(args) > 2 else 200
COUNT = "--count" in sys.argv
PPN_FREE = "--ppn-free" in sys.argv
PPN_NOI = "--ppn-noi" in sys.argv
TABLE = "--table" in sys.argv
PPN_SPREAD = "--ppn" in sys.argv or PPN_FREE
NP_TEST = 200
for a_ in sys.argv[1:]:
    if a_.startswith("--np="):
        NP_TEST = int(a_[5:])
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION not in (111, 112, 113, 114, 115):            # 11-19：版112 は CL 256・別々の体からが網に入った ⇒ 差し替えなしで同じ列を出す
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版111・112 の上で回す")
if B.VERSION >= 112:          # 版113・114 も CL 256（版112 から変えていない）
    assert TB.N_CL == NCL, f"版112 の CL は {TB.N_CL} 体（差し替えはしない）⇒ {TB.N_CL} を渡す"
else:
    assert TB.N_CL == 5 and TB.N_NEW["CL"] == 5, "N_CL が思っていた値でない"
TB.N_CL = NCL
TB.N_NEW["CL"] = NCL
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(NP_TEST if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
assert b.n["CL"] == NCL and len(b.cl_spk) == NCL, "CL の体数が入っていない"
nr_ = len(b.cl_ch)
K = len(b.cl_w) // nr_
DISTINCT = NCL >= K
if DISTINCT and B.VERSION == 111:                                             # 受け手ごとに別々の CL の体から K 本（重なりなし）
    gc_ = torch.Generator(device="cpu"); gc_.manual_seed(112)
    src = torch.empty(nr_, K, dtype=torch.long)
    for a0 in range(0, nr_, 512):
        src[a0:a0 + 512] = torch.rand(min(512, nr_ - a0), NCL, generator=gc_).argsort(1)[:, :K]
    assert bool((src.sort(1).values[:, 1:] != src.sort(1).values[:, :-1]).all()), "重なりが残っている"
    b.cl_src = src.reshape(-1).to(device=b.cl_src.device, dtype=b.cl_src.dtype)
print(f"CL {NCL} 体 ／ CL → 皮質 受け手 {nr_}・1体 {K} 本・{'別々の体から' if DISTINCT else '重なりあり（今の引き方）'}"
      f" ／ 網の中の辺 {len(g.e_src):,}・下からの線 {len(b.but.w):,}・CL の線 {len(b.cl_w):,}・体 {g.n_ch_all:,}・受け手の行 {g.n_recv_all:,}", flush=True)
if COUNT:
    sys.exit(0)
PPS = slice(int(b.off_ch[NAMES.index("PPN")]), int(b.off_ch[NAMES.index("PPN")]) + b.n["PPN"])
if PPN_SPREAD:
    import math
    o_ = 0
    for k_ in TB.NUC:
        if k_ == "PPN":
            break
        o_ += b.n[k_]
    npp = b.n["PPN"]
    v_ = b.nuc_i0[o_:o_ + npp]
    assert o_ + npp <= len(b.nuc_i0) and float(v_.max() - v_.min()) <= 1e-6 * float(v_.max()), "PPN の i0 の場所が思っていた形でない"
    print(f"PPN の i0（ばらす前・ノブ込み）{float(v_[0]):.5f}", flush=True)
    if PPN_FREE:
        kn_ = float(TB.KNOBS.get("nuc_vinf", {}).get("PPN", 1.0))
        KF = kn_                                         # 既定：ノブの倍率はそのまま（容器で切り離した PPN：倍率 1 で 約5.5Hz・1.5 で 約9Hz【測】⇒ 2.81 で 約18Hz の見込み【計算】）
        for a_ in sys.argv[1:]:
            if a_.startswith("--ppn-k="):
                KF = float(a_[8:])
        b.nuc_i0[o_:o_ + npp] *= KF / kn_                # ノブの倍率を外し、切り離した PPN が約 23Hz になる倍率 KF を掛ける（容器で合わせた値を渡す）
        nn_ = g.nucnet
        _s, d_, sg_, _dl = nn_.flat()
        op_ = int(b.off_ch[NAMES.index("PPN")])
        msk = (d_ >= op_) & (d_ < op_ + npp)
        assert int(msk.sum()) > 0 and bool((sg_[msk] < 0).all()), "PPN へ入る線が思っていた形でない（抑制だけのはず）"
        nn_.w[msk] = 0.0
        nn_.w0[msk] = 0.0
        print(f"PPN を入力から切り離した：PPN へ入る抑制の線 {int(msk.sum()):,} 本の重さを 0・i0 のノブの倍率 {kn_:.3f} を外して {KF:.3f} を掛けた", flush=True)
    if PPN_NOI:
        ip_ = NAMES.index("PPN")
        lo_, hi_ = int(g.off_recv[ip_]), int(g.off_recv[ip_ + 1])

        def ppn_noi():                                   # PPN の中の 抑制の体 → 体 の辺の重さを 0（刈り込みのあとにも当て直す：11-17 23:23 の回で 1000歩から率が 25 → 7Hz）
            n0 = 0
            for a0 in range(0, len(g.e_recv), 1 << 22):      # 11-18：4M ずつ・32bit のまま比べる（1つの GPU の仕事を短く）
                r_ = g.e_recv[a0:a0 + (1 << 22)]
                m_ = g.e_inh[a0:a0 + (1 << 22)] & (r_ >= lo_) & (r_ < hi_)
                w_ = g.w[a0:a0 + (1 << 22)]
                n0 += int((m_ & (w_ != 0)).sum())
                w_[m_] = 0.0
                L.beat()
            return n0
        n0 = ppn_noi()
        print(f"PPN の中の 抑制の体 → 体（全部の行）の辺 {n0:,} 本の重さを 0（刈り込みの歩のあとにも当て直す）", flush=True)
    M, SD = 23.2, 15.6                                   # 【ヒト】JNNP 2010（32体）
    sg2 = math.log(1.0 + (SD / M) ** 2)
    gp_ = torch.Generator(device="cpu"); gp_.manual_seed(113)
    hz = torch.exp(math.log(M) - sg2 / 2 + math.sqrt(sg2) * torch.randn(npp, generator=gp_)).clamp(0.5, 90.0)
    vinf = 1.0 / (1.0 - torch.exp(-(1000.0 / hz) / TB.TAU_M_NUC))
    ratio = (vinf / TB.NUC_VINF["PPN"]).to(b.nuc_i0.device, b.nuc_i0.dtype)
    b.nuc_i0[o_:o_ + npp] *= ratio
    print(f"PPN {npp} 体：i0 を体ごとにばらした（引いた率 平均 {float(hz.mean()):.1f}・SD {float(hz.std()):.1f}・中央 {float(hz.median()):.1f}Hz・"
          f"i0 の倍率 {float(ratio.min()):.2f}〜{float(ratio.max()):.2f}・中央 {float(ratio.median()):.2f}）", flush=True)
n_all = g.n_ch_all
DT = 10.67 / 1000.0
CX = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質" and n != "運動"]
HIP = [n for n in ("歯状回", "CA3", "CA1") if n in NAMES]
SL = {nm: slice(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm]) for nm in CX + ["運動"] + HIP}

ALL = CX + ["運動"]
ch0_name = {int(b.off_ch[NAMES.index(nm)]): nm for nm in NAMES}
is_rep_r = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev)
is_rep_r[g.rep_i.long()] = True
RR = {}
for nm in ALL:
    i = NAMES.index(nm)
    rr = torch.arange(int(g.off_recv[i]), int(g.off_recv[i + 1]), device=dev)
    RR[nm] = rr[is_rep_r[rr]]
tab = b.but
paths = {nm: [] for nm in ALL}                           # 領野 → [(切り出し, 受け手の数)]
for i, k in enumerate(tab.keys):
    nm = ch0_name.get(int(b.bu[k].ch0), "?")
    if nm in paths:
        paths[nm].append((slice(int(tab.eo[i]), int(tab.eo[i + 1])), max(1, tab.nt[i])))
cl_on = bool(getattr(b, "cl_on", False))
if cl_on:
    nr = len(b.cl_ch)
    kcl = len(b.cl_w) // nr
    assert kcl * nr == len(b.cl_w) and bool((b.cl_recv.long().view(nr, kcl) == torch.arange(nr, device=dev)[:, None]).all()), "CL の並びが思っていた形でない"
    cl_ch = b.cl_ch.long()
    CLM = {nm: (cl_ch >= SL[nm].start) & (cl_ch < SL[nm].stop) for nm in ALL}


def weights():
    q = g.rate_ema / g.target_recv.clamp(min=1e-9)
    clw = b.cl_w.view(nr, kcl).sum(1, dtype=torch.float64) if cl_on else None
    out = {}
    for nm in ALL:
        wa = sum(float(tab.w[s].sum(dtype=torch.float64)) / nt for s, nt in paths[nm])
        cw = float(clw[CLM[nm]].mean()) if cl_on and bool(CLM[nm].any()) else float("nan")
        out[nm] = (float(q[RR[nm]].median()), wa, cw)
    return out


cx_mask = torch.zeros(n_all, dtype=torch.bool, device=dev)
for nm in CX:
    cx_mask[SL[nm]] = True
CLS = slice(int(b.off_ch[NAMES.index("CL")]), int(b.off_ch[NAMES.index("CL")]) + NCL)
for a_ in sys.argv[1:]:
    if TEST and a_.startswith("--rw="):                 # 容器の確かめだけ：刈り込みの間隔を縮める
        g.rewire_every = int(a_[5:])
noi_log = []
noi_n, noi_s = [0], [0.0]
NOI_EVERY = 10 if TEST else 100
rw_last = [g.t // NOI_EVERY]
quiet, big, tot = 0, 0.0, 0.0
pp_big, pp_tot, cl_big, cl_tot = 0.0, 0.0, 0.0, 0.0
st0 = dict(life.stats)
if TABLE:
    off_ = [int(x) for x in b.off_ch]
    lo_t = torch.tensor(off_[:-1], device=dev); hi_t = torch.tensor(off_[1:], device=dev)
    size_t = np.array([off_[i + 1] - off_[i] for i in range(len(NAMES))], dtype=np.float64)
    RT = torch.zeros(NS, len(NAMES), device=dev)
    z1 = torch.zeros(1, device=dev)
acc = torch.zeros(n_all, device=dev)
print(f"版 {B.VERSION} ／ {NS} 歩・刻み {EV} 歩（{EV * DT:.1f} 秒）／ 皮質（運動の外）{len(CX)} 領野 ／ 体 {n_all:,}", flush=True)
print(f"  {'歩':>6s} ｜ 皮質 {'率Hz':>6s}{'q':>6s}{'下からw':>8s}{'CLw':>7s} ｜ 運動 {'率Hz':>6s}{'q':>6s}{'下からw':>8s}{'CLw':>7s} ｜ {'CA1':>5s}{'CL':>6s} ｜ 黙った歩・一度に撃つ分 ｜ PPN Hz・一度・CL 一度 ｜ 門 ｜ 声・自発・言えた", flush=True)
for i in range(1, NS + 1):
    life.step(learn=True)
    if PPN_NOI and g.t // NOI_EVERY != rw_last[0]:         # 100歩ごと（11-18：容器で、戻るのが刈り込みの歩ちょうどではなかった）
        rw_last[0] = g.t // NOI_EVERY
        import time as _tm
        t0_ = _tm.perf_counter()
        nz_ = ppn_noi()
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()                     # 11-18：当て直しの GPU の仕事を、次の歩の時間（守りの 0.5秒）に入れない
        L.beat()
        if nz_:
            noi_log.append((life.t, nz_, round(_tm.perf_counter() - t0_, 3)))
        noi_n[0] += 1; noi_s[0] = max(noi_s[0], _tm.perf_counter() - t0_)
    f = g._any[0][:n_all].float()
    acc.add_(f)
    if TABLE:
        c_ = torch.cat([z1, f.cumsum(0)])
        RT[i - 1] = c_[hi_t] - c_[lo_t]
    fr = float(f[cx_mask].mean())                        # この歩に皮質（運動の外）で撃った体の割合
    quiet += int(fr == 0.0)
    tot += fr
    if fr >= 0.2:
        big += fr
    fp = float(f[PPS].mean()); fc = float(f[CLS].mean())  # PPN・CL：体の 50% 以上が同じ歩に撃った歩に入る発火の割合を数える
    pp_tot += fp; cl_tot += fc
    if fp >= 0.5:
        pp_big += fp
    if fc >= 0.5:
        cl_big += fc
    if i % EV == 0:
        L.beat()
        r = acc / EV / DT
        W = weights()
        L.beat()
        cx = np.array([[float(r[SL[nm]].mean()), *W[nm]] for nm in CX]).mean(0)
        mo = [float(r[SL["運動"]].mean()), *W["運動"]]
        th = g.third.reshape(-1).tolist() if torch.is_tensor(g.third) else list(g.third)
        print(f"  {life.t:6d} ｜      {cx[0]:6.2f}{cx[1]:6.2f}{cx[2]:8.3f}{cx[3]:7.3f} ｜      {mo[0]:6.2f}{mo[1]:6.2f}{mo[2]:8.3f}{mo[3]:7.3f} ｜ "
              f"{float(r[SL['CA1']].mean()):5.2f}{float(r[CLS].mean()):6.2f} ｜ {quiet / EV:5.0%}・{(big / tot if tot > 0 else 0.0):5.0%} ｜ {float(r[PPS].mean()):5.1f}・{(pp_big / pp_tot if pp_tot > 0 else 0.0):4.0%}・{(cl_big / cl_tot if cl_tot > 0 else 0.0):4.0%} ｜ "
              f"{'・'.join(f'{x:.2f}' for x in th)} ｜ {life.stats['voices'] - st0['voices']}・{life.stats['spont'] - st0['spont']}・{life.stats['ok'] - st0['ok']}/{life.stats['tried'] - st0['tried']}", flush=True)
        acc.zero_(); quiet, big, tot = 0, 0.0, 0.0; pp_big, pp_tot, cl_big, cl_tot = 0.0, 0.0, 0.0, 0.0; st0 = dict(life.stats)
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
if PPN_NOI:
    print(f"0 に当て直した（{noi_n[0]} 回見た・一番長くかかった {noi_s[0]:.3f}秒）：戻っていた辺（歩, 本, 秒）{noi_log}", flush=True)
if TABLE:
    L.beat()
    Xt = RT.cpu().numpy().astype(np.float64) / np.maximum(size_t, 1.0)[None, :]
    np.savez_compressed("cl112_table.npz", rate=Xt.astype(np.float32), names=np.array(NAMES), dt=DT, version=B.VERSION)
    print("cl112_table.npz に 歩 × 領野 の表を書いた", flush=True)
print("\n（終わり）", flush=True)
