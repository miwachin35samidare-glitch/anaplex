"""版109：どこが最初に無限大・NaN になるか、何が伸びて撃ちすぎになるかを見る（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_nan109.py [番の上限=70] [書き出す間隔の番=5]
  srccx109（11-11 07:30）：版109 で皮質は 1番から撃つ（目標の 1.4〜3.8倍）⇒ 30番で ほぼ全部が 毎歩撃つ（約93Hz）⇒ 60番で 下から・持ち越し・駆動が NaN、表現 0Hz
    VRAM の山 15.92GB（GPU 全体）・OOM の警告
  【コード】下からの線の駆動は受け手ごとの和を累積和の差で出す（torch_brain._segsum、版95b）⇒ 組のどこか1本が無限大だと、後ろの受け手が全部 NaN
           持ち越し（v_carry）は NaN になると閾値を越えないまま残る（撃たないまま）
           下からの線の学び：入力が先の項は 滑る境目のずれが上限 0.197（ln）で、ドーパミンの割合 0.45 以上では強まりが残る。受け手が先の窓・海馬の線には境目が無い。重さの上限は無い
  5歩ごと（最大・最小だけを GPU の上で見て印を立て、20歩ごとに1回だけ読む）：下からの線の重さ・短期の状態・網の中の辺・持ち越し・率の平均・CL の重さ・核どうしの重さ が有限か
    ⇒ 最初に有限でなくなったら、その歩・どの線（道の名）／どの領野かを出して止める
  間隔ごと：領野ごとの表現の率（Hz・目標比）・ドーパミンの門（third）・重さの大きい下からの道 上位10・VRAM
  止め：VRAM の取り置き 14.0GB（2^30 バイト、vram_peak と同じ単位）を越えたら止める（画面を守る）・life.py の止めもそのまま
    ⟲ 11-11 19:50：10^9 バイトで 12.5 にしていて、20歩目（取り置き 12.77 ＝ 13.71×10^9）で止まった
  学びの2つの項（線ごと・歩の和、ln）：同じ歩（入力が先・送り手と受け手がそろう）の項 ／ 受け手が先の窓の項（境目のずれが無い）
    ⟲ 11-11 19:30 の Claude の読み「門が開くと撃ちすぎでも強まる」は計算の誤り：ずれが頭打ち（0.197）でも同じ歩の項は門 1 まで 0 以下
    ⇒ どちらの項が重さを伸ばしているかを測る
  ⚠️ 値を読むのは20歩ごとと間隔ごとだけ（重くしない。11-11 00:55 の止めの反省）
"""
import os, sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NMAX = int(args[0]) if args else 70
EV = int(args[1]) if len(args) > 1 else 5
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION not in (107, 109):   # 版109
    raise SystemExit(f"⚠️ 版 {B.VERSION}")
VMAX = float(os.environ.get("ANAPLEX_VMAX_GB", "14.0"))   # 2^30 バイト（logs/vram_peak の「GB」と同じ単位）。全体 15.92 のうち画面の分を残す
GIB = float(2 ** 30)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
n_all = g.n_ch_all
DT = 10.67 / 1000.0
but = b.but
nn_ = getattr(g, "nucnet", None)


def named():
    out = [("下からの線の重さ", but.w), ("下からの線の短期 state", but.state), ("下からの線の短期 fac", but.fac),
           ("網の中の辺", g.w), ("表現・誤差の持ち越し", g.v_carry), ("抑制の持ち越し", g.v_carry_inh), ("率の平均", g.rate_ema)]
    if getattr(b, "cl_on", False):
        out.append(("CL の重さ", b.cl_w))
    if nn_ is not None and hasattr(nn_, "w"):
        out.append(("核どうしの重さ", nn_.w))
    return out


NT = len(named())
eo = list(but.eo) if not torch.is_tensor(but.eo) else but.eo.tolist()
A_, C_ = TB._CL_LN_NODA, TB._CL_LN_DA
NP_ = len(but.keys)
pid_e = torch.repeat_interleave(torch.arange(NP_, device=dev), torch.tensor([eo[j + 1] - eo[j] for j in range(NP_)], device=dev))
acc_co = torch.zeros(NP_, dtype=torch.float64, device=dev)    # 同じ歩の項の和（ln）
acc_pf = torch.zeros(NP_, dtype=torch.float64, device=dev)    # 受け手が先の窓の項の和（ln）
acc_n = torch.zeros(NP_, dtype=torch.float64, device=dev)     # 同じ歩でそろった回数
L0 = but.learn


def learn_w(anyr, cl_spk, d):
    """_BUTable.learn と同じ式で、2つの項を線ごとに足す（重さは変えない。元の learn をそのあと呼ぶ）"""
    src = anyr if cl_spk is None else torch.cat([anyr, cl_spk])
    pidx = but._pidx if but._pidx is not None else torch.cat([(torch.arange(n, device=dev) + (c if c >= 0 else anyr.shape[0])) for c, n in zip(but.ch0, but.nt)])
    post = src[pidx]
    pre_now = but.pre_now
    pre_e = pre_now | but.pre_prev
    post_e = post[but.tgt] > 0
    dd_ = torch.as_tensor(d, dtype=but.w.dtype, device=dev)
    g_on = (2.0 * dd_ - 1.0).clamp(0.0, 1.0)
    sh_ = but.sh if getattr(but, "sh", None) is not None else 0.0
    ln_ctx = ((1.0 - TB._dd(dd_)) * A_ + TB._dd(dd_) * C_ - sh_) / TB.CL_PAIRS
    ln_pp = torch.where(but.hippo, TB.LN_PRE_HIP * g_on, ln_ctx)
    k = but.post_age.clamp(max=TB.WIN_MAXK + 1)[but.tgt]
    pf = but.tab[but.hippo.long(), k] * (pre_now & ~post_e).float() * g_on
    co_m = (pre_e & post_e)
    acc_co.index_add_(0, pid_e, (ln_pp * co_m.float()).double())
    acc_pf.index_add_(0, pid_e, pf.double())
    acc_n.index_add_(0, pid_e, co_m.double())
    return L0(anyr, cl_spk, d)


but.learn = learn_w


def terms(k=12):
    co, pf, n = acc_co.cpu().numpy(), acc_pf.cpu().numpy(), acc_n.cpu().numpy()
    cnt_e = np.array([eo[j + 1] - eo[j] for j in range(NP_)], dtype=float).clip(min=1)
    tot = (co + pf) / cnt_e                 # 線1本あたりの ln の和（窓の間）
    o = np.argsort(-tot)[:k]
    s_ = "  ".join(f"{but.keys[j]} 計{tot[j]:+.3f}（同じ歩{co[j] / cnt_e[j]:+.3f}・受け手が先{pf[j] / cnt_e[j]:+.3f}・そろい{n[j] / cnt_e[j]:.0f}回）" for j in o)
    acc_co.zero_(); acc_pf.zero_(); acc_n.zero_()
    return s_


flag = torch.zeros(NT, dtype=torch.bool, device=dev)
rep_n = torch.zeros(n_all, device=dev)
cnt = [0, 0]                    # 窓の歩・20歩ごとの数え
cx = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質"]
SHOW = cx + [n for n in ("視床", "CA1", "CA3", "歯状回", "黒質", "VTA", "背内側視床", "視床前核", "VA", "CL", "扁桃体基底", "海馬支脚") if n in NAMES]


def path_of(i):
    for j in range(len(but.keys)):
        if eo[j] <= i < eo[j + 1]:
            return but.keys[j]
    return "?"


def area_of_ch(i):
    for j, nm in enumerate(NAMES):
        if int(b.off_ch[j]) <= i < int(b.off_ch[j]) + b.n[nm]:
            return nm
    return "?"


def area_of_recv(r):
    for j, nm in enumerate(NAMES):
        if int(g.off_recv[j]) <= r < int(g.off_recv[j + 1]):
            return nm
    return "?"


def top_paths(k=10):
    mx = torch.stack([but.w[eo[j]:eo[j + 1]].max() if eo[j + 1] > eo[j] else torch.zeros((), device=dev) for j in range(len(but.keys))]).float().cpu().numpy()
    mn = torch.stack([but.w[eo[j]:eo[j + 1]].mean() if eo[j + 1] > eo[j] else torch.zeros((), device=dev) for j in range(len(but.keys))]).float().cpu().numpy()
    o = np.argsort(-np.nan_to_num(mx, nan=np.inf))[:k]
    hip = [bool(but.hippo[eo[j]]) if eo[j + 1] > eo[j] else False for j in o]
    return "  ".join(f"{but.keys[j]} 最大{mx[j]:.3g}・平均{mn[j]:.3g}{'（海馬の窓）' if h else ''}" for j, h in zip(o, hip))


def report(tag):
    ns = max(1, cnt[0])
    hz = (rep_n / ns / DT)
    q = g.rate_ema / g.target_recv.clamp(min=1e-9)
    cells = []
    for nm in SHOW:
        i = NAMES.index(nm)
        oc, n = int(b.off_ch[i]), b.n[nm]
        lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
        cells.append(f"{nm} {float(hz[oc:oc + n].mean()):.1f}Hz（{float(q[lo:hi].median()):.2f}）")
    th = g.third.reshape(-1).float().cpu().numpy() if torch.is_tensor(g.third) else g.third
    res = torch.cuda.memory_reserved() / GIB if str(dev).startswith("cuda") else 0.0
    print(f"\n── {tag}（{life.t}歩）── VRAM 取り置き {res:.2f}GB ／ 門 third {np.round(th, 3).tolist()}", flush=True)
    print("  率（窓の平均・括弧は 率の平均 ÷ 目標の中央）：" + " ／ ".join(cells), flush=True)
    print("  下からの道 重さの最大 上位10：" + top_paths(), flush=True)
    print("  学びで伸びた道 上位12（窓の間の ln の和・線1本あたり ＝ 同じ歩の項 ＋ 受け手が先の窓の項）：" + terms(), flush=True)
    rep_n.zero_()
    cnt[0] = 0
    return res


stop = [None]
vstop = [False]


INJ = TEST and os.environ.get("ANAPLEX_NAN_TEST") == "1"   # 容器の試しだけ：書き出しを確かめるため 1本に無限大を入れる


def until(lf):
    if INJ and cnt[1] == 30:
        but.w[eo[3]] = float("inf")
    rep_n.add_(g._any[0][:n_all].float())
    cnt[0] += 1
    cnt[1] += 1
    if cnt[1] % 5 == 0:                     # 5歩ごと：最大・最小だけで見る（NaN は最大に伝わる・無限大は最大か最小に出る。大きい写しを作らない）
        for k, (nm, t) in enumerate(named()):
            flag[k] |= ~(torch.isfinite(t.amax()) & torch.isfinite(t.amin()))
    if cnt[1] % 20 == 0:
        fl = flag.cpu().tolist()
        if any(fl):
            stop[0] = fl
            return True
        if str(dev).startswith("cuda") and torch.cuda.memory_reserved() / GIB > VMAX:   # 取り置きは CPU 側で読める（待たない）
            vstop[0] = True
            return True
    return False


print(f"版 {B.VERSION} ／ 上限 {NMAX}番・{EV}番ごと ／ VRAM の止め {VMAX}GB", flush=True)
done = 0
while done < NMAX:
    k = min(EV, NMAX - done)
    life.run_plans(k, learn=True, until=until)
    if stop[0] is not None:
        break
    if vstop[0]:
        report(f"{done}番〜（途中）")
        print(f"🛑 VRAM の取り置き ＞ {VMAX}GB で止めた（画面を守る）", flush=True)
        break
    done += k
    res = report(f"{done}番")
    if res > VMAX:
        print(f"🛑 VRAM の取り置き {res:.2f}GB ＞ {VMAX}GB で止めた（画面を守る）", flush=True)
        break

if stop[0] is not None:
    print(f"\n🛑 有限でない値が出た（{life.t}歩・{done}番〜）", flush=True)
    for k, (nm, t) in enumerate(named()):
        if not stop[0][k]:
            continue
        bad = ~torch.isfinite(t.reshape(-1))
        nb = int(bad.sum())
        i0 = int(torch.nonzero(bad)[0])
        kinds = f"inf {int(torch.isinf(t).sum())}・NaN {int(torch.isnan(t).sum())}"
        if t is but.w or t is but.state or t is but.fac:
            ps = sorted({path_of(int(i)) for i in torch.nonzero(bad).squeeze(1)[:2000].tolist()})
            where = f"道 {ps[:12]}"
        elif t is g.w:
            rs = sorted({area_of_recv(int(g.e_recv[int(i)])) for i in torch.nonzero(bad).squeeze(1)[:2000].tolist()})
            where = f"受け手の領野 {rs[:12]}"
        elif t is g.v_carry:
            where = f"領野 {sorted({area_of_ch(int(i) % n_all) for i in torch.nonzero(bad).squeeze(1)[:2000].tolist()})[:12]}"
        elif t is g.rate_ema:
            where = f"受け手の領野 {sorted({area_of_recv(int(i)) for i in torch.nonzero(bad).squeeze(1)[:2000].tolist()})[:12]}"
        else:
            where = f"最初の番号 {i0}"
        print(f"  ✗ {nm}：{nb} 個（{kinds}）・{where}", flush=True)
    print("  下からの道 重さの最大 上位10：" + top_paths(), flush=True)
    report("止めた時点")
print("\n（終わり）", flush=True)
