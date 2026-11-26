"""版116c（11-27）：時間細胞の写しの輪を開いて、錐体が人の率の写しを受けたときに何Hz で撃つかを測る（網は変えない・保存しない）
  python3 measure_clamp116.py [歩=3000] [--exc=3.3] [--inh=4.7] [--noclamp]
  11-27 07:5x：見本の受け手の興奮を送り手の種類（時間細胞 自分・外／表現→表現 rr／ほか）で分けて出す行を足した。--noclamp で差し替えない（比べる元）
  何のため（組み表_版116_案 §37）：毎歩撃つ形の根が「写しの輪」（写す元が撃つほど写しも撃つ）か、
    「写しの重さ・足し方」（イ・ウ）かを分ける
    皮質20 の時間細胞の元の行（自分の行・外の行・抑制の行）の撃ちを、網の撃ちではなく人の率のくじに差し替える（測るためだけ）
      自分の行・外の行 ＝ 錐体 3.3Hz【ヒト】Verzhbinsky 2024 ／ 抑制の行 ＝ 4.7Hz【ヒト】同（--exc・--inh で変える）
    ⇒ 錐体が人の率のまわりに落ち着けば輪が元・まだ毎歩撃てば重さ・足し方が元
  出すものは measure_tc116 と同じ（「元の行が撃った割合」は網の本当の撃ち・「遅れの体」はくじから）
  毎歩に足す仕事：measure_tc116 の分 ＋ 元の行 約2万行のくじ1回（足し込みなし）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, torch_group2 as G, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 3000
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 116 or not TB.V116 or not getattr(TB, "V116C", False):
    raise SystemExit(f"⚠️ 版 {B.VERSION}（V116 {TB.V116}・V116C {getattr(TB, 'V116C', None)}）：この道具は版116c で回す")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
W = g.w.device
Lc = g.rr_L_ch
Lrow = g.rr_L_row.long()
nL = int(Lc.numel())
assert len(g.ch.areas) == len(g.areas) and all(x is y for x, y in zip(g.ch.areas, g.areas)), "時間細胞の並びが領野の並びと違う"

# 送り手の種類（遅れの体だけ）：1 自分の行・2 外の行・3 抑制の行 ／ 元の行の種類（g.ch.age の行）
cat_s = torch.zeros(g.n_send_all, dtype=torch.int8, device=W)
cat_r = torch.zeros(int(g.ch.off_cpu[-1]), dtype=torch.int8, device=W)
for nm in TB.CX_TC:
    i = NAMES.index(nm); a = g.areas[i]
    per, nc = a.ch.n_cell + a.ch.n_ramp, a.ch.n_cell
    lo = int(g.off_send[i]); r0 = int(g.ch.off_cpu[i])
    kind = torch.ones(a.n_src, dtype=torch.int8, device=W)
    kind[a.n_ch:a.n_ch + a.n_ext] = 2
    kind[a.n_ch + a.n_ext:] = 3
    cat_r[r0:r0 + a.n_src] = kind
    loc = lo + torch.arange(a.n_src, device=W).unsqueeze(1) * per + torch.arange(nc, device=W)
    cat_s[loc.reshape(-1)] = kind.unsqueeze(1).expand(-1, nc).reshape(-1)
# 版116c の測り：皮質20 の元の行の撃ちを、時間細胞へ渡す所だけ人の率のくじに差し替える（網の撃ちそのものは変えない）
def _opt(k, d):
    for a_ in sys.argv[1:]:
        if a_.startswith("--" + k + "="):
            return float(a_.split("=", 1)[1])
    return d
EXC_HZ, INH_HZ = _opt("exc", 3.3), _opt("inh", 4.7)
P_ROW = torch.zeros(int(g.ch.off_cpu[-1]), device=W)
P_ROW[cat_r == 1] = EXC_HZ * 0.01067
P_ROW[cat_r == 2] = EXC_HZ * 0.01067
P_ROW[cat_r == 3] = INH_HZ * 0.01067
CL_MASK = cat_r > 0
gen_c = torch.Generator(device=W); gen_c.manual_seed(1127)
REAL = {"f": None}
_ch_step0 = type(g.ch).step


NOCLAMP = "--noclamp" in sys.argv


def _ch_step_clamp(self, fired_all):
    if self is g.ch and NOCLAMP:
        REAL["f"] = (fired_all > 0).float()
    if self is g.ch and not NOCLAMP:
        REAL["f"] = (fired_all > 0).float()
        lot = (torch.rand(fired_all.shape, device=fired_all.device, generator=gen_c) < P_ROW).to(fired_all.dtype)
        fired_all = torch.where(CL_MASK, lot, fired_all)
    return _ch_step0(self, fired_all)


type(g.ch).step = _ch_step_clamp
print(("［差し替えなし（--noclamp）：比べる元］" if NOCLAMP else "") + f"［測るための差し替え{'：しない' if NOCLAMP else ''}］皮質20 の時間細胞の元の行の撃ち ⇒ くじ：自分・外の行 {EXC_HZ}Hz・抑制の行 {INH_HZ}Hz（網の撃ちは変えない）", flush=True)
n_s = [int((cat_s == k).sum()) for k in (1, 2, 3)]
n_r = [int((cat_r == k).sum()) for k in (1, 2, 3)]

# 見本の受け手（表現の体 256、固定の種）
gen = torch.Generator(device="cpu"); gen.manual_seed(1126)
SM = torch.randperm(nL, generator=gen)[:min(256, nL)].to(W)
nS = int(SM.numel())
smap = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=W)
smap[Lrow[SM]] = torch.arange(nS, device=W)
samp = {"src": None}


DEC = {"e": None, "acc": torch.zeros(6, device=W, dtype=torch.float64), "n": 0}
DEC_NAMES = ("時間細胞 自分", "時間細胞 外", "表現→表現（rr）", "ほかの網の辺", "調整の側（駆動に入らない）")
nTc = int(g.ch.spikes().numel())


def dec_rebuild():
    # 見本の受け手（表現の体 256）に入る興奮の辺を全部（刈り込みのあと作り直す）
    out = []
    for s0 in range(0, len(g.w), 1 << 22):
        r = g.e_recv[s0:s0 + (1 << 22)].long()
        k = (smap[r] >= 0) & ~g.e_inh[s0:s0 + (1 << 22)]
        out.append(torch.nonzero(k).squeeze(1) + s0)
    e = torch.cat(out); sp = g.e_src[e].long()
    cat = torch.full_like(sp, 3)
    cat[sp < nTc] = torch.where(cat_s[sp[sp < nTc]] == 2, 1, 0)
    cat[(sp >= nTc) & g.rr_send[sp]] = 2
    cat[g.mod_send[sp]] = 4
    DEC.update(e=e, sp=sp, rv=g.e_recv[e].long(), cat=cat, n_e=int(e.numel()))


def dec_step(act):
    # この歩の送り手の撃ち act（時間細胞の写し ＋ 遅れの列）で、見本の受け手の興奮を種類ごとに（÷閾値・恒常性の倍率込み）
    e, sp, rv, cat = DEC["e"], DEC["sp"], DEC["rv"], DEC["cat"]
    amp = g.c_exc[rv] * g.w[e] * g.send_gain[sp] * act[sp]
    if g._amp_buf is not None and len(g._amp_buf) == len(g.w):
        amp = amp * g._amp_buf[e]
    sc = g.scale[rv] * g.scale_up[rv]
    if g.exc_gain is not None:
        sc = sc * g.exc_gain[rv]
    th = (g.fire_th * g.th_gain)
    th = th[g.rr_L_ch[SM]] if torch.is_tensor(th) and th.numel() > 1 else torch.as_tensor(th, device=W)
    box = torch.zeros(5 * nS, device=W, dtype=amp.dtype).index_add_(0, cat * nS + smap[rv], amp * sc).reshape(5, nS)
    DEC["acc"][:5] += (box / th).mean(1).double()
    DEC["n"] += 1


def dec_print():
    if DEC["n"] == 0:
        return
    v = (DEC["acc"][:5] / DEC["n"]).cpu().numpy()
    tot = v[:4].sum()
    chk = float(DEC["acc"][5] / DEC["n"])
    print("  興奮を送り手の種類で（見本の受け手 %d・辺 %s 本・÷閾値・割る前・歩の頭の送り手）：" % (nS, f"{DEC['n_e']:,}") +
          "・".join(f"{a} {x:.2f}" for a, x in zip(DEC_NAMES, v)) + f"・駆動に入る分の和 {tot:.2f}（同じ見本の網の値 {chk:.2f}。容器では 0.7〜0.9倍しか合わない：周の中の学びの足し分ほかは数えていない ⇒ 種類の割合で読む）", flush=True)
    DEC["acc"].zero_(); DEC["n"] = 0


def samp_rebuild():
    idx = b._tc_idx.long()
    out = []
    for s0 in range(0, idx.numel(), 1 << 22):
        e = idx[s0:s0 + (1 << 22)]
        out.append(e[smap[g.e_recv[e].long()] >= 0])
    e = torch.cat(out)
    samp.update(src=b._tc_src, e=e, sp=g.e_src[e].long(), bx=smap[g.e_recv[e].long()] * 2 + (cat_s[g.e_src[e].long()].long() - 1),
                c=g.c_exc[g.e_recv[e].long()])
    assert bool((cat_s[samp["sp"]] >= 1).all() & (cat_s[samp["sp"]] <= 2).all()), "見本の辺に興奮の行の遅れの体でない送り手"


# _drive_core の出力を周ごとに足す（グラフの中に入る形：決まった置き場への足し込みだけ）
ACC = torch.zeros(4, nL, device=W)
CNT = torch.zeros(1, device=W)
_run0 = G._run


def _run_spy(fn, *a):
    out = _run0(fn, *a)
    if fn is G._drive_core:
        exc, ihn, exc_mod, drive = out[0], out[1], out[2], out[5]
        ACC[0].add_((exc - exc_mod)[Lrow])
        ACC[1].add_(ihn[Lrow])
        ACC[2].add_(drive[Lc])
        CNT.add_(1.0)
    return out


G._run = _run_spy
cap = {}
_step0 = G.Group2.step


def _step_spy(self, x_all, ext_all, learn=True):
    cap["x"] = x_all
    cap["buf"] = self._buf_flat_get().clone()
    return _step0(self, x_all, ext_all, learn)


G.Group2.step = _step_spy

KEYS = ("率", "元1", "元2", "元3", "遅1", "遅2", "遅3", "興奮", "時自", "時外", "下から", "CL", "1+抑制", "代わり", "駆動", "時割", "上興", "上抑")
SUM = {k: 0.0 for k in KEYS}
SUMT = torch.zeros(len(KEYS), device=W, dtype=torch.float64)
last = {"t": 0}
W_I = float(TB.W_INH_BU)


def window(t):
    L.beat()
    v = (SUMT / max(t - last["t"], 1)).cpu().numpy()
    d = dict(zip(KEYS, v))
    hz = lambda p: p / 0.01067
    print(f"\n━━ {last['t']}〜{t} 歩 ━━", flush=True)
    print(f"  皮質20 の表現 {hz(d['率']):.2f}Hz", flush=True)
    print(f"  元の行が撃った割合（歩あたり）  自分の行 {d['元1']:.1%}（{hz(d['元1']):.1f}Hz）・外の行 {d['元2']:.1%}（{hz(d['元2']):.1f}Hz）・"
          f"抑制の行 {d['元3']:.1%}（{hz(d['元3']):.1f}Hz）", flush=True)
    print(f"  遅れの体が撃った割合          自分の行 {d['遅1']:.1%}・外の行 {d['遅2']:.1%}・抑制の行 {d['遅3']:.1%}", flush=True)
    print(f"  押し ÷閾値（割る前）：網の中の興奮 {d['興奮']:.2f}（うち時間細胞 自分の行 {d['時自']:.2f}・外の行 {d['時外']:.2f}：見本）・"
          f"下から {d['下から']:.2f}・CL {d['CL']:.2f}", flush=True)
    print(f"  1＋抑制 {d['1+抑制']:.2f}（代わりの抑制 {d['代わり']:.2f}・切片の中ほか {d['1+抑制'] - 1 - d['代わり']:.2f}）", flush=True)
    print(f"  割ったあと：駆動 ÷閾値 {d['駆動']:.2f}・うち時間細胞 {d['時割']:.2f}（見本）", flush=True)
    print(f"  上限：時間細胞が全部撃ったとき ÷閾値 {d['上興']:.1f}（見本の中央の平均）・代わりの抑制が全部撃ったとき 1＋抑制 {1 + d['上抑']:.1f} ⇒ 比 {d['上興'] / max(d['上抑'], 1e-9):.2f}", flush=True)
    dec_print()
    SUMT.zero_(); last["t"] = t
    L.beat()


print(f"版 {B.VERSION} ／ {NS} 歩 ／ 皮質20 の表現の体 {nL:,}・見本 {nS}", flush=True)
print(f"遅れの体 自分の行 {n_s[0]:,}・外の行 {n_s[1]:,}・抑制の行 {n_s[2]:,} ／ 元の行 自分 {n_r[0]:,}・外 {n_r[1]:,}・抑制 {n_r[2]:,}", flush=True)
marks = {m for m in (200, 1000) if m < NS} | {NS}
nT = int(g.ch.spikes().numel())                      # 時間細胞の送り手は送り手の番号の頭に並ぶ（torch_group2 1001行）
assert bool((cat_s[nT:] == 0).all()), "時間細胞でない送り手に印"
m1, m2, m3 = (cat_s[:nT] == 1), (cat_s[:nT] == 2), (cat_s[:nT] == 3)
r1, r2, r3 = (cat_r == 1), (cat_r == 2), (cat_r == 3)
for i in range(1, NS + 1):
    ACC.zero_(); CNT.zero_()
    life.step(learn=True)
    if samp["src"] is not b._tc_src:
        samp_rebuild()
    if DEC["e"] is None or DEC.get("w_len") != len(g.w) or DEC.get("src") is not b._tc_src:
        dec_rebuild(); DEC["w_len"] = len(g.w); DEC["src"] = b._tc_src
    dec_step(torch.cat([b._tc_now.float(), cap["buf"].float()]))
    th = (g.fire_th * g.th_gain)
    th = th[Lc] if torch.is_tensor(th) and th.dim() > 0 and th.numel() > 1 else torch.as_tensor(th, device=W).expand(nL)
    np_ = CNT.clamp(min=1.0)
    exc, ihn, drv = ACC[0] / np_, ACC[1] / np_, ACC[2] / np_
    spk = b._tc_now                                    # この歩に届いた時間細胞の撃ち（_v116_pre で取った列）
    fr = REAL["f"]                                     # 元の行の本当の撃ち（くじに差し替える前）
    s = samp
    val = s["c"] * g.w[s["e"]] * spk[s["sp"]]
    box = torch.zeros(nS * 2, device=W, dtype=val.dtype).index_add_(0, s["bx"], val).reshape(nS, 2)
    top = torch.zeros(nS * 2, device=W, dtype=val.dtype).index_add_(0, s["bx"], s["c"] * g.w[s["e"]]).reshape(nS, 2).sum(1)
    ths = th[SM]
    DEC["acc"][5] += (exc[SM] / ths).mean().double()
    row = torch.stack([
        g._any[0][Lc].float().mean(),
        fr[r1].mean(), fr[r2].mean(), fr[r3].mean(),
        spk[m1].float().mean(), spk[m2].float().mean(), spk[m3].float().mean(),
        (exc / th).mean(), (box[:, 0] / ths).mean(), (box[:, 1] / ths).mean(),
        (cap["x"][Lc] * g.x_gain[Lc] / th).mean(), (g.cl_drive[Lc] / th).mean(),
        (1.0 + ihn).mean(), g.ext_ihn[Lrow].mean(),
        (drv / th).mean(), (box.sum(1) / ths / (1.0 + ihn[SM])).mean(),
        (top / ths).median(), torch.as_tensor(TB.N_TC_INH * W_I, device=W),
    ]).double()
    SUMT += row
    if i % 1000 == 0:
        L.beat()
    if i in marks:
        window(i)
if str(dev).startswith("cuda"):
    print(f"  ［VRAM］PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 2 ** 30:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
