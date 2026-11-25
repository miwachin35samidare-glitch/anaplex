"""版116（11-26）：皮質20 の表現の体が毎歩撃つ形の中身を分ける（arkhe の ~/anaplex で。網は変えない・保存しない）
  python3 measure_tc116.py [歩=3000]
  何のため（組み表_版116_案 §22・§23）：検算は「代わりの興奮 ÷ 代わりの抑制 の上限の比 ≒ 1.8」だったが、
    抑制の行も 80% 撃つと置いた上の側の見積もり。網の値で次を分ける
  出すもの（区間ごと・皮質20 の平均）
    1 元の行が撃った割合（歩あたり）：自分の行・外の行（ほかの領野から届いた入力）・抑制の行（切片の中の抑制の体）
    2 遅れの体が撃った割合：同じ3つ
    3 押し（÷閾値 fire_th × th_gain、1＋抑制 で割る前）：網の中の興奮の和（_drive_core の exc − exc_mod、周の平均）・
      うち時間細胞（見本の 256体で数えた。自分の行・外の行に分ける。⚠️ 短期の増減と send_gain を入れない）・下から・CL
    4 1＋抑制（周の平均）：全部・うち代わりの抑制（ext_ihn、版116 C）・残り（切片の中）
    5 割ったあと：駆動 ÷閾値（_drive_core の drive、周の平均。持ち越し込み）・時間細胞の分 ÷ (1＋抑制)（見本）
    6 上限の比：時間細胞の興奮が全部撃ったとき ÷閾値（見本）と、代わりの抑制が全部撃ったときの 1＋抑制
  毎歩に足す仕事：_drive_core の出力から 3,256体を4つ足す（周ごと・グラフの中に入る）・見本 256体の時間細胞の辺 約190万本の
    index_add（箱 512）・元の行の歳の最小（送り元 数十万行）。刈り込みのあと見本の辺を作り直す（2,456万本を 1<<22 ずつ）
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
if B.VERSION != 116 or not TB.V116:
    raise SystemExit(f"⚠️ 版 {B.VERSION}（V116 {TB.V116}）：この道具は版116 で回す")
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
n_s = [int((cat_s == k).sum()) for k in (1, 2, 3)]
n_r = [int((cat_r == k).sum()) for k in (1, 2, 3)]

# 見本の受け手（表現の体 256、固定の種）
gen = torch.Generator(device="cpu"); gen.manual_seed(1126)
SM = torch.randperm(nL, generator=gen)[:min(256, nL)].to(W)
nS = int(SM.numel())
smap = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=W)
smap[Lrow[SM]] = torch.arange(nS, device=W)
samp = {"src": None}


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
    th = (g.fire_th * g.th_gain)
    th = th[Lc] if torch.is_tensor(th) and th.dim() > 0 and th.numel() > 1 else torch.as_tensor(th, device=W).expand(nL)
    np_ = CNT.clamp(min=1.0)
    exc, ihn, drv = ACC[0] / np_, ACC[1] / np_, ACC[2] / np_
    spk = b._tc_now                                    # この歩に届いた時間細胞の撃ち（_v116_pre で取った列）
    fr = (g.ch.age.min(1).values == 0).float()        # この歩に撃った元の行（歩の終わりの歳）
    s = samp
    val = s["c"] * g.w[s["e"]] * spk[s["sp"]]
    box = torch.zeros(nS * 2, device=W, dtype=val.dtype).index_add_(0, s["bx"], val).reshape(nS, 2)
    top = torch.zeros(nS * 2, device=W, dtype=val.dtype).index_add_(0, s["bx"], s["c"] * g.w[s["e"]]).reshape(nS, 2).sum(1)
    ths = th[SM]
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
