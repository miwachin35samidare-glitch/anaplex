"""版111：率の時間の流れを、短い刻みで並べる（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_wave111.py [歩=15000] [刻み=200]
  何のため：term111 の 100番で、区切りのあと 50歩の皮質の率が 0.00〜0.04Hz なのに、約21秒のならし（率÷目標）は 1.46〜1.72 だった
    （data/arkhe_term111_1116.txt）⇒ 撃つ時期と黙る時期があるはず。50歩の窓 1つでは分からないので、刻みごとの率を始めから並べる
  出すもの（刻みごとに1行。数えは行ごとに 0 に戻す）
    皮質（運動の外）の表現の率：領野の平均の 平均・一番低い領野・一番高い領野 ／ 運動 ／ 歯状回・CA3・CA1
    その刻みで皮質が1発も撃たなかった歩の割合 ／ 門（third、行の終わりの値）／ その刻みの 声・自発・言えた／試した
  毎歩に足す仕事：表現の発火（体の数 110,406）を1本の数えへ足す 1回・皮質の発火の和 1回。線・辺には触らない・少ない箱への足し込みなし・
    網と同じ大きさの写しなし（measure_talk110 より軽い）
  歩は life.step（見張り・休み・止めはそのまま働く）。グラフで回す（ふだんと同じ）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 15000
EV = int(args[1]) if len(args) > 1 else 200
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 111:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版111 で確かめた（版111 で回す）")

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
n_all = g.n_ch_all
DT = 10.67 / 1000.0
CX = [n for n in NAMES if TB.PRIOR_TYPE.get(n) == "皮質" and n != "運動"]
HIP = [n for n in ("歯状回", "CA3", "CA1") if n in NAMES]
SL = {nm: slice(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm]) for nm in CX + ["運動"] + HIP}
cx_mask = torch.zeros(n_all, dtype=torch.bool, device=dev)
for nm in CX:
    cx_mask[SL[nm]] = True

acc = torch.zeros(n_all, device=dev)
print(f"版 {B.VERSION} ／ {NS} 歩・刻み {EV} 歩（{EV * DT:.1f} 秒）／ 皮質（運動の外）{len(CX)} 領野 ／ 体 {n_all:,}", flush=True)
print(f"  {'歩':>6s} ｜ 皮質 {'平均':>6s}{'低い':>6s}{'高い':>6s}{'黙った歩':>6s} ｜ {'運動':>6s} ｜ {'歯状回':>5s}{'CA3':>6s}{'CA1':>7s} ｜ 門 ｜ 声・自発・言えた", flush=True)
st0 = dict(life.stats)
quiet = 0
for i in range(1, NS + 1):
    life.step(learn=True)
    f = g._any[0][:n_all].float()
    acc.add_(f)
    quiet += int(float(f[cx_mask].sum()) == 0.0)
    if i % EV == 0:
        L.beat()
        r = acc / EV / DT
        m = [float(r[SL[nm]].mean()) for nm in CX]
        th = g.third.reshape(-1).tolist() if torch.is_tensor(g.third) else list(g.third)
        d = {k: life.stats[k] - st0[k] for k in life.stats}
        print(f"  {life.t:6d} ｜      {np.mean(m):6.2f}{min(m):6.2f}{max(m):6.2f}{quiet / EV:7.0%} ｜ {float(r[SL['運動']].mean()):6.2f} ｜ "
              + "".join(f"{float(r[SL[x]].mean()):6.2f}" for x in HIP)
              + f" ｜ {'・'.join(f'{x:.2f}' for x in th)} ｜ {d['voices']}・{d['spont']}・{d['ok']}/{d['tried']}", flush=True)
        acc.zero_(); quiet = 0; st0 = dict(life.stats)
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
