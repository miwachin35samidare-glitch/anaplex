"""起きているあいだに「疲れ」（下方調整が戻すはずの量）がたまっているかを見る（arkhe の ~/anaplex で  python3 read_fatigue.py ）

state.pt を読むだけ。網を回さない・書き換えない
領野ごとに出すもの
  発火率 ÷ 目標   細胞自身の発火率（rate_ema、約2000歩の移動平均）÷【ヒト】目標 0.45%/歩。1 より上が「たまった分」
                  表現・誤差・抑制に分けて出す
  下方調整なら    いま眠って下方調整が動いたら、1歩で scale に掛かる倍率（1 より下 ＝ 下げる側）
  興奮の重み      受け手の領野ごとの、興奮性の辺の重みの平均（0 から始まり、学ぶと伸びる）
  scale          恒常性の掛け率（起きているあいだは止めているので 1.000 のはず）
使い方：喃語を足すたびに回して並べると、たまり方（増え方）が見える
"""
import torch
import torch_area as T
import babble as B
from torch_brain import Brain, NAMES

DEV = "cuda"
prior = T.Prior.load("prior_v1.npz", dev=DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
done, log = B.load(b)
g = b.g
tgt = float(g.scaling_target)
a_ = 1.0 / g.homeo_tau
print(f"喃語 {done} 回の網（版 {B.VERSION}）  目標 {tgt:.4f}/歩  脳幹の状態 {b.bs.state}  起きていた歩数 {getattr(b.bs, 'n_wake', '?')}", flush=True)
print(f"  {'領野':4} {'発火率÷目標（表現／誤差／抑制）':>28} {'下方調整なら':>10} {'興奮の重み':>9} {'scale':>7}", flush=True)
for i, nm in enumerate(NAMES):
    lo, hi = int(g.off_recv[i]), int(g.off_recv[i + 1])
    a = b.areas[nm]
    r = g.rate_ema[lo:hi] / tgt
    err, rep, inh = r[:a.n_ch], r[a.n_ch:2 * a.n_ch], r[2 * a.n_ch:]
    f = ((1 - 0.01 * (g.rate_ema[lo:hi] / tgt - 1)).clamp(0.5, 1.5) ** a_).mean()
    m = (g.e_recv >= lo) & (g.e_recv < hi) & ~g.e_inh
    w = float(g.w[m].mean()) if bool(m.any()) else 0.0
    print(f"  {nm:4} {float(rep.mean()):8.2f} ／{float(err.mean()):6.2f} ／{float(inh.mean()):6.2f}"
          f" {float(f):12.6f} {w:10.3f} {float(g.scale[lo:hi].mean()):7.3f}", flush=True)
