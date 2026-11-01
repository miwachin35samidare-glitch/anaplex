"""版103：高次の皮質を撃たせている「網の辺」の出どころ（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_net103.py [歩=30]
  measure_drive103（11-01 18:07）：眼窩前頭・中側頭後・STS1・扁桃体・前頭前野 は 網の辺 が閾値の 1.6〜4.6倍
  ⇒ その辺の送り手がどこか（どの領野の・時間細胞か遅れの列か）を、受け手 ＝ 表現の体の行について数える
  1歩の頭で撃っている送り手（_base_on）× 重さ × 接触 × 送り手の利得 で足す（短期の増減は入れない・興奮だけ。調整する側の送り手は除く）
  ⚠️ 見るのは割合（どこから来るか）。量そのものは周ごとの足し込みと同じではない
"""
import sys, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 30
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
TGT = ("眼窩前頭", "中側頭後", "STS1", "高次", "前頭前野", "45", "44", "前帯状", "運動", "扁桃体基底")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(60):
    life.step()
g = b.g
A = len(g.areas)
ch_off, buf_off = [0], [0]
for a in g.areas:
    ch_off.append(ch_off[-1] + a.ch.n_cells); buf_off.append(buf_off[-1] + a.n_src * a.D)
bb = ch_off[-1]
lab = torch.empty(g.n_send_all, dtype=torch.long, device=dev)
for i in range(A):
    lab[ch_off[i]:ch_off[i + 1]] = i
    lab[bb + buf_off[i]:bb + buf_off[i + 1]] = i
kind = (torch.arange(g.n_send_all, device=dev) >= bb).long()          # 0 時間細胞 ／ 1 遅れの列
key = lab * 2 + kind
sel = {}
for nm in TGT:
    i = NAMES.index(nm); lo = int(g.off_recv[i]); nc = g.areas[i].n_ch
    r = g.e_recv.long()
    m = (r >= lo + nc) & (r < lo + 2 * nc) & ~g.e_inh & ~g.mod_send[g.e_src.long()]
    sel[nm] = torch.nonzero(m).squeeze(1)
acc = {nm: torch.zeros(2 * A, device=dev, dtype=torch.float64) for nm in TGT}
for _ in range(NS):
    life.step()
    on = g._base_on if getattr(g, "_base_on", None) is not None else None
    if on is None:
        continue
    for nm, e in sel.items():
        s = g.e_src[e].long()
        amp = (g.w[e] * g.c_exc[g.e_recv[e].long()] * g.send_gain[s] * on[s].float()).double()
        acc[nm].index_add_(0, key[s], amp)
KN = ("時間細胞", "遅れの列")
print(f"版 {B.VERSION} ／ 暮らし {NS}歩 ／ 表現の体の行へ来る網の辺（興奮・駆動する側）の出どころ（割合）", flush=True)
for nm in TGT:
    v = acc[nm]; tot = float(v.sum())
    if tot <= 0:
        print(f"  {nm}：撃った送り手なし", flush=True); continue
    top = torch.argsort(v, descending=True)[:8].tolist()
    own = float(v[2 * NAMES.index(nm)] + v[2 * NAMES.index(nm) + 1]) / tot
    parts = "・".join(f"{NAMES[k // 2]}の{KN[k % 2]} {float(v[k]) / tot:.0%}" for k in top if float(v[k]) > 0)
    print(f"  {nm}（辺 {len(sel[nm]):,}・自分の領野から {own:.0%}）：{parts}", flush=True)
