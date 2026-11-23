"""版103：発火の率が時間とともにどう動くか（arkhe の ~/anaplex で。0から・保存しない・網は変えない）
  python3 measure_rate103.py [起きる歩=3000] [眠る歩=0] [区切り=250]
  例）python3 measure_rate103.py 3000 3000   起きて 3000歩 → 眠って 3000歩 → 起きて 区切り2つぶん
  ねらい：数値合わせのノブを回す前に、measure_fire103 の率（0から 160歩め）が
          学び（興奮の重さは 0 から育つ）と恒常性（眠りで下げる）で落ち着く途中の値か、続く値かを見る
  出すもの：区切りごとに 1歩の時間（ms）・撃った辺/歩・領野ごとの表現の率（Hz）・恒常性の倍率（表現の行の平均）・興奮の重さの和（表現の行1つあたり。0 から育つ）
  ⚠️ 率は見るだけ。ノブを回すかは、この並びを見てから
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import sleep as S
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NW = int(args[0]) if len(args) > 0 else 3000
NSL = int(args[1]) if len(args) > 1 else 0
W = int(args[2]) if len(args) > 2 else 250
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
A = len(g.areas)
deg = (g.start[1:] - g.start[:-1]).float()
fired = torch.zeros((), device=dev)
P0 = type(g)._fired_edges


def fe(self):
    out = P0(self)
    on = getattr(self, "_base_on", None)
    if on is not None and len(on) == len(deg) and not getattr(self, "_fe_seen", False):
        fired.add_((on.float() * deg).sum())
        self._fe_seen = True
    return out


type(g)._fired_edges = fe
sync = (lambda: torch.cuda.synchronize()) if dev != "cpu" else (lambda: None)
cols = []          # (札, ms/歩, 撃った辺/歩, {領野: Hz}, {領野: 倍率})


def window(tag, n):
    rep_n = torch.zeros(g.n_ch_all, device=dev)
    fired.zero_()
    sync(); t0 = time.time()
    for _ in range(n):
        g._fe_seen = False
        life.step()
        rep_n += g._any[0][:g.n_ch_all].float()
    sync(); ms = (time.time() - t0) / n * 1000.0
    hz, sc, ww = {}, {}, {}
    wsum = torch.zeros(g.n_recv_all, device=dev).index_add_(0, g.e_recv.long(), g.w * (~g.e_inh).float())
    for i, nm in enumerate(NAMES):
        nc, oc = g.areas[i].n_ch, int(b.off_ch[i])
        lo = int(g.off_recv[i])
        hz[nm] = float(rep_n[oc:oc + nc].mean()) / n / (10.67 / 1000.0)
        s_ = g.scale if torch.is_tensor(g.scale) else None
        sc[nm] = float(s_[lo + nc:lo + 2 * nc].mean()) if s_ is not None and s_.numel() == g.n_recv_all else float("nan")
        ww[nm] = float(wsum[lo + nc:lo + 2 * nc].mean())
    cols.append((tag, ms, float(fired) / n, hz, sc, ww))
    print(f"  {tag:>8s}  {ms:6.1f}ms/歩  撃った辺 {float(fired) / n:12,.0f}/歩", flush=True)


print(f"版 {B.VERSION} ／ 0から ／ 起きる {NW}歩 ・ 眠る {NSL}歩 ・ 区切り {W}歩", flush=True)
done = 0
while done < NW:
    n = min(W, NW - done)
    window(f"起{done + n}", n)
    done += n
if NSL > 0:
    sync(); t0 = time.time()
    S.sleep(b, n_step=NSL, report_every=max(1, NSL // 4), verbose=False)
    sync()
    print(f"  眠った {b.bs.last_slept}/{NSL}歩（{(time.time() - t0) / max(1, NSL) * 1000:.1f}ms/歩）", flush=True)
    b.bs.set_state(BS.WAKE)
    for k in range(2):
        window(f"後{(k + 1) * W}", W)

print("\n表現の率（Hz、目標 皮質 3.3）。区切りごと", flush=True)
print(f"  {'領野':<8s}" + "".join(f"{c[0]:>8s}" for c in cols), flush=True)
order = sorted(NAMES, key=lambda nm: -max(c[3][nm] for c in cols))
for nm in order:
    print(f"  {nm:<8s}" + "".join(f"{c[3][nm]:8.1f}" for c in cols), flush=True)
print("\n恒常性の倍率（表現の行の平均。1 ＝ 動いていない）。率の高い 15 領野", flush=True)
print(f"  {'領野':<8s}" + "".join(f"{c[0]:>8s}" for c in cols), flush=True)
for nm in order[:15]:
    print(f"  {nm:<8s}" + "".join(f"{c[4][nm]:8.3f}" for c in cols), flush=True)
print("\n興奮の重さの和（表現の行1つあたり。学びで 0 から育つ）。率の高い 15 領野", flush=True)
print(f"  {'領野':<8s}" + "".join(f"{c[0]:>8s}" for c in cols), flush=True)
for nm in order[:15]:
    print(f"  {nm:<8s}" + "".join(f"{c[5][nm]:8.2f}" for c in cols), flush=True)
