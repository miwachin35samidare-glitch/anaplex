"""版103g：重さが育ったあとの「表現の体を撃たせている入力」を分ける（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_drive103g.py [先に回す番=50] [測る歩=40]     （グラフを撮らずに回す。値は同じ）
  measure_loop103（11-02 21:26）：44・補足・高次・連合は、自分に戻る輪の辺が全部そろって撃っても閾値 14.3 に届かない
    （44 3.3〜3.8・補足 0.7〜2.6・高次 1.7〜2.2・連合 1.4〜5.4）のに 27〜88Hz ⇒ 撃たせているのは輪の外
    measure_drive103（11-01）は 0から 60歩め（重さ 0 のまま・版103f）で測った ⇒ 版103g で重さが育ったあとに測り直す
  ⟲ 「越えた Hz」の列（不応期の前の周ごとの数。率ではない、50 §2）は外し、撃った率（Hz）を出す
  【み】11-01 17:55「速さへの大きい方を優先しよう」⇒ 撃った辺は暴走の分だけ多い（measure_fire103）。率を人に合わせる前に、症状から原因を分ける
  _drive_core（周ごと・表現の体が撃つか）の入力を、撃った体について分ける（式は _drive_core と同じ）：
    網   領野の中・領野どうしの網の辺（exc − 調整する側）÷（1＋抑制）
    下から 下からの線（x × x_gain）÷（1＋抑制）
    CL・i0 CL → 皮質・核の自分で撃つ分（cl_drive）÷（1＋抑制）
    持ち越し 前の周の足し合わせの残り（v_carry）
    自発   閾値に届かずに自発のくじで撃った
  領野ごとに：表現の体が閾値を越えた回数（不応期の前・周ごと。1歩あたり・周あたりの Hz）・越えたときの入力の内訳（閾値に対する割合の平均）・自発だけで撃った割合
  ⚠️ 不応期（3周）の前の数なので、毎周越える体は 937Hz と出る（実際に撃つのは 3周に1回まで）
"""
import os, sys, collections, numpy as np, torch
os.environ.setdefault("ANAPLEX_GRAPH", "eager")
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WARM = int(args[0]) if args else 50
NS = int(args[1]) if len(args) > 1 else 40
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
if WARM > 0:
    life.run_plans(WARM, learn=True)          # measure_loop103 と同じ回し方（学びながら・眠りなし）
g = b.g
A = len(NAMES)
ch_area = torch.cat([torch.full((b.n[nm],), i, dtype=torch.long, device=dev) for i, nm in enumerate(NAMES)])
acc = {k: torch.zeros(A, device=dev, dtype=torch.float64) for k in ("撃った", "網", "下から", "CL・i0", "持ち越し", "自発だけ", "周")}
rep_n = torch.zeros(g.n_ch_all, device=dev)
on = [False]
R0 = G._run


def run(fn, *a):
    out = R0(fn, *a)
    if fn is G._drive_core and on[0]:
        (both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry, fire_th, th_gain,
         complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_recv) = a
        exc, ihn, exc_mod2, p, pr, drive, rep = out
        ih = (1.0 + ihn[rep_i]).double()
        th = (fire_th * th_gain).double()
        xx = x_all if nuc_x is None else x_all + nuc_x
        parts = {"網": (exc - exc_mod2)[rep_i].double() / ih, "下から": (xx * x_gain).double() / ih,
                 "CL・i0": (cl_drive.double() / ih) if cl_drive is not None else torch.zeros_like(ih),
                 "持ち越し": v_carry.double() if v_carry is not None else torch.zeros_like(ih)}
        f = rep > 0
        below = f & (drive.double() <= th)
        acc["撃った"].index_add_(0, ch_area, f.double())
        acc["自発だけ"].index_add_(0, ch_area, below.double())
        for k, v in parts.items():
            acc[k].index_add_(0, ch_area, torch.where(f, v / th.clamp(min=1e-9), torch.zeros_like(v)))
        acc["周"] += 1.0
    return out


G._run = run
on[0] = True
t0 = life.t
for _ in range(NS):
    life.step()
    rep_n.add_(g._any[0][:g.n_ch_all].float())     # 表現の体が撃ったか（measure_loop103 と同じ数え方）
on[0] = False
ns_real = max(1, life.t - t0)
G._run = R0
n_pass = float(acc["周"][0])
fired = acc["撃った"].cpu().numpy()
rows = []
for i, nm in enumerate(NAMES):
    n = b.n[nm]
    if fired[i] <= 0:
        continue
    oc, nc = int(b.off_ch[i]), g.areas[i].n_ch
    hz = float(rep_n[oc:oc + nc].mean()) / ns_real / (10.67 / 1000.0)    # 撃った率（1歩に1回まで数える）
    m = {k: float(acc[k][i]) / fired[i] for k in ("網", "下から", "CL・i0", "持ち越し")}
    rows.append((fired[i] / NS, nm, n, hz, m, float(acc["自発だけ"][i]) / fired[i]))
print(f"版 {B.VERSION} ／ 先に {WARM}番（{t0}歩）・測る {NS}歩・周 {n_pass / NS:.0f}/歩 ／ 撃ったときの入力（閾値 ＝ 1 に対する割合の平均）", flush=True)
print("  網 ＝ 撃たせる側の網の辺（時間細胞は版103g で調整する側なので入らない）／ 持ち越し ＝ 前の周の足し合わせの残り（中身は前の周の網・下から・CL）", flush=True)
print(f"  {'領野':<8s} {'越えた/歩':>10s} {'体':>6s} {'率Hz':>10s} {'網':>6s} {'下から':>6s} {'CL・i0':>6s} {'持ち越し':>6s} {'自発だけ':>7s}", flush=True)
for fn_, nm, n, hz, m, sp in sorted(rows, key=lambda r: -r[0]):
    print(f"  {nm:<8s} {fn_:10,.0f} {n:6,d} {hz:10.1f} {m['網']:6.2f} {m['下から']:6.2f} {m['CL・i0']:6.2f} {m['持ち越し']:6.2f} {sp:7.1%}", flush=True)
