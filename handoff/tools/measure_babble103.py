"""版103：喃語が言えるか（arkhe の ~/anaplex で。0から・学びながら・保存しない）
  python3 measure_babble103.py [番=800] [区切り=100]
  ANAPLEX_CX_TC=drive python3 measure_babble103.py   ⇒ 皮質の時間細胞が撃たせる側（版103f と同じ網）
  【み】11-01「抜いて喃語が言えるかではあるんだよな」
  暮らし（life.py、作り物の声）を 0 から回し、区切りごとに 言えた/試した・声・自発・1歩の時間・表現の率（いくつかの領野）。
  最後に健診（life.health 20番：自分・録音の聴覚の誤差、supp ＝ 録音 ÷ 自分）
  番 800 ＝ day.py の1回の覚醒（WAKE 800。そのあと眠る）。健診は day.py の「眠る前」と同じ life.health()
  ⚠️ 0 からの網（数値を合わせていない）。眠りは入れない。2つの網を同じ乱数で比べるためのもの
"""
import os, sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NT = int(args[0]) if len(args) > 0 else 800
W = int(args[1]) if len(args) > 1 else 100
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
SHOW = ("運動", "補足", "聴覚", "連合", "高次", "前頭前野", "眼窩前頭", "44", "中側頭後", "被殻")
idx = {nm: NAMES.index(nm) for nm in SHOW if nm in NAMES}
sync = (lambda: torch.cuda.synchronize()) if dev != "cpu" else (lambda: None)
print(f"版 {B.VERSION} ／ 皮質の時間細胞 {'撃たせる側（103f）' if os.environ.get('ANAPLEX_CX_TC') == 'drive' else '調整する側（103g）'}"
      f" ／ 0から {NT}番", flush=True)
print(f"  {'番':>5s} {'言えた':>8s} {'声':>4s} {'自発':>4s} {'ms/歩':>6s} " + "".join(f"{nm:>7s}" for nm in idx), flush=True)
done, ok_all, tr_all = 0, 0, 0
while done < NT:
    n = min(W, NT - done)
    st0 = dict(life.stats); t0s = life.t
    rep_n = torch.zeros(g.n_ch_all, device=dev)
    P0 = life.step

    def step_count(*a, **k):
        P0(*a, **k)
        rep_n.add_(g._any[0][:g.n_ch_all].float())

    life.step = step_count
    sync(); t0 = time.time()
    life.run_plans(n, learn=True)
    sync()
    life.step = P0
    ns = max(1, life.t - t0s)
    ms = (time.time() - t0) / ns * 1000.0
    d = {k: life.stats[k] - st0[k] for k in life.stats}
    ok_all += d["ok"]; tr_all += d["tried"]
    hz = []
    for nm, i in idx.items():
        nc, oc = g.areas[i].n_ch, int(b.off_ch[i])
        hz.append(float(rep_n[oc:oc + nc].mean()) / ns / (10.67 / 1000.0))
    done += n
    print(f"  {done:5d} {d['ok']:>3d}/{d['tried']:<4d} {d['voices']:4d} {d['spont']:4d} {ms:6.1f} " + "".join(f"{h:7.1f}" for h in hz), flush=True)
h = life.health()
print(f"合計 言えた {ok_all}/{tr_all} ／ 眠る前の健診自分 {h['自分']:.4f} ・ 録音 {h['録音']:.4f} ・ supp {h['supp']:.2f}"
      f" ・ 言えた {h['言えた']}/{h['試した']}", flush=True)
