"""版93 の合わせ：網の本当の入力で、中脳の核の i0 をふだんの率に合わせ直す（1回だけ回す道具）
  python3 calib_v93.py [歩=3000]            （arkhe の ~/anaplex で。0から・学びながら・保存しない）
  python3 calib_v93.py 300 --test           （コンテナで道具を確かめるとき：CPU・相手 200）
手順
  1 網を [歩] 回し、中脳へ入る外の発火（前頭前野の表現・側坐核の表現＋誤差・上丘の誤差・PPN、撃った周つき）を写し取る
     喃語と同じ回し方（自分の声 → 耳）。中脳の学びは止める（重さ 1 のまま）
  2 写し取った入力を、中脳の核だけに何度も流し直して、視床下核の辺縁 → 境 → 外側手綱核 → 吻内側被蓋核 → 黒質 の順に
     i0 を二分法で的の率（19.3・33・30・17.8・4.56Hz）へ（外側手綱核は i0 ≥ 0、吻内側被蓋核は入力なしの 6.7Hz を守らず、ふだん 17.8Hz に）
     ⚠️ i0 だけを動かす（強さ g は bench_v93h のまま）。i0 が 0 で足りないときは「下限」と出す ⇒ 相談
  3 結果を midbrain_i0.json に書く（midbrain.py が組むときに読む。消せば版93 の初めの値）
"""
import sys, json, copy, time, torch, numpy as np
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T
import midbrain as MB
from torch_brain import Brain

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NSTEP = int(args[0]) if args else 3000
if TEST:
    B.DEV = "cpu"

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
rec = []
orig = b.mb.step


def grab(ext, learn=True):
    rec.append({k: v.detach().to("cpu", torch.uint8) for k, v in ext.items()})
    return orig(ext, learn=False)


b.mb.step = grab
print(f"1 網を回して中脳の入力を写す（{NSTEP} 歩）", flush=True)
t0 = time.time()
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
while len(rec) < NSTEP:                      # 健診2 と同じ暮らし（自分の声 ＋ 返し手）
    say(rng, True)
    if rp is not None:
        B.reply(b, rp, rng_r)
print(f"  {len(rec)} 歩（{time.time() - t0:.0f}秒）／ 入力の率（Hz）：" + "・".join(
    f"{k} {float(torch.stack([r[k].float().sum(0).mean() for r in rec]).mean()) * 1000 / 10.67:.2f}" for k in rec[0]), flush=True)

n_src = dict(b.mb.n_src)
MB.I0_FILE = "__合わせる間は読まない__.json"   # 流し直す核は版93 の初めの i0 から（前の midbrain_i0.json を読まない）
dev = "cpu"


def rates(i0):
    m = MB.Midbrain(n=b.mb.n, dev=dev, gen=torch.Generator().manual_seed(1))
    m.wire(n_src)
    for nm, v in i0.items():
        a, z = m.off[nm], m.off[nm] + m.n[nm]
        m.i0[a:z] = v
    warm = min(200, len(rec) // 5)
    for k, r in enumerate(rec):
        if k == warm:
            m.rate_sum.zero_(); m.steps = 0
        m.step({kk: vv.float() for kk, vv in r.items()}, learn=False)
    return m.rates_hz()


i0 = {k: v["i0"] for k, v in MB.NUC.items()}
print("2 i0 を合わせる（写した入力を流し直す）", flush=True)
for nm in ("視床下核", "境", "外側手綱核", "吻内側被蓋核", "黒質"):
    lo, hi = 0.0, 0.3
    tgt = MB.HZ[nm]
    for _ in range(12):
        i0[nm] = (lo + hi) / 2
        r = rates(i0)[nm]
        lo, hi = (i0[nm], hi) if r < tgt else (lo, i0[nm])
    i0[nm] = (lo + hi) / 2
    r = rates(i0)[nm]
    flag = "  ⚠️ i0 が下限（0）でも率が高い ⇒ 相談" if i0[nm] < 1e-4 and r > tgt * 1.1 else ""
    print(f"  {nm}：i0 {MB.NUC[nm]['i0']:.5f} → {i0[nm]:.5f} ⇒ {r:.1f}Hz（的 {tgt}）{flag}", flush=True)
MB.I0_FILE = "midbrain_i0.json"
json.dump({"i0": i0, "steps": len(rec), "test": TEST}, open("midbrain_i0.json", "w"), ensure_ascii=False, indent=1)
print("3 midbrain_i0.json に書いた（消せば版93 の初めの値）", flush=True)
print("  合わせたあとの率：" + "・".join(f"{k} {v:.1f}" for k, v in rates(i0).items()), flush=True)
