"""版43b の働きの試し（arkhe）。音A → 間 → 音B を くり返し、後半で B を 25% 抜く
  python3 measure_v43b.py [回数=100]
見るもの：前頭前野に音が届くか（的）／「来なかった誤差」が B を抜いた時に撃つか／上丘の誤差／中脳の核の率／門"""
import sys, time, torch, numpy as np
import babble as B, torch_area as T
from torch_brain import Brain, NAMES, N_OMIT
torch.manual_seed(0)
N_TR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
dev = B.DEV
rng = np.random.default_rng(1)
A = torch.tensor((rng.random(B.N_EAR) < 0.15).astype(np.float32), device=dev)
Bs = torch.tensor((rng.random(B.N_EAR) < 0.15).astype(np.float32), device=dev)
Z = torch.zeros(B.N_EAR, device=dev); cmd = torch.zeros(b.n["運動"], device=dev)
lo = int(b.off_ch[NAMES.index("前頭前野")]); npf = b.n["前頭前野"]
orig = b.g.step; box = {}
def hook(x, e, learn):
    out = orig(x, e, learn); box["out"] = out; return out
b.g.step = hook
L = 50
rec = torch.zeros(N_TR, L, 12, device=dev)
omits = []
t0 = time.time()
for tr in range(N_TR):
    om = tr >= N_TR // 2 and rng.random() < 0.25
    omits.append(om)
    for s in range(L):
        ear = A if 5 <= s < 10 else (Bs if (20 <= s < 25 and not om) else Z)
        b.step(ear, cmd, learn=True)
        pe = box["out"][2]; xt = b.g.last_xt
        rec[tr, s] = torch.stack([b.omit.sum(), b.up["上丘"].mean(), b.gate.reshape(()),
                                  *[b.mb.spk[k].mean() for k in ("黒質", "視床下核", "境", "外側手綱核", "吻内側被蓋核")],
                                  xt[lo:lo + npf].sum(), pe[lo:lo + N_OMIT].max(), b.up["高次"].mean(), b.up["前頭前野"].mean()])
    if tr % 20 == 0:
        print(tr, f"{time.time()-t0:.0f}秒", flush=True)
r = rec.cpu().numpy(); om = np.array(omits)
HZ = 1000 / 10.67
def show(sel, name):
    m = r[sel]
    print(f"{name}（{len(m)}回）")
    for a, z, lab in ((0, 5, "前"), (5, 12, "A"), (12, 20, "間"), (20, 30, "Bの時"), (30, 50, "後")):
        x = m[:, a:z].reshape(-1, 12).mean(0)
        print(f"  {lab:4s} 来なかった{x[0]:5.2f} 上丘誤差{x[1]:.3f} 門{x[2]:.2f} 黒質{x[3]*HZ:5.1f} 視床下{x[4]*HZ:5.1f} "
              f"境{x[5]*HZ:5.1f} 手綱{x[6]*HZ:5.1f} 吻内{x[7]*HZ:5.1f} | 前頭前野の的{x[8]:5.1f} p最大{x[9]:.2f} "
              f"高次誤差{x[10]:.3f} 前頭前野誤差{x[11]:.3f}")
idx = np.arange(N_TR)
show(idx < 5, "最初の5回")
show((idx >= N_TR // 2) & ~om, "後半 B あり")
show((idx >= N_TR // 2) & om, "後半 B 抜き")
print(f"1歩 {(time.time()-t0)/N_TR/L*1000:.1f}ms  VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB" if dev != "cpu" else "")
