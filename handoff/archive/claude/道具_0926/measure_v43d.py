"""版43d の試し（arkhe）：口が運動の領野を読んで言う。学ぶにつれて、選んだとおりに言えるか
  python3 measure_v43d.py [喃語の回数=100]
覚えは残さない（state.pt は読まない・書かない）。返し手も返す（babble.py と同じ回し方）
10回ごとに出すもの
  言えた      選んだモーラ列どおりに言えた回数 ／ 声を出そうとした回数
  指令の体    語の指令が入っている歩で、指令した体のうち運動の領野で撃った割合
  指令外      同じ歩で、指令していない体が撃った数（1歩あたり）
  抑制の重み  運動の領野の表現の体1つに入る抑制の重みの合計（学びで育つか）
⟲ コンテナ（相手 200、CPU）では 8回で 抑制 0 → 36、指令の体 1.00 → 0.4 前後、学びを止めると 1.00 のまま（09-29）"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
lo = int(b.off_ch[NAMES.index("運動")]); nm = b.n["運動"]
is_m = torch.zeros(g.n_recv_all, dtype=torch.bool, device=B.DEV)
is_m[torch.nonzero(g.is_rep).squeeze(1)[lo:lo + nm]] = True
m_inh = is_m[g.e_recv.long()] & g.e_inh


def inh_w():
    return float(g.w[m_inh].sum()) / nm      # 辺の並びは刈り込み（rewire）で変わらない前提。変わったら作り直す


box = {"hit": 0.0, "want": 0.0, "extra": 0.0, "n": 0}
ostep = b.step


def step(ear, cmd, *a, **k):
    r = ostep(ear, cmd, *a, **k)
    if float(cmd.sum()) > 0:
        rep = b.motor_rep
        box["hit"] += float((rep * cmd).sum()); box["want"] += float(cmd.sum())
        box["extra"] += float((rep * (1 - cmd)).sum()); box["n"] += 1
    return r


b.step = step
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
ok = tried = 0; ex = ""; t0 = time.time()
print("回   言えた  指令の体  指令外/歩  抑制の重み/体  例（選んだ → 言えた）")
for i in range(N):
    ear, cmd, n, want, got = say(rng, True)
    if want:
        tried += 1; ok += int(want == got)
        ex = f"{want} → {got}"
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        m_inh = is_m[g.e_recv.long()] & g.e_inh
        print(f"{i+1:4d}  {ok:2d}/{tried:2d}   {box['hit']/max(box['want'],1):6.2f}   {box['extra']/max(box['n'],1):7.1f}"
              f"   {inh_w():10.2f}    {ex}  （{(time.time()-t0)/60:.1f}分）", flush=True)
        ok = tried = 0; box.update(hit=0.0, want=0.0, extra=0.0, n=0)
print(f"VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB")
