"""版51：今の網で、相手 9,100 が随伴放電に要るかを測り直す（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51u.py [喃語の回数=30] [相手の数 …（既定 9100 3000 1000）]
  （コンテナで道具を確かめるとき：python3 measure_v51u.py 2 200 100 --test ⇒ CPU・作り物の声）
【み】10-06「今の網で9100が要るかの測り直し」
前の測り（決まったこと 09-11、1領野・耳40体・area3 の作り）：相手 40 → 500 → 3,000 → 9,000 で supp 1.01 → 1.08 → 1.06 → 1.21
  「別々の時刻を運ぶ相手が要る ＝ 相手の数そのものが要る」⇒ 今の網（版51、全領野）で同じかは測っていない
相手の数（n_partner）を変えると変わるもの：表現・誤差の体の相手（その 16% が抑制）。変わらないもの：抑制の体の相手 180／194、
  視床の中継どうし 64、CA3 の 16,100
回し方（measure_v51h と同じ）：言う（写しあり）→ 同じ耳の入力を録音として聞く（写しなし）→ 返し手が返せば聞く。どれも学びながら
⚠️ 10-05 measure_v51h：今の網の supp は 10回ごとに 0.4〜1.9 と揺れる ⇒ 10回ごとの値と、全部を足した値の両方を出す
並べるもの（相手の数ごと）
  supp ＝ 録音の聴覚の誤差 ÷ 自分の聴覚の誤差（10回ごと ／ 全部）
  自分 ／ 録音 の聴覚の誤差（耳の発火 1本あたり）
  届く発火：聴覚の誤差の体1体に、1歩に届いた興奮の本数（自分で言う歩。09-11 の「届く発火」と同じ量。人 約135）
  言えた ／ 言おうとした
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 30
PS = [int(v) for v in args[1:]] or [9100, 3000, 1000]
if TEST:
    B.DEV = "cpu"
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
orig = G.Group2._pass
ctx = {"on": False, "arr": 0.0, "k": 0}


def hooked(self, x_all, ext_all, learn, advance, first):
    if ctx["on"] and advance:
        e = self._fired_edges()
        if len(e):
            r = self.e_recv[e].long()
            m = (r >= ctx["lo"]) & (r < ctx["hi"]) & ~self.e_inh[e]
            ctx["arr"] += float(m.sum()) / ctx["n"]
        ctx["k"] += 1
    return orig(self, x_all, ext_all, learn, advance, first)


G.Group2._pass = hooked
rows = []
for P in PS:
    torch.manual_seed(0)
    b = Brain(prior, n_partner=P, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    _, say, rp = B.voices(b, TEST)
    g = b.g; ia = NAMES.index("聴覚")
    lo = int(g.off_recv[ia]); ctx.update(lo=lo, hi=lo + g.areas[ia].n_ch, n=g.areas[ia].n_ch, arr=0.0, k=0)
    rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
    tot = np.zeros(3); blk = np.zeros(3); ok = tried = 0
    print(f"── 相手 {P}（辺 {len(g.w):,}）", flush=True)
    t0 = time.time()
    for i in range(N):
        ctx["on"] = True
        out = say(rng, True)
        ctx["on"] = False
        ear, cmd, n_own = out[0], out[1], out[2]
        if len(out) > 4 and out[3]:
            tried += 1; ok += int(out[3] == out[4])
        if ear.sum() > 0:
            n_rec = B.run(b, cmd, ear, False, True)
            v = np.array([n_own, n_rec, float(ear.sum())])
            tot += v; blk += v
        if rp is not None:
            B.reply(b, rp, rng_r)
        if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
            k = max(ctx["k"], 1)
            print(f"  {i+1:4d}回（{(time.time()-t0)/60:.1f}分）  supp {blk[1]/max(blk[0],1e-9):5.2f}"
                  f"  自分 {blk[0]/max(blk[2],1):.4f} ／ 録音 {blk[1]/max(blk[2],1):.4f}（耳1本あたり）"
                  f"  届く発火 {ctx['arr']/k:7.1f}本/歩  言えた {ok}/{tried}", flush=True)
            blk[:] = 0; ctx["arr"] = 0.0; ctx["k"] = 0
    rows.append((P, tot[1] / max(tot[0], 1e-9), ok, tried))
    del b, g
    if B.DEV == "cuda":
        torch.cuda.empty_cache()
print("まとめ（全部を足した supp）  " + "  ".join(f"相手 {P}: {s:.2f}（言えた {o}/{t}）" for P, s, o, t in rows), flush=True)
print("前の測り（09-11、1領野・耳40体）：相手 500 1.08 ／ 3,000 1.06 ／ 9,000 1.21", flush=True)
