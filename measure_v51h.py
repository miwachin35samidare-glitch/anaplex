"""版51：網の健康診断（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51h.py [喃語の回数=50]
喃語の段のゴール 1（論点台帳 10-05）：随伴放電が学べて、自分の声・録音・返し手の声を聞き分けられる
回し方：言う（写しあり）→ 同じ耳の入力を録音として聞く（写しなし）→ 返し手が返せば聞く。どれも学びながら（measure_v49b と同じ）
10回ごとに出すもの
  1 領野ごとの 表現・誤差・抑制の体 の発火率（1歩あたり）と、表現が 40% を越えた歩の割合（一斉発火）
     比べる人の値（1歩 10.67ms）：新皮質 錐体 3.5%（Verma 2024 覚醒 3.30Hz）・抑制 5.0%（4.70Hz）／
       海馬・嗅内 錐体 3.3%・抑制 5.4%（Le Van Quyen 2008）／ 視床の中継 約5〜30%（運動視床など。聴覚の視床は無い）
  2 聴覚の誤差（音のある歩 1歩あたりの体数）：自分 ／ 録音 ／ 返し手。supp ＝ 録音 ÷ 自分、返し手 ÷ 自分
読み方：表現と抑制の体は最後の周の値（tb に足す発火を tau 1 の欄から戻す。版50 の並び cat([rep, rep, inh])）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain, NAMES

if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
NC = int(g.off_ch[-1]); off_inh = np.cumsum([0] + [a.n_inh for a in g.areas])
idx = {nm: NAMES.index(nm) for nm in NAMES}
d0 = g.decay[:, 0].clone()
st = {"tag": "他"}
acc = {}          # 領野 → [表, 誤, 抑, 40%超, 歩]
aud = {"自分": [0.0, 0], "録音": [0.0, 0], "返し手": [0.0, 0]}
orig_step = Brain.step


def step(self, ear, cmd, eye=None, learn=True):
    tb0 = g.tb[:, 0].clone()
    out = orig_step(self, ear, cmd, eye, learn)
    v = (g.tb[:, 0] - tb0 * d0) / (1 - d0)
    for nm, i in idx.items():
        a = g.areas[i]; c = int(g.off_ch[i]); h = 2 * NC + int(off_inh[i])
        rep = float((v[NC + c:NC + c + a.n_ch] > 0.5).float().mean())
        err = float((self.up[nm] > 0).float().mean()) if nm != "視床" else float("nan")
        inh = float((v[h:h + a.n_inh] > 0.5).float().mean())
        s = acc.setdefault(nm, [0.0, 0.0, 0.0, 0.0, 0])
        s[0] += rep; s[1] += 0.0 if err != err else err; s[2] += inh; s[3] += rep > 0.4; s[4] += 1
    if st["tag"] in aud and float(torch.as_tensor(ear).float().sum()) > aud_floor["v"]:
        aud[st["tag"]][0] += float((out[1] > 0).sum()); aud[st["tag"]][1] += 1
    return out


aud_floor = {"v": 0.0}
Brain.step = step
HUMAN = {**{nm: "錐体 3.5 抑 5.0" for nm in ("聴覚", "運動", "視覚", "連合", "高次", "前頭前野")},
         **{nm: "錐体 3.3 抑 5.4" for nm in ("歯状回", "CA3", "CA1", "嗅内浅", "嗅内深")},
         "視床": "中継 5〜30", "背内側視床": "中継 5〜30"}
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
for i in range(N):
    st["tag"] = "自分"; ear, cmd, n, want, got = say(rng, True)
    if i == 0:
        aud_floor["v"] = float(np.asarray(ear).sum(1).min()) + 1e-6   # 無音の耳の強さ（一番小さい値）
    if np.asarray(ear).sum() > 0:
        st["tag"] = "録音"; B.run(b, cmd, ear, False, True)
    st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0:
        print(f"── {i+1}回目まで（この10回）")
        print("   領野        表現    誤差    抑制    表現40%超の歩   人（%/歩）")
        for nm in NAMES:
            s = acc[nm]; k = max(s[4], 1)
            e = "  ―  " if nm == "視床" else f"{s[1]/k:5.1%}"
            print(f"   {nm:6s}  {s[0]/k:6.1%}  {e:>6s}  {s[2]/k:6.1%}   {s[3]/k:6.1%}        {HUMAN.get(nm, '人の値なし')}")
        r = {k: (v[0] / v[1] if v[1] else float('nan')) for k, v in aud.items()}
        print(f"   聴覚の誤差（音のある歩 1歩あたり）自分 {r['自分']:.1f} ／ 録音 {r['録音']:.1f} ／ 返し手 {r['返し手']:.1f}"
              f"  ⇒ 録音÷自分 {r['録音']/r['自分']:.2f}  返し手÷自分 {r['返し手']/r['自分']:.2f}"
              f"  （歩 {aud['自分'][1]} ／ {aud['録音'][1]} ／ {aud['返し手'][1]}）", flush=True)
        acc.clear()
        for v in aud.values():
            v[0] = 0.0; v[1] = 0
