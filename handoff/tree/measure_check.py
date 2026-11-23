"""健診：目標値をまとめて測る（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_check.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_check.py 4 --test ⇒ CPU・相手 200・作り物の声・返し手なし）
目標値と出どころは claude/13_健診の目標値.md（【み】10-07「目標値をすべて測る測定器」）
版を変えたら必ずこれを回す。外れが出たら、原因を分ける道具（measure_v51s・v51t など）はそのあと
並べるもの（10回ごとに 1・2、最後に全部）
  1 領野ごとの率（1歩あたり。最後の周）表現 ／ 誤差 ／ 抑制の体 と人の値、外れの印
  2 ゴール1：聴覚の誤差（音のある歩 1歩あたり）自分 ／ 録音 ／ 返し手（録音は2回以上前の自分の声からでたらめ。measure_v51w と同じ）
  3 揃って撃つ：細胞ごとに「自分の発火」と「同じ領野のほかの体の同じ歩の発火数」の重なりが、前後 3〜12歩ずらしたときより多いか
      （20通りのずらしの中で一番多ければ p＝1/21＜0.05）。ずらしを近くに限るのは、音のある・なしで一緒に上下する分を除くため【C】
      揃う細胞の割合 表現（興奮性）／ 抑制の体 ⇔【ヒト】覚醒の γ に位相が揃う細胞 RS 9%・FS 13%（Le Van Quyen 2016）【C の当てはめ】
      あわせて、1歩に撃った割合の最大と上位1%（人の値なし）
  4 下から越えた体のうち撃てた割合・表現の体の抑制 ihn（皮質）
  5 中脳の核の率（Hz）／ 言えた ／ 1歩の時間 ／ VRAM 取り置きの山
⚠️ 模型の耳は録音と全く同じ音を聞く（人は骨からも聞く）⇒ 自分と録音の差は写しと蹴りからしか出ない（【み】10-07）
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (51, 52, 53):     # 版52・53 は網の作りが版51 と同じ
    sys.exit(f"⚠️ babble.py が版51・52 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2

# ── 目標値（13_健診の目標値.md と同じ。（下, 上）は1歩あたりの割合。None は人の値なし）──
NEO = ("聴覚", "運動", "視覚", "連合", "高次", "前頭前野")
HIP = ("CA3", "CA1", "嗅内浅", "嗅内深")
TGT = {}
for nm in NEO:     # 【ヒト】Verma 2024 覚醒 錐体 3.30Hz・抑制 4.70Hz（±半分〜2倍を範囲とみる【C】）
    TGT[nm] = {"表現": (0.0176, 0.070), "誤差": (0.0176, 0.070), "抑制": (0.025, 0.100)}
for nm in HIP:     # 【ヒト】Le Van Quyen 2008 錐体 3.1Hz・抑制 5.1Hz
    TGT[nm] = {"表現": (0.0165, 0.066), "誤差": (0.0165, 0.066), "抑制": (0.027, 0.108)}
TGT["歯状回"] = {"表現": (0.0, 0.025), "誤差": None, "抑制": None}            # 【ヒト】項目に応じるのは 2.5%未満
TGT["視床"] = {"表現": (0.05, 0.30), "誤差": None, "抑制": None}              # 【ヒト】運動視床・内側・腹側 約5〜30%
for nm in ("背内側視床", "側坐核", "上丘"):
    TGT[nm] = {"表現": None, "誤差": None, "抑制": None}
MB_HZ = {"黒質": 4.56, "視床下核": 36.0, "境": 33.0, "外側手綱核": 30.0, "吻内側被蓋核": 17.8}   # 黒質・視床下核【ヒト】ほか【マカク】
COUP = {"表現": 0.09, "抑制": 0.13}           # 【ヒト】LVQ 2016 覚醒 γ の位相に揃う RS 9%・FS 13%

torch.manual_seed(0)
if not TEST:
    torch.cuda.reset_peak_memory_stats()
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
d0 = g.decay[:, 0].clone()
IDX = {nm: NAMES.index(nm) for nm in NAMES}
rows = {}
for nm, i in IDX.items():
    a = g.areas[i]; lo = int(g.off_recv[i])
    rows[nm] = (slice(lo + a.n_ch, lo + 2 * a.n_ch), slice(lo + 2 * a.n_ch, lo + 2 * a.n_ch + a.n_inh))
rep_all = torch.cat([torch.arange(rows[nm][0].start, rows[nm][0].stop) for nm in NAMES]).to(g.dev)
inh_all = torch.cat([torch.arange(rows[nm][1].start, rows[nm][1].stop) for nm in NAMES]).to(g.dev)
cut_rep = np.cumsum([0] + [rows[nm][0].stop - rows[nm][0].start for nm in NAMES])
cut_inh = np.cumsum([0] + [rows[nm][1].stop - rows[nm][1].start for nm in NAMES])

st = {"tag": "他"}; floor = {"v": None}
rate = {nm: np.zeros(4) for nm in NAMES}          # 表現・誤差・抑制・歩
aud = {k: [0.0, 0] for k in ("自分", "録音", "返し手")}
aud_all = {k: [0.0, 0] for k in aud}
mb = {k: 0.0 for k in MB_HZ}; mb_n = [0]
spk_rep, spk_inh = [], []                          # 揃いを測るため（ビットに詰めて CPU に置く）
CORT = ("聴覚", "連合", "高次", "前頭前野", "運動")
fire = {nm: np.zeros(4) for nm in CORT}            # 的・撃てた・ihn の和・歩
orig_step, orig_pass = Brain.step, G.Group2._pass
rep_recv = torch.nonzero(g.is_rep).squeeze(1)
slc = {nm: slice(int(g.off_ch[IDX[nm]]), int(g.off_ch[IDX[nm] + 1])) for nm in CORT}
tm = {"t": 0.0, "n": 0}


def hooked_pass(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if first else None
    out = orig_pass(self, x_all, ext_all, learn, advance, first)
    if first and len(e):
        r = self.e_recv[e].long(); inh = self.e_inh[e]
        amp = 4.0 * self.w[e] * self._amp[e] * self.send_gain[self.e_src[e].long()]
        ihn = torch.zeros(self.n_recv_all, device=self.dev).index_add_(0, r[inh], amp[inh]) * self.scale
        for nm, s in slc.items():
            tgt = (x_all[s] * self.x_gain[s]) > self.fire_th[s]
            k = int(tgt.sum()); f = fire[nm]
            f[3] += 1
            if k:
                f[0] += k; f[1] += float(out[0][s][tgt].sum()); f[2] += float(ihn[rep_recv[s]][tgt].sum())
    return out


def step(self, ear, cmd, eye=None, learn=True):
    tb0 = g.tb[:, 0].clone()
    t0 = time.time()
    out = orig_step(self, ear, cmd, eye, learn)
    tm["t"] += time.time() - t0; tm["n"] += 1
    v = (g.tb[:, 0] - tb0 * d0) / (1 - d0) > 0.5
    for nm in NAMES:
        r = rate[nm]
        r[0] += float(v[rows[nm][0]].float().mean()); r[2] += float(v[rows[nm][1]].float().mean())
        if nm != "視床":
            r[1] += float((self.up[nm] > 0).float().mean())
        r[3] += 1
    spk_rep.append(np.packbits(v[rep_all].cpu().numpy())); spk_inh.append(np.packbits(v[inh_all].cpu().numpy()))
    for k in MB_HZ:
        mb[k] += float(self.mb.spk[k].mean())
    mb_n[0] += 1
    if st["tag"] in aud and floor["v"] is not None and float(torch.as_tensor(ear).float().sum()) > floor["v"]:
        n = float((out[1] > 0).sum())
        for d in (aud, aud_all):
            d[st["tag"]][0] += n; d[st["tag"]][1] += 1
    return out


Brain.step = step
G.Group2._pass = hooked_pass


def mark(x, rng_):
    if rng_ is None:
        return "  値なし"
    lo, hi = rng_
    return "  ○" if lo <= x <= hi else ("  ▼低い" if x < lo else "  ▲高い")


def show_rates():
    print("   領野        表現（人）            誤差（人）            抑制の体（人）", flush=True)
    for nm in NAMES:
        r = rate[nm]; k = max(r[3], 1); t = TGT[nm]
        cells = []
        for j, key in ((0, "表現"), (1, "誤差"), (2, "抑制")):
            x = r[j] / k
            if nm == "視床" and key == "誤差":
                cells.append(f"{'―':>8s}{'':12s}"); continue
            hu = "" if t[key] is None else f"{t[key][0]:.1%}〜{t[key][1]:.1%}"
            cells.append(f"{x:7.2%}（{hu or '―'}）{mark(x, t[key])}")
        print(f"   {nm:6s}  " + "  ".join(cells), flush=True)
        r[:] = 0


def show_aud(d, head):
    r = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in d.items()}
    print(f"   {head} 聴覚の誤差（音のある歩 1歩あたり）自分 {r['自分']:.2f} ／ 録音 {r['録音']:.2f} ／ 返し手 {r['返し手']:.2f}"
          f"  ⇒ 録音÷自分 {r['録音']/r['自分']:.2f}（＞1 が人の向き）  返し手÷自分 {r['返し手']/r['自分']:.2f}"
          f"  （歩 {d['自分'][1]}／{d['録音'][1]}／{d['返し手'][1]}）", flush=True)


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []; ok = tried = 0
t_start = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if len(out) > 4 and out[3]:
        tried += 1; ok += int(out[3] == out[4])
    if floor["v"] is None:
        floor["v"] = float(ear.sum(1).min()) + 1e-6
    if ear.sum() > 0:
        past.append((cmd, ear))

    def rec():
        if len(past) > DELAY:
            c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
            st["tag"] = "録音"; B.run(b, c0, e0, False, True)

    def rep():
        if rp is not None:
            st["tag"] = "返し手"; B.reply(b, rp, rng_r)

    for f in ((rec, rep) if rng_p.random() < 0.5 else (rep, rec)):
        f()
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        print(f"── {i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回", flush=True)
        show_rates()
        show_aud(aud, "ゴール1")
        for v in aud.values():
            v[0] = 0.0; v[1] = 0

# ── 最後に全部 ──
print("\n══ 全部 ══", flush=True)
show_aud(aud_all, "ゴール1（全部）")
print(f"   言えた {ok}/{tried}", flush=True)
print("   下から越えた体のうち撃てた ／ その体の抑制 ihn（人の値なし）  " + "  ".join(
    f"{nm} {f[1]/max(f[0],1):.0%}／{f[2]/max(f[0],1):.2f}" for nm, f in fire.items()), flush=True)
k = max(mb_n[0], 1)
print("   中脳の核（Hz、人・マカク）  " + "  ".join(
    f"{nm} {mb[nm]/k*1000/T.DT_MS:.1f}（{hz}）" for nm, hz in MB_HZ.items()), flush=True)
print(f"   1歩 {tm['t']/max(tm['n'],1)*1000:.1f}ms（耳の1歩 10.67ms）", end="", flush=True)
if not TEST:
    print(f"  VRAM 取り置きの山 {torch.cuda.max_memory_reserved()/2**30:.2f}GB（12GB まで）", end="")
print("", flush=True)

# ── 揃って撃つ（全部の歩で） ──
SR = np.unpackbits(np.stack(spk_rep), axis=1)[:, :cut_rep[-1]].astype(np.float32)
SI = np.unpackbits(np.stack(spk_inh), axis=1)[:, :cut_inh[-1]].astype(np.float32)
Tn = SR.shape[0]
LAGS = [k for k in range(-12, 13) if abs(k) >= 3]


def coupled(S):
    """自分の発火と同じ歩のほかの体の発火数の重なりが、前後 3〜12歩ずらしたどれよりも多い細胞の割合（10発以上撃った体のうち）"""
    act = S.sum(0) >= 10
    if act.sum() == 0 or Tn < 200:
        return float("nan"), 0
    S = S[:, act]
    P = S.sum(1)
    c = S.T @ P - S.sum(0)
    ge = np.zeros(S.shape[1])
    for k in LAGS:
        ck = S.T @ np.roll(P, k) - (S * np.roll(S, k, axis=0)).sum(0)
        ge += ck >= c
    p = (1 + ge) / (1 + len(LAGS))
    return float((p < 0.05).mean()), int(act.sum())


print(f"   揃って撃つ（{Tn}歩。前後3〜12歩より同じ歩に揃う細胞の割合）⇔【ヒト】覚醒 RS 9%・FS 13%【C の当てはめ】"
      "  ／ 1歩に撃った割合の最大・上位1%（人の値なし）", flush=True)
for j, nm in enumerate(NAMES):
    sr = SR[:, cut_rep[j]:cut_rep[j + 1]]; si = SI[:, cut_inh[j]:cut_inh[j + 1]]
    cr, nr = coupled(sr); ci, ni = coupled(si)
    fr = sr.mean(1)
    print(f"   {nm:6s}  表現 {cr:5.1%}（{nr}体）  抑制の体 {ci:5.1%}（{ni}体）"
          f"  ｜ 表現が1歩に撃った割合 最大 {fr.max():5.1%}・上位1% {np.quantile(fr, 0.99):5.1%}", flush=True)
print("○ 人の範囲 ／ ▲▼ 外れ ／ 値なし ＝ 人の値が無い（外れとは言わない）。範囲は人の値の半分〜2倍【C】", flush=True)
