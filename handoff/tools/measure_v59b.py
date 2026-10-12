"""撃った表現の体の「駆動 ÷ 閾値」r の分布と、人の f-I に通したときの1歩に2発の割合（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は書き換えない。
暮らしと試しは measure_check2 と同じ）
  python3 measure_v59b.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v59b.py 3 --test）
【み】10-11 0時48分「いいよ」
【ヒト】Barzó・Szöts ほか 2024 eLife（L2/3 錐体、大人 189体）：閾値の電流 208〜263pA・f-I の傾き 0.13〜0.18Hz/pA
【ヒト】Ison ほか 2011：10ms 未満の間隔の割合 錐体 0.17〜0.25（海馬 0.25）・嗅内 0.07
【計算】f(r) ＝ 0.15Hz/pA × 230pA × (r − 1)【C の当てはめ：模型の駆動を細胞体の電流に見立てる】
  1歩の中で最初に撃った周 k（1〜10）から、残り (10 − k) × 1.102ms のうちに次の発火が来る（1/f ≦ 残り）なら「人なら2発」
並べるもの（領野ごと・10回ごと。表現の体、下から＋横の駆動で越えて撃った体だけ。自発・見込みで足した分は除く）
  r の 10・50・90% 点 ／ r ＞ 3.7 の割合 ／ 人の f-I なら1歩に2発の割合 ／ 模型の実際の2発以上の割合 ／ 人の f-I で1歩に撃つ見込み（f × 10.67ms、1 で頭打ち）の平均
"""
import sys, time, inspect, textwrap, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (59,):
    sys.exit(f"⚠️ babble.py が版59 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
N_PROBE = 3
SLOPE, RHEO = 0.15, 230.0            # 【ヒト】Barzó・Szöts 2024 大人の f-I の傾き（Hz/pA）・閾値の電流（pA）
PASS_MS = T.SYN_DELAY_MS

src = textwrap.dedent(inspect.getsource(G.Group2._pass))
MARK = "    self.pred_rep = pr\n"
if src.count(MARK) != 1:
    sys.exit("⚠️ _pass に差し込む印が見つからない")
src_i = src.replace(MARK, "    _PROBE(self, locals())\n" + MARK, 1)
ns = dict(G.__dict__)
box = {"f": None}
ns["_PROBE"] = lambda self, L: box["f"] and box["f"](self, L)
exec(src_i, ns)
G.Group2._pass = ns["_pass"]

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
off = [int(v) for v in g.off_ch]
area_of_ch = torch.cat([torch.full((a.n_ch,), i, dtype=torch.long) for i, a in enumerate(g.areas)]).to(g.dev)
st = {"tag": "他"}
cur = {"first": None, "r": None, "cnt": None}
rec = None


def new_rec():
    return {"r": [[] for _ in NAMES], "k": [[] for _ in NAMES], "two": [[] for _ in NAMES]}


def probe(self, L):
    if st["tag"] == "他":
        return
    rep, drive = L["rep"], L["drive"]
    th = self.fire_th * self.th_gain
    if L["first"]:
        n = rep.shape[0]
        cur["first"] = torch.zeros(n, dtype=torch.long, device=self.dev)     # 最初に撃った周（0 ＝ まだ）
        cur["r"] = torch.zeros(n, device=self.dev)
        cur["cnt"] = torch.zeros(n, device=self.dev)
        cur["pass"] = 0
    cur["pass"] += 1
    fired = rep > 0
    new = fired & (cur["first"] == 0) & (drive > th)              # 駆動で越えて、この歩で初めて撃った体
    cur["first"] = torch.where(new, torch.full_like(cur["first"], cur["pass"]), cur["first"])
    cur["r"] = torch.where(new, drive / th, cur["r"])
    cur["cnt"] += fired.float()
    if L["advance"]:
        m = cur["first"] > 0
        if bool(m.any()):
            ia = area_of_ch[m]; rr = cur["r"][m]; kk = cur["first"][m]; cc = cur["cnt"][m]
            for i in range(len(NAMES)):
                s = ia == i
                if bool(s.any()):
                    rec["r"][i].append(rr[s].cpu().numpy()); rec["k"][i].append(kk[s].cpu().numpy())
                    rec["two"][i].append((cc[s] >= 2).cpu().numpy())


box["f"] = probe


def show(head):
    print(f"── {head}  r ＝ 駆動 ÷ 閾値（撃った体）  人の f-I：f ＝ {SLOPE}Hz/pA × {RHEO:.0f}pA × (r − 1)", flush=True)
    for i, nm in enumerate(NAMES):
        if not rec["r"][i]:
            print(f"   {nm:5s}（駆動で撃った体なし）", flush=True)
            continue
        r = np.concatenate(rec["r"][i]); k = np.concatenate(rec["k"][i]); two = np.concatenate(rec["two"][i])
        f = SLOPE * RHEO * np.clip(r - 1.0, 0, None)
        remain = (10 - k) * PASS_MS                               # 最初の発火から歩の終わりまで（ms）
        two_h = (f > 0) & (1000.0 / np.maximum(f, 1e-9) <= remain)
        pstep = np.minimum(1.0, f * 10.67 / 1000.0)
        q = np.percentile(r, [10, 50, 90])
        print(f"   {nm:5s} 体×歩 {len(r):7d}  r {q[0]:5.2f}・{q[1]:5.2f}・{q[2]:5.2f}  r＞3.7 {np.mean(r > 3.7):5.1%}"
              f"  ｜ 2発：人の f-I なら {two_h.mean():5.1%} ／ 模型 {two.mean():5.1%}"
              f"  ｜ 人の f-I で1歩に撃つ見込み {pstep.mean():5.1%}", flush=True)


rec = new_rec()
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t_start = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if ear.sum() > 0:
        past.append((cmd, ear))
    if rp is not None:                           # 暮らし：返し手が返せば聞く（babble.py と同じ）
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        for _ in range(N_PROBE):                 # 試し：10回ごとに録音を3つ。学びは止めない
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
        show(f"{i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回")
        rec = new_rec()
print("読み：「人の f-I なら」が人の 17〜25%（錐体）に近ければ、当てはめは合う ⇒ f-I（撃ったあと 1/f 待つ）を入れる根拠になる", flush=True)
print("     「人の f-I で1歩に撃つ見込み」は、f-I を入れたときに今撃っている体がどれだけ撃つかの目安（r が歩をまたいで続く前提）", flush=True)
