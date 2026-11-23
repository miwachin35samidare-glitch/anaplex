"""版67：運動の表現の「指令の無い体」を何が撃たせているか（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v67c.py [喃語の回数=20] [K_PM=20]      （コンテナ：python3 measure_v67c.py 2 20 --test）
【み】10-12「イ」：版はまだ作らない。指令の無い体の内訳を先に測る（K_PM と一緒に直す）
【測】measure_v67b：K_PM 0 でも 指令の無い体が 窓で 10.5%・指令の無い歩で 22%（的 3.5%）
歩を3つに分ける（どれも指令の入っていない体だけを数える）
  準備・蹴り  ：言う前で指令がまだ入らず、運動視床の蹴りが入っている歩
  窓          ：語の指令が入っている歩（指令の入った体は除く。参考に指令の体も別の行に）
  黙り・聞く  ：指令も蹴りも無い歩（返し手を聞く・黙る・声のあと）
並べるもの（1周目、表現の体1体・1歩あたり）
  率 ／ 駆動の中身：運動前野（送り手の数）・蹴り・CL・横の興奮・抑制 ihn・閾値
  撃った体の訳（それを除くと越えなかった割合）：CL・横・蹴り・運動前野 ／ CL だけで越えた割合
K_PM は torch_brain の値を上書きして測る（既定 20 ＝ measure_v67b で指令の体が 35Hz になった値）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (67, 68):
    sys.exit(f"⚠️ babble.py が版67・68 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 20
TB.K_PM = int(args[1]) if len(args) > 1 else (20 if B.VERSION == 67 else TB.K_PM)   # 版68 は torch_brain の値（27）
if TEST:
    B.DEV = "cpu"
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=200 if TEST else 9100, hippo_scale=4.0 if TEST else 20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
s = slice(int(g.off_ch[NAMES.index("運動")]), int(g.off_ch[NAMES.index("運動") + 1]))
cur = {"kind": None, "on": None, "pm": None}
KEYS = ("歩体", "撃った", "前野", "蹴り", "CL", "横", "ihn", "閾値",
        "CLが要った", "横が要った", "蹴りが要った", "前野が要った", "CLだけで越えた")
acc = {}

opm = b._premotor


def premotor(cmd):
    v = opm(cmd)
    cur["pm"] = v
    return v


b._premotor = premotor
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if first else None      # 1周目の辺（orig が終わると _e_cache は消える）
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not first or cur["kind"] is None:
        return out
    both, mod = self._sum(e, False)
    exc = both[:self.n_recv_all] * (self.scale * self.scale_up); ihn = both[self.n_recv_all:] * self.scale
    lat = (exc - mod * (self.scale * self.scale_up))[self.rep_i][s]
    cl = getattr(self, "cl_drive", None)
    cl = cl[s] if cl is not None else torch.zeros_like(x_all[s])
    xg = self.x_gain[s]
    down = x_all[s] * xg
    pm = cur["pm"] * xg
    kick = down - pm
    den = 1.0 + ihn[self.rep_i][s]; th = self.fire_th[s] * self.th_gain
    f = self._any[0][s] > 0
    tot = down + cl + lat
    groups = [(cur["kind"], ~cur["on"])]
    if cur["kind"] == "窓":
        groups.append(("窓・指令の体", cur["on"]))
    for key, m in groups:
        a = acc.setdefault(key, dict.fromkeys(KEYS, 0.0))
        fm = f & m
        a["歩体"] += float(m.sum()); a["撃った"] += float(fm.sum())
        a["前野"] += float(pm[m].sum()); a["蹴り"] += float(kick[m].sum()); a["CL"] += float(cl[m].sum())
        a["横"] += float(lat[m].sum()); a["ihn"] += float((den - 1.0)[m].sum()); a["閾値"] += float(th[m].sum())
        a["CLが要った"] += float((fm & ((tot - cl) / den <= th)).sum())
        a["横が要った"] += float((fm & ((tot - lat) / den <= th)).sum())
        a["蹴りが要った"] += float((fm & ((tot - kick) / den <= th)).sum())
        a["前野が要った"] += float((fm & ((tot - pm) / den <= th)).sum())
        a["CLだけで越えた"] += float((fm & (cl / den > th)).sum())
    return out


G.Group2._pass = hooked
ostep = b.step


def step(ear, cmd, *a, **k):
    c = torch.as_tensor(cmd).to(b.motor_rep.device).reshape(-1)
    on = c > 0
    if bool(on.any()):
        cur["kind"] = "窓"
    elif b.kick_left > 0:
        cur["kind"] = "準備・蹴り"
    else:
        cur["kind"] = "黙り・聞く"
    cur["on"] = on
    out = ostep(ear, cmd, *a, **k)
    cur["kind"] = None
    return out


b.step = step
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
ok = tried = 0
for i in range(N):
    ear, cmd, n, want, got = say(rng, True)
    if want:
        tried += 1; ok += int(want == got)
    if rp is not None:
        B.reply(b, rp, rng_r)
print(f"K_PM {TB.K_PM}   言えた {ok}/{tried}   的：指令の無い体 3.5%/歩（3.3Hz）・指令の体 37.3%/歩（35Hz）")
print("   歩の種類        歩体数     率     前野   蹴り    CL     横     ihn   閾値  ｜ 撃った体の訳 CL・横・蹴り・前野が要った ｜ CLだけで越えた")
for key in ("準備・蹴り", "窓", "窓・指令の体", "黙り・聞く"):
    a = acc.get(key)
    if a is None:
        continue
    k = max(a["歩体"], 1); f = max(a["撃った"], 1)
    print(f"   {key:12s} {int(a['歩体']):9d}  {a['撃った']/k:6.2%}  {a['前野']/k:5.2f}  {a['蹴り']/k:5.2f}  {a['CL']/k:5.2f}  {a['横']/k:5.2f}"
          f"  {a['ihn']/k:5.2f}  {a['閾値']/k:5.2f}  ｜ {a['CLが要った']/f:4.0%}・{a['横が要った']/f:4.0%}・{a['蹴りが要った']/f:4.0%}・{a['前野が要った']/f:4.0%}"
          f"  ｜ {a['CLだけで越えた']/f:4.0%}")
print("読み：「要った」はそれを除くと越えなかった割合（抑制はそのまま）。両方要る体もあるので和は 100% にならない")
