"""版67：表現の体を撃たせているものを分ける（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v67a.py [喃語の回数=20]      （コンテナ：python3 measure_v67a.py 2 --test）
【測】measure_check2 50（版67、10-12）：運動の表現 31〜33%（版66 8〜11%）⇒ 口が型の2倍超で黙り、言えた 0/43・耳は全部の歩で無音
並べるもの（運動・聴覚・連合・高次・前頭前野。1周目の値、表現の体1体・1歩あたり）
  駆動の中身：下から（入口 × x_gain。運動は運動前野の送り手の数）／ CL（版67）／ 横の興奮（調整する側を除く）／ 抑制 ihn ／ 閾値 × th_gain
  撃った体の訳：CL が無ければ撃たなかった ／ 横が無ければ撃たなかった ／ 下からだけで撃てた
  運動は 指令あり（語の指令の窓）と 指令なし に分ける
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (67,):
    sys.exit(f"⚠️ babble.py が版67 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 20
if TEST:
    B.DEV = "cpu"
AREAS = ("運動", "聴覚", "連合", "高次", "前頭前野")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=200 if TEST else 9100, hippo_scale=4.0 if TEST else 20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
sl = {nm: slice(int(g.off_ch[NAMES.index(nm)]), int(g.off_ch[NAMES.index(nm) + 1])) for nm in AREAS}
cur = {"cmd": False}
KEYS = ("歩体", "撃った", "下", "CL", "横", "ihn", "閾値", "CLが要った", "横が要った", "下だけ")
acc = {}
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if first else None      # 1周目の辺（orig が終わると _e_cache は消える）
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not first:
        return out
    both, mod = self._sum(e, False)
    exc = both[:self.n_recv_all] * (self.scale * self.scale_up); ihn = both[self.n_recv_all:] * self.scale
    lat = (exc - mod * (self.scale * self.scale_up))[self.rep_i]
    cl = getattr(self, "cl_drive", None)
    cl = cl if cl is not None else torch.zeros_like(x_all)
    rep = self._any[0]
    for nm, s in sl.items():
        key = nm + ("・指令あり" if (nm == "運動" and cur["cmd"]) else ("・指令なし" if nm == "運動" else ""))
        a = acc.setdefault(key, dict.fromkeys(KEYS, 0.0))
        down = x_all[s] * self.x_gain[s]; den = 1.0 + ihn[self.rep_i][s]; th = self.fire_th[s] * self.th_gain
        f = rep[s] > 0
        a["歩体"] += down.numel(); a["撃った"] += float(f.sum())
        a["下"] += float(down.sum()); a["CL"] += float(cl[s].sum()); a["横"] += float(lat[s].sum())
        a["ihn"] += float(ihn[self.rep_i][s].sum()); a["閾値"] += float(th.sum())
        a["CLが要った"] += float((f & ((down + lat[s]) / den <= th)).sum())
        a["横が要った"] += float((f & ((down + cl[s]) / den <= th)).sum())
        a["下だけ"] += float((f & (down / den > th)).sum())
    return out


G.Group2._pass = hooked
ostep = b.step


def step(ear, cmd, *a, **k):
    cur["cmd"] = float(torch.as_tensor(cmd).sum()) > 0
    return ostep(ear, cmd, *a, **k)


b.step = step
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
ok = tried = 0
for i in range(N):
    ear, cmd, n, want, got = say(rng, True)
    if want:
        tried += 1; ok += int(want == got)
    if rp is not None:
        B.reply(b, rp, rng_r)
print(f"言えた {ok}/{tried}   CL → 皮質 の重さ 平均 {float(b.cl_w.mean()):.3f}")
print("   領野・場面        率      下から   CL      横      ihn    閾値   ｜ 撃った体の訳 CLが要った・横が要った・下だけで撃てた")
for key, a in acc.items():
    k = max(a["歩体"], 1); f = max(a["撃った"], 1)
    print(f"   {key:12s} {a['撃った']/k:6.2%}  {a['下']/k:6.2f}  {a['CL']/k:6.2f}  {a['横']/k:6.2f}  {a['ihn']/k:5.2f}  {a['閾値']/k:5.2f}"
          f"  ｜ {a['CLが要った']/f:5.0%}・{a['横が要った']/f:5.0%}・{a['下だけ']/f:5.0%}")
print("読み：CL・横の要否は「それを除いても越えたか」（抑制はそのまま）。両方要る体もあるので和は 100% にならない")
