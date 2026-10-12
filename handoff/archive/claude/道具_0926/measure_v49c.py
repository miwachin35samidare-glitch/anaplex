"""版49：視床・CA1・上丘の錐体の側（表現の体）が、なぜ人より速く撃つかを分ける（arkhe の ~/anaplex で）
  python3 measure_v49c.py [喃語の回数=50]
measure_v49 と同じ回し方（0から・学びながら・返し手あり・保存しない）。比べるために聴覚・連合（新皮質）も並べる
表現の体が撃った歩を、撃たせたもので3つに分ける（表現の体の撃ち方 drive ＝ (横の興奮 ＋ 下からの入力 × 重さ) ÷ (1 ＋ 抑制) ＞ 閾値）
  下だけ     下からの入力 × 重さ ÷ (1 ＋ 抑制) だけで閾値を越えた
  下＋横     下からだけでは越えず、横の興奮（上からの戻り・写しを除く）を足して越えた
  越えずに   閾値を越えていないのに撃った（自発 ／ 海馬の補完）
並べるもの（10回ごと、歩の平均）
  率        表現の体が撃った割合（1歩）。人：視床の単一細胞 約5〜30%（MGB は値なし）／ CA1 錐体 3.1Hz ＝ 3.3%（Le Van Quyen 2008）／ 上丘 値なし
  内訳      撃った歩のうち 下だけ／下＋横／越えずに の割合
  下が来た  下からの入力が 0 でない体の割合 ／ 来たときの 入力 × 重さ の平均
  横・抑制  横の興奮の平均・抑制（ihn）の平均（表現の体）
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
AREAS = ("視床", "CA1", "上丘", "聴覚", "連合")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
rep_recv = torch.nonzero(g.is_rep).squeeze(1)          # 表現の受け手（並びは入口の並びと同じ）
sl = {nm: slice(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm]) for nm in AREAS}
K = ("n", "fire", "bottom", "both", "none", "xon", "xval", "lat", "ihn")
acc = {nm: dict.fromkeys(K, 0.0) for nm in AREAS}
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges()
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not advance:
        return out
    exc = torch.zeros(self.n_recv_all, device=self.dev)
    ihn = torch.zeros(self.n_recv_all, device=self.dev)
    if len(e):
        r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
        amp = torch.where(inh, 4.0, self.c_exc[r]) * self.w[e] * self._amp[e] * self.send_gain[se]
        keep = ~inh & ~self.mod_send[se]
        exc.index_add_(0, r[keep], amp[keep]); ihn.index_add_(0, r[inh], amp[inh])
    exc = exc * self.scale; ihn = ihn * self.scale
    lat = exc[rep_recv]; ih = ihn[rep_recv]
    bot = x_all * self.x_gain
    th = self.fire_th * self.th_gain
    fired = out[0] > 0
    up_b = bot / (1.0 + ih) > th
    up_all = (lat + bot) / (1.0 + ih) > th
    for nm in AREAS:
        s = sl[nm]; f = fired[s]; d = acc[nm]
        d["n"] += 1; d["fire"] += float(f.float().mean())
        d["bottom"] += float((f & up_b[s]).float().sum()); d["both"] += float((f & ~up_b[s] & up_all[s]).float().sum())
        d["none"] += float((f & ~up_all[s]).float().sum())
        xo = x_all[s] > 0
        d["xon"] += float(xo.float().mean())
        d["xval"] += float(bot[s][xo].mean()) if bool(xo.any()) else 0.0
        d["lat"] += float(lat[s].mean()); d["ihn"] += float(ih[s].mean())
    return out


G.Group2._pass = hooked


def line(nm):
    d = acc[nm]; k = max(d["n"], 1); tot = max(d["bottom"] + d["both"] + d["none"], 1)
    return (f"{nm} 率 {d['fire']/k:5.1%}  内訳 下だけ {d['bottom']/tot:4.0%} 下＋横 {d['both']/tot:4.0%} 越えずに {d['none']/tot:4.0%}"
            f"  下が来た {d['xon']/k:5.1%}（入力×重さ {d['xval']/k:6.1f}）  横 {d['lat']/k:6.1f}  ihn {d['ihn']/k:5.2f}"
            f"  閾値 {float(g.fire_th[sl[nm]].mean()):.1f}")


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    say(rng, True)
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        print(f"{i+1:4d}回  （{(time.time()-t0)/60:.1f}分）", flush=True)
        for nm in AREAS:
            print("    " + line(nm), flush=True)
        for nm in AREAS:
            acc[nm] = dict.fromkeys(K, 0.0)
