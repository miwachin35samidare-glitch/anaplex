"""版49：海馬の流れを段ごとに測る（CA3 が 1%/歩 しか撃たない原因を分ける。arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v49g.py [喃語の回数=30]
【測】measure_v49f（10-04）：CA3 1.0〜1.2%/歩 ⇒ CA1 を下から撃たせられない
流れ（torch_brain の作り）
  高次・前頭前野 → 嗅内浅（細胞体へ、1体 8本 × 7.14・閾値 14.3）
  嗅内浅 層 II（頭 20体）の誤差 → 貫通路（歯状回1体が 2本）→ 歯状回は足した値の上位 2.5%（8体）だけ。値が 0 なら撃たない
  歯状回 → 苔状線維（CA3 1体が 4本、どれか撃ったら 1）× 重さ 57.3 ＞ 閾値 28.6（÷(1＋抑制)）
  CA3 → シャッファー（CA1 1体が 4本、どれか撃ったら 1）× 1 ＞ 閾値 1.0
並べるもの（10回ごと、歩の平均）
  嗅内II誤  嗅内浅 層 II（頭 20体）の誤差の体が撃った割合 ＝ 歯状回へ行くもの
  各領野   表（表現の体が撃った割合）／ 下が来た（下からの入力が 0 でない体の割合）／
          来たとき（来た体の 入力×重さ の平均）／ 下で越え（来た体のうち 下だけで閾値を越えた割合）／ ihn ／ 横 ／ 閾値
          内訳（撃った歩のうち 下だけ／下＋横／越えずに）
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
import torch_brain as TB
from torch_brain import Brain, NAMES

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
AREAS = ("高次", "嗅内浅", "歯状回", "CA3", "CA1")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
rep_recv = torch.nonzero(g.is_rep).squeeze(1)
sl = {nm: slice(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm]) for nm in AREAS}
lo_ec = int(b.off_ch[NAMES.index("嗅内浅")])
K = ("n", "rep", "xon", "xsum", "xcnt", "upb", "ihn", "lat", "bottom", "both", "none")
acc = {nm: dict.fromkeys(K, 0.0) for nm in AREAS}
ec2 = {"n": 0.0, "err": 0.0}
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
        s = sl[nm]; f = fired[s]; d = acc[nm]; xo = x_all[s] > 0
        d["n"] += 1; d["rep"] += float(f.float().mean()); d["xon"] += float(xo.float().mean())
        d["xsum"] += float(bot[s][xo].sum()); d["xcnt"] += float(xo.sum())
        d["upb"] += float((up_b[s] & xo).float().sum())
        d["ihn"] += float(ih[s].mean()); d["lat"] += float(lat[s].mean())
        d["bottom"] += float((f & up_b[s]).float().sum()); d["both"] += float((f & ~up_b[s] & up_all[s]).float().sum())
        d["none"] += float((f & ~up_all[s]).float().sum())
    ec2["n"] += 1; ec2["err"] += float((out[1][lo_ec:lo_ec + TB.EC_II] > 0).float().mean())
    return out


G.Group2._pass = hooked


def line(nm):
    d = acc[nm]; k = max(d["n"], 1); tot = max(d["bottom"] + d["both"] + d["none"], 1); xc = max(d["xcnt"], 1)
    return (f"{nm:5s} 表 {d['rep']/k:5.1%}  下が来た {d['xon']/k:5.1%}（来たとき {d['xsum']/xc:6.1f}・下で越え {d['upb']/xc:4.0%}）"
            f"  ihn {d['ihn']/k:5.2f}  横 {d['lat']/k:6.1f}  閾値 {float(g.fire_th[sl[nm]].mean()):.1f}"
            f"  内訳 {d['bottom']/tot:3.0%}／{d['both']/tot:3.0%}／{d['none']/tot:3.0%}")


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
t0 = time.time()
for i in range(N):
    say(rng, True)
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        print(f"{i+1:4d}回（{(time.time()-t0)/60:.1f}分）  嗅内II誤 {ec2['err']/max(ec2['n'],1):5.1%}", flush=True)
        for nm in AREAS:
            print("    " + line(nm), flush=True)
        for nm in AREAS:
            acc[nm] = dict.fromkeys(K, 0.0)
        ec2.update(n=0.0, err=0.0)
print("内訳 ＝ 下だけ／下＋横／越えずに。歯状回は上位 2.5%（8体）で撃つので 表 ≦ 2.5%", flush=True)
