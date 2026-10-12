"""版49：CA1 の表現の体の閾値（ノブ）を振って、CA1 の率を見る（arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v49e.py [喃語の回数=30] [閾値 …（既定 1 16 19 22）]
【み】10-03「16〜22の間で振ってみればいいんじゃないの」
  閾値 ＝ 人の繋がり何本ぶんで撃つか。⚠️ ノブ【C】
  分子：静止から閾値まで 約18mV（【ヒト】CA1 錐体 静止 −63.4mV Mertens 2024 ／ 閾値 約 −45mV Butola 2025 図1e から読み）
  分母：人の CA3 → CA1 の1本は見つからない ⇒ 1.12mV（皮質 Hunt 2023）16本 〜 0.80mV（CA3 Watson 2025）22本
  1 は今の値（比べるため）
閾値ごとに 0から組み直して、measure_v49 と同じ回し方（学びながら・返し手あり）
並べるもの（10回ごと、歩の平均）
  表   CA1 の表現の体が撃った割合     【ヒト】CA1 錐体 3.1Hz ＝ 3.3%/歩（Le Van Quyen 2008）
  抑   CA1 の抑制の体が撃った割合     【ヒト】海馬の抑制細胞 5.1Hz ＝ 5.4%/歩（同）
  ihn  CA1 の表現の体の抑制の平均
  内訳 撃った歩のうち 下だけ ／ 下＋横 ／ 越えずに（自発・補完）
  横   表現の体の横の興奮の平均
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
THS = [float(v) for v in sys.argv[2:]] or [1.0, 16.0, 19.0, 22.0]
NM = "CA1"
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
orig = G.Group2._pass
ctx = {}
K = ("n", "rep", "inh", "ihn", "lat", "bottom", "both", "none")


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges()
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not advance:
        return out
    c = ctx
    exc = torch.zeros(self.n_recv_all, device=self.dev)
    ihn = torch.zeros(self.n_recv_all, device=self.dev)
    if len(e):
        r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
        amp = torch.where(inh, 4.0, self.c_exc[r]) * self.w[e] * self._amp[e] * self.send_gain[se]
        keep = ~inh & ~self.mod_send[se]
        exc.index_add_(0, r[keep], amp[keep]); ihn.index_add_(0, r[inh], amp[inh])
    exc = exc * self.scale; ihn = ihn * self.scale
    s = c["sl"]
    lat = exc[c["rep_recv"]][s]; ih = ihn[c["rep_recv"]][s]
    bot = (x_all * self.x_gain)[s]
    th = (self.fire_th * self.th_gain)[s]
    f = out[0][s] > 0
    up_b = bot / (1.0 + ih) > th
    up_all = (lat + bot) / (1.0 + ih) > th
    inh_spk = self.bufs[c["ai"]][c["inh_rows"], 0]          # この歩の抑制の体の発火（進めたあとの遅れの頭）
    d = c["acc"]
    d["n"] += 1; d["rep"] += float(f.float().mean()); d["inh"] += float(inh_spk.mean())
    d["ihn"] += float(ih.mean()); d["lat"] += float(lat.mean())
    d["bottom"] += float((f & up_b).float().sum()); d["both"] += float((f & ~up_b & up_all).float().sum())
    d["none"] += float((f & ~up_all).float().sum())
    return out


G.Group2._pass = hooked

for th in THS:
    torch.manual_seed(0)
    b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    g = b.g
    ai = NAMES.index(NM); a = g.areas[ai]
    lo = int(b.off_ch[ai]); s = slice(lo, lo + b.n[NM])
    g.fire_th[s] = th                       # 表現の体の閾値（Group2 が使う値）
    b.areas[NM].fire_th = th
    ctx.update(sl=s, ai=ai, rep_recv=torch.nonzero(g.is_rep).squeeze(1),
               inh_rows=slice(a.n_ch + a.n_ext, a.n_src), acc=dict.fromkeys(K, 0.0))
    _, say, rp = B.voices(b, False)
    rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
    print(f"── CA1 の閾値 {th:g}（人の繋がり何本ぶん）", flush=True)
    t0 = time.time()
    for i in range(N):
        say(rng, True)
        B.reply(b, rp, rng_r)
        if (i + 1) % 10 == 0:
            d = ctx["acc"]; k = max(d["n"], 1); tot = max(d["bottom"] + d["both"] + d["none"], 1)
            print(f"  {i+1:4d}回（{(time.time()-t0)/60:.1f}分）  表 {d['rep']/k:5.1%}  抑 {d['inh']/k:5.1%}  ihn {d['ihn']/k:5.2f}"
                  f"  内訳 下だけ {d['bottom']/tot:4.0%} 下＋横 {d['both']/tot:4.0%} 越えずに {d['none']/tot:4.0%}"
                  f"  横 {d['lat']/k:6.1f}", flush=True)
            ctx["acc"] = dict.fromkeys(K, 0.0)
    del b, g
    if B.DEV == "cuda":
        torch.cuda.empty_cache()
print("人：表 3.3%（3.1Hz）・抑 5.4%（5.1Hz）Le Van Quyen 2008", flush=True)
