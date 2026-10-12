"""版49：CA3 → CA1（シャッファー）を「K本を拾って撃った数を足す」にして測る（arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v49f.py [喃語の回数=30] [CA1 の閾値=20] [K:重さ …（既定 8:1 27:1 27:4）]
【み】10-03「A」（54体から K本を拾い、撃った数を足す。新皮質の領野と同じ「本数のまま」）
  今の形：CA1 の1体が CA3 から 4本、どれか撃ったら 1（重さ 1）
  A の形：CA1 の1体が CA3 54体から K本（重複なし）を拾い、撃った数 × 重さ を下からの入力にする
    K     ⚠️ ノブ。人の CA1 が CA3 から何本受けるかは人で無い（09-23・10-03 探した）。54 に近いほど どの CA1 も同じ入力になる
    重さ  ⚠️ ノブ。CA3 → CA1 の1本の重さ。1 ＝ 人の繋がり1本。模型の CA3 は 54体しかなく本数が入り切らないので、動くために盛る値
          ⟲ 模型の1体は人のニューロン1個（「人の多数の細胞を代表する」ではない【み】09-28）
  閾値  ⚠️ ノブ。20 ＝ 静止から閾値まで 約20mV（【ヒト】Mertens 2024 補足図 S3、09-23）÷ 繋がり1本 1mV 前後（人に値なし）
網の中は書き換えない：Group2 に渡る入口の CA1 の分だけを、1歩前の CA3（成功率 0.90 を通した後）から作り直す
  あわせて CA1 の誤差の的を「下からの入力で撃つか」（0か1）にする（数を的にすると予測が追いつけない。他の領野と同じ形）
並べるもの（10回ごと、歩の平均）
  CA3   CA3 の表現の体が撃った割合【ヒト】CA3 9.6Hz ＝ 10%/歩（Watson 2025 ⚠️ 切片の値）
  表・抑  CA1 の表現・抑制の体が撃った割合【ヒト】3.3% ・ 5.4%（Le Van Quyen 2008）
  下     CA1 の1体に届いた 撃った数 × 重さ の平均 ／ その体ごとのばらつき（標準偏差。0 に近いほど どの体も同じ入力）
  下で越え  下からだけで閾値を越えた体の割合（1歩あたり）
  内訳   撃った歩のうち 下だけ ／ 下＋横 ／ 越えずに
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
CFG = [(int(a), float(c)) for a, c in (v.split(":") for v in sys.argv[3:])] or [(8, 1.0), (27, 1.0), (27, 4.0)]
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
orig = G.Group2._pass
ctx = {}
K_ = ("n", "ca3", "rep", "inh", "bot", "botsd", "upb", "bottom", "both", "none")


def hooked(self, x_all, ext_all, learn, advance, first):
    c = ctx
    b = c["b"]; s = c["s"]
    x_all = x_all.clone()
    x_all[s] = b.ca3_prev[c["sch"]].sum(1) * c["gain"]     # A の形：撃った数 × 重さ（x_gain は 1 にしてある）
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
    lat = exc[c["rep_recv"]][s]; ih = ihn[c["rep_recv"]][s]
    bot = (x_all * self.x_gain)[s]
    th = (self.fire_th * self.th_gain)[s]
    f = out[0][s] > 0
    up_b = bot / (1.0 + ih) > th
    up_all = (lat + bot) / (1.0 + ih) > th
    d = c["acc"]
    d["n"] += 1
    d["ca3"] += float((out[0][c["s3"]] > 0).float().mean())
    d["rep"] += float(f.float().mean())
    d["inh"] += float(self.bufs[c["ai"]][c["inh_rows"], 0].mean())
    d["bot"] += float(bot.mean()); d["botsd"] += float(bot.std())
    d["upb"] += float(up_b.float().mean())
    d["bottom"] += float((f & up_b).float().sum()); d["both"] += float((f & ~up_b & up_all).float().sum())
    d["none"] += float((f & ~up_all).float().sum())
    return out


G.Group2._pass = hooked

for k, gain in CFG:
    torch.manual_seed(0)
    b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    g = b.g
    ai = NAMES.index("CA1"); a = g.areas[ai]
    lo = int(b.off_ch[ai]); s = slice(lo, lo + b.n["CA1"])
    lo3 = int(b.off_ch[NAMES.index("CA3")]); s3 = slice(lo3, lo3 + b.n["CA3"])
    n3 = b.n["CA3"]
    if k > n3:
        sys.exit(f"⚠️ K {k} が CA3 の体数 {n3} を超える")
    g.fire_th[s] = TH; b.areas["CA1"].fire_th = TH
    g.x_gain[s] = 1.0
    # 誤差の的は「下からの入力で撃つか」（0か1）にする（他の領野と同じ）。数のままだと的が 1 を超え、
    #   予測（1 を超えない）が追いつけず重みが伸び続ける（torch_brain._fire の注記。版12〜16 の連合・高次）
    g.tgt_by_fire[s] = True
    gen = torch.Generator(device=B.DEV); gen.manual_seed(7)
    sch = torch.stack([torch.randperm(n3, device=B.DEV, generator=gen)[:k] for _ in range(b.n["CA1"])])
    ctx.update(b=b, s=s, s3=s3, ai=ai, sch=sch, gain=gain, rep_recv=torch.nonzero(g.is_rep).squeeze(1),
               inh_rows=slice(a.n_ch + a.n_ext, a.n_src), acc=dict.fromkeys(K_, 0.0))
    _, say, rp = B.voices(b, False)
    rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
    print(f"── K {k} 本 × 重さ {gain:g}（CA1 の閾値 {TH:g}、CA3 は {n3} 体）", flush=True)
    t0 = time.time()
    for i in range(N):
        say(rng, True)
        B.reply(b, rp, rng_r)
        if (i + 1) % 10 == 0:
            d = ctx["acc"]; n = max(d["n"], 1); tot = max(d["bottom"] + d["both"] + d["none"], 1)
            print(f"  {i+1:4d}回（{(time.time()-t0)/60:.1f}分）  CA3 {d['ca3']/n:5.1%}  表 {d['rep']/n:5.1%}  抑 {d['inh']/n:5.1%}"
                  f"  下 {d['bot']/n:5.2f}（ばらつき {d['botsd']/n:4.2f}）  下で越え {d['upb']/n:5.1%}"
                  f"  内訳 下だけ {d['bottom']/tot:4.0%} 下＋横 {d['both']/tot:4.0%} 越えずに {d['none']/tot:4.0%}", flush=True)
            ctx["acc"] = dict.fromkeys(K_, 0.0)
    del b, g
    if B.DEV == "cuda":
        torch.cuda.empty_cache()
print("人：CA3 10%（9.6Hz）・CA1 表 3.3%・抑 5.4%", flush=True)
