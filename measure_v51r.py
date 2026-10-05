"""版51：皮質の表現が下から撃てない理由を分ける（arkhe の ~/anaplex で。0から・学びながら・返し手あり・保存しない・網は書き換えない）
  python3 measure_v51r.py [喃語の回数=30]
【測】measure_v51h（10-05）：聴覚・連合・高次・前頭前野の表現が 0.4〜0.5%/歩（自発 0.32%/歩 とほぼ同じ。人 3.5%）
表現の撃ち方（torch_group2）：撃つ ＝ (横の興奮（調整する側を除く）＋ 下 × x_gain) ÷ (1 ＋ 抑制) ＞ 閾値 × th_gain
  的（下から越え）＝ 下 × x_gain ＞ 閾値（抑制なし）
並べるもの（領野 聴覚・連合・高次・前頭前野、10回ごと。1周目の値）
  的が立った（下から越えた）体について
    撃てた割合 ／ 下×x_gain ／ 横の興奮 ／ 抑制 ihn ／ 届いた抑制の本数 ／ 閾値×th_gain
    抑制が無ければ撃てた割合（(下×x_gain ＋ 横) ＞ 閾値×th_gain）
  的が立っていないのに撃った体の割合（横・自発・補完）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
AREAS = ("聴覚", "連合", "高次", "前頭前野")
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
rep_recv = torch.nonzero(g.is_rep).squeeze(1)          # 表現の受け手の番号（入口の並び＝ch の順）
sl = {nm: slice(int(g.off_ch[NAMES.index(nm)]), int(g.off_ch[NAMES.index(nm) + 1])) for nm in AREAS}
keys = ("的", "撃てた", "下", "横", "ihn", "本数", "閾値", "抑制なしなら", "的なし歩体", "的なしで撃った")
acc = {nm: dict.fromkeys(keys, 0.0) for nm in AREAS}
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if first else None
    out = orig(self, x_all, ext_all, learn, advance, first)
    if not first:
        return out
    exc = torch.zeros(self.n_recv_all, device=self.dev); ihn = torch.zeros_like(exc); cnt = torch.zeros_like(exc)
    if len(e):
        r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
        amp = torch.where(inh, 4.0, self.c_exc[r]) * self.w[e] * self._amp[e] * self.send_gain[se]
        keep = ~inh & ~self.mod_send[se]
        exc.index_add_(0, r[keep], amp[keep]); ihn.index_add_(0, r[inh], amp[inh])
        cnt.index_add_(0, r[inh], torch.ones_like(amp[inh]))
    exc = exc * self.scale; ihn = ihn * self.scale
    rep = out[0]
    for nm, s in sl.items():
        rr = rep_recv[s]
        down = x_all[s] * self.x_gain[s]; th = self.fire_th[s] * self.th_gain
        tgt = down > self.fire_th[s]
        a = acc[nm]; k = int(tgt.sum())
        a["的"] += k; a["的なし歩体"] += int((~tgt).sum()); a["的なしで撃った"] += float(rep[s][~tgt].sum())
        if k:
            a["撃てた"] += float(rep[s][tgt].sum()); a["下"] += float(down[tgt].sum()); a["横"] += float(exc[rr][tgt].sum())
            a["ihn"] += float(ihn[rr][tgt].sum()); a["本数"] += float(cnt[rr][tgt].sum()); a["閾値"] += float(th[tgt].sum())
            a["抑制なしなら"] += float(((down + exc[rr]) > th)[tgt].sum())
    return out


G.Group2._pass = hooked
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
for i in range(N):
    say(rng, True)
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        print(f"── {i+1}回目まで（この10回）  的が立った体について（1体あたりの平均）")
        print("   領野     的の数   撃てた  抑制なしなら  下×x_gain   横    抑制ihn  届いた抑制  閾値×th   的なしで撃った")
        for nm in AREAS:
            a = acc[nm]; k = max(a["的"], 1)
            print(f"   {nm:5s} {int(a['的']):7d}  {a['撃てた']/k:6.1%}   {a['抑制なしなら']/k:6.1%}     {a['下']/k:6.2f}  {a['横']/k:6.2f}  {a['ihn']/k:6.3f}"
                  f"  {a['本数']/k:7.1f}本  {a['閾値']/k:6.2f}   {a['的なしで撃った']/max(a['的なし歩体'],1):6.2%}", flush=True)
        for a in acc.values():
            for x in a: a[x] = 0.0
