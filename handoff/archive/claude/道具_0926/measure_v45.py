"""版45 の試し（arkhe）：指令の体が撃たない回があるとき、何が駆動を割っているか
  python3 measure_v45.py [喃語の回数=100]
版45 の measure_v43d 100（10-01 07時）：90回目までの10回だけ 0/8・指令の体 0.12。前後は 0.87〜0.97
覚えは残さない。measure_v43d と同じ乱数で回す（同じ発話の並びになる）
語の指令が入っている歩で、指令した体について（最初の周の値）
  駆動の上   下からの指令 × x_gain ＋ 横の興奮（上からの戻り・写しは除く）
  抑制の本数 届いた抑制の辺の数 ／ ihn（抑制の和。1＋ihn で割る）
  閾値       fire_th × th_gain（覚醒で動く）
  撃った     指令した体のうち撃った割合
10回ごとの平均と、指令の体が 0.5 未満だった発話は1回ずつ出す"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
lo = int(b.off_ch[NAMES.index("運動")]); nm = b.n["運動"]
rep_recv = torch.nonzero(g.is_rep).squeeze(1)                 # 表現の体 → 受け手の番号
m_recv = rep_recv[lo:lo + nm]
cur = {"cmd": None}
acc = {"up": 0.0, "cnt": 0.0, "ihn": 0.0, "th": 0.0, "hit": 0.0, "k": 0}
utt = {"up": 0.0, "cnt": 0.0, "ihn": 0.0, "th": 0.0, "hit": 0.0, "k": 0}
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges()
    out = orig(self, x_all, ext_all, learn, advance, first)
    c = cur["cmd"]
    if first and c is not None and len(e):
        on = torch.nonzero(c > 0).squeeze(1)
        rr = m_recv[on]
        r = self.e_recv[e].long(); se = self.e_src[e].long(); inh = self.e_inh[e]
        amp = torch.where(inh, 4.0, self.c_exc[r]) * self.w[e] * self._amp[e] * self.send_gain[se]
        keep = ~inh & ~self.mod_send[se]
        exc = torch.zeros(self.n_recv_all, device=self.dev); exc.index_add_(0, r[keep], amp[keep])
        ihn = torch.zeros(self.n_recv_all, device=self.dev); ihn.index_add_(0, r[inh], amp[inh])
        cnt = torch.zeros(self.n_recv_all, device=self.dev); cnt.index_add_(0, r[inh], torch.ones_like(amp[inh]))
        ch = lo + on
        up = exc[rr] * self.scale[rr] + x_all[ch] * self.x_gain[ch]
        ih = ihn[rr] * self.scale[rr]
        th = self.fire_th[ch] * self.th_gain
        for d in (acc, utt):
            d["up"] += float(up.mean()); d["cnt"] += float(cnt[rr].mean()); d["ih"] = 0
            d["ihn"] += float(ih.mean()); d["th"] += float(th.mean())
            d["hit"] += float(b_rep_next(on)); d["k"] += 1
    return out


def b_rep_next(on):
    return 0.0          # 撃ったかは step の後で足す（下）


G.Group2._pass = hooked
ostep = b.step


def step(ear, cmd, *a, **k):
    cur["cmd"] = cmd if float(cmd.sum()) > 0 else None
    r = ostep(ear, cmd, *a, **k)
    if cur["cmd"] is not None:
        f = float((b.motor_rep * cmd).sum() / cmd.sum())
        acc["hit"] += f; utt["hit"] += f
    cur["cmd"] = None
    return r


b.step = step


def line(d):
    k = max(d["k"], 1)
    return (f"駆動の上 {d['up']/k:6.1f}  抑制 {d['cnt']/k:6.1f}本 ihn {d['ihn']/k:5.2f}  "
            f"上÷(1＋ihn) {d['up']/k/(1+d['ihn']/k):6.1f}  閾値 {d['th']/k:5.2f}  撃った {d['hit']/k:4.2f}")


def clear(d):
    for x in d: d[x] = 0.0


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
ok = tried = 0; t0 = time.time()
print("回   言えた  （指令の歩の平均）")
for i in range(N):
    clear(utt)
    ear, cmd, n, want, got = say(rng, True)
    if want:
        tried += 1; ok += int(want == got)
        if utt["k"] and utt["hit"] / utt["k"] < 0.5:
            print(f"   ✗ {i+1:3d}回目 {want} → {got}  {line(utt)}  覚醒 {float(b.bs.arousal):.2f}", flush=True)
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        print(f"{i+1:4d}  {ok:2d}/{tried:2d}   {line(acc)}  （{(time.time()-t0)/60:.1f}分）", flush=True)
        ok = tried = 0; clear(acc)
print(f"VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB" if torch.cuda.is_available() else "")
