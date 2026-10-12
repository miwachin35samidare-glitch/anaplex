"""版49：嗅内浅 II の誤差が、どの歩にまとまって撃つかを測る（arkhe の ~/anaplex で。保存しない・網は書き換えない）
  python3 measure_v49i.py [喃語の回数=10]
【測】measure_v49h（10-04）：嗅内浅 II の誤差は 9割の歩で全員黙り、残りの歩でまとまって撃つ（最大 20体 全員）
嗅内浅の下からの入力（torch_brain.step）：高次の誤差 → 1体 8本 ＋ 前頭前野の誤差 → 1体 8本、重さ 7.14、閾値 14.3（2本で越える）
  誤差の的 ＝ 下からの入力 × 重さ ＞ 閾値。誤差の体はその的と予測の差で撃つ
並べるもの
  1 嗅内浅 II の誤差が撃った歩の割合を、場面ごとに
      話す（say）の 準備（蹴りの入っている歩）／ 声 ／ 聞く（返し手 reply）
      発話の頭からの歩数（0〜9 ／ 10〜49 ／ 50〜102 ／ 103〜）
      耳に届いた音の強さ（4つに等分）
  2 嗅内浅 II が撃った歩と黙った歩で、1歩前の 高次の誤差・前頭前野の誤差 の数、嗅内浅 II の的（下から越えた体の数）を比べる
  3 撃つ歩が続く長さ（何歩まとまって撃つか）
"""
import sys, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

if B.VERSION != 49:
    sys.exit(f"⚠️ babble.py が版49 ではない（版 {B.VERSION}）")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
a_ec = b.areas["嗅内浅"]
TH_EC, XG_EC = float(a_ec.fire_th), float(a_ec.x_gain)
st = {"phase": "", "t": 0}
rec = []      # (場面, 発話の頭からの歩, 音の強さ, 蹴り, 高次誤, 前頭前野誤, 嗅内II的, 嗅内II誤)
orig_step, orig_new = Brain.step, Brain.new_utterance


def new_utt(self, speaking=True):
    st["t"] = 0
    return orig_new(self, speaking)


def step(self, ear, cmd, eye=None, learn=True):
    hi = int((self.up["高次"] > 0).sum()); pf = int((self.up["前頭前野"] > 0).sum())
    x_ec = self._proj(self.up["高次"], self.p_h2ecs) + self._proj(self.up["前頭前野"], self.p_pfc2ecs)
    tgt = int((x_ec[:TB.EC_II] * XG_EC > TH_EC).sum())
    kick = self.kick_left > 0
    out = orig_step(self, ear, cmd, eye, learn)
    ec = int((self.up["嗅内浅"][:TB.EC_II] > 0).sum())
    rec.append((st["phase"], st["t"], float(torch.as_tensor(ear).float().sum()), kick, hi, pf, tgt, ec))
    st["t"] += 1
    return out


Brain.step = step
Brain.new_utterance = new_utt
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
for i in range(N):
    st["phase"] = "話す"; st["t"] = 0; say(rng, True)
    st["phase"] = "聞く"; st["t"] = 0; B.reply(b, rp, rng_r)   # 聞く側は new_utterance を呼ぶか確かめていないので、ここで 0 に戻す

ph = np.array([r[0] for r in rec]); t = np.array([r[1] for r in rec]); en = np.array([r[2] for r in rec])
kick = np.array([r[3] for r in rec]); hi = np.array([r[4] for r in rec]); pf = np.array([r[5] for r in rec])
tg = np.array([r[6] for r in rec]); ec = np.array([r[7] for r in rec]); on = ec > 0
n_hi, n_pf = b.n["高次"], b.n["前頭前野"]


def rate(m, name):
    m = np.asarray(m, bool)
    return f"{name} {on[m].mean():5.1%}（{m.sum()}歩）" if m.any() else f"{name} ―（0歩）"


print(f"歩 {len(rec)}  嗅内浅 II の誤差が撃った歩 {on.mean():.1%}")
print("1 場面ごと（撃った歩の割合）")
print("   " + "  ".join([rate((ph == '話す') & kick, "話す・準備"), rate((ph == '話す') & ~kick, "話す・声"), rate(ph == '聞く', "聞く")]))
print("   " + "  ".join([rate(t < 10, "頭 0〜9"), rate((t >= 10) & (t < 50), "10〜49"), rate((t >= 50) & (t < 103), "50〜102"), rate(t >= 103, "103〜")]))
q = np.quantile(en, [0.25, 0.5, 0.75])
print("   音の強さ " + "  ".join([rate(en <= q[0], "弱い1/4"), rate((en > q[0]) & (en <= q[1]), "2/4"),
                                rate((en > q[1]) & (en <= q[2]), "3/4"), rate(en > q[2], "強い1/4")]))
print("2 撃った歩 と 黙った歩（1歩前の数の平均）")
for name, m in (("撃った", on), ("黙った", ~on)):
    if m.any():
        print(f"   {name}  高次の誤差 {hi[m].mean():6.1f}体（{hi[m].mean()/n_hi:5.1%}）  前頭前野の誤差 {pf[m].mean():6.1f}体（{pf[m].mean()/n_pf:5.1%}）"
              f"  嗅内浅 II の的 {tg[m].mean():5.2f}体  撃った数 {ec[m].mean():5.2f}体")
runs, c = [], 0
for v in on:
    if v:
        c += 1
    elif c:
        runs.append(c); c = 0
if c:
    runs.append(c)
r = np.array(runs) if runs else np.array([0])
print(f"3 続けて撃つ長さ  回数 {len(runs)}  1歩 {np.mean(r==1):5.1%}  2〜3歩 {np.mean((r>=2)&(r<=3)):5.1%}  4〜9歩 {np.mean((r>=4)&(r<=9)):5.1%}  10歩以上 {np.mean(r>=10):5.1%}  最長 {r.max()}")
