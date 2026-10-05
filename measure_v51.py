"""版51 の試し（arkhe）：measure_v49 と同じ（版の確認だけ 50）。measure_v48b と同じ回し方（0から・学びながら・返し手あり・保存しない）
  python3 measure_v51.py [喃語の回数=100]
選んだとおりに言えなかった回を全部出す（measure_v48 は「指令の体が半分も撃たなかった回」だけだった）
  口が読む窓（指令が入ってから CMD_LEAD 歩）の終わりの筋の力（版49 の口と同じ）で、
  指令した体のうち力が READ_TH 未満の数（落ち）と、指令していない体のうち READ_TH 以上の数（混ざり）
  10回ごと：指令の体が指令の歩に撃った割合（人の話すあいだ 35Hz ＝ 37%/歩）
覚えは残さない（0から組んで回す）。measure_v45 と同じ乱数（同じ発話の並び）
版45 の measure_v45 100（10-01 夜）：言えた 73/87 前後、言えない回は抑制の体が蹴りで 10〜15%（形A）か
  自分の履歴で 54〜73%（形B）撃っていた
出すもの（10回ごと）
  言えた ／ 指令の歩の 駆動の上・抑制の本数・ihn・閾値・撃った（measure_v45 と同じ）
  運動の抑制の体  指令の歩に撃った割合 ／ 全部の歩 ／ 続けて撃った最長の歩数
  抑制の体の発火率（恒常性の移動平均）  新皮質の人は覚醒 4.70Hz ＝ 5.0%/歩（Verma 2024）／ 海馬・嗅内の人は 5.1Hz ＝ 5.4%/歩（Le Van Quyen 2008）
                                       視床・背内側視床・側坐核・上丘は人の値なし（新皮質の代用で組んだ）
  錐体 → 抑制の体 の大きい繋がり（4.21mV 超）の割合（人の分布なら約 6%。連発で弱まると減る）"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 51:
    sys.exit(f"⚠️ babble.py が版51 ではない（版 {B.VERSION}）。apply_v51.py を先に")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, False)
g = b.g
im = NAMES.index("運動"); am = b.areas["運動"]
lo = int(b.off_ch[im]); nm = b.n["運動"]
inh_lo = am.n_ch + am.n_ext                                   # 運動の送り元の並び：表現・ext・抑制
rep_recv = torch.nonzero(g.is_rep).squeeze(1)
m_recv = rep_recv[lo:lo + nm]
cur = {"cmd": None}
acc = {"up": 0.0, "cnt": 0.0, "ihn": 0.0, "th": 0.0, "hit": 0.0, "k": 0}
utt = {"up": 0.0, "cnt": 0.0, "ihn": 0.0, "th": 0.0, "hit": 0.0, "k": 0}
inh_acc = {"cmd": 0.0, "kc": 0, "all": 0.0, "ka": 0}
import config as C
import mouth as MO
win = {"list": [], "n": 0, "cmd": None}
run = torch.zeros(am.n_inh, device=B.DEV)
run_max = {"v": 0}
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
            d["up"] += float(up.mean()); d["cnt"] += float(cnt[rr].mean())
            d["ihn"] += float(ih.mean()); d["th"] += float(th.mean()); d["k"] += 1
    return out


G.Group2._pass = hooked
ostep = b.step


def step(ear, cmd, *a, **k):
    global run
    cur["cmd"] = cmd if float(cmd.sum()) > 0 else None
    r = ostep(ear, cmd, *a, **k)
    spk = g.bufs[im][inh_lo:, 0]                              # この歩に運動の抑制の体が撃ったか
    run = (run + 1) * spk
    run_max["v"] = max(run_max["v"], int(run.max()))
    f_inh = float(spk.mean())
    inh_acc["all"] += f_inh; inh_acc["ka"] += 1
    if cur["cmd"] is not None:
        f = float((b.motor_rep * cmd).sum() / cmd.sum())
        if win["n"] < C.CMD_LEAD:
            win["list"].append(b.motor_rep.clone())
            win["cmd"] = cmd.clone(); win["n"] += 1
        acc["hit"] += f; utt["hit"] += f
        inh_acc["cmd"] += f_inh; inh_acc["kc"] += 1
    cur["cmd"] = None
    return r


b.step = step


def line(d):
    k = max(d["k"], 1)
    return (f"駆動の上 {d['up']/k:6.1f}  抑制 {d['cnt']/k:6.1f}本 ihn {d['ihn']/k:5.2f}  "
            f"上÷(1＋ihn) {d['up']/k/(1+d['ihn']/k):6.1f}  閾値 {d['th']/k:5.2f}  撃った {d['hit']/k:4.2f}")


def inh_line():
    out = []
    for nm_ in ("聴覚", "運動", "連合", "高次", "前頭前野", "歯状回", "CA3", "CA1", "嗅内浅", "視床", "背内側視床", "側坐核", "上丘"):
        i = NAMES.index(nm_)
        m = g.is_inh_r & (g.area_of == i)
        out.append(f"{nm_} {float(g.rate_ema[m].mean()):.2%}")
    ei = g.is_inh_r[g.e_recv.long()] & ~g.e_inh
    big = float(((g.w[ei] * g.c_exc[g.e_recv[ei].long()]) > T.INH_LTD_TH).float().mean())
    return (f"運動の抑制の体 指令の歩 {inh_acc['cmd']/max(inh_acc['kc'],1):.1%} ／ 全部 {inh_acc['all']/max(inh_acc['ka'],1):.1%}"
            f" ／ 続けて最長 {run_max['v']}歩    抑制の体の率 " + " ".join(out) + f"    大きい繋がり {big:.1%}")


def clear(d):
    for x in d: d[x] = 0.0


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
ok = tried = 0; t0 = time.time()
print("回   言えた  （指令の歩の平均）")
for i in range(N):
    clear(utt); win.update(list=[], n=0, cmd=None)
    ear, cmd, n, want, got = say(rng, True)
    if want:
        tried += 1; ok += int(want == got)
        if want != got and win["n"]:
            fr = torch.tensor(MO.muscle_force(torch.stack(win["list"]).cpu().numpy()), device=B.DEV); c = win["cmd"] > 0
            miss = int(((fr < MO.READ_TH) & c).sum()); extra = int(((fr >= MO.READ_TH) & ~c).sum())
            print(f"   ✗ {i+1:3d}回目 {want} → {got}  指令 {int(c.sum())}体 落ち {miss} 混ざり {extra}  {line(utt)}"
                  f"  覚醒 {float(b.bs.arousal):.2f}", flush=True)
    B.reply(b, rp, rng_r)
    if (i + 1) % 10 == 0:
        print(f"{i+1:4d}  {ok:2d}/{tried:2d}   {line(acc)}  （{(time.time()-t0)/60:.1f}分）", flush=True)
        print(f"        {inh_line()}", flush=True)
        ok = tried = 0; clear(acc); clear(inh_acc); run_max["v"] = 0
print(f"VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB" if torch.cuda.is_available() else "")
