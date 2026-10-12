"""版47：babble.py の最後の測り（学ばない）で 0/15 になった理由を分ける（arkhe の ~/anaplex で）
  python3 measure_v47b.py frozen [回数=20]   state.pt（喃語100回のあと）を読み、学ばずに言わせる（babble.py の測りと同じ）
  python3 measure_v47b.py learn  [回数=20]   同じ state.pt から、学びながら言わせる
  ⚠️ state.pt は書き換えない（保存しない）。返し手は返さない（babble.py の測りと同じ）
  言わせる列は babble.py の測りと同じ（乱数 10000＋i）
分けたいこと
  学ばないことが効いているのか（frozen だけ言えない）／ 100回育った網がもう言えないのか（両方言えない）
出すもの（1回ごと）
  選んだ → 言えた ／ 指令の歩の 駆動の上・抑制の本数・ihn・閾値・撃った ／ 運動の抑制の体がその発話の指令の歩に撃った割合
5回ごと：領野ごとの抑制の体の率（恒常性の移動平均）"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

if B.VERSION != 47:
    sys.exit(f"⚠️ babble.py が版47 ではない（版 {B.VERSION}）。apply_v47.py を先に")
MODE = sys.argv[1] if len(sys.argv) > 1 else "frozen"
if MODE not in ("frozen", "learn"):
    sys.exit("frozen か learn")
LEARN = MODE == "learn"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
done, _ = B.load(b)
print(f"state.pt 喃語 {done} 回（版 {B.VERSION}）を読んだ。{'学ばずに' if not LEARN else '学びながら'} {N} 回言わせる（保存しない）", flush=True)
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


ok = tried = 0; t0 = time.time()
for i in range(N):
    clear(utt); clear(inh_acc)
    ear, cmd, n, want, got = say(np.random.default_rng(10000 + i), LEARN)
    if want:
        tried += 1; ok += int(want == got)
        m = "○" if want == got else "✗"
        print(f"  {m} {i+1:3d}回目 {want} → {got}  {line(utt)}  運動の抑制の体 指令の歩 {inh_acc['cmd']/max(inh_acc['kc'],1):.1%}"
              f"  覚醒 {float(b.bs.arousal):.2f}", flush=True)
    else:
        print(f"    {i+1:3d}回目 黙る", flush=True)
    if (i + 1) % 5 == 0:
        print(f"        {inh_line()}", flush=True)
print(f"{MODE}  言えた {ok}/{tried}  （{(time.time()-t0)/60:.1f}分）", flush=True)
