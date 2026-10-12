"""焼き直した素子は、どの道で海馬・側坐核の表現の発火を上げるか（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は書き換えない。
暮らしと試しは measure_check2 と同じ）
  python3 measure_v55a.py [喃語の回数=50]                              ← 版55（全部 焼き直した素子）
  ANAPLEX_PRIOR_V1=海馬,側坐核 python3 measure_v55a.py                  ← 海馬・側坐核だけ前の素子（比べる）
  （コンテナで道具を確かめるとき：python3 measure_v55a.py 3 --test）
【み】10-10 15時半「一応Cの確認からでいいけど」
【測】measure_check2（10-10）：CA1・嗅内深・側坐核・歯状回の約2倍は、海馬・側坐核自身の焼き直した素子から（上流からではない）
素子（見込み）が表現の発火に届く道（torch_group2._pass。コードから）
  1 見込みで足す：CA3・CA1 だけ。drive ＋ 0.9 × p ＞ 閾値 で撃つ（下からの入力があるとき）⇒ p が下がれば 減る向き
  2 誤差の体：的 − p ＞ 0.5 で撃ち、下流の入口になる ⇒ p が下がれば 増える向き
  3 学び：跡 ＋＝ η × (的 − p)（撃った辺）⇒ p が的の率より下なら辺が育ち、横の興奮が増えて表現が撃つ ⇒ 増える向き
     起きているあいだの恒常性は上げるだけ（版53）。健診の中では下げる側が無い
並べるもの（領野ごと、10回ごと。表現は最後の周）
  表現の率と、撃った訳：下からだけで越える ／ 横の興奮が要る ／ 見込みで足す ／ 自発
  表現の体の p ／ 横の興奮（表現の体の平均）／ ihn ／ 誤差の体の率と p
  学び：撃った辺の (的 − p) の平均と ＋ の割合（表現の体へ・誤差の体へ）
  重み：表現の体へ入る興奮の辺の平均（自分の時間 ／ 自分の遅れ ／ 外から）／ 起きて上げた倍率 scale_up の平均
読み：前の素子と比べて、2倍の領野で「横の興奮が要る」と重みが増え、押しが ＋ に寄れば 3（学び）。誤差の体が増えて下流が上がれば 2
"""
import sys, time, inspect, textwrap, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (55,):
    sys.exit(f"⚠️ babble.py が版55 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
N_PROBE = 3
AREAS = ("聴覚", "歯状回", "CA1", "嗅内深", "側坐核")
SRC = ("自分の時間", "自分の遅れ", "外から")
WHY = ("下からだけ", "横が要る", "見込みで足す", "自発")

# ── _pass に測る行を1つ差し込む（学びの直前。計算は変えない）──
src = textwrap.dedent(inspect.getsource(G.Group2._pass))
MARK = "        if learn and len(e):\n"
src_i = src.replace("    if learn and len(e):\n", "    _PROBE(self, locals())\n    if learn and len(e):\n", 1)
if src_i.count("_PROBE(") != 1:
    sys.exit("⚠️ _pass に差し込む印が見つからない")
ns = dict(G.__dict__)
probe_box = {"f": None}
ns["_PROBE"] = lambda self, L: probe_box["f"] and probe_box["f"](self, L)
exec(src_i, ns)
G.Group2._pass = ns["_pass"]

torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev

# 領野ごとの場所
info = {}
ch_off = np.cumsum([0] + [x.ch.n_cells for x in g.areas])
buf_base = int(ch_off[-1])
buf_off = buf_base + np.cumsum([0] + [x.n_src * x.D for x in g.areas])
send_area = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
send_cat = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    if per:
        row = torch.arange(a.ch.n_cells, device=dev) // per
        send_cat[ch_off[i]:ch_off[i + 1]] = torch.where(row < a.n_ch, 0, -1)
    row = torch.arange(a.n_src * a.D, device=dev) // a.D
    send_cat[buf_off[i]:buf_off[i + 1]] = torch.where(row < a.n_ch, 1, torch.where(row < a.n_ch + a.n_ext, 2, -1))
    send_area[ch_off[i]:ch_off[i + 1]] = i
    send_area[buf_off[i]:buf_off[i + 1]] = i
for nm in AREAS:
    i = NAMES.index(nm); a = g.areas[i]
    c0 = int(g.off_ch[i])
    info[nm] = dict(i=i, ch=slice(c0, c0 + a.n_ch), lo=int(g.off_recv[i]), hi=int(g.off_recv[i + 1]))

st = {"tag": "他"}
K = ("steps", "rep", "w0", "w1", "w2", "w3", "pr", "lat", "ihn", "err", "perr", "tgt",
     "nr", "pr_sum", "pr_pos", "ne", "pe_sum", "pe_pos")
acc = {nm: dict.fromkeys(K, 0.0) for nm in AREAS}


def probe(self, L):
    if st["tag"] == "他":
        return
    e, err_sig, p = L["e"], L["err_sig"], L["p"]
    if L["learn"] and len(e):                      # 学び：撃った辺の (的 − p)。抑制の体へ・抑制から は学ばない
        r = self.e_recv[e].long()
        ok = ~self.e_inh[e] & ~self.is_inh_r[r]
        for nm in AREAS:
            d = info[nm]
            m = ok & (r >= d["lo"]) & (r < d["hi"])
            if not bool(m.any()):
                continue
            s = err_sig[r[m]]; isr = self.is_rep[r[m]]
            A = acc[nm]
            A["nr"] += float(isr.sum()); A["pr_sum"] += float(s[isr].sum()); A["pr_pos"] += float((s[isr] > 0).sum())
            A["ne"] += float((~isr).sum()); A["pe_sum"] += float(s[~isr].sum()); A["pe_pos"] += float((s[~isr] > 0).sum())
    if not L["advance"]:
        return
    exc, exc_mod, ihn, x_all, drive, rep, pr = L["exc"], L["exc_mod"], L["ihn"], L["x_all"], L["drive"], L["rep"], L["pr"]
    th = self.fire_th * self.th_gain
    ihr = ihn[self.rep_i]
    bottom = (x_all * self.x_gain / (1.0 + ihr)) > th
    by_drive = drive > th
    fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > th) & (x_all > 0) & ~by_drive
    lat = (exc - exc_mod)[self.rep_i] / (1.0 + ihr)
    errf = (err_sig[self.err_i] > T.THETA).float()
    for nm in AREAS:
        d = info[nm]; c = d["ch"]; A = acc[nm]
        rp_ = rep[c] > 0
        n = float(rp_.sum())
        A["steps"] += 1; A["rep"] += n
        A["w0"] += float((rp_ & bottom[c]).sum())
        A["w1"] += float((rp_ & by_drive[c] & ~bottom[c]).sum())
        A["w2"] += float((rp_ & fill[c]).sum())
        A["w3"] += float((rp_ & ~by_drive[c] & ~fill[c]).sum())
        A["pr"] += float(pr[c].mean()); A["lat"] += float(lat[c].mean()); A["ihn"] += float(ihr[c].mean())
        A["err"] += float(errf[c].mean()); A["perr"] += float(p[self.err_i][c].mean())
        A["tgt"] += float(L["tgt"][self.err_i][c].mean())


probe_box["f"] = probe


def weights():
    g.flush_all()                                  # 版54：跡を重みへまとめてから読む（意味は同じ）
    r = g.e_recv.long(); out = {}
    for nm in AREAS:
        d = info[nm]
        m = (r >= d["lo"]) & (r < d["hi"]) & g.is_rep[r] & ~g.e_inh
        c = send_cat[g.e_src[m].long()]; w = g.w[m]
        out[nm] = ([float(w[c == j].mean()) if int((c == j).sum()) else float("nan") for j in range(3)],
                   float((g.scale_up[d["lo"]:d["hi"]])[g.is_rep[d["lo"]:d["hi"]]].mean()))
    return out


def show(head):
    W = weights()
    print(f"── {head}", flush=True)
    for nm in AREAS:
        A = acc[nm]; s = max(A["steps"], 1); n = max(A["rep"], 1); nc = info[nm]["ch"].stop - info[nm]["ch"].start
        why = "・".join(f"{WHY[j]} {A[f'w{j}']/n:4.0%}" for j in range(4))
        print(f"   {nm:4s} 表現 {A['rep']/s/nc:6.2%}（{why}）  p {A['pr']/s:.3f}  横 {A['lat']/s:.3f}  ihn {A['ihn']/s:.2f}"
              f"  ｜ 誤差 {A['err']/s:5.2%}  的 {A['tgt']/s:5.2%}  p {A['perr']/s:.3f}", flush=True)
        wm, su = W[nm]
        print(f"        学び（的−p）表現へ {A['pr_sum']/max(A['nr'],1):+.3f}（＋{A['pr_pos']/max(A['nr'],1):4.1%}、{A['nr']/s:.0f}本/歩）"
              f"  誤差へ {A['pe_sum']/max(A['ne'],1):+.3f}（＋{A['pe_pos']/max(A['ne'],1):4.1%}、{A['ne']/s:.0f}本/歩）"
              f"  ｜ 重み " + "  ".join(f"{SRC[j]} {wm[j]:.4f}" for j in range(3)) + f"  scale_up {su:.2f}", flush=True)
    for A in acc.values():
        for k in A:
            A[k] = 0.0


import os
print(f"素子：{os.environ.get('ANAPLEX_PRIOR_V1') or '全部 焼き直し'}{'（ANAPLEX_PRIOR=v1 全部 前）' if os.environ.get('ANAPLEX_PRIOR') == 'v1' else ''}", flush=True)
rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
t_start = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if ear.sum() > 0:
        past.append((cmd, ear))
    if rp is not None:                           # 暮らし：返し手が返せば聞く（babble.py と同じ）
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        for _ in range(N_PROBE):                 # 試し：10回ごとに録音を3つ。学びは止めない
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
        show(f"{i+1}回目まで（{(time.time()-t_start)/60:.1f}分）この10回")
print("読み：2倍の領野で 横が要る・重み・学びの＋ が前の素子より上なら 学びの道（3）。誤差が増えて下流が上がるなら 誤差の道（2）", flush=True)
