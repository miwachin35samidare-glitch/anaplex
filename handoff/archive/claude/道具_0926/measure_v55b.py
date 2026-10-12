"""海馬の表現を横から撃たせているのはどの送り手か（arkhe の ~/anaplex で。0から・学びながら・保存しない・網は書き換えない。
暮らしと試しは measure_check2 と同じ）
  python3 measure_v55b.py [喃語の回数=50]                              ← 版55（全部 焼き直した素子）
  ANAPLEX_PRIOR_V1=海馬,側坐核 python3 measure_v55b.py                  ← 海馬・側坐核だけ前の素子（比べる）
  （コンテナで道具を確かめるとき：python3 measure_v55b.py 3 --test）
【み】10-10 16時11分「いいよ」
【測】measure_v55a（10-10）：CA1・歯状回の増えた発火はほぼ全部「横の興奮が要る」。CA1 に入る辺で撃つ数が約2倍。
  学びの押しは − のままで、焼き直した素子では1回に下げる力が弱い（CA1 約半分・歯状回 約1/5）
【コード】受け手の領野の中の送り手は2つの口から来る（torch_area・torch_brain）
  自分 ＝ その領野の表現の発火 ／ ext ＝ 樹状突起の口（CA1：嗅内浅 層 III の誤差 ／ 歯状回：嗅内浅 層 II の誤差 ／
  嗅内浅：嗅内深 の表現（調整する側。撃たせない）／ 嗅内深：なし）
  どちらも 時間細胞（遅れの窓）と 遅れの列（1〜D歩前。周回の中の発火を含む）で届く
  駆動 ＝ ((興奮 − 調整する側) ＋ 下から x) ÷ (1 ＋ ihn)。下から x は 歯状回 dg ／ CA1 シャッファー（CA3）／ 嗅内深 CA1 ／ 嗅内浅 高次・前頭前野
並べるもの（表現の体、最後の周、10回ごと）
  表現の率 ／ 下からの寄与 ／ ihn ／ 口ごとに：撃った辺（体あたり/歩）・その重みの平均・駆動への寄与（÷(1＋ihn)）
  撃った体の駆動のうち口ごとの割合 ／ ext の送り手の率
読み：焼き直した素子で増えた寄与がどの口か。自分（自分の表現の戻り）なら 自分で自分を撃たせる輪、ext なら 嗅内の輪
"""
import sys, time, inspect, textwrap, os, torch, numpy as np
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
AREAS = ("歯状回", "CA1", "嗅内深", "嗅内浅")
CATS = ("自分・時間細胞", "自分・遅れ", "ext・時間細胞", "ext・遅れ")

# ── _pass に測る行を1つ差し込む（学びの直前。計算は変えない）──
src = textwrap.dedent(inspect.getsource(G.Group2._pass))
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

# 送り手の口：0 自分・時間細胞 ／ 1 自分・遅れ ／ 2 ext・時間細胞 ／ 3 ext・遅れ ／ −1 抑制
ch_off = np.cumsum([0] + [x.ch.n_cells for x in g.areas])
buf_base = int(ch_off[-1])
buf_off = buf_base + np.cumsum([0] + [x.n_src * x.D for x in g.areas])
send_cat = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
for i, a in enumerate(g.areas):
    per = a.ch.n_cell + a.ch.n_ramp
    if per:
        row = torch.arange(a.ch.n_cells, device=dev) // per
        send_cat[ch_off[i]:ch_off[i + 1]] = torch.where(row < a.n_ch, 0, torch.where(row < a.n_ch + a.n_ext, 2, -1))
    row = torch.arange(a.n_src * a.D, device=dev) // a.D
    send_cat[buf_off[i]:buf_off[i + 1]] = torch.where(row < a.n_ch, 1, torch.where(row < a.n_ch + a.n_ext, 3, -1))
info = {}
for nm in AREAS:
    i = NAMES.index(nm); a = g.areas[i]
    c0 = int(g.off_ch[i])
    info[nm] = dict(i=i, ch=slice(c0, c0 + a.n_ch), lo=int(g.off_recv[i]), hi=int(g.off_recv[i + 1]),
                    ext=slice(int(b.off_ext[i]), int(b.off_ext[i + 1])), n_ext=a.n_ext)
rep_pos = torch.full((g.n_recv_all,), -1, dtype=torch.long, device=dev)   # 受け手 → 表現の体の番号（チャンネル）
rep_pos[g.rep_i] = torch.arange(len(g.rep_i), device=dev)

st = {"tag": "他"}
NC = len(CATS)
mk = lambda: {"steps": 0, "rep": 0.0, "x": 0.0, "ihn": 0.0, "ext": 0.0, "mod": 0.0,
              "n": np.zeros(NC), "w": np.zeros(NC), "c": np.zeros(NC), "cf": np.zeros(NC), "dfire": 0.0}
acc = {nm: mk() for nm in AREAS}


def probe(self, L):
    if st["tag"] == "他" or not L["advance"]:
        return
    e, exc, exc_mod, ihn, x_all, rep = L["e"], L["exc"], L["exc_mod"], L["ihn"], L["x_all"], L["rep"]
    ihr = ihn[self.rep_i]
    re_ = self.e_recv[e].long(); se_ = self.e_src[e].long()
    amp = self.c_exc[re_] * self.w[e]
    if self.stp is not None:
        amp = amp * self._amp[e]
    amp = amp * self.send_gain[se_] * (self.scale * self.scale_up)[re_]
    amp = torch.where(self.mod_send[se_], torch.zeros_like(amp), amp)        # 調整する側は撃たせない
    cat = send_cat[se_]
    pos = rep_pos[re_]
    keep = (pos >= 0) & ~self.e_inh[e] & (cat >= 0)
    pos, cat, amp, wv = pos[keep], cat[keep], amp[keep], self.w[e][keep]
    contrib = amp / (1.0 + ihr[pos])
    for nm in AREAS:
        d = info[nm]; c = d["ch"]; A = acc[nm]; n_ch = c.stop - c.start
        fired = rep[c] > 0
        A["steps"] += 1; A["rep"] += float(fired.float().mean())
        A["x"] += float((x_all[c] * self.x_gain[c] / (1.0 + ihr[c])).mean())
        A["ihn"] += float(ihr[c].mean())
        A["mod"] += float((exc_mod[self.rep_i][c] / (1.0 + ihr[c])).mean())
        if d["n_ext"]:
            A["ext"] += float((L["ext_all"][d["ext"]] > 0).float().mean())
        m = (pos >= c.start) & (pos < c.stop)
        fpos = fired[(pos - c.start).clamp(0, n_ch - 1)] & m
        for j in range(NC):
            k = m & (cat == j)
            A["n"][j] += float(k.sum()) / n_ch
            A["w"][j] += float(wv[k].sum())
            A["c"][j] += float(contrib[k].sum()) / n_ch
            A["cf"][j] += float(contrib[fpos & (cat == j)].sum())
        A["dfire"] += float(fired.sum())


probe_box["f"] = probe


def show(head):
    print(f"── {head}", flush=True)
    for nm in AREAS:
        A = acc[nm]; s = max(A["steps"], 1)
        tot_cf = max(A["cf"].sum(), 1e-9)
        ext = f"  ext の送り手 {A['ext']/s:5.2%}" if info[nm]["n_ext"] else ""
        print(f"   {nm:4s} 表現 {A['rep']/s:6.2%}  下から {A['x']/s:.3f}  ihn {A['ihn']/s:5.2f}  調整する側 {A['mod']/s:.3f}{ext}", flush=True)
        print("        " + "  ".join(
            f"{c}：撃った辺 {A['n'][j]/s:6.1f}・重み {A['w'][j]/max(A['n'][j]*(info[nm]['ch'].stop-info[nm]['ch'].start),1):.3f}"
            f"・寄与 {A['c'][j]/s:.3f}・撃った体の {A['cf'][j]/tot_cf:4.0%}" for j, c in enumerate(CATS)), flush=True)
    for k in acc:
        acc[k] = mk()


print(f"素子：{os.environ.get('ANAPLEX_PRIOR_V1') or '全部 焼き直し'}"
      f"{'（ANAPLEX_PRIOR=v1 全部 前）' if os.environ.get('ANAPLEX_PRIOR') == 'v1' else ''}", flush=True)
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
print("読み：焼き直した素子で増えた寄与がどの口か。自分（自分の表現の戻り）なら自分で自分を撃たせる輪、ext なら嗅内の輪", flush=True)
