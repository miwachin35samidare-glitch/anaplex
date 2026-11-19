"""版114：網の中の興奮の辺の上限（受け手ごとに 6mV）で、どこの辺が上限に当たるか・率とそろいがどう変わるかを見る（arkhe の ~/anaplex で。網は変えない・保存しない）
  python3 measure_cap114.py [歩=15000]                       ← 版114（上限 6mV ÷ 接触）
  ANAPLEX_WCAP=107z python3 measure_cap114.py 15000           ← 同じ道具で版113 と同じ上限（5.36 ＝ 接触 3.3 で 19.8mV）
  読むもの（1000・3000・6000・10000・終わりの歩で）
    6mV 越え・上限に当たる辺の数：受け手の行（誤差・表現・来ない体）× 送り手（時間細胞・自分の表現・外から）・多い領野（rr の辺は上限の外なので別に）
    区間ごと：皮質（運動の外 20）・運動・海馬・嗅内・扁桃体の表現の率・皮質の発火のうち 20% 以上がそろう歩に入る割合・黙った歩・1歩の時間
  毎歩に足す仕事：表現の発火（全部の領野の体）を GPU の上で足すだけ（少ない箱への足し込みなし・CPU を待たない）
  読む歩だけ、辺を 4M 本ずつ CPU へ写して numpy で数える（区切りごとに beat・読みのあと synchronize）
"""
import sys, time, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, torch_group2 as G, life as L
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 15000
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 114:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版114 で回す（ANAPLEX_WCAP=107z で版113 と同じ上限）")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
areas = g.areas
ch_off = np.cumsum([0] + [a.ch.n_cells for a in areas])
buf_base = int(ch_off[-1])
buf_off = np.cumsum([0] + [a.n_src * a.D for a in areas])
kind = np.full(g.n_send_all, -1, dtype=np.int8)
for i, a in enumerate(areas):
    kind[int(ch_off[i]):int(ch_off[i + 1])] = 0
    s = np.repeat(np.arange(a.n_src), a.D)
    kind[buf_base + int(buf_off[i]):buf_base + int(buf_off[i + 1])] = np.where(s < a.n_ch, 1, np.where(s < a.n_ch + a.n_ext, 2, 3))
assert int((kind < 0).sum()) == 0, "送り手の種類が埋まらない（並びが違う）"
rk_all = g.kind.cpu().numpy().astype(np.int8)            # 0 誤差・1 表現・2 抑制の体
rk_all[g.is_om.cpu().numpy()] = 3                           # 来ない体
area_of = g.area_of.cpu().numpy().astype(np.int16)
c_exc = g.c_exc.cpu().numpy()
rr_s = g.rr_send.cpu().numpy(); rr_r = g.rr_recv.cpu().numpy()
SK = ("時間細胞", "自分の表現", "外から")
RK = ("誤差", "表現", "抑制の体", "来ない体")
CX = [nm for nm in NAMES if TB.PRIOR_TYPE.get(nm) == "皮質" and nm != "運動"]
GROUPS = [("皮質20", CX), ("運動", ["運動"]), ("歯状回", ["歯状回"]), ("CA3", ["CA3"]), ("CA1", ["CA1"]),
          ("嗅内浅", ["嗅内浅"]), ("嗅内深", ["嗅内深"]), ("扁桃体外側", ["扁桃体外側"]), ("扁桃体基底", ["扁桃体基底"])]
GROUPS = [(k, [n for n in v if n in NAMES]) for k, v in GROUPS]
off_ch = [int(x) for x in g.off_ch]
ch_area = np.concatenate([np.full(areas[i].n_ch, i) for i in range(len(areas))])
cx_ch = torch.tensor(np.concatenate([np.arange(off_ch[NAMES.index(n)], off_ch[NAMES.index(n) + 1]) for n in CX]), device=g.w.device)
n_cx = len(cx_ch)
acc = torch.zeros(g.n_ch_all, device=g.w.device)
step_n = torch.zeros(NS + 1, device=g.w.device)
last = {"t": 0, "acc": np.zeros(g.n_ch_all), "clock": time.perf_counter(), "read": 0.0}


def edges(t):
    t0 = time.perf_counter()
    L.beat(); g.flush_all(); L.beat()
    over = np.zeros((4, 3)); cap = np.zeros((4, 3))
    over_a = np.zeros(len(NAMES)); cap_a = np.zeros(len(NAMES))
    over_rr, n_rr = 0, 0
    E, st = len(g.w), 1 << 22
    for a0 in range(0, E, st):
        sl = slice(a0, min(E, a0 + st))
        r = g.e_recv[sl].long().cpu().numpy()
        src = g.e_src[sl].long().cpu().numpy()
        w = g.w[sl].double().cpu().numpy()
        ex = ~g.e_inh[sl].bool().cpu().numpy()
        mv = w * c_exc[r] * 1.12
        rr = rr_s[src] & rr_r[r]
        n_rr += int(rr.sum()); over_rr += int((rr & (mv > 6.0 + 1e-4)).sum())
        ok = ex & ~rr & (kind[src] < 3)
        for m_, M, MA in ((ok & (mv > 6.0 + 1e-4), over, over_a), (ok & (mv > 5.99), cap, cap_a)):
            if m_.any():
                np.add.at(M, (rk_all[r[m_]], kind[src[m_]]), 1)
                MA += np.bincount(area_of[r[m_]], minlength=len(NAMES))
        L.beat()
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()                       # 読みの GPU の仕事を、次の歩の時間（守りの 0.5秒）に入れない
    L.beat()
    print(f"  ［網の中の興奮の辺（予測誤差で学ぶ辺・rr を除く）］読むのに {time.perf_counter() - t0:.1f} 秒", flush=True)
    for nm_, M, MA in (("1本 6mV 越え", over, over_a), ("上限に当たる（5.99mV 以上）", cap, cap_a)):
        cells = "  ".join(f"{RK[i]}←{SK[j]} {int(M[i, j]):,}" for i in range(4) for j in range(3) if M[i, j])
        top = np.argsort(-MA)[:10]
        print(f"    {nm_}：{int(M.sum()):,} 本 ／ {cells}", flush=True)
        print("      多い領野：" + "  ".join(f"{NAMES[i]} {int(MA[i]):,}" for i in top if MA[i]), flush=True)
    print(f"    rr の辺（上限の外）：{n_rr:,} 本のうち 6mV 越え {over_rr:,} 本", flush=True)
    return time.perf_counter() - t0


def window(t):
    a_ = acc.cpu().numpy(); dn = a_ - last["acc"]; dt = t - last["t"]
    sn = step_n[last["t"]:t].cpu().numpy()
    ms = (time.perf_counter() - last["clock"] - last["read"]) / max(dt, 1) * 1000.0
    last["acc"], last["t"] = a_.copy(), t
    hz_a = np.bincount(ch_area, weights=dn, minlength=len(NAMES)) / np.maximum(np.bincount(ch_area, minlength=len(NAMES)), 1) / max(dt, 1) / 0.01067
    n_a = np.bincount(ch_area, minlength=len(NAMES))
    parts = []
    for k, v in GROUPS:
        ii = [NAMES.index(n) for n in v]
        if ii:
            parts.append(f"{k} {float((hz_a[ii] * n_a[ii]).sum() / n_a[ii].sum()):.2f}")
    tot = float(sn.sum())
    syn = float(sn[sn >= 0.2 * n_cx].sum()) / tot if tot > 0 else float("nan")
    print(f"\n━━ {t} 歩（区間 {t - dt}〜{t}・1歩 約 {ms:.1f}ms）━━", flush=True)
    print("  表現の率 Hz：" + "  ".join(parts), flush=True)
    print(f"  皮質20：発火のうち 20% 以上がそろう歩に入る割合 {syn:.1%}（その歩 {float((sn >= 0.2 * n_cx).mean()):.1%}）／ 1体も撃たない歩 {float((sn == 0).mean()):.1%}", flush=True)


print(f"版 {B.VERSION}（上限 {'107z：5.36（接触を入れない）' if G.WCAP_107Z else '6mV ÷ 接触'}・ANAPLEX_RR={'1' if TB.RR_ON else '0'}）／ {NS} 歩 ／ 体 {g.n_ch_all:,}・辺 {len(g.w):,}", flush=True)
marks = {m for m in (1000, 3000, 6000, 10000) if m < NS} | {NS}
for i in range(1, NS + 1):
    life.step(learn=True)
    a_ = g._any[0]
    acc += a_
    step_n[i - 1] = a_[cx_ch].sum()
    if i % 1000 == 0:
        L.beat()
    if i in marks:
        window(i)
        rd = edges(i)
        last["clock"], last["read"] = time.perf_counter(), 0.0
if str(dev).startswith("cuda"):
    import vram_peak as VP
    print(f"\n  ［VRAM］GPU 全体の山（1秒ごと）{VP.peak():.2f}GB・PyTorch の取り置きの山 {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB", flush=True)
print("\n（終わり）", flush=True)
