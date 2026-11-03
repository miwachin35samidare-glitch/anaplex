"""版104（強まった後ふさがる仕組み・皮質）の確かめ。組み表_版104 §6
  python3 check_v104.py          （コンテナ・CPU・相手 200。数値（率）は読まない【み】10-18）
確かめること
  1 痕跡の形（重さの配列だけで）：痕跡 0 は全部効く ／ 強めた直後はまだ効く・育つと効かない・深いと弱まりへ ／ 撃つだけでは増えない ／
    弱めると痕跡が減る ／ τ残（3時間・6時間）
  2 網：当てる辺が皮質の領野の興奮（受け手が表現・誤差・来ない体）だけ・下からの線は受け手が皮質の道だけ・CL は全部
  3 網：痕跡 0 なら ANAPLEX_OCC=0 と重さが全部同じ（同じ網・同じ入力で 2歩）
  4 網：痕跡があると、皮質の強まりだけが g 倍になり、ほかの重さは触らない
  5 網：刈り込みのあと、残った辺の痕跡は持ち越し・生やした辺は 0・当てる辺の選び直し
"""
import sys, math, torch
sys.path.insert(0, ".")
import babble as B, torch_area as T
import occlusion as O
import torch_brain as TB
from torch_brain import Brain, NAMES
from mouth import N_CMD

B.DEV = "cpu"
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


print("1 痕跡の形（OccSet だけ）", flush=True)
W = {"w": torch.ones(4)}
s = O.OccSet(lambda: W["w"], None)
W["w"] += torch.tensor([0.1, 0.0, -0.1, 0.0]); s.apply()
chk(torch.allclose(W["w"], torch.tensor([1.1, 1.0, 0.9, 1.0])), "痕跡 0：強まりも弱まりもそのまま（g ＝ 1）")
chk(float(s.q[1]) == 0.0, "撃つだけ（重さが変わらない）では q は増えない ⇒ ①")
s = O.OccSet(lambda: W["w"], None); W["w"] = torch.ones(4); s.prev = W["w"].clone()
W["w"][0] += 10 * O.OCC_M; s.apply()                         # 大きく強めた
W["w"][1] += 0.1; s.apply()
chk(abs(float(W["w"][1]) - 1.1) < 1e-6, "強めた直後（まだ育っていない）は次の強まりも効く ⇒ ③")
s.advance(int(4 * O.TAU_GROW))
W["w"][0] += 0.1; s.apply()
chk(float(W["w"][0]) < 1.0 + 10 * O.OCC_M, f"育ったあと（τ育 の4倍）は強めると弱まりに転ぶ（g ＝ {float(O.gain(s.m[0:1].float())):.2f}）⇒ ②")
s = O.OccSet(lambda: W["w"], None); W["w"] = torch.ones(4); s.prev = W["w"].clone()
W["w"][0] += 5.0; s.apply(); W["w"][0] -= 5.0; s.apply()
s.advance(int(4 * O.TAU_GROW))
chk(float(s.m[0]) < 1e-3, "強めたあとすぐ弱めると痕跡は残らない ⇒ ⑥")
s = O.OccSet(lambda: W["w"], None); s.m.fill_(10 * O.OCC_M)
s3 = int(round(3 * 3600 * 1000 / T.DT_MS)); s.advance(s3)
g3 = float(O.gain(s.m[0:1].float()))
s.advance(s3)
g6 = float(O.gain(s.m[0:1].float()))
chk(g3 <= 0.01 and g6 >= 0.89, f"痕跡 10M から：3時間で g ＝ {g3:.2f}（まだ効かない）・6時間で g ＝ {g6:.2f}（戻る）⇒ ④⑤")
chk(abs(O.TAU_GROW - 11246) < 1 and abs(O.TAU_KEEP - 438613) < 2, f"τ育 {O.TAU_GROW:.0f}歩（2分）・τ残 {O.TAU_KEEP:.0f}歩（1.3時間）")

print("2 当てる辺", flush=True)
torch.manual_seed(0)
mk = lambda: Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b = mk()
g = b.g
gen = torch.Generator().manual_seed(1)
INS = [((torch.rand(B.N_EAR, generator=gen) < 0.2).float(), torch.zeros(N_CMD)) for _ in range(4)]
b.step(*INS[0])
oc = b.occ
chk(oc is not None and g._occ is oc, "初めの歩で作られ、Group2 の刈り込みにも繋がる")
chk(set(O.OCC_AREAS) == set(TB.CL_CORTEX) and "嗅内浅" not in O.OCC_AREAS and "嗅内深" not in O.OCC_AREAS,
    "当てる領野 ＝ 新皮質 16（CL の受け手と同じ・嗅内は外す）")
area = torch.bucketize(g.e_recv.long(), torch.as_tensor(g.off_recv[1:]).long(), right=True)
cx = torch.tensor([NAMES[int(i)] in O.OCC_AREAS for i in range(len(NAMES))])
want = cx[area] & ~g.is_inh_r.bool()[g.e_recv.long()] & ~g.e_inh.bool()
idx = oc.grp.idx.long()
chk(bool(want[idx].all()) and int(want.sum()) == len(idx), f"領野の中：{len(idx):,}本（網の辺 {len(g.w):,} の {len(idx)/len(g.w):.1%}）＝ 皮質の受け手（表現・誤差・来ない体）× 興奮の辺")
chk(not bool((cx[area] & g.e_inh.bool())[idx].any()), "抑制の辺・抑制の体への辺は入らない")
ch0 = {int(b.off_ch[NAMES.index(nm)]) for nm in O.OCC_AREAS}
keys = oc.keys_bu
chk(len(keys) > 0 and all(int(b.bu[k].ch0) in ch0 for k in keys), f"下から：受け手が皮質の道だけ（{len(keys)}本の道・線 {len(oc.sets['下から'].q):,}）")
chk(not any(k.split("→")[-1] in ("歯状回", "CA3", "CA1", "嗅内浅", "嗅内深", "尾状核", "被殻", "側坐核", "黒質", "視床下核")
            for k in keys) and "貫通路" not in keys, "下から：海馬・嗅内・線条体・核の道は入らない")
print("     道：" + "・".join(keys[:12]) + (" …" if len(keys) > 12 else ""), flush=True)
chk(oc.sets["CL"].idx is None and len(oc.sets["CL"].q) == len(b.cl_w), "CL → 皮質：全部")

print("3 痕跡 0 なら ANAPLEX_OCC=0 と同じ", flush=True)
O.OCC_ON = False
torch.manual_seed(0)
b0 = mk()
for x in INS[:3]:
    b0.step(*x)
O.OCC_ON = True
for x in INS[1:3]:
    b.step(*x)
oc.grp.apply(); [s_.apply() for s_ in oc.sets.values()]
b.g.flush_all(); b0.g.flush_all()
chk(torch.equal(b.g.w, b0.g.w) and torch.equal(b.but.w, b0.but.w) and torch.equal(b.cl_w, b0.cl_w),
    "3歩：領野の中・下から・CL の重さが全部同じ")

print("4 痕跡があると皮質の強まりだけ g 倍", flush=True)
for s_ in oc.all_sets():
    s_.m.fill_(5.0)                                   # g ＝ max(1 − 5, −0.5) ＝ −0.5
b.step(*INS[3])
b.g.flush_all()
w_all = [b.g.w.clone(), b.but.w.clone(), b.cl_w.clone()]
prevs = [oc.grp.prev.clone(), oc.sets["下から"].prev.clone(), oc.sets["CL"].prev.clone()]
oc.grp.apply(); [s_.apply() for s_ in oc.sets.values()]
for nm, w0, w1, pv, s_ in (("領野の中", w_all[0], b.g.w, prevs[0], oc.grp), ("下から", w_all[1], b.but.w, prevs[1], oc.sets["下から"]),
                           ("CL", w_all[2], b.cl_w, prevs[2], oc.sets["CL"])):
    sel = torch.ones(len(w0), dtype=torch.bool) if s_.idx is None else torch.zeros(len(w0), dtype=torch.bool).index_fill_(0, s_.idx.long(), True)
    d = (w0[sel] - pv)
    want_ = torch.where(d > 0, (pv - 0.5 * d).clamp(min=0), w0[sel])
    chk(torch.allclose(w1[sel], want_, atol=1e-6) and torch.equal(w1[~sel], w0[~sel]),
        f"{nm}：強まった {int((d > 0).sum()):,}本は弱まりへ（−0.5 倍）・弱まりはそのまま・当てない線 {int((~sel).sum()):,}本は触らない")

print("5 刈り込み", flush=True)
f = lambda: ((g.e_src.long() * 7919 + g.e_recv.long()) % 8).float() / 8.0 + 0.125   # 繋がり（送り手・受け手）ごとの目印
oc.grp.m.copy_(f()[oc.grp.idx.long()].half())
src0 = g.e_src.clone()
g.rewire()
idx = oc.grp.idx.long()
area = torch.bucketize(g.e_recv.long(), torch.as_tensor(g.off_recv[1:]).long(), right=True)
want = cx[area] & ~g.is_inh_r.bool()[g.e_recv.long()] & ~g.e_inh.bool()
chk(bool(want[idx].all()) and int(want.sum()) == len(idx) and len(oc.grp.q) == len(idx) == len(oc.grp.m) == len(oc.grp.prev),
    f"選び直し：{len(idx):,}本・q・m・prev の長さがそろう")
mm = oc.grp.m.float()
same = mm == f()[idx]
zero = mm == 0
chk(bool((same | zero).all()) and float(same.float().mean()) > 0.9,
    f"残った辺は目印を持ち越し（{float(same.float().mean()):.1%}）・ほかは 0（生やした辺 {int(zero.sum()):,}本）")
chk(torch.equal(oc.grp.prev, g.w[idx].float()), "prev は刈り込みのあとの重さ")
b.step(*INS[0])
chk(True, "刈り込みのあとも回る")

print("6 VRAM の見込み（arkhe の網は相手 9,100）", flush=True)
n_rx = len(oc.grp.q) / 200.0                       # 皮質の受け手1体の相手 200 本（この網）
est = n_rx * 9100.0
print(f"     皮質の受け手 約{n_rx:,.0f}体 × 相手 9,100 ≒ {est/1e6:.0f}百万本 × 14バイト（prev 4・q 4・m 2・番号 4）≒ {est*14/1e9:.2f}GB"
      f"（下から・CL は数十MB）⇒ measure_mem103 で確かめる", flush=True)

print("全部 ✓" if ok else "✗ がある", flush=True)
sys.exit(0 if ok else 1)
