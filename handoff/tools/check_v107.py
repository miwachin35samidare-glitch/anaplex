"""版107（落とした 13部位・調整物質）の確かめ。相手 200。率は読まない【み】10-18
  python3 check_v107.py          （GPU があれば GPU で。Triton の核の 門 2つ を通す）
  1 体数・型・並び ／ 2 線 ／ 3 樹状突起の側・駆動の並び ／ 4 門 2つ（組を全部 0 にした網と、門を同じにした網で重さが同じ）／
  5 閾値の倍率（セロトニン・アセチルコリン）／ 6 回る（nan なし）
"""
import sys, torch
sys.path.insert(0, ".")
import babble as B, torch_area as T, brainstem as BS
import torch_brain as TB
from torch_brain import Brain, NAMES
from mouth import N_CMD

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B.DEV = DEV
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


_obs = BS.Brainstem.observe


def build(mode="new"):
    def obs(self, b, *a, **k):
        if mode in ("eq", "zero"):                     # 門を黒質の 1つにそろえる
            b.gate = b.gate[:1].expand(2).clone()
        return _obs(self, b, *a, **k)
    BS.Brainstem.observe = obs
    torch.manual_seed(0)
    b = Brain(T.Prior.load("prior_v1.npz", dev=DEV), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
    if mode == "zero":
        b.g.recv_grp.zero_()
    return b


def run(b, n):
    gen = torch.Generator().manual_seed(1)
    for _ in range(n):
        b.step((torch.rand(B.N_EAR, generator=gen) < 0.2).float().to(DEV), torch.zeros(N_CMD, device=DEV), learn=True)


print(f"（{DEV}）", flush=True)
b = build()
n = b.n
print("1 体数・型・並び", flush=True)
want = {"島前": 226, "島後": 157, "縁上回": 377, "海馬傍": 125, "脳梁膨大後": 22,
        "中線核": 20, "視床前核": 35, "海馬支脚": 90, "乳頭体": 1, "分界条床核": 19, "視床網様核": 198,
        "VTA": 256, "背側縫線": 256, "正中縫線": 256, "中隔GABA": 16, "中隔ACh": 10, "Ch4": 17, "線条体ACh": 34,
        "ストリオソーム": 511, "尾状核": 848, "被殻": 929}
bad = {k: n.get(k) for k, v in want.items() if n.get(k) != v}
chk(not bad, f"体数 {len(want)} 部位が組み表どおり" + (f"（違う：{bad}）" if bad else ""))
chk(NAMES[-(5 + 6 + 7):] == TB.CX107 + TB.G107B + TB.G107C, "NAMES の最後に 版107a・b・c の部位")
chk(all(TB.PRIOR_TYPE[k] == "皮質" for k in TB.CX107) and all(TB.PRIOR_TYPE[k] == "視床" for k in TB.TH107)
    and TB.PRIOR_TYPE["海馬支脚"] == "海馬" and TB.PRIOR_TYPE["VTA"] == "黒質", "型")
chk(all(b.areas[k].ch.n_cell == 0 for k in TB.TH107 + TB.NC107), "視床の部位・核は時間細胞を持たない")
chk(b.areas["Ch4"].n_inh == 2, f"Ch4 の中の抑制の体 1割（{b.areas['Ch4'].n_inh}）")

print("2 線", flush=True)
ks = TB.P107_CX + TB.P107_TH + TB.P107B_TH
chk(all(k in b.bu for k, _t in ks), f"8本の道 {len(ks)} 本がみな網にある")
chk(all(len(b.bu[k].w) == min(8, n[k.split("→")[0]]) * n[t] for k, t in ks), "受け手1体 K_NEW 8（送り手が 8 未満なら全部）")
kp = {r: b.k_cx.get(f"縁上回→{r}") for r in ("被殻", "尾状核", "側坐核")}
chk(kp == {"被殻": 537, "尾状核": 354, "側坐核": 81}, f"縁上回 → 線条体（頭頂の帯）{kp}")
chk(set(b.k106) == {"眼窩前頭→ストリオソーム", "前帯状→ストリオソーム", "島前→ストリオソーム"} and sum(b.k106.values()) == 2700,
    f"ストリオソーム ← 辺縁の皮質 2,700（{b.k106}）")
chk(not any(k in b.bu for k in ("背内側視床→ストリオソーム", "CL→ストリオソーム")), "版106 の 背内側視床・CL → ストリオソーム は無い")
nl = {(s_[0], t_, sg_) for s_, t_, sg_, *_ in TB.NUC_LINES}
need = [("乳頭体", "視床前核", 1), ("乳頭体", "中線核", 1), ("中線核", "側坐核", 1), ("中線核", "分界条床核", 1),
        ("中線核", "ストリオソーム", 1), ("扁桃体中心", "分界条床核", -1), ("分界条床核", "側坐核", -1),
        ("分界条床核", "PAG", -1), ("分界条床核", "黒質", -1), ("分界条床核", "VTA", -1), ("側坐核", "VTA", -1),
        ("扁桃体中心", "VTA", -1), ("吻内側被蓋核", "VTA", -1), ("前帯状", "背側縫線", 1), ("眼窩前頭", "正中縫線", 1),
        ("島前", "Ch4", 1), ("扁桃体中心", "Ch4", -1), ("背内側視床", "Ch4", 1), ("CM", "線条体ACh", 1), ("Pf", "線条体ACh", 1)] + \
       [("視床網様核", t_, -1) for t_ in TB.TH_ALL107]
miss = [p for p in need if p not in nl]
chk(not miss, f"64本の線 {len(need)} 本（符号込み）" + (f"（無い：{miss}）" if miss else ""))
chk(len(b.sep_lines) == 5 and all(len(r_) in (nt_, 2 * nt_) for _k, _s, r_, nt_, *_ in b.sep_lines),
    "海馬 ⇄ 中隔（抑制の体どうし）5 本：中隔GABA → 歯状回・CA3・CA1 ／ CA3・CA1 → 中隔GABA")

print("3 樹状突起の側・駆動の並び", flush=True)
for r_, ss_ in TB.EXT107_POST.items():
    chk(b.n_ext[r_] - b.ext_old[r_] == sum(n[s] for s in ss_), f"{r_} の後ろに {'・'.join(ss_)}")
chk(all(b.areas[k].n_ext == b.n_ext[k] for k in NAMES), "領野の樹状突起の側の数 ＝ n_ext")
o = 0
good = True
for nm in NAMES:
    good = good and b._x_off[nm] == o
    o += n[nm]
chk(good, f"_x_off は NAMES の順・体数の和 {o}")

print("4 門 2つ", flush=True)
rg = b.g.recv_grp
vta_areas = [nm for i, nm in enumerate(NAMES) if int(rg[int(b.g.off_recv[i])]) == 1]
chk("側坐核" in vta_areas and "前頭前野" in vta_areas and "CA1" in vta_areas and "扁桃体基底" in vta_areas
    and "尾状核" not in vta_areas and "ストリオソーム" not in vta_areas and "黒質" not in vta_areas,
    f"VTA の門の組 {len(vta_areas)} 領野（側坐核・皮質・海馬・扁桃体。線条体の背側・核は黒質）")
run(b, 6)
chk(tuple(b.gate.shape) == (2,) and tuple(b.g.third.reshape(-1).shape) == (2,), f"門は (黒質, VTA) ＝ {[round(float(v), 3) for v in b.gate]}")
if DEV == "cpu":
    bz, be = build("zero"), build("eq")
    run(bz, 6); run(be, 6)
    bz.g.flush_all(); be.g.flush_all()
    d0 = float((bz.g.w - be.g.w).abs().max())
    chk(d0 == 0.0, f"組を全部 0 にした網と門を同じにした網の重さが同じ（最大の差 {d0}）")
    del bz, be
else:   # GPU は足し込みの順が揺れて同じ網でも重さがずれる（11-05 check_v107_gpu A）⇒ 核は check_v107_gpu の B で比べる
    print("     （GPU：組 0 と門を同じの比べは check_v107_gpu で。同じ網を2回でも差 0.41 が出る）", flush=True)

print("5 閾値の倍率", flush=True)
tm = b.th_mod
o_ = lambda nm: int(b.off_ch[NAMES.index(nm)])
chk(bool(torch.isfinite(tm).all()) and float(tm.min()) > 0.0, f"倍率 {float(tm.min()):.3f}〜{float(tm.max()):.3f}")
w5 = b.w5
chk(float(w5[o_("聴覚")]) > 0 > float(w5[o_("嗅内浅")]) and float(w5[o_("CA1")]) < 0 and float(w5[o_("視床")]) > 0,
    "セロトニンの向き：聴覚・視床は上げ・嗅内・海馬は下げ（2A − 1A）")
chk(float(b.wach4[o_("聴覚")]) > 0 and float(b.wach4[o_("被殻")]) == 0 and float(b.wsep[o_("CA1")]) == 1.0
    and float(b.wcn[o_("尾状核")]) == 1.0 and float(b.wpu[o_("被殻")]) == 1.0, "アセチルコリンの行き先（Ch4・中隔・線条体の介在）")
print("     ならした撃った割合：" + "・".join(f"{k} {float(v):.3f}" for k, v in b.m107.items()), flush=True)

print("6 回る", flush=True)
chk(all(bool(torch.isfinite(v).all()) for v in (b.g.w, b.g.v_carry)), "重さ・膜に nan が無い")
print("全部 ✓" if ok else "✗ がある", flush=True)
sys.exit(0 if ok else 1)
