"""版105（線条体・側坐核・核の入口の盛りを外す）の確かめ。コンテナ・CPU・相手 200。数値（率）は読まない【み】10-18
  python3 check_v105.py
確かめること
  1 盛り（x_gain）：線条体・側坐核・核は 1（1本 ＝ 人の1本）・ほかの領野は今のまま
  2 盛りを外した領野へ入る 8本の道（K_NEW）は X_KEEP に全部入っていて、道ごとに X_NEW を掛けて今の強さのまま
  3 回る（X_KEEP の道は 届いた分 × X_NEW が足される）
  ⚠️ ANAPLEX_X1=0 で版104 と重さが同じことは、コンテナで版104 の木と並べて確かめた（6歩の重さの和が一致、作業記録）
"""
import sys, torch
sys.path.insert(0, ".")
import babble as B, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES
from mouth import N_CMD

B.DEV = "cpu"
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
print("1 盛り（x_gain）", flush=True)
xg = {nm: float(g.areas[i].x_gain) for i, nm in enumerate(NAMES)}
chk(all(xg[k] == 1.0 for k in TB.X1_AREAS), f"線条体・側坐核・核（{len(TB.X1_AREAS)} 領野）は 1")
keep = {"前頭前野": TB.X_NEW, "高次": TB.HUMAN_TH / 2.0, "聴覚": TB.HUMAN_TH / 8.5, "プルキンエ": TB.X_NEW, "扁桃体中心": TB.X_NEW,
        "CM": TB.X_NEW, "上丘": TB.X_NEW, "連合": 1.0, "CA3": 2 * TB.HUMAN_TH_CA3}
chk(all(abs(xg[k] - v) < 1e-9 for k, v in keep.items()), "ほかは今のまま（前頭前野・CM・上丘・扁桃体中心・プルキンエ X_NEW ／ 高次・聴覚・CA3 は今の値 ／ 連合 1）")

print("2 盛りを外した領野へ入る道", flush=True)
ch0 = {int(b.off_ch[NAMES.index(nm)]): nm for nm in TB.X1_AREAS}
rows = []
for k, s in b.bu.items():
    nm = ch0.get(int(s.ch0))
    if nm is None:
        continue
    rows.append((nm, k, len(s.w) / s.n_tgt))
hum = {k for k, _t in TB.P103_STR}                     # 版103：皮質 → 線条体を人の比で分けた道（本数がたまたま 8 のものもある）
k8 = {k for nm, k, l in rows if abs(l - TB.K_NEW) < 1e-9 and k not in hum}
chk(k8 == set(TB.X_KEEP), f"8本の道（K_NEW。人の比で分けた皮質 → 線条体は除く）＝ X_KEEP：{'・'.join(sorted(k8))}")
other = sorted({(nm, int(round(l))) for nm, k, l in rows if k not in k8})
print("     盛りなしになる道の 受け手1体の本数：" + "・".join(f"{nm}{l}" for nm, l in other[:20]) + (" …" if len(other) > 20 else ""), flush=True)
chk(all(k in hum or abs(l - TB.K_NEW) > 1e-9 for nm, k, l in rows if k not in k8), "ほかの道は 人の本数・64本の分け（8本は人の比で分けた皮質 → 線条体のときだけ）")

print("3 回る", flush=True)
raw = {}
D0 = TB._BUTable.drive


def drive_w(self, gi, vs):
    out = D0(self, gi, vs)
    i0, i1 = self._span(gi)
    for j, i in enumerate(range(i0, i1)):
        raw[self.keys[i]] = out[j].clone()
    return out


TB._BUTable.drive = drive_w
gen = torch.Generator().manual_seed(1)
for _ in range(3):
    b.step((torch.rand(B.N_EAR, generator=gen) < 0.2).float(), torch.zeros(N_CMD), learn=True)
TB._BUTable.drive = D0
chk(all(torch.allclose(b._bo[k], raw[k] * TB.X_NEW) for k in TB.X_KEEP), "X_KEEP の道は 届いた分 × X_NEW（受け手の x_gain 1 と合わせて今の強さ）")
chk(all(torch.equal(b._bo[k], raw[k]) for k in raw if k not in TB.X_KEEP and k in b._bo), "ほかの道はそのまま")
print("全部 ✓" if ok else "✗ がある", flush=True)
sys.exit(0 if ok else 1)
