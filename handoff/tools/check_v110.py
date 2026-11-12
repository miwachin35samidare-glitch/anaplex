"""版110 の確かめ（コンテナ・CPU・網は組まない）：下からの線・CL の学びの式に、受け手が先の項の境目が入ったか
  python3 check_v110.py
  見ること（式は torch_brain の _bu_learn_core・_cl_learn_core をそのまま呼ぶ）
    1 撃たない受け手（窓の外）：入力が撃っても重さは変わらない（どの q でも）
    2 受け手が窓の中（2歩前）に撃ち、入力が今撃つ：版109 の値 − ずれ／50 になる
    3 q ＝ 1（的どおり）：版109 と同じ
    4 海馬の線：ずれ 0 なので版109 と同じ
    5 CL：受け手が1歩前に撃ち、入力が今撃つ：(ln1.21 − ずれ)／75
"""
import math, torch
import babble as B, torch_brain as TB

assert B.VERSION == 110, B.VERSION
ok = True


def chk(name, got, want, tol=1e-6):
    global ok
    g, w = float(got), float(want)
    good = abs(g - w) <= tol * max(1.0, abs(w))
    ok &= good
    print(f"{'✓' if good else '✗'} {name}：{g:.6f}（思っていた値 {w:.6f}）", flush=True)


def bu(q, age, hippo=False, d=0.7, pre_prev=False):
    """1本の線：入力が今撃つ・受け手は今撃たない・受け手は age 歩前に撃った"""
    src = torch.zeros(1)                        # 受け手は今撃たない
    pidx = torch.zeros(1, dtype=torch.long)
    pre_now = torch.ones(1, dtype=torch.bool)
    pre_prev_ = torch.full((1,), pre_prev, dtype=torch.bool)
    tgt = torch.zeros(1, dtype=torch.long)
    hip = torch.full((1,), hippo, dtype=torch.bool)
    tab = torch.zeros(2, TB.WIN_MAXK + 2)
    for k, v in TB.WIN_CTX.items():
        tab[0, k] = v
    for k, v in TB.WIN_HIP.items():
        tab[1, k] = v
    post_age = torch.full((1,), age, dtype=torch.long)
    sh = torch.zeros(1) if hippo else TB._bcm_shift(torch.tensor([q]))
    fac, _ = TB._bu_learn_core(src, pidx, pre_now, pre_prev_, tgt, hip, tab, post_age, torch.tensor(d), sh)
    return math.log(float(fac))


g_on = 2 * 0.7 - 1
for q in (0.0, 0.3, 3.0):
    chk(f"1 撃たない受け手（13歩前）q {q}", bu(q, 13), 0.0)
sh3 = float(TB._bcm_shift(torch.tensor([3.0])))
sh03 = float(TB._bcm_shift(torch.tensor([0.3])))
chk("2 撃ちすぎ q 3・2歩前", bu(3.0, 2), math.log(1.26) / 50 * g_on - sh3 / 50)
chk("2 撃ちすぎ q 3・6歩前", bu(3.0, 6), math.log(1.16) / 50 * g_on - sh3 / 50)
chk("2 静か q 0.3・2歩前", bu(0.3, 2), math.log(1.26) / 50 * g_on - sh03 / 50)
chk("3 的どおり q 1・2歩前（版109 と同じ）", bu(1.0, 2), math.log(1.26) / 50 * g_on)
chk("4 海馬の線・2歩前（版109 と同じ）", bu(3.0, 2, hippo=True), TB.WIN_HIP[2] * g_on)
chk("4 海馬の線・10歩前（版109 と同じ）", bu(3.0, 10, hippo=True), TB.WIN_HIP[10] * g_on)


def cl(q, post_prev):
    any0 = torch.zeros(1)                       # 受け手は今撃たない
    cl_ch = torch.zeros(1, dtype=torch.long)
    cl_spk = torch.ones(1); cl_pre_prev = torch.zeros(1)
    cl_src = torch.zeros(1, dtype=torch.long); cl_recv = torch.zeros(1, dtype=torch.long)
    cl_post_prev = torch.full((1,), 1.0 if post_prev else 0.0)
    sh = TB._bcm_shift(torch.tensor([q]))
    w, _ = TB._cl_learn_core(any0, cl_ch, cl_spk, cl_pre_prev, cl_src, cl_recv, cl_post_prev, torch.tensor(1.4), torch.ones(1), sh)
    return math.log(float(w))


chk("5 CL 撃ちすぎ q 3・受け手が1歩前", cl(3.0, True), (math.log(1.21) - sh3) / TB.CL_PAIRS)
chk("5 CL 撃たない受け手 q 3", cl(3.0, False), 0.0)
chk("5 CL 的どおり q 1（版109 と同じ）", cl(1.0, True), math.log(1.21) / TB.CL_PAIRS)
print("\n" + ("全部 ✓" if ok else "✗ がある"), flush=True)
