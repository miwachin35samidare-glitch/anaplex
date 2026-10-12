"""B（表現の的 ＝ 自分がこの歩に撃ったか）を最小構成で試す（09-26 夕）

最小構成  diag_rep.py と同じ：聴覚1領野（40帯）・相手500。下からは帯ごとの本数、9本以上で越える（重さ 1.68・閾値 14.3）
          越える歩は 1歩 約1%
見込み    代役の網。定数項を焼いた見込みの値（p 約0.12、check_prior 09-26 arkhe）に寄せる
          ⚠️ 焼いたものではない。結論を見込みの値に寄せない
条件      版35 ／ 版35 ＋ 興奮 0以上 ／ B ／ B ＋ 興奮 0以上（興奮 0以上 は【み】09-26 決めた形）
見るもの  下から越えた歩：表現が撃った割合 ／ 誤差が撃った割合
          越えない歩：表現が撃った割合（自発 0.32% を大きく超えたら、的が学ぶ重みに依存して暴走している）
"""
import sys
import time
import numpy as np
import torch
from torch_area import Area, Prior
from torch_group2 import Group2

torch.set_num_threads(4)
N_CH, N_PARTNER = 40, 500
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
EVERY = 2000
RATE = 0.01
P_PRIOR = 0.12


def make(cond, seed=0):
    torch.manual_seed(seed)
    prior = Prior(dev="cpu")
    with torch.no_grad():
        prior.b2.fill_(float(np.log(P_PRIOR / (1 - P_PRIOR))))
    a = Area(n_ch=N_CH, prior=prior, n_partner=N_PARTNER, dev="cpu", seed=seed,
             fire_th=14.3, x_gain=1.68, tgt_by_fire=True)
    g = Group2([a])
    g.rep_tgt_own = cond.startswith("B")
    g.exc_nonneg = "0以上" in cond
    return g


def run(cond, seed=0):
    g = make(cond, seed)
    rng = np.random.default_rng(seed)
    log = []
    z = lambda: dict(nh=0, rh=0.0, eh=0.0, nn=0, rn=0.0, en=0.0)
    acc = z()
    t0 = time.time()
    for t in range(STEPS):
        x = rng.integers(0, 4, N_CH).astype(np.float32)
        hit = rng.random(N_CH) < RATE
        x[hit] = rng.integers(9, 11, int(hit.sum()))
        g.step(torch.tensor(x), torch.zeros(0))
        m = torch.tensor(hit)
        rep, err = g.last_rep, g.last_err
        acc["nh"] += int(m.sum()); acc["rh"] += float(rep[m].sum()); acc["eh"] += float(err[m].sum())
        acc["nn"] += int((~m).sum()); acc["rn"] += float(rep[~m].sum()); acc["en"] += float(err[~m].sum())
        if (t + 1) % EVERY == 0:
            we = g.w[~g.e_inh & g.is_rep[g.e_recv]]
            wi = g.w[g.e_inh & g.is_rep[g.e_recv]]
            h, n = max(acc["nh"], 1), max(acc["nn"], 1)
            log.append((t + 1, acc["rh"] / h, acc["eh"] / h, acc["rn"] / n, acc["en"] / n,
                        float(we.mean()), float((we < 0).float().mean()), float(wi.mean()),
                        float(g.scale[g.is_rep].mean())))
            acc = z()
    return log, time.time() - t0


if __name__ == "__main__":
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 else \
        ["版35", "版35・興奮0以上", "B", "B・興奮0以上"]
    for c in conds:
        log, sec = run(c)
        print(f"\n[{c}]  {sec:.0f}秒")
        print("   歩    越えた歩:表現  誤差   越えない歩:表現  誤差   興奮の重み  負の割合  抑制の重み  scale")
        for r in log:
            print(f"  {r[0]:6d}   {r[1]:7.1%}  {r[2]:6.1%}   {r[3]:9.2%}  {r[4]:6.2%}   {r[5]:9.4f}  {r[6]:6.1%}  {r[7]:9.4f}  {r[8]:6.3f}")
        sys.stdout.flush()
