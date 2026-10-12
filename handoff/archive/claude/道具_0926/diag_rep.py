"""切り分け：表現のニューロンが下から撃てなくなる原因（版35 の式のまま）

最小構成  聴覚1領野（40帯）。下からの入力は帯ごとの本数（0〜10）。9本以上で越える（重さ 1.68・閾値 14.3、版29）
          越える歩はまれ（人の範囲 1歩 0.5〜2% に合わせて 約1%）
          ⚠️ 素子の見込み（prior）は焼いたもの（prior_v1.npz）ではなく代役。見込みの値そのものに結論を寄せない
条件      中身は版35 のまま。切り分けのために1つずつ止めて比べる（直しではない）
  そのまま
  抑制の学びを止める   抑制の重みを毎歩 0 に戻す
  恒常性を止める       scale を 1 に固定
  見込みを的の率に     代役の見込みの定数項を、的の率（約1%）のロジットに寄せる
見るもの  下から越えた歩のうち、表現が下から撃てた割合 ／ そのときの興奮の合計・抑制の合計・scale
"""
import sys
import time
import numpy as np
import torch
from torch_area import Area, Prior
from torch_group2 import Group2

torch.set_num_threads(2)
N_CH, N_PARTNER, STEPS, EVERY = 40, 500, int(sys.argv[1]) if len(sys.argv) > 1 else 12000, 2000
RATE = 0.01


def make(cond, seed=0):
    torch.manual_seed(seed)
    prior = Prior(dev="cpu")
    if cond == "見込みを的の率に":
        with torch.no_grad():
            prior.b2.fill_(float(np.log(RATE / (1 - RATE))))
    a = Area(n_ch=N_CH, prior=prior, n_partner=N_PARTNER, dev="cpu", seed=seed,
             fire_th=14.3, x_gain=1.68, tgt_by_fire=True)
    g = Group2([a])
    if cond == "恒常性を止める":
        g.scaling_on = False
    if cond == "興奮の重みを0以上に":
        g.exc_nonneg = True
    return g


def run(cond, seed=0):
    g = make(cond, seed)
    rng = np.random.default_rng(seed)
    log = []
    acc = dict(n=0, below=0, exc=0.0, ihn=0.0)
    t0 = time.time()
    for t in range(STEPS):
        x = rng.integers(0, 4, N_CH).astype(np.float32)          # ふだんは 0〜3本（越えない）
        hit = rng.random(N_CH) < RATE
        x[hit] = rng.integers(9, 11, int(hit.sum()))               # まれに 9〜10本（越える）
        g.step(torch.tensor(x), torch.zeros(0))
        if cond == "抑制の学びを止める":
            g.w[g.e_inh] = 0.0
        # 最後の周の値で見る（歩の出力と同じ周）
        m = torch.tensor(hit)
        if m.any():
            acc["n"] += int(m.sum())
            acc["below"] += float(g.last_rep_below[m].sum())
            acc["exc"] += float(g.last_exc[g.is_rep][m].sum())
            acc["ihn"] += float(g.last_ihn[g.is_rep][m].sum())
        if (t + 1) % EVERY == 0:
            n = max(acc["n"], 1)
            rep_scale = float(g.scale[g.is_rep].mean())
            we = g.w[~g.e_inh & g.is_rep[g.e_recv]]
            wi = g.w[g.e_inh & g.is_rep[g.e_recv]]
            log.append((t + 1, acc["below"] / n, acc["exc"] / n, acc["ihn"] / n, rep_scale,
                        float(we.mean()), float((we < 0).float().mean()), float(wi.mean())))
            acc = dict(n=0, below=0, exc=0.0, ihn=0.0)
    return log, time.time() - t0


if __name__ == "__main__":
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 else \
        ["そのまま", "抑制の学びを止める", "恒常性を止める", "見込みを的の率に"]
    for c in conds:
        log, sec = run(c)
        print(f"\n[{c}]  {sec:.0f}秒")
        print("   歩     下から撃てた  興奮の合計  抑制の合計  scale   興奮の重み平均  負の割合  抑制の重み平均")
        for r in log:
            print(f"  {r[0]:6d}   {r[1]:7.1%}   {r[2]:9.3f}  {r[3]:9.3f}  {r[4]:6.3f}   {r[5]:10.4f}   {r[6]:6.1%}   {r[7]:10.4f}")
        sys.stdout.flush()
