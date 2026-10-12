"""最小の随伴放電：案2（表現は下からだけ ＋ 重みは 0 以上）で、誤差のニューロンが自分の声で黙るか

最小構成  聴覚1領野（40帯）＋ 指令の写し 8本（樹状突起へ）
  音節 8種。音節ごとに決まった5帯が、4歩のあいだ 9〜10本そろって越える（下から越え）
  自分で言う：写しが音の 5歩前に1回撃つ ／ 録音で聞く：同じ音、写しなし
  間は 30〜70歩。越えない歩の帯は 0〜3本
学ぶ   自分で言うだけ（喃語）を N 回
測る   学んだあと、自分で言う・録音で聞く を各 200回。越えた歩のうち誤差が撃った割合と supp（録音 ÷ 自分）
       表現が下から撃てた割合も
⚠️ 素子の見込みは代役（焼いたものではない）。見込みが高い場合の条件も置く
"""
import sys
import time
import numpy as np
import torch
from torch_area import Area, Prior
from torch_group2 import Group2

torch.set_num_threads(1)
N_CH, N_CMD, N_SYL, LAT = 40, 8, 8, 5


def make(cond, seed=0):
    torch.manual_seed(seed)
    prior = Prior(dev="cpu")
    if "見込み高" in cond:
        with torch.no_grad():
            prior.b2.fill_(2.0)
    a = Area(n_ch=N_CH, prior=prior, n_ext=N_CMD, n_partner=500, dev="cpu", seed=seed,
             fire_th=14.3, x_gain=1.68, tgt_by_fire=True)
    g = Group2([a])
    if "案2" in cond:
        g.rep_below_only = True
        g.exc_nonneg = True
    return g


def utterance(rng, syl_bands, own):
    s = int(rng.integers(N_SYL))
    gap = int(rng.integers(30, 70))
    T = gap + LAT + 4
    x = rng.integers(0, 4, (T, N_CH)).astype(np.float32)
    ext = np.zeros((T, N_CMD), np.float32)
    on = gap + LAT
    for t in range(on, on + 4):
        x[t, syl_bands[s]] = rng.integers(9, 11, len(syl_bands[s]))
    if own:
        ext[gap, s] = 1.0
    hit = np.zeros((T, N_CH), bool)
    hit[on:on + 4, syl_bands[s]] = True
    return x, ext, hit


def play(g, rng, syl_bands, own, learn):
    x, ext, hit = utterance(rng, syl_bands, own)
    n_hit = err_hit = rep_hit = 0
    for t in range(len(x)):
        rep, err, _ = g.step(torch.tensor(x[t]), torch.tensor(ext[t]), learn=learn)
        h = torch.tensor(hit[t])
        if h.any():
            n_hit += int(h.sum())
            err_hit += float(err[h].sum())
            rep_hit += float(g.last_rep_below[h].sum())
    return n_hit, err_hit, rep_hit


def run(cond, n_train, seed=0):
    g = make(cond, seed)
    rng = np.random.default_rng(seed)
    syl_bands = [np.sort(rng.choice(N_CH, 5, replace=False)) for _ in range(N_SYL)]
    t0 = time.time()
    for _ in range(n_train):
        play(g, rng, syl_bands, True, True)
    res = {}
    for own in (True, False):
        r = np.random.default_rng(1000)
        tot = [0, 0.0, 0.0]
        for _ in range(100):
            n, e, rp = play(g, r, syl_bands, own, False)
            tot[0] += n; tot[1] += e; tot[2] += rp
        res[own] = (tot[1] / max(tot[0], 1), tot[2] / max(tot[0], 1))
    return res, time.time() - t0


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["版35", "案2", "版35・見込み高", "案2・見込み高"]
    for c in conds:
        res, sec = run(c, n_train)
        (eo, ro), (er, rr) = res[True], res[False]
        print(f"[{c}] 喃語 {n_train}回  誤差 自分 {eo:.1%} ／ 録音 {er:.1%}  supp {er/max(eo,1e-9):.2f}"
              f"   表現が下から撃てた 自分 {ro:.1%} ／ 録音 {rr:.1%}   {sec:.0f}秒", flush=True)
