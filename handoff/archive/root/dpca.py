"""dPCA で分ける（人と同じ測り方）
【ヒト】BrainGate では、成分ごとに「その中の何%が語によらないか」を見る
  一番大きい成分の 98.7%（患者T5）／ 87.3%（患者T8）が語によらない
  8つの成分で全体の 45.1%（T5）／ 8.4%（T8）を説明
"""
import numpy as np


def dpca(X, n_comp=8):
    """X [語, 歩, 体] → 成分ごとの (全体に占める割合, その中の語によらない割合)"""
    X = X - X.reshape(-1, X.shape[2]).mean(0)
    ci = X.mean(0, keepdims=True).repeat(X.shape[0], 0)   # 語によらない部分
    cd = X - ci                                            # 語ごとの部分
    tot = (X ** 2).sum()
    out = []
    for part, name in ((ci, "語によらない"), (cd, "語ごと")):
        M = part.reshape(-1, X.shape[2])
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        for i in range(min(n_comp, len(S))):
            ax = Vt[i]
            proj_all = (X.reshape(-1, X.shape[2]) @ ax)
            var_all = (proj_all ** 2).sum()
            var_ci = ((ci.reshape(-1, X.shape[2]) @ ax) ** 2).sum()
            out.append((name, i, var_all / tot, var_ci / max(var_all, 1e-12)))
    out.sort(key=lambda r: -r[2])
    return out
