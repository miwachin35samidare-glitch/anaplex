"""焼いた見込み（prior_v1.npz）が、どの状態でどれだけの予測を出すか（arkhe 用。網は組まない）

案2（興奮の重みも 0 以上）では、誤差のニューロンの予測は見込みより下げられない。
見込みが 0.5 を越える状態があると、その状態では誤差のニューロンが撃てなくなる（最小構成で確認、09-26）
⇒ 焼いた見込みがどの状態で 0.5 を越えるかを見る

見るもの
  1 何も撃っていない状態（履歴 0）
  2 1回撃ってから k 歩あと（k ＝ 0〜300）
  3 毎歩 r の率で撃ち続けた定常（r ＝ 0.3%〜100%）
使い方  python3 check_prior.py
"""
import numpy as np
import torch
import torch_area as T

prior = T.Prior.load("prior_v1.npz", dev="cpu")
TAUS = T.TAUS
dec = torch.exp(-1.0 / TAUS)


def p_of(tb):
    return float(torch.sigmoid(prior.logit(tb.unsqueeze(0)))[0])


print("焼いた見込み（prior_v1.npz）")
print(f"  パラメータ {sum(v.numel() for v in prior.parameters())}")
print(f"\n1 何も撃っていない状態          p ＝ {p_of(torch.zeros(len(TAUS))):.3f}")

print("\n2 1回撃ってから k 歩あと")
tb = torch.zeros(len(TAUS))
tb = tb * dec + 1.0 * (1 - dec)
ks = {0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300}
for k in range(301):
    if k in ks:
        print(f"  k ＝ {k:3d}   p ＝ {p_of(tb):.3f}")
    tb = tb * dec

print("\n3 毎歩 r の率で撃ち続けた定常（期待値の履歴）")
for r in (0.003, 0.0045, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    tb = torch.full((len(TAUS),), float(r))
    print(f"  r ＝ {r:6.2%}   p ＝ {p_of(tb):.3f}")

# 0.5 を越える状態があるか（履歴の各成分 0〜1 をでたらめに振って探す）
g = torch.Generator().manual_seed(0)
tbs = torch.rand(200000, len(TAUS), generator=g)
p = torch.sigmoid(prior.logit(tbs))
print(f"\n4 履歴をでたらめに振った 20万通り  p の最小 {float(p.min()):.3f} ／ 中央 {float(p.median()):.3f}"
      f" ／ 最大 {float(p.max()):.3f} ／ 0.5 を越える割合 {float((p > 0.5).float().mean()):.1%}")
