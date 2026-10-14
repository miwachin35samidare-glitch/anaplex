"""版79 の確かめ（arkhe の GPU で。~/anaplex のファイルは書き換えない・保存しない）
  python3 check_v79_gpu.py [歩=100]
グラフそのものは GPU でしか撮れないので、同じ種・同じ声で3回まわして並べる
  a：グラフなし（ANAPLEX_GRAPH=eager：グラフと同じ形で、撮らずに回す）
  b：グラフあり（ANAPLEX_GRAPH=1、ふだん）
  c：グラフなしをもう1回（GPU は足す順が毎回少し違うので、a と c の違いが「丸めの範囲」の目安）
⇒ a と b の違いが、a と c の違いと同じくらいなら、グラフは値を変えていない
   （コンテナでは CPU で版78 と eager が全部一致することを確かめ済み）
"""
import os, sys, subprocess, tempfile, pathlib
import numpy as np

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RUN = r'''
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
NS = int(sys.argv[2]); rows = []; k = 0
while k < NS:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(len(ear)):
        if k >= NS:
            break
        out = b.step(torch.tensor(ear[t], device=B.DEV), torch.tensor(cmd[t], device=B.DEV), learn=True)
        rows.append(torch.cat([o.flatten() for o in out]).cpu().numpy() > 0)
        k += 1
g = b.g
g.flush_all()
np.savez(sys.argv[1], fire=np.array(rows), w=g.w.cpu().numpy(), rate=g.rate_ema.cpu().numpy(),
         pc=np.array(int(g._pc_t)), t=np.array(g.t))
print(f"  ANAPLEX_GRAPH={G.GRAPH}：{k}歩 ／ 撮ったグラフ {getattr(g, 'n_capture', 0)}個 ／ "
      f"箱の大きさ {len(getattr(g, 'gp_hist', {}) or {})}種", flush=True)
'''
import babble
if babble.VERSION < 79:
    sys.exit(f"⚠️ babble.py が版79 より前（版 {babble.VERSION}）")
tmp = pathlib.Path(tempfile.mkdtemp(prefix="check_v79_"))
res = {}
for name, mode in (("a", "eager"), ("b", "1"), ("c", "eager")):
    out = tmp / f"{name}.npz"
    env = dict(os.environ, ANAPLEX_GRAPH=mode)
    p = subprocess.run([sys.executable, "-c", RUN, str(out), str(NS)], cwd=pathlib.Path.cwd(), env=env)
    if p.returncode != 0:
        sys.exit(f"⚠️ {name}（ANAPLEX_GRAPH={mode}）が止まった。この出力を貼ってください")
    res[name] = np.load(out)


def cmp(x, y, label):
    fx, fy = x["fire"], y["fire"]
    d = (fx != fy).any(1)
    first = int(np.argmax(d)) if d.any() else None
    nd = (fx != fy).sum(1)
    dw = np.abs(x["w"] - y["w"])
    dr = np.abs(x["rate"] - y["rate"])
    print(f"{label}：発火の違う歩 {int(d.sum())}/{len(d)}" + (f"（最初 {first}歩目・{int(nd[first])}本 ／ 1歩の平均 {nd.mean():.1f}本）" if first is not None else "")
          + f" ／ 網の重さ 差の最大 {dw.max():.2e}・違う本数 {(dw > 0).sum():,} ／ 発火率の平均 差の最大 {dr.max():.2e}", flush=True)
    return d.sum(), dw.max()


print(f"発火の本数（1歩）{res['a']['fire'].shape[1]:,} ／ 撃った割合 {res['a']['fire'].mean():.4f}", flush=True)
ab = cmp(res["a"], res["b"], "a（なし）と b（グラフ）")
ac = cmp(res["a"], res["c"], "a（なし）と c（なし・もう1回）")
same_clock = all(int(res[k]["pc"]) == int(res["a"]["pc"]) and int(res[k]["t"]) == int(res["a"]["t"]) for k in "bc")
print(f"周の番号・歩の数 一致 {same_clock}", flush=True)
if ab[0] == 0 and ab[1] == 0:
    print("⇒ 一致（グラフは値を変えていない）", flush=True)
elif ab[0] <= max(ac[0], 1) * 2 and same_clock:
    print("⇒ グラフあり・なしの違いは、なし同士の違い（GPU の足す順）と同じくらい", flush=True)
else:
    print("⚠️ グラフあり・なしの違いが、なし同士より大きい。この出力を貼ってください", flush=True)
