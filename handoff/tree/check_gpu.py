"""GPU でしか動かない形の確かめ（arkhe の GPU で。~/anaplex のファイルは書き換えない・保存しない）
  python3 check_gpu.py 名前 前の値 新しい値 [歩=100]
    例 python3 check_gpu.py ANAPLEX_SCATTER 0 1     （版81：足し込みを Triton の核で）
       python3 check_gpu.py ANAPLEX_GRAPH eager 1    （版79：グラフ。check_v79_gpu と同じ）
同じ種・同じ声で3回まわして並べる
  a：前の値 ／ b：新しい値 ／ c：前の値をもう1回（GPU は足す順が毎回少し違うので、a と c の違いが「丸めの範囲」の目安）
⇒ a と b の違いが、a と c の違いと同じくらいなら、新しい形は値を変えていない
"""
import os, sys, subprocess, tempfile, pathlib
import numpy as np

if len(sys.argv) < 4:
    sys.exit("使い方：python3 check_gpu.py 名前 前の値 新しい値 [歩=100]")
VAR, OLD, NEW = sys.argv[1:4]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
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
import os
print(f"  {sys.argv[3]}={os.environ.get(sys.argv[3])}：{k}歩 ／ 撮ったグラフ {getattr(g, 'n_capture', 0)}個 ／ "
      f"箱の大きさ {len(getattr(g, 'gp_hist', {}) or {})}種", flush=True)
'''
import babble
if babble.VERSION < 79:
    sys.exit(f"⚠️ babble.py が版79 より前（版 {babble.VERSION}）")
tmp = pathlib.Path(tempfile.mkdtemp(prefix="check_v79_"))
res = {}
for name, mode in (("a", OLD), ("b", NEW), ("c", OLD)):
    out = tmp / f"{name}.npz"
    env = dict(os.environ, **{VAR: mode})
    p = subprocess.run([sys.executable, "-c", RUN, str(out), str(NS), VAR], cwd=pathlib.Path.cwd(), env=env)
    if p.returncode != 0:
        sys.exit(f"⚠️ {name}（{VAR}={mode}）が止まった。この出力を貼ってください")
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
ab = cmp(res["a"], res["b"], f"a（{OLD}）と b（{NEW}）")
ac = cmp(res["a"], res["c"], f"a（{OLD}）と c（{OLD}・もう1回）")
same_clock = all(int(res[k]["pc"]) == int(res["a"]["pc"]) and int(res[k]["t"]) == int(res["a"]["t"]) for k in "bc")
print(f"周の番号・歩の数 一致 {same_clock}", flush=True)
if ab[0] == 0 and ab[1] == 0:
    print("⇒ 一致（新しい形は値を変えていない）", flush=True)
elif ab[0] <= max(ac[0], 1) * 2 and same_clock:
    print("⇒ 新しい形との違いは、前の形同士の違い（GPU の足す順）と同じくらい", flush=True)
else:
    print("⚠️ 新しい形との違いが、前の形同士より大きい。この出力を貼ってください", flush=True)
