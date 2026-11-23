"""版76 の確かめ：版75 と版76 を、放出のくじを期待値に置き換えた形で並べ、発火・重さが一致するかを見る
（~/anaplex で。版75 のまま・apply_v76.py を置いた状態で。~/anaplex のファイルは書き換えない。CPU で数分）
  python3 check_v76.py [歩=300]
やること
  1 ~/anaplex の .py を2つの仮の場所へ写す（a：版75 のまま ／ b：apply_v76.py を当てる）
  2 どちらも同じ種・小さい網（相手 200）・作り物の声で、CPU で回す
     torch.binomial（短期の増減の放出のくじ）を「数 × 確率」（期待値）に置き換える
     ⇒ 版76 は式を変えていない（スイッチを消した・辺の計算を torch.compile でまとめた）ので、同じ値になるはず
     （CPU の torch.compile は C++ で作る。作れない環境なら「⚠️ torch.compile が … で動かない」と出て、まとめずに比べる）
  3 毎歩の 聴覚・運動 の表現と誤差の発火、最後の下からのシナプスの重さ・短期の増減の状態 を比べる
"""
import os, sys, shutil, subprocess, tempfile, pathlib
import numpy as np

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
HOME = pathlib.Path.cwd()
RUN = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import torch
torch.binomial = lambda count, prob, generator=None: count * prob     # くじを期待値に
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain
B.DEV = "cpu"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
NS = int(sys.argv[3]); rows = []; k = 0
while k < NS:
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(bool(np.any(cmd)))
    for t in range(len(ear)):
        if k >= NS:
            break
        out = b.step(torch.tensor(ear[t]), torch.tensor(cmd[t]), learn=True)
        rows.append(torch.cat([o.flatten() for o in out]).numpy() > 0)
        k += 1
if hasattr(b, "bu_unpack"):
    b.bu_unpack()
keys = sorted(b.bu)
np.savez(sys.argv[2], fire=np.array(rows), keys=np.array(keys),
         w=np.concatenate([b.bu[q].w.numpy() for q in keys]),
         state=np.concatenate([b.bu[q].stp.state.numpy() for q in keys]),
         last=np.concatenate([b.bu[q].stp.last.numpy() for q in keys]),
         t=np.array([b.bu[q].t for q in keys]))
print(f"  {B.VERSION}：{k}歩", flush=True)
'''

if not (HOME / "apply_v76.py").exists():
    sys.exit("⚠️ apply_v76.py が ~/anaplex に無い")
tmp = pathlib.Path(tempfile.mkdtemp(prefix="check_v76_"))
for name in ("a", "b"):
    (tmp / name).mkdir()
    for f in HOME.glob("*.py"):
        shutil.copy(f, tmp / name / f.name)
r = subprocess.run([sys.executable, "apply_v76.py"], cwd=tmp / "b", capture_output=True, text=True)
print(r.stdout.splitlines()[0] if r.stdout else r.stderr, flush=True)
if r.returncode != 0:
    sys.exit("⚠️ 写しに版76 が当たらなかった")
res = {}
for name in ("a", "b"):
    out = tmp / f"{name}.npz"
    p = subprocess.run([sys.executable, "-c", RUN, str(tmp / name), str(out), str(NS)], cwd=HOME)
    if p.returncode != 0:
        sys.exit(f"⚠️ {name}（{'版75' if name == 'a' else '版76'}）が止まった")
    res[name] = np.load(out)
a, b = res["a"], res["b"]
fa, fb = a["fire"], b["fire"]
diff_steps = np.nonzero((fa != fb).any(1))[0]
print(f"発火（聴覚・運動の表現と誤差、{len(fa)}歩 × {fa.shape[1]}本）：違う歩 {len(diff_steps)}"
      + (f"（最初 {diff_steps[0]}歩目・{(fa[diff_steps[0]] != fb[diff_steps[0]]).sum()}本）" if len(diff_steps) else ""), flush=True)
dw = np.abs(a["w"] - b["w"])
print(f"下からのシナプスの重さ {len(a['w']):,}本：差の最大 {dw.max():.2e} ／ 1e-6 を越えた本数 {(dw > 1e-6).sum()}", flush=True)
ds = np.abs(a["state"] - b["state"])
print(f"短期の増減の状態：差の最大 {ds.max():.2e} ／ 最後に撃った歩の違い {(a['last'] != b['last']).sum()}本 ／ 道ごとの歩の数 一致 {bool((a['t'] == b['t']).all())}", flush=True)
ok = len(diff_steps) == 0 and dw.max() < 1e-5 and ds.max() < 1e-5 and (a["last"] == b["last"]).all()
print("⇒ " + ("一致（版76 は 版75 と同じ値）" if ok else "⚠️ 一致しない。当てないで、この出力を貼ってください"), flush=True)
shutil.rmtree(tmp, ignore_errors=True)
