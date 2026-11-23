"""版115 の確かめ（容器・CPU・相手 200）：駆動 x を番号の表1回で組んでも、版114 の組み立てと同じか
  python3 check_v115.py [歩=200]   （子の python を3つ起こす）
  1 ANAPLEX_V115_CHECK=1：毎歩 両方の組み立てを作り、差の最大（x の値の大きさとも並べる）
  2 ANAPLEX_V115=0 と 1 を同じ種で 歩 回：毎歩の発火（表現・誤差・抑制）の違い・終わりの重さ（網・下から・CL）・跡・発火率の差
  （版84 の確かめと同じ型：違い 0・差は丸めの範囲）
"""
import os, sys, subprocess, json

NS = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 200
CHILD = r'''
import sys, json, hashlib, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB
from torch_brain import Brain
B.DEV = "cpu"
NS = int(sys.argv[1]); MODE = sys.argv[2]
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
diffs, xmax, spk = [], [], []
for i in range(NS):
    life.step()
    if MODE == "check":
        diffs.append((getattr(b, "_x115_diff", float("nan")), getattr(b, "_x115_rel", float("nan"))))
    else:
        spk.append(torch.cat([g._any[0], g._any[1], g._any[2]]).to(torch.uint8).numpy())
if MODE == "check":
    print(json.dumps(dict(diffs=diffs, tab=getattr(b, "_x115_tab", None) is not None)))
else:
    np.savez(f"/tmp/chk115_{MODE}.npz", spk=np.stack(spk), w=g.w.numpy(), elig=g.elig.numpy(), bu=b.but.w.numpy(),
             cl=b.cl_w.numpy(), rate=g.rate_ema.numpy())
    print(json.dumps(dict(ok=True)))
'''


def run(mode, env):
    p = subprocess.run([sys.executable, "-c", CHILD, str(NS), mode], env=dict(os.environ, **env), capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not line:
        sys.exit(f"{mode} で止まった：{p.stderr[-1500:]}")
    return json.loads(line[-1])


ok = True


def ck(name, c, extra=""):
    global ok
    ok &= bool(c)
    print(("✓ " if c else "✗ ") + name + extra, flush=True)


import numpy as np
r = run("check", {"ANAPLEX_V115": "1", "ANAPLEX_V115_CHECK": "1"})
if "--only1" in sys.argv:
    d = np.array(r["diffs"], dtype=float)
    print(f"割った差 最大 {np.nanmax(d[:, 1]):.2e}・差そのもの 最大 {np.nanmax(d[:, 0]):.2e}", flush=True)
    sys.exit(0)
d = np.array(r["diffs"], dtype=float)
ck("1 番号の表を使った（初めの歩に作った）", r["tab"])
ck("1 毎歩 x の差（版114 の組み立てと）が float32 の丸めの範囲（値の大きさで割って 1e-5 未満）",
   np.isfinite(d).all() and d[:, 1].max() < 1e-5,
   f"：割った差 最大 {np.nanmax(d[:, 1]):.2e}・中央 {np.nanmedian(d[:, 1]):.2e} ／ 差そのもの 最大 {np.nanmax(d[:, 0]):.2e}（{len(d)} 歩）")
run("v0", {"ANAPLEX_V115": "0"})
run("v1", {"ANAPLEX_V115": "1"})
a, b = np.load("/tmp/chk115_v0.npz"), np.load("/tmp/chk115_v1.npz")
nd = (a["spk"] != b["spk"]).any(1)
ck("2 発火（表現・誤差・抑制）が毎歩同じ", not nd.any(), f"：違う歩 {int(nd.sum())} ／ {len(nd)}（初めに違った歩 {int(np.argmax(nd)) + 1 if nd.any() else '—'}）")
for k in ("w", "elig", "bu", "cl", "rate"):
    m = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())
    s = float(np.abs(a[k]).max())
    ck(f"2 終わりの {k} の差が丸めの範囲（値の大きさの 1e-5 未満）", m <= 1e-5 * max(s, 1e-12), f"：差の最大 {m:.2e}・値の最大 {s:.2e}")
print("全部 ✓" if ok else "✗ あり", flush=True)
