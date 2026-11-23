"""版115b の確かめ（容器・CPU・相手 200）：PPN → CL の速い分・ゆっくりの分
  python3 check_v115b.py [歩=40] [版115 の木]
  1 跡の式：ゆっくりの分を1発（体0）と一定（体1）で入れ、立ち上がりの歩・減りの時定数・落ち着いた値を式と比べる（網を組まない）
  2 既定（ANAPLEX_PPNCL_FAST なし）で 版115 の木と 歩 回 同じ（発火・重さ）
  3 ANAPLEX_PPNCL_FAST=0.5 で 歩 回る・跡が進む・値が有限
"""
import os, sys, subprocess, json, types
import numpy as np, torch

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OLD = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/w/v115"
ok = True


def ck(name, c, extra=""):
    global ok
    ok &= bool(c)
    print(("✓ " if c else "✗ ") + name + extra, flush=True)


# ── 1 跡の式 ──
os.environ["ANAPLEX_PPNCL_FAST"] = "0.5"
import torch_brain as TB
f_, g_, w_ = TB.PPNCL_FAST, TB.PPNCL_G, TB.PPNCL_M_W
fake = types.SimpleNamespace()
T_ = 2000
outs = []
for t in range(T_):
    u = torch.zeros(2)
    if t == 0:
        u[0] = 1.0
    u[1] = 1.0
    outs.append(TB.Brain._ppncl_slow(fake, u).clone())
o = torch.stack(outs).numpy()
dn, dm = TB.PPNCL_DLY
first = int(np.argmax(o[:, 0] > 0))
an, am = fake._pc_a
k = (1 - f_) * g_ / (1 + w_)
ck("1 1発の立ち上がりはニコチンの遅れの歩", first == dn, f"：{first} 歩（式 {dn}）")
ck("1 その歩の値は (1−a_n)・k", abs(o[dn, 0] - (1 - an) * k) < 1e-6, f"：{o[dn, 0]:.3e}（式 {(1 - an) * k:.3e}）")
ck("1 ムスカリンの分は遅れの歩に足される", abs((o[dm, 0] - o[dm - 1, 0] * an) - (1 - am) * w_ * k) < 1e-5 * k + 1e-9,
   f"：{dm} 歩")
r = o[dn + 1:dm - 1, 0] / o[dn:dm - 2, 0]
ck("1 ニコチンの分の減りは exp(−1歩／0.43秒)", np.allclose(r, an, rtol=1e-4), f"：比 {r.mean():.5f}（式 {an:.5f}）")
ck("1 一定の入力の落ち着いた値は (1−f)・g（強さの平均を変えない）",
   abs(o[-1, 1] - (1 - f_) * g_ * (1 + w_ * (1 - am ** (T_ - dm))) / (1 + w_)) < 1e-4,
   f"：{o[-1, 1]:.4f}（ずっと先で {(1 - f_) * g_:.4f}）")

CHILD = r'''
import sys, json, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB
from torch_brain import Brain
B.DEV = "cpu"
NS, OUT = int(sys.argv[1]), sys.argv[2]
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
spk = []
for i in range(NS):
    life.step()
    spk.append(torch.cat([g._any[0], g._any[1], g._any[2]]).to(torch.uint8).numpy())
np.savez(OUT, spk=np.stack(spk), w=g.w.numpy(), bu=b.but.w.numpy(), cl=b.cl_w.numpy(), rate=g.rate_ema.numpy(),
         pc_i=getattr(b, "_pc_i", -1), pcn=(b._pc_n.numpy() if getattr(b, "_pc_n", None) is not None else np.zeros(1)),
         pcm=(b._pc_m.numpy() if getattr(b, "_pc_m", None) is not None else np.zeros(1)))
print(json.dumps(dict(ok=True)))
'''


def run(cwd, out, env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("ANAPLEX_PPNCL")}
    e.update(env)
    p = subprocess.run([sys.executable, "-c", CHILD, str(NS), out], cwd=cwd, env=e, capture_output=True, text=True)
    if not any(l.startswith("{") for l in p.stdout.splitlines()):
        sys.exit(f"{cwd} {env} で止まった：{p.stderr[-1500:]}")
    return np.load(out)


here = os.path.dirname(os.path.abspath(__file__))
a = run(OLD, "/tmp/chk115b_old.npz", {})
b = run(here, "/tmp/chk115b_def.npz", {})
c = run(here, "/tmp/chk115b_slow.npz", {"ANAPLEX_PPNCL_FAST": "0.5"})
nd = (a["spk"] != b["spk"]).any(1)
ck("2 既定は版115 と発火が毎歩同じ", not nd.any(), f"：違う歩 {int(nd.sum())} ／ {NS}")
for k in ("w", "bu", "cl", "rate"):
    ck(f"2 既定は版115 と終わりの {k} が同じ", np.array_equal(a[k], b[k]))
ck("3 FAST=0.5 で回り、跡が歩の数だけ進む", int(c["pc_i"]) == NS, f"：{int(c['pc_i'])} 歩")
ck("3 跡と重さが有限", all(np.isfinite(c[k]).all() for k in ("pcn", "pcm", "w", "bu", "cl", "rate")),
   f"：ニコチンの跡 平均 {c['pcn'].mean():.3e}・ムスカリン {c['pcm'].mean():.3e}")
nd2 = (a["spk"] != c["spk"]).any(1)
print(f"  （参考）FAST=0.5 で版115 と発火が違う歩 {int(nd2.sum())} ／ {NS}（初め {int(np.argmax(nd2)) + 1 if nd2.any() else '—'} 歩）", flush=True)
print("全部 ✓" if ok else "✗ あり", flush=True)
