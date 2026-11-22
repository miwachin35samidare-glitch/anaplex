"""版115 の GPU での確かめ（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 check_v115_gpu.py [歩=100]   （子の python を3つ順に起こす：版114 の組み立て・もう1回・版115）
  check_v79_gpu と同じ型：同じ種で組んで 歩 回し、毎歩の発火（領野ごとの撃った体の数）と終わりの重さを並べる。
  GPU は足し込みの順が回ごとに揺れる ⇒ 「版114 の組み立て同士」の差を丸めの目安にして、版115 との差がその範囲かを見る
  毎歩に足す仕事：領野ごとの撃った体の数を読む（累積和 1回・境の読み 1回）。少ない箱への足し込みなし
"""
import os, sys, subprocess, json
import numpy as np

TEST = "--test" in sys.argv                          # 容器で道具が回るかだけ（CPU・相手 200）
_a = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(_a[0]) if _a else 100
CHILD = r'''
import sys, json, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, life as L
from torch_brain import Brain, NAMES
NS, OUT, TEST = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
if TEST:
    B.DEV = "cpu"
sync = torch.cuda.synchronize if str(B.DEV).startswith("cuda") else (lambda: None)
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
oc = torch.tensor([int(x) for x in b.off_ch], device=B.DEV)
z = torch.zeros(1, dtype=torch.float64, device=B.DEV)
rows = []
for i in range(NS):
    life.step()
    c = torch.cat([z, g._any[0][:g.n_ch_all].double().cumsum(0)])
    rows.append((c[oc[1:]] - c[oc[:-1]]).cpu().numpy())
    sync()
    if (i + 1) % 20 == 0:
        L.beat()
idx = torch.arange(0, len(g.w), 997, device=B.DEV)
np.savez(OUT, spk=np.stack(rows), w=g.w[idx].cpu().numpy(), bu=b.but.w[::97].cpu().numpy(), cl=b.cl_w.cpu().numpy(),
         rate=g.rate_ema.cpu().numpy(), x115=bool(getattr(b, "_x115_tab", None) is not None))
print(json.dumps(dict(ok=True)), flush=True)
'''


def run(tag, env):
    out = f"/tmp/chk115gpu_{tag}.npz"
    p = subprocess.run([sys.executable, "-c", CHILD, str(NS), out, "1" if TEST else "0"], env=dict(os.environ, **env), capture_output=True, text=True)
    if not any(l.startswith("{") for l in p.stdout.splitlines()):
        sys.exit(f"{tag} で止まった：{p.stderr[-1500:]}")
    print(f"  {tag} 済み", flush=True)
    return np.load(out)


print(f"版115 の GPU の確かめ：{NS} 歩 × 3", flush=True)
a0 = run("版114の組み立て", {"ANAPLEX_V115": "0"})
a1 = run("版114の組み立て・もう1回", {"ANAPLEX_V115": "0"})
c = run("版115", {"ANAPLEX_V115": "1"})


def cmp(x, y):
    d = (x["spk"] != y["spk"]).any(1)
    first = int(np.argmax(d)) + 1 if d.any() else None
    out = {"違う歩": int(d.sum()), "初めに違った歩": first}
    for k in ("w", "bu", "cl", "rate"):
        out[k] = float(np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)).max())
    return out


base, new = cmp(a0, a1), cmp(a0, c)
print(f"\n使った組み立て：版114 {bool(a0['x115'])}・{bool(a1['x115'])} ／ 版115 {bool(c['x115'])}（版115 だけ True のはず）", flush=True)
print(f"  {'':<22s} {'発火の違う歩':>10s} {'初め':>6s} {'網の重さ':>10s} {'下から':>10s} {'CL':>10s} {'率':>10s}", flush=True)
for nm, r in (("版114 同士（丸めの目安）", base), ("版114 と 版115", new)):
    print(f"  {nm:<22s} {r['違う歩']:>10d} {str(r['初めに違った歩']):>6s} {r['w']:10.2e} {r['bu']:10.2e} {r['cl']:10.2e} {r['rate']:10.2e}", flush=True)
ok = (not a0["x115"]) and (not a1["x115"]) and bool(c["x115"])
ok &= new["違う歩"] <= max(base["違う歩"], 0) or (base["初めに違った歩"] is not None and new["初めに違った歩"] is not None
                                                  and new["初めに違った歩"] >= base["初めに違った歩"] * 0.5)
print(("✓" if ok else "⚠️") + " 版115 の差は 版114 同士の揺れと同じくらいか（違う歩が多いときは 初めに違った歩 を並べて読む）", flush=True)
