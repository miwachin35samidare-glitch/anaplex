"""版114c の確かめ（容器・CPU・相手 200）。ANAPLEX_CX_TC と ANAPLEX_KCL を変えて網を組み、作りの違いを数える
  python3 check_v114c.py          （子の python を4つ起こす：既定・drive・drive103f・KCL 10）
"""
import os, sys, subprocess, json

CHILD = r'''
import json, hashlib, torch, numpy as np
import babble as B, torch_area as T, torch_brain as TB
from torch_brain import Brain, NAMES
B.DEV = "cpu"
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
g = b.g
h = lambda t: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]
cx = [n for n in TB.CX_TC]
ms = g.mod_send
cxm, cxn = 0, 0
for nm in cx:
    i = NAMES.index(nm); a = b.areas[nm]; per = a.ch.n_cell + a.ch.n_ramp
    lo = int(g.off_send[i]); cxm += int(ms[lo:lo + (a.n_ch + a.n_ext) * per].sum()); cxn += (a.n_ch + a.n_ext) * per
nr = len(b.cl_ch); k = len(b.cl_w) // nr
s_ = b.cl_src.view(nr, k).sort(1).values
print(json.dumps(dict(edges=h(g.e_src) + h(g.e_recv), w=h(g.w), mod_all=int(ms.sum()), mod_cx=cxm, rows_cx=cxn,
      k=k, n_cl_lines=len(b.cl_w), nr=nr, distinct=bool((s_[:, 1:] != s_[:, :-1]).all()) if k > 1 else True,
      tc_mod=len(TB.TC_MOD), tc_inh=len(TB.TC_INH))))
'''

runs = {"既定": {}, "drive": {"ANAPLEX_CX_TC": "drive"}, "drive103f": {"ANAPLEX_CX_TC": "drive103f"}, "KCL10": {"ANAPLEX_KCL": "10"}}
R = {}
for k, e in runs.items():
    p = subprocess.run([sys.executable, "-c", CHILD], env=dict(os.environ, **e), capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not line:
        sys.exit(f"{k} で止まった：{p.stderr[-800:]}")
    R[k] = json.loads(line[-1])
    print(k, R[k], flush=True)

ok = True
def ck(name, c, extra=""):
    global ok
    ok &= bool(c)
    print(("✓ " if c else "✗ ") + name + extra, flush=True)

d, v, f, c = R["既定"], R["drive"], R["drive103f"], R["KCL10"]
ck("1 drive：網の辺・初めの重さは既定と同じ（抑制の体の相手の取り方は変えない）", v["edges"] == d["edges"] and v["w"] == d["w"])
ck("2 既定：皮質の時間細胞・傾きの細胞の送り手の行は全部 調整の印", d["mod_cx"] == d["rows_cx"], f"：{d['mod_cx']:,} ／ {d['rows_cx']:,} 行")
ck("3 drive：その行のうち調整の印が外れ、外れた数は網全体の減りと同じ（ほかの領野の印は変わらない）",
   0 < d["mod_cx"] - v["mod_cx"] == d["mod_all"] - v["mod_all"],
   f"：外れた {d['mod_cx'] - v['mod_cx']:,} 行・残る {v['mod_cx']:,} 行（drive103f でも調整のままの行）")
ck("4 drive103f：調整の印は drive と同じ・抑制の体の相手は前の作り（辺が既定と違う）", f["mod_all"] == v["mod_all"] and f["edges"] != d["edges"])
ck("5 KCL 10：皮質の表現の体1つに CL の線 10本・別々の体から", c["k"] == 10 and c["n_cl_lines"] == c["nr"] * 10 and c["distinct"],
   f"：受け手 {c['nr']:,}・線 {c['n_cl_lines']:,}")
ck("6 既定：CL 40本", d["k"] == 40)
print("全部 ✓" if ok else "✗ あり", flush=True)
