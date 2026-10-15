"""版92 を当てる前に：組む途中と刈り込みの結果が版91 と1ビットも違わないか ＋ GPU のメモリの山（arkhe の ~/anaplex で。版91 のまま回す）
  python3 check_v92_gpu.py
  （コンテナ：python3 check_v92_gpu.py --test ⇒ CPU・相手 200）
~/anaplex のファイルは書き換えない：/tmp に作業場を2つ作る
  a ＝ 版91 のまま（~/anaplex へ繋ぐだけ）／ b ＝ babble・torch_group2・torch_area を写して apply_v92 を当てたもの ／ c ＝ a をもう1回（毎回同じになるかの目安）
それぞれの中で
  1 同じ種で網を組む ⇒ 辺の配列（e_src・e_recv・w・e_inh・start・短期可塑性5本）の指紋（sha256）
  2 w を決まった乱数（0・0.25・0.5・0.75・1 の同点だらけ）、発火の長い平均を目標の 0.5〜1.5倍にして、刈り込みを3回 ⇒ 切った本数と辺の配列の指紋
  3 GPU の山：組む途中（使った量の山）・刈り込みの増え
⇒ a・b・c の指紋が全部同じなら、版92 は値を変えていない
"""
import os, sys, subprocess, tempfile, shutil, json

HERE = os.path.dirname(os.path.realpath(__file__))
TEST = "--test" in sys.argv
CHILD = r'''
import sys, json, hashlib, numpy as np, torch
import babble as B, torch_area as T
TEST = sys.argv[1] == "1"
if TEST:
    B.DEV = "cpu"
dev = B.DEV; gpu = dev == "cuda"
from torch_brain import Brain
GB = 1024 ** 3
def fp(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
def arrays(g):
    d = {k: fp(getattr(g, k)) for k in ("e_src", "e_recv", "w", "e_inh", "start", "elig", "k_start")}
    if g.stp is not None:
        d.update({"stp." + k: fp(getattr(g.stp, k)) for k in ("row", "state", "fac", "last", "deplete")})
    return d
torch.manual_seed(0)
if gpu:
    torch.cuda.reset_peak_memory_stats()
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
g = b.g
out = {"版": B.VERSION, "辺": len(g.w), "組んだ": arrays(g)}
if gpu:
    torch.cuda.synchronize()
    out["組む途中の山"] = torch.cuda.max_memory_allocated() / GB
gen = torch.Generator(device=dev); gen.manual_seed(7)
with torch.no_grad():
    g.w.copy_(torch.randint(0, 5, (len(g.w),), device=dev, generator=gen).float() / 4)
    if gpu:
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
    # 静か・撃ちすぎの両方を通すため、長い発火の平均を目標の 0.5〜1.5倍に（決まった乱数）
    g.rate_long.copy_(g.target_recv * (0.5 + torch.rand(len(g.rate_long), device=dev, generator=gen)))
    for _ in range(3):
        g.rewire()
    if gpu:
        torch.cuda.synchronize()
        out["刈り込みの増え"] = (torch.cuda.max_memory_allocated() - m0) / GB
out["切った"] = int(getattr(g, "n_cut", -1))
out["刈り込みのあと"] = arrays(g)
print("JSON" + json.dumps(out, ensure_ascii=False), flush=True)
'''


def workdir(tag, patch):
    d = tempfile.mkdtemp(prefix=f"anaplex_v92{tag}_")
    for f in os.listdir(HERE):
        if f.startswith(".") or f in ("state.pt", "state.pt.tmp"):
            continue
        if patch and f in ("babble.py", "torch_group2.py", "torch_area.py", "apply_v92.py"):
            shutil.copy2(os.path.join(HERE, f), os.path.join(d, f))
        else:
            os.symlink(os.path.join(HERE, f), os.path.join(d, f))
    if patch:
        if not os.path.exists(os.path.join(d, "apply_v92.py")):
            sys.exit("⚠️ apply_v92.py が ~/anaplex に無い")
        r = subprocess.run([sys.executable, "apply_v92.py"], cwd=d, capture_output=True, text=True)
        if "✅" not in r.stdout:
            sys.exit("⚠️ 作業場で apply_v92 が当たらなかった：\n" + r.stdout + r.stderr)
    with open(os.path.join(d, "_child92.py"), "w") as f:
        f.write(CHILD)
    return d


res = {}
dirs = []
try:
    for tag, patch in (("a", False), ("b", True), ("c", False)):
        d = workdir(tag, patch); dirs.append(d)
        r = subprocess.run([sys.executable, "_child92.py", "1" if TEST else "0"], cwd=d, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("JSON")]
        if not line:
            sys.exit(f"⚠️ {tag} が止まった：\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        res[tag] = json.loads(line[0][4:])
        x = res[tag]
        mem = (f" ／ 組む途中の山 {x['組む途中の山']:.2f}GB・刈り込みの増え {x['刈り込みの増え']:.2f}GB"
               if "組む途中の山" in x else "")
        print(f"{tag}（版 {x['版']}）：辺 {x['辺']:,} ／ 切った {x['切った']:,}{mem}", flush=True)
finally:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

bad = 0
for part in ("組んだ", "刈り込みのあと"):
    for k in res["a"][part]:
        va, vb, vc = res["a"][part][k], res["b"][part][k], res["c"][part][k]
        if not (va == vb == vc):
            bad += 1
            print(f"  ⚠️ {part} {k}：a {va} ／ b {vb} ／ c {vc}", flush=True)
if res["a"]["切った"] != res["b"]["切った"]:
    bad += 1
print("✅ a・b・c の指紋が全部同じ ⇒ apply_v92.py を当ててよい" if bad == 0 else f"⚠️ 違い {bad} 個 ⇒ 当てずに結果を Claude へ", flush=True)
