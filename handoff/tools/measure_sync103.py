"""版103 の速さの段 ①：1歩の中で CPU が GPU を待つ場所（版54b で一番効いた手の測り。元は measure_sync.py 版54）
  python3 measure_sync103.py [歩=20]      （arkhe の ~/anaplex で。保存しない・網は書き換えない）
  過去に効いた順【測】40 §1：
    版54b  CPU が GPU を待つ所 230 → 22回/歩 ⇒ 37.4 → 25.0ms（値も足す順も同じ）
    版78〜90 形を毎歩同じにして HIP グラフ・Triton で1つの核に ⇒ 23.2 → 9.5ms
    版95b  足し込みの渋滞を受け手の順の累積和に ⇒ 256 → 25.7ms
  今（版103b）はグラフを撮らなくても 66.8ms とほぼ同じ（measure_prof103）＝ 待ちでグラフの外が詰まっている疑い ⇒ まず待つ場所を数える
並べるもの
  1 待つ場所（ファイル:行・中身・1歩あたりの回数）多い順 ／ 2 その行を呼んだ自分のコードの行（1つ上）
"""
import sys, collections, linecache, warnings, traceback, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
from torch_brain import Brain

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 20
if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(60):                      # 温め（compile・グラフ）
    life.step()

MINE = ("torch_brain", "torch_group2", "torch_area", "brainstem", "life", "babble", "mouth", "voice")
cnt, caller = collections.Counter(), collections.defaultdict(collections.Counter)
orig = warnings.showwarning


def show(message, category, filename, lineno, file=None, line=None):
    key = (filename, lineno)
    cnt[key] += 1
    st = [f for f in traceback.extract_stack()[:-1] if any(m in f.filename for m in MINE)]
    if len(st) >= 2:
        up = st[-2] if (st[-1].filename, st[-1].lineno) == key else st[-1]
        caller[key][f"{up.filename.split('/')[-1]}:{up.lineno}"] += 1


warnings.showwarning = show
torch.cuda.set_sync_debug_mode(1)
with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.showwarning = show
    for _ in range(NS):
        life.step()
torch.cuda.set_sync_debug_mode(0)
warnings.showwarning = orig
out = [f"待つ場所（版 {B.VERSION}・暮らし {NS}歩、多い順）合計 {sum(cnt.values()) / NS:.0f}回/歩"]
for (f, ln), c in cnt.most_common(40):
    src = linecache.getline(f, ln).strip()
    up = "・".join(f"{k}×{v // NS if v >= NS else v}" for k, v in caller[(f, ln)].most_common(2))
    out.append(f"  {c / NS:6.1f}回/歩  {f.split('/')[-1]}:{ln}  {src[:100]}" + (f"  ← {up}" if up else ""))
txt = "\n".join(out)
print(txt, flush=True)
open("data_sync103.txt", "w").write(txt + "\n")
print("（data_sync103.txt に書いた）", flush=True)
