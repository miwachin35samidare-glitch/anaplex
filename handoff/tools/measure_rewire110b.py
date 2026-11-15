"""版110：刈り込み（Group2.rewire）の中を、文の並びごとに区切って 時間と VRAM の山を測る（arkhe の ~/anaplex で。保存しない・式は変えない）
  python3 measure_rewire110b.py [回=2] [あいだの歩=300]
  11-14 measure_rewire110：合計 1.72秒のうち、包んだ呼びは 0.35秒・包んでいない所が 1.37秒（山 +2.7GB）⇒ その中を分ける
  やり方：torch_group2 の rewire の元の文（inspect で読む）に、区切りの印 _tk("名前") を差し込んだ写しを作って、それを呼ぶ。
    差し込む場所の文が 1つずつ見つからなければ止まる（式は元の文のまま。足すのは印だけ）
    印は GPU を待ってから 時間・前の印からの使用の山（呼ぶ前の使用からの増え）を覚える
  ⚠️ 印のたびに GPU を待つ分、合計は少し長く出る。毎歩に足す仕事：なし
"""
import sys, time, inspect, textwrap, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_group2 as G
import torch_brain as TB, life as L
from torch_brain import Brain

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 2
GAP = int(args[1]) if len(args) > 1 else 300
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
CUDA = str(dev).startswith("cuda")
if B.VERSION != 110:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：版110 で確かめた道具")
GB = 1024 ** 3
rows = []
ST = {"t": 0.0, "base": 0.0}


def _tk(label):
    if CUDA:
        torch.cuda.synchronize()
    now = time.perf_counter()
    pk = (torch.cuda.max_memory_allocated() / GB - ST["base"]) if CUDA else 0.0
    rs = torch.cuda.max_memory_reserved() / GB if CUDA else 0.0
    rows.append((label, now - ST["t"], pk, rs))
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    ST["t"] = time.perf_counter()


# 区切りの印を差し込む（この文の「前」に入れる）
MARKS = [
    ("        n = self.n_recv_all\n", "A 取り置きを返す・学びの分を足す（flush_all）"),
    ("        R = float(self.rewire_every)\n", "B 受け手ごとの興奮・抑制の本数（bincount 2回）"),
    ("        key = self.w.clone()\n", "C 切るかのくじ・静か／撃ちすぎ"),
    ("        key.uniform_(generator=self.gen)\n", "D 一番弱い興奮の辺（重さの写し＋選び）"),
    ("        cut_e = (do_e | (over & do_i)) & (weak_e >= 0)\n", "E でたらめな抑制の辺（乱数＋選び）"),
    ("        ar = self.area_of[self.e_recv[cuts].long()]\n", "F 切る辺を決める・数える"),
    ("        o = self._order_after_cut(cuts, new)", "G 生やす先を引く（領野ごとの輪）"),
    ("        self.e_src[cuts] = new.to(self.e_src.dtype)\n", "H 差し込みの並び（_order_after_cut）"),
    ("        if self._amp_buf is not None and len(self._amp_buf) == len(self.w):\n", "I 5つの配列を新しい並びへ写す"),
    ("        pos = torch.empty_like(o)\n", "J 抑制の辺の印を作り直す（index_select）ほか"),
    ("        cuts = pos[cuts].long()\n", "K1 新しい位置の表を作る（pos[o] = 番号）"),
    ("        self.w[cuts] = 0.0 ", "K2 切った辺の新しい位置を読む（pos[cuts]）"),
    ("        self.elig[cuts] = 0.0\n", "L 開通の重さ（切った辺 0・抑制の辺に W_INH・抑制の体へ）"),
    ("        if self.stp is not None:\n", "M1 切った辺の学びの跡を 0（elig[cuts]）"),
    ("            self.stp.redraw(cuts,", "M2 短期の並べ直し（stp.reorder、5本）"),
    ("        self.start.copy_(self._starts())", "M3 短期の引き直し（stp.redraw）"),
]
src = textwrap.dedent(inspect.getsource(G.Group2.rewire))
src = src.replace("@torch.no_grad()\n", "", 1)
ind = "    "
for anchor, label in MARKS:
    a = anchor[4:]                                       # dedent で 4 つ減っている
    ind = a[:len(a) - len(a.lstrip())]                   # 差し込む文と同じ字下げ
    if src.count(a) != 1:
        raise SystemExit(f"⚠️ 差し込む場所が {src.count(a)} 個：{a.strip()[:60]}（何も測っていない）")
    src = src.replace(a, f'{ind}_tk("{label}")\n{a}')
src = src.rstrip("\n") + f'\n{ind}_tk("N 始まりの表（_starts）・取り置きを返す")\n'
src = src.replace("def rewire(self):", "def rewire_timed(self):", 1)
ns = G.__dict__
ns["_tk"] = _tk
exec(compile(src, "rewire_timed", "exec"), ns)
rewire_timed = torch.no_grad()(ns["rewire_timed"])

torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
print(f"版 {B.VERSION} ／ 辺 {len(g.w):,} 本・受け手の行 {g.n_recv_all:,}・領野 {len(g.areas)} ／ 刈り込み {N} 回・あいだ {GAP} 歩", flush=True)
for k in range(N):
    for _ in range(GAP):
        life.step(learn=True)
    L.beat()
    if CUDA:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows.clear()
    cut0 = getattr(g, "n_cut", 0)
    ST["base"] = torch.cuda.memory_allocated() / GB if CUDA else 0.0
    rs0 = torch.cuda.memory_reserved() / GB if CUDA else 0.0
    t0 = ST["t"] = time.perf_counter()
    rewire_timed(g)
    tot = time.perf_counter() - t0
    L.beat()
    print(f"\n━━ 刈り込み {k + 1} 回目（{life.t} 歩のあと）：合計 {tot:.3f} 秒・切った辺 {getattr(g, 'n_cut', 0) - cut0:,} 本"
          f"・呼ぶ前の使用 {ST['base']:.2f}GB・取り置き {rs0:.2f}GB ━━", flush=True)
    print(f"  {'区切り':<44s}{'秒':>8s}{'使用の山の増え':>10s}{'取り置きの山':>9s}", flush=True)
    for label, dt, pk, rs in rows:
        print(f"  {label:<44s}{dt:8.3f}{pk:10.2f}{rs:11.2f}", flush=True)
# 小さな試し：900 か所へ値を書くだけの時間（重さ w と学びの跡 elig で。値は書き戻す）
idx = torch.randint(0, len(g.w), (900,), device=dev, generator=g.gen if False else None)
for nm in ("w", "elig"):
    t_ = getattr(g, nm)
    old = t_[idx].clone()
    if CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_[idx] = 0.0
    if CUDA:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    t_[idx] = old
    print(f"［試し］{nm}[900 か所] = 0.0：{dt * 1000:.1f} ms（型 {t_.dtype}・形 {tuple(t_.shape)}・続いた置き場所 {t_.is_contiguous()}）", flush=True)
# 試し2：刈り込みの外で、同じ形の文を1つずつ（値は変えない：同じ並び o ＝ 0,1,2,… で写し、同じ値を書き戻す）
#   K2（pos[cuts] を読む 0.47秒）・M1（elig[cuts] = 0 0.67秒）が、その文そのものの時間か、前の大きな仕事の持ち越しかを分ける


def _t(label, fn, twice=True):
    if CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    t1 = time.perf_counter()                 # 文を出し終えた（GPU は待っていない）
    if CUDA:
        torch.cuda.synchronize()
    t2 = time.perf_counter()                 # GPU を待ち終えた
    if CUDA and twice:
        torch.cuda.synchronize()
    t3 = time.perf_counter()                 # もう1回待った（0 に近いはず）
    print(f"［試し2］{label:<46s} 出す {1000 * (t1 - t0):7.1f} ms・待つ {1000 * (t2 - t1):7.1f} ms・もう1回待つ {1000 * (t3 - t2):6.1f} ms", flush=True)
    return r


E_ = len(g.w)
L.beat()
o_ = _t("並び o を作る（0,1,2,…・32bit）", lambda: torch.arange(E_, dtype=torch.int32, device=dev))
pos_ = _t("pos を置く（empty）", lambda: torch.empty_like(o_))


def _scat():
    pos_[o_] = torch.arange(E_, device=dev, dtype=o_.dtype)


_t("pos[o] = 番号（刈り込みの K1 と同じ文）", _scat)
c_ = _t("pos[900 か所] を読む（K2 の文）", lambda: pos_[idx])
c_ = _t(".long()", lambda: c_.long())
L.beat()


def _del():
    global pos_
    del pos_


_t("del pos", _del)
old_e = g.elig[idx].clone()
_t("elig を並び o で写す（I の文・値は同じ）", lambda: g.elig.copy_(g.elig[o_]))


def _wr():
    g.elig[idx] = old_e


_t("elig[900 か所] に書く（M1 の文・同じ値）", _wr)
_t("もう一度 elig[900 か所] に書く", _wr)
L.beat()
old_w = g.w[idx].clone()
_t("w を並び o で写す", lambda: g.w.copy_(g.w[o_]))


def _ww():
    g.w[idx] = old_w


_t("w[900 か所] に書く（L の文・同じ値）", _ww)
_t("始まりの表 _starts（N の文）", lambda: g._starts())
del o_
L.beat()
if g.stp is not None:
    print("［短期の表］" + "・".join(f"{nm} {tuple(getattr(g.stp, nm).shape)} {getattr(g.stp, nm).dtype}" for nm in ("row", "state", "fac", "last", "deplete")), flush=True)
print("\n（終わり）", flush=True)
