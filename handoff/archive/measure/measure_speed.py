"""1歩を縮める直しを、意味を変えずに入れられるかを試す（arkhe の ~/anaplex で。ファイルは書き換えない・保存しない）
  python3 measure_speed.py [喃語の回数=3]
  （コンテナで道具を確かめるとき：python3 measure_speed.py 1 --test ⇒ CPU・相手 200）
【み】10-07「6.4倍を縮められないか。以前短縮したときの手法を入れ忘れていたりしないか確認して」
前に縮めたとき（決まったこと 09-11）の手法と、今のコードを突き合わせた
  済  送り手ごとに届け先を連ねて持ち、撃った送り手の辺だけ触る（sparse.py → Group2.start）
  済  短期可塑性は最後に触った歩からまとめて減らす（LazyDecay → STP.release_at）
  済  領野をまとめて1回で処理する（group.py → Group2）
  済  時間細胞の発火の表は1歩に1回だけ作る（Group2._base）
  済  CA3 の外側の引き込みのループは外したまま
  漏れ  「1歩は呼び出しの回数（と待ち合わせ）が支配」（09-11）なのに、torch に移したとき次の待ち合わせが入った
    A  Group2 の領野の境目（off_ch・off_src・off_recv）が GPU の配列。周回ごとに int() で読むたび GPU を待つ
       1周 15領野 × 2回 × 10周 ＋ 歩の終わり 約60回 ＝ 1歩 約360回（numpy の group.py では CPU の数だった）
    B  if self.complete.any()：周回ごとに GPU を待つ（1歩 10回）
    C  撃った辺の一覧を、周回ごとに最初から作り直す（時間細胞と buf の分は1歩のあいだ変わらないのに）
       ⇒ 1周目に作った一覧を残し、2周目からは周回の中で新しく撃った送り手の辺だけ足す
【測】10-07 arkhe：もと 66.8 ／ AB 61.7 ／ ABC 60.1ms（1割しか縮まない）。もと＝もと2 は一致、AB は 150歩目から違った
  ⇒ 2つ目の版：AB を2回（毎回同じか）／ 足し算の順を決まったものにして もと と AB を比べる（中身が同じか）／ どこに時間を使っているか
やること（初版）：同じ乱数・同じ声（作り物）で、もとのまま ／ もとのまま2回目（GPU の足し算の順のゆらぎを見る）／ A＋B ／ A＋B＋C を
  別々のプロセスで回し、1歩の時間と、撃った体（表現・誤差）が もとのまま と何本違うかを出す
  A・B は中身が同じなので一致するはず。C は辺の並びが変わるので足し算の順だけ変わる（もとのまま2回目と同じ程度なら意味は同じ）
"""
import sys, os, time, subprocess, tempfile, numpy as np

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 3
MODES = ("もと", "もと2", "AB", "AB2", "ABC", "もと決", "AB決", "ABC決")


def child(mode, out_path):
    import torch
    if mode.endswith("決"):
        torch.use_deterministic_algorithms(True, warn_only=True)   # 足し算の順を決まったものに（遅くなってよい）
    src = open("torch_group2.py", encoding="utf-8").read()
    if mode.startswith("AB"):
        old = "        if self.complete.any():           # 入力が無いところは予測が足しになる（撃たせはしない）\n"
        if src.count(old) != 1:
            sys.exit("⚠️ torch_group2.py に complete.any() の行が1つではない。止める")
        src = src.replace(old, "        if self._complete_any:           # 入力が無いところは予測が足しになる（撃たせはしない）\n")
    m = type(sys)("torch_group2"); m.__file__ = "torch_group2.py"; sys.modules["torch_group2"] = m
    exec(compile(src, "torch_group2.py", "exec"), m.__dict__)
    import babble as B, brainstem as BS, torch_area as T
    from torch_brain import Brain
    if TEST:
        B.DEV = "cpu"
    dev = B.DEV
    torch.manual_seed(0)
    prior = T.Prior.load("prior_v1.npz", dev=dev)
    b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
    b.bs.set_state(BS.WAKE)
    g = b.g
    if mode.startswith("AB"):
        g.off_ch, g.off_src, g.off_recv = (g.off_ch.cpu().numpy(), g.off_src.cpu().numpy(), g.off_recv.cpu().numpy())
        g._complete_any = bool(g.complete.any())
    if mode.startswith("ABC"):
        G2 = m.Group2

        def expand(self, fired):
            empty = torch.empty(0, dtype=torch.long, device=self.dev)
            if len(fired) == 0:
                return empty
            lens = self.start[fired + 1] - self.start[fired]
            keep = lens > 0
            fired, lens = fired[keep], lens[keep]
            if len(fired) == 0:
                return empty
            base = self.start[fired].repeat_interleave(lens)
            off = torch.arange(int(lens.sum()), device=self.dev) - torch.cat(
                [torch.zeros(1, dtype=torch.long, device=self.dev), lens.cumsum(0)[:-1]]).repeat_interleave(lens)
            return base + off

        def fired_edges(self):
            if self._e_cache is not None:
                return self._e_cache
            if self._base is None or getattr(self, "_base_of", None) is not self._base:
                if self._base is None:
                    self._base = torch.cat([self.ch.spikes()] + [bb.reshape(-1) for bb in self.bufs])
                self._base_of = self._base
                self._base_on = self._base > 0
                self._base_e = expand(self, torch.nonzero(self._base_on).squeeze(1))
            if self._within is None:
                self._e_cache = self._base_e
            else:
                new = torch.nonzero((self._within > 0) & ~self._base_on).squeeze(1)
                self._e_cache = torch.cat([self._base_e, expand(self, new)]) if len(new) else self._base_e
            return self._e_cache

        G2._fired_edges = fired_edges
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    rec_rep, rec_err, ms = [], [], []
    orig = Brain.step

    def step(self, ear, cmd, eye=None, learn=True):
        sync(); t0 = time.time()
        out = orig(self, ear, cmd, eye, learn)
        sync(); ms.append((time.time() - t0) * 1000)
        return out

    def pas_hook(orig_pass):
        def pas(self, x_all, ext_all, learn, advance, first):
            out = orig_pass(self, x_all, ext_all, learn, advance, first)
            if advance:
                rec_rep.append(np.packbits((out[0] > 0).cpu().numpy())); rec_err.append(np.packbits((out[1] > 0).cpu().numpy()))
            return out
        return pas

    m.Group2._pass = pas_hook(m.Group2._pass)
    Brain.step = step
    rec_rep.clear()
    inv = B.make_inventory(np.random.default_rng(0))
    rng = np.random.default_rng(0)
    for _ in range(N):
        mel, cmd, _ = B.utter(inv, rng, b.n["運動"])
        B.run(b, cmd, B.encode(mel), True, True)
    np.savez(out_path, ms=np.array(ms), rep=np.stack(rec_rep), err=np.stack(rec_err))


if len(args) > 1 and args[1] == "PROF":
    import torch
    import babble as B, brainstem as BS, torch_area as T
    from torch_brain import Brain
    from torch.profiler import profile, ProfilerActivity
    torch.manual_seed(0)
    b = Brain(T.Prior.load("prior_v1.npz", dev=B.DEV), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    b.new_utterance(True)
    for t in range(20):
        b.step(torch.tensor(ear[t], device=B.DEV), torch.tensor(cmd[t], device=B.DEV), learn=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for t in range(20, 70):
            b.step(torch.tensor(ear[t], device=B.DEV), torch.tensor(cmd[t], device=B.DEV), learn=True)
        torch.cuda.synchronize()
    print("\n── 50歩の内訳（GPU の時間の多い順）", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15), flush=True)
    print("── 50歩の内訳（CPU の時間の多い順）", flush=True)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15), flush=True)
    sys.exit(0)

if len(args) > 1 and args[1] in MODES:          # 子のプロセス
    child(args[1], args[2])
    sys.exit(0)

tmp = tempfile.mkdtemp()
res = {}
for mode in MODES:
    p = os.path.join(tmp, f"{mode}.npz")
    cmd = [sys.executable, __file__, str(N), mode, p] + (["--test"] if TEST else [])
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"⚠️ {mode} が止まった")
    res[mode] = np.load(p)
    d = res[mode]
    print(f"{mode:4s}  {len(d['ms'])}歩  1歩 平均 {d['ms'][20:].mean():6.1f}ms（中央 {np.median(d['ms'][20:]):6.1f}）"
          f"  （作るのを含めて {time.time()-t0:.0f}秒）", flush=True)
for ref, mode in (("もと", "もと2"), ("もと", "AB"), ("AB", "AB2"), ("AB", "ABC"), ("もと決", "AB決"), ("もと決", "ABC決")):
    base, d = res[ref], res[mode]
    n = min(len(base["rep"]), len(d["rep"]))
    dr = int((np.unpackbits(base["rep"][:n]) != np.unpackbits(d["rep"][:n])).sum())
    de = int((np.unpackbits(base["err"][:n]) != np.unpackbits(d["err"][:n])).sum())
    tot = int(np.unpackbits(base["rep"][:n]).sum())
    first = next((k for k in range(n) if (base["rep"][k] != d["rep"][k]).any() or (base["err"][k] != d["err"][k]).any()), None)
    print(f"  {ref} と {mode}：違う発火 表現 {dr}本・誤差 {de}本（もとの表現の発火 {tot}本）  最初に違った歩 {first}", flush=True)
print("読み方：もと≠AB でも AB＝AB2 なら、AB は毎回同じ結果を出す（足し算の順が GPU の待ち方で変わった見込み）", flush=True)
print("        もと決＝AB決 なら、AB は計算の中身を変えていない（違いは足し算の順のゆらぎが網で大きくなっただけ）", flush=True)

# ── どこに時間を使っているか（もとのまま、50歩） ──
if not TEST:
    subprocess.run([sys.executable, __file__, str(1), "PROF", os.path.join(tmp, "p.npz")])
