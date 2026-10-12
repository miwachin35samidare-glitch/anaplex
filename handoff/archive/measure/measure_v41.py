"""版41 の切り分け（arkhe の ~/anaplex で  python3 measure_v41.py 形 回数 ）⚠️ 保存しない（state.pt を書き換えない）

版41 の喃語100回：supp 1.34 に上がったが、誤差そのものが 自分 62.9% ／ 録音 84.2%（版39 は 11.2% ／ 12.7%）に増えた。
門の平均が 2.1〜2.6（ふだんの分 1 のはずが、上乗せがほぼずっと出ている）
どれが効いているかを、0から同じ回数だけ回して分ける（声の乱数は同じ列）
  full      版41 のまま
  k0        上乗せなし（門 ＝ 1。学んだ跡は残る）
  noreply   返し手なし
  noarousal 感度なし（閾値の倍率 ＝ 1。眠気の覆いは起きているだけなので関係しない）
  base      上の3つを全部なし（残るのは 学んだ跡・刈り込みの直し）
  notrace   base ＋ 学んだ跡なし（その歩の学びをそのまま重みへ。版39 の学び方）
  oldrewire base ＋ 刈り込みを版39 の形に戻す（並べ直す前の位置で重みを 0 にする不具合ごと。跡の 0 は正しい位置）
  v39       base ＋ 上の2つ ＋ 版39 の驚き（驚きで学ぶ速さそのものを上げる。8歩後・その歩だけ・上限なし・K＝1）
            起きているあいだは版39 と同じになるはず。版39 の 11.2% に戻るかの確かめ
  v39fix    v39 の刈り込みだけ直した形（跡なし ＋ 版39 の驚き ＋ 刈り込みの直し）
  ⚠️ notrace・oldrewire・v39 は arkhe の torch_group2.py をこの中だけ書き換えて読む（ファイルは変えない）
使い方：python3 measure_v41.py k0 40 ／ base 100 のあと notrace 100・oldrewire 100・v39 100・v39fix 100  （40回で約20分。形ごとに回して並べる）
"""
import sys
import time
import types
import numpy as np
import torch

MODES = ("full", "k0", "noreply", "noarousal", "base", "notrace", "oldrewire", "v39", "v39fix")
mode = sys.argv[1] if len(sys.argv) > 1 else "full"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
if mode not in MODES:
    sys.exit("形は " + " / ".join(MODES))


def swap(src, old, new, name, f="torch_group2.py"):
    if src.count(old) != 1:
        sys.exit(f"⚠️ {f} に「{name}」の行が1つ見つからない（{src.count(old)}個）。回していない")
    return src.replace(old, new)


if mode in ("notrace", "oldrewire", "v39", "v39fix"):
    src = open("torch_group2.py", encoding="utf-8").read()
    if mode in ("notrace", "v39", "v39fix"):       # 学んだ跡を使わず、その歩の学びをそのまま重みへ（third は base なので 1）
        src = swap(src, "            self.elig.mul_(self.elig_decay)\n"
                        "            self.elig[e] += torch.where(self.e_inh[e], -gg * shunt, gg)\n",
                   "            self.w[e] += torch.where(self.e_inh[e], -gg * shunt, gg) * self.third\n", "跡にためる")
        src = swap(src, "            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))\n",
                   "", "跡から重みへ")
    if mode in ("oldrewire", "v39"):     # 版39 の刈り込み：並べ直す前の位置 cuts で重みを 0 にする
        src = swap(src, "        cuts = pos[cuts]\n", "", "刈り込みの位置の直し")
        src = swap(src, "        self.elig[cuts] = 0.0\n", "        self.elig[pos[cuts]] = 0.0\n", "跡の 0")
    m = types.ModuleType("torch_group2")
    m.__file__ = "torch_group2.py"
    exec(compile(src, "torch_group2.py（measure で書き換え）", "exec"), m.__dict__)
    sys.modules["torch_group2"] = m
    print(f"torch_group2.py をこの中だけ書き換えた（{mode}）", flush=True)

if mode in ("v39", "v39fix"):              # 版39 の驚き：学ぶ速さ ＝ 1 ＋ 1 ×（8歩前の誤差 ÷ いつも − 1 の正の分）
    src = open("brainstem.py", encoding="utf-8").read()
    src = swap(src, "            self.g.eta_gain = torch.ones((), device=self.dev)\n"
                    "            # 覚醒（版41）",
               "            self.g.eta_gain = 1.0 + 1.0 * self.surprise   # measure の v39：版39 の驚き（SURPRISE_K の 0 とは別）\n"
               "            # 覚醒（版41）", "起きているあいだの学ぶ速さ", "brainstem.py")
    m = types.ModuleType("brainstem")
    m.__file__ = "brainstem.py"
    exec(compile(src, "brainstem.py（measure で書き換え）", "exec"), m.__dict__)
    sys.modules["brainstem"] = m
    print(f"brainstem.py をこの中だけ書き換えた（{mode}）", flush=True)

import torch_area as T
import babble as B
import brainstem as BS
from torch_brain import Brain

BASE = ("base", "notrace", "oldrewire", "v39", "v39fix")
if mode in ("k0",) + BASE:
    BS.SURPRISE_K = 0.0
if mode in ("noarousal",) + BASE:
    BS.Brainstem._set_gain = lambda self: setattr(self.g, "th_gain", torch.ones((), device=self.dev))
reply_on = mode not in ("noreply",) + BASE

prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
from voice import Voice, Replier
from mouth import Mouth
vo, mo = Voice(), Mouth()
rp = Replier(vo) if reply_on else None


def speak(r):
    morae, vol = vo.random_morae(r)
    mel = vo.say(morae, vol)
    return mel, vo.cmd_stream(morae, vol, len(mel), mo), vo.encode


rng, rng_r = np.random.default_rng(0), np.random.default_rng(1000)
print(f"形 {mode}  喃語 {n} 回（0から・保存しない）", flush=True)
t0 = time.time()
for i in range(n):
    mel, cmd, enc = speak(rng)
    B.run(b, cmd, enc(mel), True, True)
    if rp is not None:
        B.reply(b, rp, rng_r)
    if (i + 1) % 20 == 0:
        print(f"  {i+1} 回  門 {b.bs.gate_mean():.3f}  覚醒 {float(b.bs.arousal):.2f}  "
              f"経過 {(time.time()-t0)/60:.1f}分", flush=True)
o, r, s = B.measure(b, speak)
print(f"形 {mode}  喃語 {n} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  1歩 {(time.time()-t0)/max(b.g.t,1)*1000:.0f}ms", flush=True)
