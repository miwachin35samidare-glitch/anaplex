"""版42 の切り分け（arkhe の ~/anaplex で  python3 measure_v42.py 形 回数 ）⚠️ 保存しない（state.pt を書き換えない）

0から同じ回数だけ回して並べる（声の乱数は同じ列）。版41 までの測りは 01_現在地.md
形は 切るものを ＋ でつなぐ（09-28 21時から。例：k0+noarousal）。何もつながないのが full
  k0        学びの門の上乗せなし（門 ＝ 1）
  noreply   返し手なし
  noarousal 感度なし（閾値の倍率 ＝ 1）
  nolc      驚き→学ぶ速さなし（版41 と同じ）
  notrace   学んだ跡なし（その歩の学びをそのまま重みへ）
  oldrewire 刈り込みを版39 の形に戻す（並べ直す前の位置で重みを 0 にする不具合ごと）
  base      k0＋noreply＋noarousal の略（base+notrace のようにもつなげる）
  ⚠️ 前の「notrace」「oldrewire」は base 込みだったが、今は それだけを切る。前と同じ形は base+notrace
  ⚠️ notrace・oldrewire は arkhe の torch_group2.py をこの中だけ書き換えて読む（ファイルは変えない）
使い方：python3 measure_v42.py k0+noarousal 100（100回で約35〜60分）
"""
import sys
import time
import types
import numpy as np
import torch

FLAGS = ("k0", "noreply", "noarousal", "nolc", "notrace", "oldrewire")
mode = sys.argv[1] if len(sys.argv) > 1 else "full"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
off = set()
for f in mode.split("+"):
    if f == "full":
        continue
    if f == "base":
        off |= {"k0", "noreply", "noarousal"}
    elif f in FLAGS:
        off.add(f)
    else:
        sys.exit("切るものは " + " / ".join(FLAGS) + " / base を ＋ でつなぐ（何も切らないのは full）")
print("切るもの：" + ("なし" if not off else "・".join(sorted(off))), flush=True)


def swap(src, old, new, name, f="torch_group2.py"):
    if src.count(old) != 1:
        sys.exit(f"⚠️ {f} に「{name}」の行が1つ見つからない（{src.count(old)}個）。回していない")
    return src.replace(old, new)


if off & {"notrace", "oldrewire"}:
    src = open("torch_group2.py", encoding="utf-8").read()
    if "notrace" in off:       # 学んだ跡を使わず、その歩の学びをそのまま重みへ（門があれば その歩に掛ける）
        src = swap(src, "            self.elig.mul_(self.elig_decay)\n"
                        "            self.elig[e] += torch.where(self.e_inh[e], -gg * shunt, gg)\n",
                   "            self.w[e] += torch.where(self.e_inh[e], -gg * shunt, gg) * self.third\n", "跡にためる")
        src = swap(src, "            self.w.addcmul_(self.elig, (self.third * (1.0 - self.elig_decay)).reshape(()))\n",
                   "", "跡から重みへ")
    if "oldrewire" in off:     # 版39 の刈り込み：並べ直す前の位置 cuts で重みを 0 にする
        src = swap(src, "        cuts = pos[cuts]\n", "", "刈り込みの位置の直し")
        src = swap(src, "        self.elig[cuts] = 0.0\n", "        self.elig[pos[cuts]] = 0.0\n", "跡の 0")
    m = types.ModuleType("torch_group2")
    m.__file__ = "torch_group2.py"
    exec(compile(src, "torch_group2.py（measure で書き換え）", "exec"), m.__dict__)
    sys.modules["torch_group2"] = m
    print(f"torch_group2.py をこの中だけ書き換えた（{mode}）", flush=True)

import torch_area as T
import babble as B
import brainstem as BS
from torch_brain import Brain

if "LC_K" not in vars(BS):
    sys.exit("⚠️ 版42 ではない（先に apply_v42.py）")
if "nolc" in off:
    BS.LC_K = 0.0
if "k0" in off:
    BS.SURPRISE_K = 0.0
if "noarousal" in off:
    BS.Brainstem._set_gain = lambda self: setattr(self.g, "th_gain", torch.ones((), device=self.dev))
reply_on = "noreply" not in off

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
