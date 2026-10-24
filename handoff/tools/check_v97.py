"""版97 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない
  python3 check_v97.py
確かめること
  1 口の筋の式が mouth.muscle_force と同じ（1歩ずつ）
  2 耳の音の重ね方：重なった歩は強さ（パワ−）の和
  3 口：運動の領野が1つのモーラの型で撃ち続けると、力が越えた歩でそのモーラを読み、声を鳴らす（鳴っている間は読まず、鳴り終われば又読む）
  4 返し手：口が出した声に返す（合図の外の声＝自発にも）・遅れは声の終わりから
  5 続けて流す：合図（足場）が入り、補助輪の指令が声の CMD_LEAD 歩前から入る・網が回る（合図の間を短くして）
  6 健診：自分・録音・返し手の誤差が出る（録音は2回以上前）
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
import babble as B, brainstem as BS, torch_area as T, mouth as MO, life as L, config as C
from torch_brain import Brain

B.DEV = "cpu"
ok = True


def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    ok = ok and bool(cond)


print("1 口の筋の式", flush=True)
rng = np.random.default_rng(0)
sp = (rng.random((60, 50)) < 0.3).astype(float)
m = L.Muscle(50, "cpu")
for t in range(60):
    y = m.step(torch.tensor(sp[t]))
chk(np.abs(y.numpy() - MO.muscle_force(sp)).max() < 1e-9, "muscle_force と同じ（1歩ずつ・前もってまとめた掛け算）")

print("2 音の重ね方", flush=True)
mx = L.Mixer(float(np.log(L.EPS)))
a = np.full((5, C.N_MELS), np.log(0.3), np.float32); bb = np.full((5, C.N_MELS), np.log(0.2), np.float32)
mx.add(0, a, "自分"); mx.add(2, bb, "返し手")
v0, k0 = mx.at(0); mx.at(1); v2, k2 = mx.at(2)
chk(abs(float(np.exp(v0[0])) - 0.3) < 1e-4 and k0 == ["自分"], "1つだけ鳴る歩はその音")
chk(abs(float(np.exp(v2[0])) - 0.5) < 1e-4 and sorted(k2) == ["自分", "返し手"], "重なった歩は強さの和（0.3 ＋ 0.2）")

print("3・4 口と返し手", flush=True)
prior = T.Prior.load("prior_v1.npz", dev="cpu")
torch.manual_seed(0)
b = Brain(prior, n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
b.bs.set_state(BS.WAKE)


class R0(np.random.Generator):
    pass


life = B.life_of(b, True, np.random.default_rng(1), np.random.default_rng(2))
mo = life.mo
want = [(7, 0.07, 0.12, float(np.log(300.0)))]
pat = torch.tensor(mo.encode(want, 1.0)[0])
spoke, ends = [], []
for t in range(40):
    n0 = life.stats["voices"]
    life._mouth(pat, t)
    if life.stats["voices"] > n0:
        spoke.append(t); ends.append(life.speak_end)
chk(len(spoke) >= 1, f"型で撃ち続けると力が越えた歩で読む（{spoke}）")
chk(all(spoke[i + 1] >= ends[i] for i in range(len(spoke) - 1)), f"自分の声が鳴っている間は読まない（読んだ歩 {spoke}・鳴り終わり {ends}）")
chk(life.stats["spont"] == len(spoke), f"合図の外の声は自発に数える（{life.stats['spont']}）")
srcs = [k for _, _, k in life.mix.src]
print(f"  鳴っている音：{srcs}（返し手 {life.stats['replies']}）", flush=True)
# 返し手が返す回まで回して、遅れが声の終わりからかを見る
n_try, got = 0, None
while got is None and n_try < 50:
    life.mix.src = []; life.speak_end = -1
    life._speak(want, 1.0, 1000)
    rr = [s for s in life.mix.src if s[2] == "返し手"]
    if rr:
        own = [s for s in life.mix.src if s[2] == "自分"][0]
        got = (rr[0][0], own[0] + len(own[1]))
    n_try += 1
chk(got is not None and got[0] == got[1], f"返し手は声（後ろの静けさ込み）の終わりから遅れを数える（{got}）")

print("5 続けて流す", flush=True)
L.CUE_MEAN = 60.0                                    # ⚠️ 確かめだけ：合図の間を短く（人は約 1,250歩）
life2 = B.life_of(b, True, np.random.default_rng(3), np.random.default_rng(4))
seen_cmd, cue_seen, steps = 0, 0, 0
orig = b.step


def spy(ear, cmd, eye=None, learn=True):
    global seen_cmd, cue_seen
    seen_cmd += int(float(cmd.sum()) > 0)
    cue_seen += int(b.cue_t is not None)
    return orig(ear, cmd, eye=eye, learn=learn)


b.step = spy
life2.run_plans(2, learn=True)
b.step = orig
st = life2.stats
chk(st["plans"] >= 2 and seen_cmd > 0 and cue_seen > 0, f"{life2.t}歩・合図 {st['plans']}・指令の入った歩 {seen_cmd}・合図の歩 {cue_seen}・声 {st['voices']}（自発 {st['spont']}）")

print("6 健診", flush=True)
for k in range(3):                                   # 録音を3つ作る
    life2.recs.append(L.FakeVoice().sound(want, 1.0))
h = life2.health(n=2)
print(f"  （見るだけ）{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} }", flush=True)
chk(isinstance(h["supp"], float) and "言えた" in h, "健診が回る")
print("✅ 版97 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
