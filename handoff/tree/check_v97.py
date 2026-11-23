"""版97 の確かめ（コンテナ・CPU・相手 200・作り物の声）。数値（率）は読まない
  python3 check_v97.py
確かめること
  1 口の筋の式が mouth.muscle_force と同じ（1歩ずつ）
  2 耳の音の重ね方：重なった歩は強さ（パワ−）の和
  3 口：運動の領野が1つのモーラの型で撃ち続けると、力が越えた歩でそのモーラを読み、声を鳴らす（鳴っている間は読まず、鳴り終われば又読む）
  4 返し手：口が出した声に返す（合図の外の声＝自発にも）・遅れは声の終わりから
  5 続けて流す：番の並びは版96 までのまま（言う番・黙る番）。合図と補助輪の指令は言う番の中だけ・学ぶ暮らしに録音は無い
  6 健診：学びは止めない・10番ごとに録音3つ（2回以上前）・録音の番に指令は入らない
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

print("5 続けて流す（版96 までの番の並び。番の決まりだけを見るので網は作り物：b.step を呼んだ記録を取る）", flush=True)


class StubBrain:
    """網の代わり（番・合図・指令・学びの引数だけを記録）。口は撃たないので声は出ない ⇒ 声は _speak で入れる"""

    def __init__(self, n):
        self.motor_rep = torch.zeros(n); self.cue_t = None; self.log = []

    def go(self):
        self.cue_t = 0

    def step(self, ear, cmd, eye=None, learn=True):
        self.log.append((self.L.turn["kind"], float(cmd.sum()) > 0, self.cue_t is not None, learn))
        if self.cue_t is not None:
            self.cue_t += 1
            if self.cue_t > 94:
                self.cue_t = None
        return None, torch.zeros(4), None, None


sb = StubBrain(mo.n_cmd)
life2 = L.Life(sb, L.FakeVoice(), mo, L.FakeReplier(L.FakeVoice()), np.random.default_rng(3), np.random.default_rng(4), "cpu")
sb.L = life2
orig_new = life2._new_turn


def new_turn(t, rec_test):
    tu = orig_new(t, rec_test)
    if tu["kind"] == "言う":                               # 口の代わり：準備の終わりに選んだモーラを言わせる
        tu["_say_at"] = t + C.GO_LEAD
    return tu


life2._new_turn = new_turn
orig_step = life2.step


def step(learn=True, rec_test=False):
    tu = life2.turn
    if tu is not None and tu["kind"] == "言う" and tu.get("_say_at") == life2.t and tu["said"] is None:
        life2._speak([(m, 0.07, 0.12, float(np.log(300.0))) for m in tu["want"]], 1.0, life2.t)
    orig_step(learn, rec_test)


life2.step = step
life2.run_plans(30, learn=True)
st = life2.stats; log = sb.log
kinds = {k for k, *_ in log}
chk(st["turns"] == 30 and 0 < st["plans"] < 30, f"{life2.t}歩・番 {st['turns']}（言う {st['plans']}・黙る {st['turns'] - st['plans']}）声 {st['voices']} 返し手 {st['replies']}")
chk(all(k == "言う" for k, c, *_ in log if c), "語の指令（補助輪）は「言う」番の中だけ")
chk(all(k == "言う" for k, c, q, _ in log if q), "合図は「言う」番の中だけ")
chk("録音" not in kinds, "学ぶ暮らし（rec_test なし）に録音は鳴らない")
chk(st["replies"] > 0, f"返し手が返した（{st['replies']}）")

print("6 健診（学びは止めない・10番ごとに録音3つ）", flush=True)
sb.log.clear()
h = life2.health(n=10)
log = sb.log
print(f"  （見るだけ）{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} }", flush=True)
chk(h["録音の数"] == 3 and h["番"] == 10, f"10番で録音 {h['録音の数']}（3）")
chk(all(l for *_, l in log), "健診のあいだも学ぶ（learn=True）")
chk(any(k == "録音" for k, *_ in log) and not any(c for k, c, *_ in log if k == "録音"), "録音の番に補助輪の指令は入らない（口は回る）")

print("7 本物の網で1番（回る）", flush=True)
life3 = B.life_of(b, True, np.random.default_rng(5), np.random.default_rng(6))
life3.run_plans(1, learn=True)
chk(life3.t > 0, f"{life3.t}歩 回った（番 {life3.stats['turns']}・声 {life3.stats['voices']}）")
print("✅ 版97 の確かめ 全部通った" if ok else "⚠️ 通らない所がある", flush=True)
