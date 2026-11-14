"""版110：暮らしの歩を「声・準備・返し手・間」に分けて、運動（と運動の発火が届く受け手）の率を測る（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_talk110.py [番の並び=10,100]
  何のため：恒常性の目標を話している・いないで分ける（10 §6【み】11-14）。人の値（Kennedy 2011：分の区間まるごと・声の出ない1人、
    作業 9.6・音楽 5.7Hz）と並べる前に、網の暮らしで各状態の歩の割合と率を知る（組み表_版111_案 §3b）
  歩の分け（その歩に使われた番と鳴っていた音から。上が先）【C】
    声        自分の声が鳴っている歩（口が読んでから声の終わりまで。後ろの静けさ 20歩は入れない）
    準備・指令  言う番で、語の指令（補助輪）が入っていて声はまだの歩
    準備・合図  言う番で、合図だけ（指令の前）の歩
    返し手     返し手の声が鳴っている歩（自分の声と重ならない歩）
    録音       録音の番
    間        それ以外（黙る番・声のあとの静けさ・番の終わり待ち）
  出すもの（区切りごと。区切りの頭で数えを 0 に戻す）
    状態ごとの歩の数と割合 ／ 運動の表現の率：体の平均・体の中央・1Hz 未満の体の割合 ／ 指令の体・それ以外の体の平均（言う番だけ）
    運動の発火が届く受け手（聴覚・高次・補足・前補足・44・縁上回）の表現の率（体の平均）／ 全部の歩の平均（分の区間まるごとの見方）
    番・声・自発・言えた の数
  毎歩に足す仕事：状態の番号を決める（Python の数個の比べ）・表現の発火（体の数 110,406）を状態の行へ足す 1回・
    指令の体の発火の和 2回（840体）。線・辺には触らない・少ない箱への足し込みなし・網と同じ大きさの写しなし（term109 の数えと同じ重さ）
  歩は life.step（見張り・休み・止めはそのまま働く）。グラフで回す（ふだんと同じ）
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB, life as L, config as C
from torch_brain import Brain, NAMES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
WIN = [int(x) for x in (args[0] if args else "10,100").split(",")]
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
if B.VERSION != 110:
    raise SystemExit(f"⚠️ 版 {B.VERSION}：この道具は版110 で確かめた（版110 で回す）")

ST = ("声", "準備・指令", "準備・合図", "返し手", "録音", "間")
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
g = b.g
n_all = g.n_ch_all
DT = 10.67 / 1000.0
AREAS = [nm for nm in ("運動", "聴覚", "高次", "補足", "前補足", "44", "縁上回") if nm in NAMES]
SL = {nm: slice(int(b.off_ch[NAMES.index(nm)]), int(b.off_ch[NAMES.index(nm)]) + b.n[nm]) for nm in AREAS}
MS = SL["運動"]

acc = torch.zeros(len(ST), n_all, device=dev)           # 状態 × 体：表現の発火の数
n_st = [0] * len(ST)
cmd_f = torch.zeros(2, device=dev)                       # 言う番で指令の入っている歩：指令の体の発火の和・それ以外の体の発火の和
cmd_n = [0.0, 0.0]                                       # その体の数 × 歩
kinds_now = []
_at0 = life.mix.at


def _at(t):
    mel, kinds = _at0(t)
    kinds_now[:] = kinds
    return mel, kinds


life.mix.at = _at


def one_step():
    """1歩進めてから、その歩の状態を決める（番の替わり・鳴っていた音は歩の中で決まるので、進めたあとに読む）"""
    t = life.t
    se = life.speak_end                                  # 進める前の声の終わり（この歩に口が読んだ声は次の歩から鳴る）
    v0 = life.stats["voices"]
    life.step(learn=True)
    tu = life.turn                                       # この歩に使われた番
    spoke = life.stats["voices"] > v0                    # この歩に口が読んだ ⇒ この歩のあいだは声はまだ
    one = None
    if t < se:
        s = 0
    elif tu["kind"] == "言う" and (tu["said"] is None or spoke):
        s = 1 if (t - tu["t0"]) >= C.GO_LEAD - C.CMD_LEAD else 2
    elif "返し手" in kinds_now:
        s = 3
    elif tu["kind"] == "録音" or "録音" in kinds_now:
        s = 4
    else:
        s = 5
    if tu["kind"] == "言う" and s in (0, 1):
        one = tu["one"]
    f = g._any[0][:n_all].float()
    acc[s].add_(f)
    n_st[s] += 1
    if one is not None:
        m = f[MS]
        on = one > 0
        cmd_f[0] += (m * on).sum()
        cmd_f[1] += (m * ~on).sum()
        k = float(on.sum())
        cmd_n[0] += k
        cmd_n[1] += b.n["運動"] - k


def report(label, st0):
    L.beat()
    tot = max(1, sum(n_st))
    d = {k: life.stats[k] - st0[k] for k in life.stats}
    print(f"\n━━ {label}（{tot} 歩 ＝ {tot * DT:.0f} 秒）番 {d['turns']}・言う番 {d['plans']}・声 {d['voices']}・自発 {d['spont']}"
          f"・言えた {d['ok']}/{d['tried']}・返し手 {d['replies']} ━━", flush=True)
    print(f"  {'状態':<10s}{'歩':>8s}{'割合':>7s} ｜ 運動 {'平均Hz':>7s}{'中央Hz':>7s}{'1Hz未満':>8s} ｜ " + "".join(f"{nm:>7s}" for nm in AREAS[1:]), flush=True)
    rows = [(ST[i], acc[i], n_st[i]) for i in range(len(ST))] + [("全部の歩", acc.sum(0), tot)]
    for nm_, a, n in rows:
        if n == 0:
            print(f"  {nm_:<10s}{0:8d}{0:7.1%}", flush=True)
            continue
        r = a / n / DT                                    # 体ごとの率 Hz
        m = r[MS]
        print(f"  {nm_:<10s}{n:8d}{n / tot:7.1%} ｜      {float(m.mean()):7.2f}{float(m.median()):7.2f}{float((m < 1).float().mean()):8.0%} ｜ "
              + "".join(f"{float(r[SL[x]].mean()):7.2f}" for x in AREAS[1:]), flush=True)
    if cmd_n[0] > 0:
        print(f"  言う番で指令の入っている歩（声 ＋ 準備・指令）：指令の体 {float(cmd_f[0]) / cmd_n[0] / DT:.2f}Hz"
              f"（1歩あたり {cmd_n[0] / max(1, n_st[0] + n_st[1]):.0f} 体）・それ以外の体 {float(cmd_f[1]) / max(cmd_n[1], 1) / DT:.2f}Hz", flush=True)
    L.beat()


print(f"版 {B.VERSION} ／ 番 {WIN} ／ 体 {n_all:,}・運動 {b.n['運動']} ／ 受け手 {AREAS[1:]}", flush=True)
done = 0
for k in WIN:
    acc.zero_(); cmd_f.zero_(); cmd_n[:] = [0.0, 0.0]; n_st[:] = [0] * len(ST)
    st0 = dict(life.stats)
    goal = life.stats["turns"] + (k - done)
    while True:                                          # life.run_plans と同じ止め方
        one_step()
        if life.stats["turns"] >= goal and life.turn is not None and life._turn_done(life.t) and not life.queue:
            break
    report(f"{done}〜{k}番", st0)
    done = k
print("\n（終わり）", flush=True)
