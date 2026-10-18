"""版93：健診2（measure_check2.py）をそのまま回し、喃語1回ごとに中脳と門の動きを並べる（どれが先に動くか）
（arkhe の ~/anaplex で。版93。measure_check2.py を読み込んで回すだけで、どのファイルも書き換えない。0から・学びながら・保存しない）
  python3 measure_v93a.py [喃語の回数=50]             … そのまま（版93 の振る舞い）
  python3 measure_v93a.py [回数] --gate1               … 試し：中脳は回して学ぶが、網へ渡す門を 1 に止める
  python3 measure_v93a.py [回数] --mbfix               … 試し：中脳の核のシナプスの学びだけを止める（門は渡す）
【測】arkhe 版93 measure_check2 50（10-18）：黒質 66.0Hz（的 4.56）・視床下核 32.1・吻内側被蓋核 21.5 ／
     運動 13.6 → 99.5%・聴覚 3.7 → 68%・連合 1.9 → 22%・前頭前野 1.6 → 12.9%（10回ごと）／ 言えた 1/43（11回目から自分の音のある歩 0）
     calib_v93（網の入力を写して流し直す）では黒質 4.5Hz ⇒ 網と一緒に回すと 15倍
【コード】門 ＝ 黒質の率（225ms でならす）÷ 4.56Hz。版93 で brainstem の上限 DA_CAP（6.25倍）を消した ⇒ 門に上限が無い
        門は網の学びの三つ目の因子：Group2 の辺の学びの量 ∝ 門（torch_group2 の A）・版74 の下からの窓の開き g ＝ 門 − 1 を 0〜1 に
     ⇒ 見立て【見・未測】：黒質が上がる → 門が上がる → 網が強まる → 前頭前野が上がる → 視床下核・黒質がさらに上がる、の輪
        どこから始まるか（黒質の入力の重さが育つのか、前頭前野が先に上がるのか、ほかの領野が先か）は測っていない ⇒ これで測る
並べるもの（喃語1回ごと：自分 → 返し手、10回ごとに録音3つも入る）
  歩 ／ 門の平均・最大・門 ＞ 2 の歩の割合 ／ 核の率（Hz）／ 中脳へ入る率（Hz）：前頭前野の表現・側坐核・上丘の誤差・PPN ／
  黒質の興奮の重さの平均（送り手ごと：PPN・視床下核・前頭前野・上丘）／ ほかの核の興奮の重さ（線ごと）
"""
import sys, torch
sys.path.insert(0, ".")
import babble as B
import torch_area as T
import torch_brain as TB

if B.VERSION != 93:
    sys.exit(f"⚠️ babble.py が版93 ではない（版 {B.VERSION}）")
GATE1 = "--gate1" in sys.argv
MBFIX = "--mbfix" in sys.argv
print("試し：" + ("門を 1 に止める（--gate1）" if GATE1 else "中脳の学びを止める（--mbfix）" if MBFIX else "そのまま"), flush=True)
STEP_S = T.DT_MS / 1000.0
S = {"b": None, "cyc": 0}


def blk(mb, name):
    n = mb.n[name] if name in mb.n else mb.n_src[name]
    return mb.off[name], mb.off[name] + n


def install(b):
    mb = b.mb
    orig = mb.step
    srcs = list(mb.n_src)
    nsrc = torch.tensor([float(mb.n_src[s]) for s in srcs], device=mb.dev)
    S.update(b=b, mb=mb, srcs=srcs, nsrc=nsrc)
    reset(mb)
    # 黒質の興奮（Harnett）の送り手ごと ／ ほかの核の興奮（決め A）の線ごと
    S["hg"] = []
    for s in ("PPN", "視床下核", "前頭前野", "上丘"):
        lo, hi = blk(mb, s)
        m = (mb.kind == 1) & (mb.src >= lo) & (mb.src < hi)
        S["hg"].append((s, m))
    S["ag"] = []
    for s, t in (("前頭前野", "視床下核"), ("視床下核", "境"), ("境", "外側手綱核"), ("外側手綱核", "吻内側被蓋核")):
        ls, hs = blk(mb, s); lt, ht = blk(mb, t)
        m = (mb.kind == 2) & (mb.src >= ls) & (mb.src < hs) & (mb.tgt >= lt) & (mb.tgt < ht)
        S["ag"].append((f"{s}→{t}", m))

    def step(ext, learn=True):
        S["inp"] += torch.stack([ext[s].float().sum() for s in srcs])
        gate = orig(ext, learn=(learn and not MBFIX))
        g = gate.reshape(())
        S["gsum"] += g
        S["gmax"] = torch.maximum(S["gmax"], g)
        S["g2"] += (g > 2.0).float()
        S["n"] += 1
        return torch.ones_like(gate) if GATE1 else gate

    mb.step = step


def reset(mb):
    dev = mb.dev
    S.update(inp=torch.zeros(len(S["srcs"]), device=dev), gsum=torch.zeros((), device=dev),
             gmax=torch.zeros((), device=dev), g2=torch.zeros((), device=dev), n=0,
             rs0=mb.rate_sum.clone(), st0=mb.steps)


def report():
    mb = S.get("mb")
    if mb is None or S["n"] == 0:
        return
    n = S["n"]
    sec = (mb.steps - S["st0"]) * STEP_S
    hz = ((mb.rate_sum - S["rs0"]) / max(sec, 1e-9)).tolist()
    inp = (S["inp"] / S["nsrc"] / (n * STEP_S)).tolist()
    hw = [float(mb.w[m].mean()) if bool(m.any()) else float("nan") for _, m in S["hg"]]
    hmax = float(mb.w[mb.kind == 1].max())
    aw = [float(mb.w[m].mean()) if bool(m.any()) else float("nan") for _, m in S["ag"]]
    S["cyc"] += 1
    if S["cyc"] == 1:
        print("── 中脳（喃語1回ごと）  門：平均・最大・＞2 の歩 ／ 核 Hz：黒質・視床下核・境・外側手綱核・吻内側被蓋核 ／ "
              "入る Hz：" + "・".join(S["srcs"]) + " ／ 黒質の重さ：PPN・視床下核・前頭前野・上丘（最大）／ ほかの重さ："
              + "・".join(nm for nm, _ in S["ag"]), flush=True)
    order = ("黒質", "視床下核", "境", "外側手綱核", "吻内側被蓋核")
    names = list(mb.n)
    print(f"   中脳 {S['cyc']:3d}  歩 {n:4d}  門 {float(S['gsum'])/n:5.2f}・{float(S['gmax']):5.2f}・{float(S['g2'])/n:5.1%}  ／ "
          + " ".join(f"{hz[names.index(k)]:5.1f}" for k in order) + "  ／ "
          + " ".join(f"{v:5.2f}" for v in inp) + "  ／ "
          + " ".join(f"{v:.3f}" for v in hw) + f"（{hmax:.2f}）  ／ "
          + " ".join(f"{v:.4f}" for v in aw), flush=True)
    reset(mb)


_init = TB.Brain.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    install(self)


TB.Brain.__init__ = init
_voices = B.voices


def voices(b, test):
    r = list(_voices(b, test))
    say = r[1]

    def say2(*a, **k):
        report()                    # 前の喃語1回（自分・返し手・録音の試し）の分を出す
        return say(*a, **k)
    r[1] = say2
    return tuple(r)


B.voices = voices
src = open("measure_check2.py", encoding="utf-8").read()
exec(compile(src, "measure_check2.py", "exec"), {"__name__": "__main__", "__file__": "measure_check2.py"})
report()
print("読み：門が先に上がるか（黒質の重さ・入る率のどれが先に動くか）、--gate1 で網の暴走が消えるか", flush=True)
