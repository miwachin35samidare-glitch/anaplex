"""網の耳（400本の聴神経）は、口の連続値をどこまで聞き分けるか（arkhe。網は使わない）
  python3 measure_ear_res.py
【み】09-30「選択肢は人の耳か本人の耳になった」「差分の110段分ちゃんとグラデーションがつくかな？」
  ⇒ 耳の発火が同じなら、網は2つの値を区別できない（学ぶ手がかりが同じ）

やること：1つの値だけを細かく動かして VOICEVOX で合成し、耳の発火（400本 × 歩）を作る
  区別できる数 ＝ 出てきた発火の型が何通りあるか（型が変わった所 ＝ 本人の耳の刻み）
  型が変わった所の差の大きさ（何本×歩の発火が入れ替わったか）も出す。1本だけのちらつきか、はっきり変わるか
値
  高さ    「ア」 ln 4.0〜6.4（55〜600Hz。【み】が 08-02 に聞いて崩れなかった範囲）を 0.005 刻み（約 0.09半音）
  母音の長さ「ア」 10〜350ms を 1ms 刻み
  子音の長さ「バ」 1〜150ms を 1ms 刻み
  大きさ  「ア」 0.05〜2.0倍を 0.1dB 刻み
ほかの値は喃語の平均（子音 70ms・母音 170ms・高さ 122Hz・大きさ 1.0）に固定
最初に同じ音を2回作り、VOICEVOX が毎回同じ音を返すかを確かめる（返さなければ型の数は意味を持たない）
"""
import numpy as np
from voice import Voice

vo = Voice()
vv = vo.vv
A = next(i for i, (c, v) in enumerate(vv.inventory) if c is None and v == "a")
BA = next(i for i, (c, v) in enumerate(vv.inventory) if c == "b" and v == "a")
BASE = dict(c=0.070, v=0.170, p=4.80, vol=1.0)


def ear(mid, c, v, p, vol):
    q = vv.build_query([(mid, c, v, p)])
    q["volumeScale"] = float(vol)
    return vo.encode(vo.ear.hear(vv.synthesize(q)).astype(np.float32)) > 0


def sweep(name, unit, mid, key, vals, conv):
    pats, changes, prev = {}, [], None
    for x in vals:
        kw = dict(BASE); kw[key] = float(x)
        e = ear(mid, kw["c"], kw["v"], kw["p"], kw["vol"])
        pats.setdefault(e.tobytes(), x)
        if prev is not None and (prev.shape != e.shape or (prev != e).any()):
            n = int((prev != e).sum()) if prev.shape == e.shape else -1
            changes.append((conv(x), n))
        prev = e
    xs = np.array([c for c, _ in changes])
    ns = np.array([n for _, n in changes])
    gaps = np.diff(xs) if len(xs) > 1 else np.array([np.nan])
    print(f"\n[{name}] {len(vals)}通り試した → 耳の発火の型 {len(pats)}通り（型が変わった所 {len(changes)}か所）")
    if len(changes):
        print(f"  変わった所の間隔（{unit}）  中央 {np.nanmedian(gaps):.3g}  最小 {np.nanmin(gaps):.3g}  最大 {np.nanmax(gaps):.3g}")
        big = ns[ns >= 0]
        print(f"  変わった所で入れ替わった 本×歩  中央 {np.median(big):.0f}  最小 {big.min()}  最大 {big.max()}"
              f"  （1〜2 はちらつき程度）")
        for th in (1, 3, 10, 30):
            print(f"    入れ替わりが {th:2d} 以上の所だけ数えると {int((big >= th).sum())}か所")


# VOICEVOX が同じ音を返すか
e1 = ear(A, BASE["c"], BASE["v"], BASE["p"], BASE["vol"])
e2 = ear(A, BASE["c"], BASE["v"], BASE["p"], BASE["vol"])
same = e1.shape == e2.shape and not (e1 != e2).any()
print(f"同じ指令を2回：耳の発火は {'同じ' if same else '違う（' + str(int((e1 != e2).sum())) + ' 本×歩）'}")
print(f"  基準の音で撃った本×歩 {int(e1.sum())}（歩 {e1.shape[0]}）")

sweep("高さ", "半音", A, "p", np.arange(4.0, 6.4001, 0.005), lambda x: x * 12 / np.log(2))
sweep("母音の長さ", "ms", A, "v", np.arange(0.010, 0.3501, 0.001), lambda x: x * 1000)
sweep("子音の長さ", "ms", BA, "c", np.arange(0.001, 0.1501, 0.001), lambda x: x * 1000)
sweep("大きさ", "dB", A, "vol", 10 ** (np.arange(20 * np.log10(0.05), 20 * np.log10(2.0) + 1e-9, 0.1) / 20),
      lambda x: 20 * np.log10(x))
