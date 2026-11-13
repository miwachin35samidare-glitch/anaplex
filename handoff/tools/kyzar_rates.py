"""人の内側前頭・内側側頭の1体ごとの率を部位ごとに（Kyzar 2024 Sci Data、DANDI 000469、21人・41回・1,809体）
（arkhe で。網は使わない・何も書き換えない・GPU を使わない）
  ~/dandi-venv/bin/python kyzar_rates.py ~/data/000469
なぜ：恒常性・滑る境目の目標を領野ごとに人の値にする【み】11-13 22:58「新皮質って一様に3.3なの？」・23:00「調べて」
  （新皮質の一様 3.30Hz は Verzhbinsky 2024 の 中側頭回1人＋一次運動野1人 を合わせた値。台帳_健診の目標値 §1b）
データの作り（rutishauserlab/workingmem-release-NWB の MATLAB のとおり）
  /units/spike_times（＋ spike_times_index）・/units/electrodes → /general/extracellular_ephys/electrodes/location
  部位の名：hippocampus・amygdala・dorsal_anterior_cingulate_cortex・pre_supplementary_motor_area・ventral_medial_prefrontal_cortex（_left/_right）
  /intervals/trials/timestamps_FixationCross（Sternberg の回だけ。凝視の十字 0.9〜1秒）
出すもの（部位ごと・左右を合わせる。体ごとの値の 数・平均・中央・四分位・1Hz 未満の割合）
  ① 回の中の率 ＝ 発火数 ÷（最後の発火 − 最初の発火）（配った MATLAB の globalRate と同じ定義）
  ② 凝視の十字のあいだの率（十字から 0.9秒。Sternberg の回だけ）＝ 絵を見る前のいちばん静かな区間
⚠️ てんかんの人・課題の最中（安静ではない）・錐体と抑制を分けていない（波形で分ける形は入れていない）
⚠️ ① は記録の両端の発火で割るので、ほとんど撃たない体は高めに出る
"""
import os
import sys
import glob
import collections
import numpy as np
import h5py

FIX_S = 0.9
NAME = {"dorsal_anterior_cingulate_cortex": "前帯状（背側）", "pre_supplementary_motor_area": "前補足",
        "ventral_medial_prefrontal_cortex": "腹内側前頭", "hippocampus": "海馬", "amygdala": "扁桃体"}


def s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def ragged(f, name):
    d = f[name][()]
    ix = f[name + "_index"][()] if name + "_index" in f else None
    if ix is None:
        return [np.atleast_1d(v) for v in d]
    st = np.concatenate([[0], ix[:-1]])
    return [d[a:b] for a, b in zip(st, ix)]


def area_of(loc):
    loc = loc.lower()
    for k in NAME:
        if loc.startswith(k):
            return k
    return "その他:" + loc


src = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data/000469")
files = sorted(glob.glob(os.path.join(src, "**", "*.nwb"), recursive=True))
if not files:
    sys.exit(f"⚠️ {src} に .nwb が無い")
rate = collections.defaultdict(list)
fix = collections.defaultdict(list)
nses = collections.Counter()
for fn in files:
    with h5py.File(fn, "r") as f:
        if "units" not in f:
            print(f"（{os.path.basename(fn)}：units が無い ⇒ 飛ばす）", flush=True)
            continue
        locs = [s(v) for v in f["general/extracellular_ephys/electrodes/location"][()]]
        spk = ragged(f["units"], "spike_times")
        u = f["units"]
        el = ragged(u, "electrodes") if "electrodes" in u else [[0]] * len(spk)
        tsf = None
        if "intervals/trials" in f and "timestamps_FixationCross" in f["intervals/trials"]:
            tsf = np.asarray(f["intervals/trials/timestamps_FixationCross"][()], dtype=float)
            tsf = tsf[np.isfinite(tsf)]
        kind = "Sternberg" if tsf is not None and len(tsf) else "見るだけ"
        nses[kind] += 1
        for t, e in zip(spk, el):
            t = np.sort(np.asarray(t, dtype=float))
            ar = area_of(locs[int(np.atleast_1d(e)[0])])
            if len(t) >= 2 and t[-1] > t[0]:
                rate[ar].append(len(t) / (t[-1] - t[0]))
            if kind == "Sternberg":
                n = np.searchsorted(t, tsf + FIX_S) - np.searchsorted(t, tsf)
                fix[ar].append(n.sum() / (FIX_S * len(tsf)))
print(f"ファイル {len(files)}（Sternberg の回 {nses['Sternberg']}・見るだけの回 {nses['見るだけ']}）", flush=True)


def row(lab, v):
    v = np.asarray(v)
    if len(v) == 0:
        return f"  {lab:<14s}   0体"
    q1, q3 = np.percentile(v, [25, 75])
    return (f"  {lab:<14s}{len(v):5d}体  平均 {v.mean():6.2f}  中央 {np.median(v):6.2f}  四分位 {q1:5.2f}〜{q3:5.2f}Hz"
            f"  1Hz 未満 {np.mean(v < 1):4.0%}")


order = list(NAME) + sorted(k for k in set(rate) | set(fix) if k not in NAME)
print("\n① 回の中の率（発火数 ÷ 最初と最後の発火のあいだ）", flush=True)
for k in order:
    print(row(NAME.get(k, k), rate.get(k, [])), flush=True)
print(f"\n② 凝視の十字のあいだ（十字から {FIX_S}秒・Sternberg の回だけ）", flush=True)
for k in order:
    print(row(NAME.get(k, k), fix.get(k, [])), flush=True)
