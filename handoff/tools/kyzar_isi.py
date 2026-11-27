"""人の単一細胞の自然な発火の並びから「撃ったあとしばらく撃たない・続けて撃つとより撃たない」を測る
（Kyzar 2024 Sci Data、DANDI 000469、21人・内側前頭と内側側頭。arkhe で。網は使わない・何も書き換えない・GPU を使わない）
  ~/dandi-venv/bin/python kyzar_isi.py ~/data/000469
  python3 kyzar_isi.py selftest      （容器の確かめ：作り物の並びで、入れた戻りの形が出るか）
なぜ（組み表_版116 §41）：網の素子は閾値を越えると不応期 3周ごとに撃ち続ける（後過分極・適応なし）【み】11-27 12:38
  「撃ったらしばらく撃たなくなる、続けて撃ったらより撃たなくなる仕組みが足りないわけか」「それで進めてみて」
  切片に電流を流した値（f-I）は駆動の根拠にしない【み】10-11 ⇒ 自然な状態（課題の最中の記録）の並びで形を引く
出すもの（部位の組ごと：皮質（前帯状・前補足・腹内側前頭）／海馬／扁桃体。率で 5Hz 未満・以上に分ける【C の目安：型は波形で分けていない】）
  ① 間隔の割合：3ms 未満・10.67ms 未満（網の1歩）・21.3ms 未満（2歩）（Ison 2011 と比べる）
  ② 戻りの曲線：前の発火からの時間ごとの撃ちやすさ（ハザード ÷ その体の平均の率）の体ごとの中央。
     1 ＝ 前の発火を覚えていない。1 より下 ＝ 撃ちにくい。0.5・0.9 に戻る時間（体ごとの中央）
  ③ 続けて撃ったら：前の 200ms に撃った数（0・1・2・3以上）で分けた戻りの曲線
     ⚠️ 率の高い区間ほど前の数が多いので、③ は「状態が変わった」分も混ざる（適応だけを分けてはいない）
  ④ 前の間隔が短い（2歩 21.3ms 未満）と長い（200ms 以上）で分けた、次の間隔の戻りの曲線
⚠️ てんかんの人・課題の最中・錐体と抑制を分けていない・細胞の外の記録（撃つ体に偏る）
"""
import os
import sys
import glob
import collections
import numpy as np

EDGES_MS = np.array([0, 1, 2, 3, 4, 5, 7, 10.67, 15, 21.3, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000])
NAME = {"dorsal_anterior_cingulate_cortex": "皮質", "pre_supplementary_motor_area": "皮質",
        "ventral_medial_prefrontal_cortex": "皮質", "hippocampus": "海馬", "amygdala": "扁桃体"}
MIN_SPK = 300


def s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def ragged(f, name):
    d = f[name][()]
    ix = f[name + "_index"][()] if name + "_index" in f else None
    if ix is not None:
        ix = np.rint(np.asarray(ix, dtype=float)).astype(np.int64)   # MATLAB で書いた NWB は区切りが小数
    if ix is None:
        return [np.atleast_1d(v) for v in d]
    st = np.concatenate([[0], ix[:-1]])
    return [d[a:b] for a, b in zip(st, ix)]


def area_of(loc):
    loc = loc.lower()
    for k in NAME:
        if loc.startswith(k):
            return NAME[k]
    return None


def hazard(isi_ms, rate_hz, edges=EDGES_MS):
    """間隔の並びから、区間ごとのハザード ÷ 平均の率（区間の頭で生き残っている間隔が 20 未満なら nan）"""
    out = np.full(len(edges) - 1, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        risk = np.sum(isi_ms >= a)
        if risk < 20:
            continue
        n = np.sum((isi_ms >= a) & (isi_ms < b))
        out[i] = n / (risk * (b - a) / 1000.0) / rate_hz
    return out


def back_time(h, lev, edges=EDGES_MS):
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = np.nonzero(np.nan_to_num(h, nan=-1) >= lev)[0]
    return mid[ok[0]] if len(ok) else np.nan


def unit_stats(t):
    t = np.sort(np.asarray(t, dtype=float))
    isi = np.diff(t) * 1000.0
    isi = isi[isi > 0]
    rate = len(t) / (t[-1] - t[0])
    r = {"rate": rate, "n": len(t), "lt3": np.mean(isi < 3), "lt1step": np.mean(isi < 10.67), "lt2step": np.mean(isi < 21.3),
         "cv": np.std(isi) / np.mean(isi), "h": hazard(isi, rate)}
    r["back50"], r["back90"] = back_time(r["h"], 0.5), back_time(r["h"], 0.9)
    # ③ 前の 200ms に撃った数（その発火を除く）
    tt = t[:-1]
    k = np.searchsorted(t, tt, side="left") - np.searchsorted(t, tt - 0.2, side="left")
    nxt = np.diff(t) * 1000.0
    for kk in (0, 1, 2, 3):
        m = (k == kk) if kk < 3 else (k >= 3)
        r[f"hk{kk}"] = hazard(nxt[m & (nxt > 0)], rate) if m.sum() >= 50 else np.full(len(EDGES_MS) - 1, np.nan)
    # ④ 前の間隔が短い・長い
    prev = np.concatenate([[np.inf], np.diff(t)[:-1]]) * 1000.0
    for lab, m in (("short", prev < 21.3), ("long", prev >= 200)):
        r[f"hp_{lab}"] = hazard(nxt[m & (nxt > 0)], rate) if m.sum() >= 50 else np.full(len(EDGES_MS) - 1, np.nan)
    return r


def report(groups):
    cols = [f"{a:g}〜{b:g}" for a, b in zip(EDGES_MS[:-1], EDGES_MS[1:])]
    for key in sorted(groups):
        us = groups[key]
        if not us:
            continue
        print(f"\n━━ {key[0]}・率 {key[1]}  {len(us)}体（{MIN_SPK}発以上）", flush=True)
        q = lambda v: f"中央 {np.nanmedian(v):.3g}（{np.nanpercentile(v, 25):.3g}〜{np.nanpercentile(v, 75):.3g}）"
        print(f"  率 {q([u['rate'] for u in us])}Hz・CV {q([u['cv'] for u in us])}", flush=True)
        print(f"  ① 間隔 3ms 未満 {q([u['lt3'] for u in us])}・10.67ms 未満 {q([u['lt1step'] for u in us])}・21.3ms 未満 {q([u['lt2step'] for u in us])}", flush=True)
        print(f"  ② 0.5 に戻る {q([u['back50'] for u in us])}ms・0.9 に戻る {q([u['back90'] for u in us])}ms", flush=True)
        print("  ② 戻りの曲線（区間 ms：ハザード ÷ 平均の率、体ごとの中央・体の数）", flush=True)
        H = np.array([u["h"] for u in us])
        print("    " + "  ".join(f"{c} {np.nanmedian(H[:, i]):.2f}({int(np.sum(np.isfinite(H[:, i])))})" for i, c in enumerate(cols)), flush=True)
        for kk in (0, 1, 2, 3):
            Hk = np.array([u[f"hk{kk}"] for u in us])
            print(f"  ③ 前の 200ms に {kk}{'以上' if kk == 3 else ''}：" +
                  "  ".join(f"{c} {np.nanmedian(Hk[:, i]):.2f}({int(np.sum(np.isfinite(Hk[:, i])))})" for i, c in enumerate(cols) if i < 15), flush=True)
        for lab, nm in (("short", "前の間隔 21.3ms 未満"), ("long", "前の間隔 200ms 以上")):
            Hp = np.array([u[f"hp_{lab}"] for u in us])
            print(f"  ④ {nm}：" + "  ".join(f"{c} {np.nanmedian(Hp[:, i]):.2f}({int(np.sum(np.isfinite(Hp[:, i])))})" for i, c in enumerate(cols) if i < 15), flush=True)


def main(src):
    import h5py
    files = sorted(glob.glob(os.path.join(os.path.expanduser(src), "**", "*.nwb"), recursive=True))
    if not files:
        sys.exit(f"⚠️ {src} に .nwb が無い")
    groups = collections.defaultdict(list)
    n_all = collections.Counter()
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "units" not in f:
                continue
            locs = [s(v) for v in f["general/extracellular_ephys/electrodes/location"][()]]
            spk = ragged(f["units"], "spike_times")
            u = f["units"]
            el = ragged(u, "electrodes") if "electrodes" in u else [[0]] * len(spk)
            eidx = np.array([int(np.rint(float(np.atleast_1d(e)[0]))) for e in el])
            base = 1 if len(eidx) and (eidx.max() >= len(locs) or eidx.min() < 0) else 0
            if len(eidx) and (eidx - base).max() >= len(locs):
                sys.exit(f"⚠️ {os.path.basename(fn)}：電極の番号が表の外。部位を読めないので止める")
            for t, e in zip(spk, eidx):
                ar = area_of(locs[e - base])
                if ar is None:
                    continue
                n_all[ar] += 1
                t = np.asarray(t, dtype=float)
                if len(t) < MIN_SPK or t.max() <= t.min():
                    continue
                r = unit_stats(t)
                groups[(ar, "5Hz 未満" if r["rate"] < 5 else "5Hz 以上")].append(r)
    print(f"ファイル {len(files)}・部位の組ごとの体（発火数を問わず）：" + "・".join(f"{k} {v}" for k, v in sorted(n_all.items())), flush=True)
    report(groups)


def selftest():
    # 作り物：死に時間 2ms のあと、撃ちやすさが 1 − exp(−(t − 2)/20ms) で戻る形（戻りの 0.5 は 約 16ms、0.9 は 約 48ms）
    #         さらに前の 200ms の数 k ごとに撃ちやすさを 0.7^k 倍（続けて撃つとより撃たない）
    rng = np.random.default_rng(0)
    dt, T, base = 0.0005, 2000.0, 4.0
    t, last, hist = 0.0, -1.0, []
    out = []
    while t < T:
        since = (t - last) * 1000.0
        rec = 0.0 if since < 2 else 1 - np.exp(-(since - 2) / 20.0)
        k = sum(1 for x in hist if t - x < 0.2)
        if rng.random() < base * dt * rec * (0.7 ** k):
            out.append(t); last = t; hist.append(t); hist = [x for x in hist if t - x < 0.2]
        t += dt
    r = unit_stats(np.array(out))
    print("作り物：率 %.2fHz・3ms 未満 %.4f・0.5 に戻る %.1fms・0.9 に戻る %.1fms" % (r["rate"], r["lt3"], r["back50"], r["back90"]))
    assert r["lt3"] < 0.002, "死に時間 2ms が出ない"
    assert 10 <= r["back50"] <= 26, r["back50"]
    h0, h2 = np.nanmean(r["hk0"][8:14]), np.nanmean(r["hk2"][8:14])
    print("作り物：前の 200ms に 0発 → %.2f ／ 2発 → %.2f（30〜200ms の区間の平均。入れた比 0.49 の向き）" % (h0, h2))
    assert h2 < h0, "続けて撃つとより撃たない形が出ない"
    print("✓ 作り物で入れた戻りの形（死に時間・戻りの速さ・続けて撃つと撃ちにくい）が出た")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "~/data/000469")
