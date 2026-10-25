"""人の扁桃体と海馬の発火の列を同じ物差しで比べる（Faraut 2018、42人・1,576体）
（arkhe で。網は使わない。何も書き換えない）
  ~/anaplex の外でよい。データは Dryad（doi:10.5061/dryad.46st5）の RecogMemory_MTL_release_v2.zip（58MB）
  ~/siibra-venv/bin/python faraut_amy_hip.py ~/Downloads/RecogMemory_MTL_release_v2.zip
  ⚠️ システムの python3 は scipy（apt）と numpy 2（~/.local）が合わず止まる ⇒ siibra の箱で回す（scikit-image の依存で scipy が入っている）
  （展開したフォルダを渡してもよい）
なぜ：扁桃体（基底外側）の素子を今の海馬の型で組めるかを、人の自然な撃ち方で決める【み】10-25 23:35「イ」
  10 §2：作り直すのは挙動の違う型だけ・焼く的は人の自然な状態の列
データの作り（rutishauserlab/recogmem-release-NWB の data.py のとおり）
  Data/events/<session>/<task>/brainArea(NEW).mat ：列 0 チャンネル・1 クラスタ・2 元のクラスタ・3 部位（1 右海馬・2 左海馬・3 右扁桃体・4 左扁桃体・13 左海馬・18 右海馬）
  Data/sorted/<session>/<task>/A<ch>_cells.mat の spikes：列 0 クラスタ・列 2 時刻（µs）
  Data/events/<session>/<task>/eventsRaw.mat か events.mat の events：列 0 時刻（µs）⇒ 記録の始めと終わり
出すもの（部位ごと。体ごとの値の中央値と四分位、扁桃体と海馬の差の検定）
  率（Hz）・間隔の CV・LV（局所のばらつき）・3ms 以内の割合・10ms 未満の割合（Ison 2011 と同じ物差し）・模型の1歩（10.67ms）の箱での間隔の分布（1・2・3〜4・5〜8・9〜16・17〜64・65歩〜）
  率 5Hz 未満と以上に分けた同じ表（Ison の基準1 の率の境。波形の幅は使っていない）
⚠️ てんかんの人・課題（見た絵を覚える・思い出す）の最中の記録
"""
import os
import sys
import glob
import zipfile
import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu, ks_2samp

STEP = 10.67e-3
AREA = {1: "海馬", 2: "海馬", 13: "海馬", 18: "海馬", 3: "扁桃体", 4: "扁桃体"}
SIDE = {1: "右", 2: "左", 13: "左", 18: "右", 3: "右", 4: "左"}
EDGES = [1, 2, 3, 5, 9, 17, 65, 10 ** 9]          # 歩
EDGE_LAB = ["1歩", "2歩", "3〜4", "5〜8", "9〜16", "17〜64", "65〜"]

if len(sys.argv) < 2:
    sys.exit("使い方：python3 faraut_amy_hip.py <zip かフォルダ>")
src = os.path.expanduser(sys.argv[1])
if not os.path.exists(src):                       # 置き場所の名前が違うとき、近い候補を出して止まる
    home = os.path.expanduser("~")
    cand = []
    for d in (home, os.path.join(home, "Downloads"), os.path.join(home, "ダウンロード"), os.getcwd()):
        for pat in ("*RecogMemory*", "*46st5*", "*dryad*"):
            cand += glob.glob(os.path.join(d, pat))
    sys.exit(f"⚠️ {src} が無い。候補：" + ("／".join(sorted(set(cand))) if cand else "見つからない（Dryad から落とした zip の場所を渡す）"))
if src.endswith(".zip"):
    out = os.path.splitext(src)[0]
    if not os.path.isdir(out):
        print(f"展開：{out}", flush=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
    src = out
for zp in glob.glob(os.path.join(src, "**", "*.zip"), recursive=True):   # zip の中の zip
    d = os.path.splitext(zp)[0]
    if not os.path.isdir(d):
        print(f"展開：{d}", flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)

bas = sorted(glob.glob(os.path.join(src, "**", "events", "**", "brainArea*.mat"), recursive=True))
dirs = {}
for p in bas:                                     # 同じフォルダに NEW があれば NEW を使う
    d = os.path.dirname(p)
    if d not in dirs or os.path.basename(p) == "brainAreaNEW.mat":
        dirs[d] = p
print(f"部位の表 {len(dirs)} 個", flush=True)
if not dirs:
    sys.exit("⚠️ brainArea.mat が見つからない。フォルダの中身：" + "／".join(sorted(os.listdir(src))[:20]))


def segments(evdir, spikes_all):
    """記録した区間（µs）。events の列 2（実験の番号）ごとに、その番号の最初から最後まで。
    1回目の版は events の最初から最後までを1つの区間にした ⇒ 覚える課題と思い出す課題のあいだ（記録していない時間）まで入り、率が低く CV が大きく出た（10-25 23:44）"""
    for name in ("eventsRaw.mat", "events.mat"):
        p = os.path.join(evdir, name)
        if os.path.isfile(p):
            try:
                ev = np.asarray(loadmat(p)["events"], dtype=float)
            except Exception:
                continue
            if ev.ndim == 2 and ev.shape[1] >= 3:
                segs = [(ev[ev[:, 2] == e, 0].min(), ev[ev[:, 2] == e, 0].max()) for e in np.unique(ev[:, 2])]
            else:
                segs = [(ev[:, 0].min(), ev[:, 0].max())]
            segs = sorted((a0, b0) for a0, b0 in segs if b0 > a0)
            merged = []                               # 重なる区間は1つにまとめる（2回目の版は重なりを2度数えた、10-25 23:47）
            for a0, b0 in segs:
                if merged and a0 <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b0))
                else:
                    merged.append((a0, b0))
            return merged, name
    t = np.concatenate(spikes_all) if spikes_all else np.zeros(1)
    return [(float(t.min()), float(t.max()))], "発火の端"


def stats(parts):
    isi = np.concatenate([np.diff(x) for x in parts if len(x) > 1]) if parts else np.zeros(0)
    if len(isi) < 20:
        return None
    cv = isi.std() / isi.mean()
    lv = np.mean([3 * np.mean(((np.diff(x)[:-1] - np.diff(x)[1:]) / (np.diff(x)[:-1] + np.diff(x)[1:])) ** 2)
                  for x in parts if len(x) > 2])
    k = np.floor(isi / STEP).astype(np.int64) + 1          # 何歩目に次が来たか（同じ歩は 1）
    h = np.histogram(k, bins=EDGES)[0] / len(isi)
    return dict(cv=cv, lv=lv, b3=np.mean(isi < 3e-3), b10=np.mean(isi < 10e-3), h=h, long=np.mean(isi > 5.0))


rows = []
miss = 0
hows, nseg, durs, gaps = {}, [], [], []
for evdir, bap in dirs.items():
    sortdir = evdir.replace(os.sep + "events" + os.sep, os.sep + "sorted" + os.sep)
    try:
        ba = np.asarray(loadmat(bap)["brainArea"])
    except Exception as e:
        print(f"  ✗ {bap}：{e}", flush=True)
        continue
    cells, alls = [], []
    for r in ba:
        ch, cl, code = int(r[0]), int(r[1]), int(r[3])
        if ch == 0 or cl == 0 or code not in AREA:
            continue
        p = os.path.join(sortdir, f"A{ch}_cells.mat")
        if not os.path.isfile(p):
            miss += 1
            continue
        sp = np.asarray(loadmat(p)["spikes"], dtype=float)
        ts = np.sort(sp[sp[:, 0] == cl, 2]) * 1e-6
        if len(ts):
            cells.append((code, ts))
            alls.append(ts * 1e6)
    if not cells:
        continue
    segs, how = segments(evdir, alls)
    dur = sum(b0 - a0 for a0, b0 in segs) * 1e-6
    whole = (max(b0 for _, b0 in segs) - min(a0 for a0, _ in segs)) * 1e-6
    hows[how] = hows.get(how, 0) + 1
    nseg.append(len(segs)); durs.append(dur); gaps.append(whole - dur)
    for code, ts in cells:
        parts = [ts[(ts >= a0 * 1e-6) & (ts <= b0 * 1e-6)] for a0, b0 in segs]
        s = stats(parts)
        if s is None or dur <= 0:
            continue
        s.update(area=AREA[code], side=SIDE[code], rate=sum(len(x) for x in parts) / dur, sess=evdir)
        rows.append(s)
print(f"区間の重なりをまとめた（負の「あいだ」は出ないはず）")
print(f"区間の取り方 {hows}・区間の数 中央値 {np.median(nseg):.0f}（{min(nseg)}〜{max(nseg)}）・記録した長さ 中央値 {np.median(durs) / 60:.1f}分・区間のあいだ（数えない）中央値 {np.median(gaps) / 60:.1f}分", flush=True)
print(f"数えた体 {len(rows)}（扁桃体 {sum(r['area'] == '扁桃体' for r in rows)}・海馬 {sum(r['area'] == '海馬' for r in rows)}）・発火のファイルなし {miss}", flush=True)


def q(x):
    x = np.asarray(x)
    return f"{np.median(x):.3g}（{np.percentile(x, 25):.3g}, {np.percentile(x, 75):.3g}）"


def table(sel, title):
    print(f"\n【{title}】", flush=True)
    a = [r for r in sel if r["area"] == "扁桃体"]
    h = [r for r in sel if r["area"] == "海馬"]
    print(f"  体：扁桃体 {len(a)}・海馬 {len(h)}", flush=True)
    if len(a) < 5 or len(h) < 5:
        print("  少なすぎて比べない", flush=True)
        return
    for key, lab in (("rate", "率 Hz"), ("cv", "間隔の CV"), ("lv", "LV"), ("b3", "3ms 以内"), ("b10", "10ms 未満"), ("long", "5秒超")):
        xa, xh = [r[key] for r in a], [r[key] for r in h]
        p1 = mannwhitneyu(xa, xh).pvalue
        p2 = ks_2samp(xa, xh).pvalue
        print(f"  {lab:<8} 扁桃体 {q(xa):<26} 海馬 {q(xh):<26} 差の検定 MW p={p1:.2g}・KS p={p2:.2g}", flush=True)
    ha = np.mean([r["h"] for r in a], axis=0)
    hh = np.mean([r["h"] for r in h], axis=0)
    print("  次の発火までの歩（模型の1歩 10.67ms の箱、体ごとの割合の平均）", flush=True)
    print("    " + "  ".join(f"{l:>6}" for l in EDGE_LAB), flush=True)
    print("    扁桃体 " + "  ".join(f"{v:6.1%}" for v in ha), flush=True)
    print("    海馬   " + "  ".join(f"{v:6.1%}" for v in hh), flush=True)


table(rows, "全部")
table([r for r in rows if r["rate"] < 5], "率 5Hz 未満（錐体らしい側）")
table([r for r in rows if r["rate"] >= 5], "率 5Hz 以上（介在らしい側）")
table([r for r in rows if r["side"] == "左"], "左だけ")
table([r for r in rows if r["side"] == "右"], "右だけ")
