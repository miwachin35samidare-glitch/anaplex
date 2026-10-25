"""人の扁桃体と海馬の発火の列を同じ物差しで比べる（Faraut 2018、42人・1,576体）
（arkhe で。網は使わない。何も書き換えない）
  ~/anaplex の外でよい。データは Dryad（doi:10.5061/dryad.46st5）の RecogMemory_MTL_release_v2.zip（58MB）
  python3 faraut_amy_hip.py ~/Downloads/RecogMemory_MTL_release_v2.zip
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


def span(evdir, spikes_all):
    for name in ("eventsRaw.mat", "events.mat"):
        p = os.path.join(evdir, name)
        if os.path.isfile(p):
            try:
                ev = np.asarray(loadmat(p)["events"], dtype=float)
                t = ev[:, 0]
                return float(t.min()), float(t.max()), name
            except Exception:
                pass
    t = np.concatenate(spikes_all) if spikes_all else np.zeros(1)
    return float(t.min()), float(t.max()), "発火の端"


def stats(ts):
    isi = np.diff(ts)
    if len(isi) < 20:
        return None
    cv = isi.std() / isi.mean()
    lv = 3 * np.mean(((isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:])) ** 2)
    k = np.floor(isi / STEP).astype(np.int64) + 1          # 何歩目に次が来たか（同じ歩は 1）
    h = np.histogram(k, bins=EDGES)[0] / len(isi)
    return dict(cv=cv, lv=lv, b3=np.mean(isi < 3e-3), b10=np.mean(isi < 10e-3), h=h)


rows = []
miss = 0
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
    t0, t1, how = span(evdir, alls)
    dur = (t1 - t0) * 1e-6
    for code, ts in cells:
        ts = ts[(ts >= t0 * 1e-6) & (ts <= t1 * 1e-6)]
        s = stats(ts)
        if s is None or dur <= 0:
            continue
        s.update(area=AREA[code], side=SIDE[code], rate=len(ts) / dur, sess=evdir)
        rows.append(s)
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
    for key, lab in (("rate", "率 Hz"), ("cv", "間隔の CV"), ("lv", "LV"), ("b3", "3ms 以内"), ("b10", "10ms 未満")):
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
