"""マーモセット（Majka 2020 の公開データ）の全部の注入から、区画ごとの階層の段を1つの当てはめで出す（arkhe で。網は使わない）
  python3 marmoset_hier.py
【み】10-23 22:35「マーモセットにないなら他をあたって」⇒ マカク（F6 ⇄ F5/44 の層）は見つからない ⇒ 組を1つずつ読むのでなく、全部の組から段を出す
  （Markov 2014 J Comp Neurol「Anatomy of hierarchy」・Theodoni 2022 Cereb Cortex と同じ考え：SLN の組ごとの値を「受け手の段 − 送り手の段」で説明する）
式：logit(SLN_ij) ＝ h(受け手) − h(送り手)。組ごとの重さは細胞の数（上 ＋ 下）の平方根【C】。V1 を 0 に固定。SLN は 0.02〜0.98 に切る【C】
  ⚠️ Markov・Theodoni は probit・ベータ回帰（ばらつきの形を当てる）。ここは重み付きの最小二乗で近い形 ⇒ 段の並びを見る（値の大きさは比べない）
前補足・補足を分ける：受け手は注入の場所（前 CJ170-DY・CJ164-CTBgr ／ 後ろ CJ94-FB ／ 間 CJ113-FE は外す）
                      送り手は A6M の細胞の前後の座標で 前・中・後ろ の3つ（中は外す）【C】。座標の向きは V1 と 8b の細胞から決める
確かめ：V1 ＜ V2 ＜ V4 ／ AuA1 ＜ ベルト（AuML・AuCL・AuAL）の並びになるか
"""
import json, csv, io, sys, math, urllib.request, collections
import numpy as np

BASE = "http://analytics.marmosetbrain.org/static/data/"
SPLIT_T = {"CJ170-DY": "A6M前", "CJ164-CTBgr": "A6M前", "CJ94-FB": "A6M後", "CJ113-FE": None}
MIN_CELLS = 20


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def rows(txt):
    body = "\n".join(l for l in txt.lstrip("﻿").splitlines() if not l.lstrip().startswith("#"))
    rd = csv.reader(io.StringIO(body))
    head = [h.strip().lower() for h in next(rd, [])]
    cs = head.index("source") if "source" in head else None
    kl = next((i for i, h in enumerate(head) if "laminar" in h), None)
    ka = head.index("a-p") if "a-p" in head else None
    if cs is None or kl is None:
        return
    for r in rd:
        if len(r) > max(cs, kl):
            ap = None
            if ka is not None and len(r) > ka:
                try:
                    ap = float(r[ka])
                except ValueError:
                    pass
            yield r[cs].strip(), r[kl].strip(), ap


inj = json.loads(get("injections_static.json"))
print(f"注入 {len(inj)} 回を読む（細胞の表を1つずつ）", flush=True)
raw = {}                                         # 注入 → [(送り手, 層, A-P)]
ap6m, apv1, ap8b = [], [], []
for n, (k, v) in enumerate(sorted(inj.items())):
    try:
        cells = list(rows(get(f"individual_injections/indivdual_cells_injection_{k}.csv")))
    except Exception as e:
        print(f"  ⚠️ {k} が読めない：{e}")
        continue
    raw[k] = (v.get("abbrev"), cells)
    for s, l, ap in cells:
        if ap is None:
            continue
        if s == "A6M":
            ap6m.append(ap)
        elif s == "V1":
            apv1.append(ap)
        elif s == "A8b":
            ap8b.append(ap)
    if (n + 1) % 20 == 0:
        print(f"  {n + 1}/{len(inj)}", flush=True)

front_big = np.median(ap8b) > np.median(apv1) if ap8b and apv1 else True
lo, hi = (min(ap6m), max(ap6m)) if ap6m else (0, 1)
w = (hi - lo) / 3 or 1.0
print(f"座標の向き：V1 {np.median(apv1) if apv1 else '―'} ／ 8b {np.median(ap8b) if ap8b else '―'} ⇒ {'大きいほど前' if front_big else '小さいほど前'}")


def src_name(s, ap):
    if s != "A6M" or ap is None:
        return s
    j = min(2, int((ap - lo) / w))
    part = (("後ろ", "中", "前") if front_big else ("前", "中", "後ろ"))[j]
    return {"前": "A6M前", "後ろ": "A6M後", "中": None}[part]


cnt = collections.defaultdict(lambda: [0, 0])
for k, (tgt, cells) in raw.items():
    t = SPLIT_T.get(k, tgt) if tgt == "A6M" else tgt
    if t is None:
        continue
    for s, l, ap in cells:
        s2 = src_name(s, ap)
        if s2 is None or s2 == t or (s == "A6M" and tgt == "A6M"):
            continue
        if l in ("1", "+1"):
            cnt[(t, s2)][0] += 1
        elif l == "-1":
            cnt[(t, s2)][1] += 1

pairs = [(t, s, u, d) for (t, s), (u, d) in cnt.items() if u + d >= MIN_CELLS]
areas = sorted({a for t, s, _, _ in pairs for a in (t, s)})
ix = {a: i for i, a in enumerate(areas)}
if "V1" not in ix:
    sys.exit("⚠️ V1 が無い")
A, y, wt = [], [], []
for t, s, u, d in pairs:
    sln = min(0.98, max(0.02, u / (u + d)))
    row = np.zeros(len(areas)); row[ix[t]] += 1; row[ix[s]] -= 1
    A.append(row); y.append(math.log(sln / (1 - sln))); wt.append(math.sqrt(u + d))
A, y, wt = np.array(A), np.array(y), np.array(wt)
keep = [i for i in range(len(areas)) if i != ix["V1"]]                 # V1 を 0 に固定
sol, *_ = np.linalg.lstsq(A[:, keep] * wt[:, None], y * wt, rcond=None)
h = np.zeros(len(areas)); h[keep] = sol
# 組の数（その区画が受け手か送り手として出た組）
deg = collections.Counter(a for t, s, _, _ in pairs for a in (t, s))
res = y - A @ h
print(f"\n組 {len(pairs)}（細胞 {MIN_CELLS} 以上）・区画 {len(areas)} ／ 当てはめの残り（logit）の中央 {np.median(np.abs(res)):.2f}")


def show(names):
    for a in names:
        if a in ix:
            print(f"  {a:<8} 段 {h[ix[a]]:+6.2f}（組 {deg[a]}）")
        else:
            print(f"  {a:<8} ―（組が無い）")


print("\n══ 確かめ（V1 ＜ V2 ＜ V4 ／ AuA1 ＜ ベルト のはず）")
show(["V1", "V2", "V4", "AuA1", "AuML", "AuCL", "AuAL"])
print("\n══ 補足運動野まわり（段が大きいほど上）")
show(["A6M前", "A6M後", "A4ab", "A4c", "A6DR", "A6DC", "A6Va", "A6Vb", "A8b", "A8aD", "A8aV", "A46D", "A46V", "A9", "A10",
      "A24c", "A24d", "A32", "A45", "A47L"])
print("\n══ 全部の区画（段の順）")
for a in sorted(areas, key=lambda a: h[ix[a]]):
    print(f"  {h[ix[a]]:+6.2f}  {a}（組 {deg[a]}）")
