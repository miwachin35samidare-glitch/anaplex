"""マーモセットの公開データ（Majka 2020 Nat Commun、marmosetbrain.org。登録なし）から、区画の組の両向きの SLN を数える（arkhe で。網は使わない）
  python3 marmoset_sln.py
【み】10-23 2:57「今手に入らなかったところは降りていいよ」⇒ 7（上下）を霊長類で。マカクの層のデータ（core-nets）は登録制 ⇒ マーモセット
SLN ＝ 送り手の区画で、層4より上（Laminar position +1）の細胞の数 ÷（上 ＋ 下 −1）。0（決まらない）は数えない
  SLN ＞ 0.5 ＝ 上り（送り手が下）／ ＜ 0.5 ＝ 下り（送り手が上）／ 0.5 前後 ＝ 横（Markov 2014 の読み方）
読み方を先に確かめる：上下が分かっている組（V1 → V2 → V4、A1 → ベルト）で SLN が上りの側で ＞ 0.5 になるか
版2（10-23 20:04）：A6M を前・後ろに分ける
  受け手：注入の場所で（Majka 2021 bioRxiv 2021.05.21.445121）前 ＝ CJ170-DY・CJ164-CTBgr（前補足にあたる）／ 後ろ ＝ CJ94-FB（補足にあたる）／ 間 ＝ CJ113-FE
  送り手：A6M の細胞を前後の座標（A-P）で 前・中・後ろ の3つに（A6M の細胞の A-P の幅を3等分【C】）
区画【C】当てはめ（マーモセット）
  前補足・補足 A6M（ひとまとめ。4回の注入）／ 一次運動 A4ab ／ 運動前 A6DR・A6DC・A6Va・A6Vb ／ 前頭前野 A8b・A8aD・A46D・A9 ／ 帯状 A24c・A24d・A32
"""
import json, csv, io, sys, urllib.request, collections

BASE = "http://analytics.marmosetbrain.org/static/data/"
AREAS = ["A6M", "A4ab", "A6DR", "A6DC", "A6Va", "A6Vb", "A8b", "A8aD", "A46D", "A9", "A24c", "A24d", "A32"]
CHECK = ["V1", "V2", "V4", "AuA1", "AuML", "AuCL", "AuAL"]
PAIRS_CHECK = [("V1", "V2"), ("V2", "V4"), ("V1", "V4"), ("AuA1", "AuML"), ("AuA1", "AuCL"), ("AuA1", "AuAL")]
PAIRS = [("A8b", "A6M"), ("A46D", "A6M"), ("A9", "A6M"), ("A8aD", "A6M"), ("A6M", "A4ab"), ("A6M", "A6DR"),
         ("A6M", "A6DC"), ("A6M", "A6Va"), ("A6M", "A6Vb"), ("A6Va", "A4ab"), ("A6DR", "A4ab"), ("A8b", "A6DR"),
         ("A46D", "A6DR"), ("A8b", "A46D"), ("A24c", "A6M"), ("A24d", "A6M"), ("A46D", "A6Va")]


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


inj = json.loads(get("injections_static.json"))
want = set(AREAS) | set(CHECK)
by_area = collections.defaultdict(list)
for k, v in inj.items():
    if v.get("abbrev") in want:
        by_area[v["abbrev"]].append(k)
print("注入（的の区画：回数）" + "・".join(f"{a} {len(by_area[a])}" for a in AREAS + CHECK))
SPLIT_T = {"CJ170-DY": "A6M前", "CJ164-CTBgr": "A6M前", "CJ94-FB": "A6M後", "CJ113-FE": "A6M間"}
AP6M, CELLS6M = [], []
APREF = collections.defaultdict(list)                     # 座標の向きを決める：V1（後ろ）と 8b（前）の細胞の A-P                                    # 送り手 A6M の細胞（受け手, A-P, 層）
cnt = collections.defaultdict(lambda: [0, 0, 0])          # (target, source) → [上, 下, 決まらない]
per = collections.defaultdict(list)                       # (target, source) → 注入ごとの SLN
for a, ks in by_area.items():
    for k in ks:
        try:
            txt = get(f"individual_injections/indivdual_cells_injection_{k}.csv")
        except Exception as e:                                # 一つ落ちても続ける
            print(f"  ⚠️ {k} が読めない：{e}")
            continue
        loc = collections.defaultdict(lambda: [0, 0])
        body = "\n".join(l for l in txt.lstrip("\ufeff").splitlines() if not l.lstrip().startswith("#"))   # 頭の説明（#）を飛ばす
        rd = csv.reader(io.StringIO(body))
        head = [h.strip().lower() for h in next(rd, [])]
        cs = [i for i, h in enumerate(head) if h == "source"]
        kl = [i for i, h in enumerate(head) if "laminar" in h]
        if not cs or not kl:                                  # 表の形が違う ⇒ ヘッダーを出して飛ばす
            print(f"  ⚠️ {k} の表の形が違う（飛ばす）：{head[:10]} ／ 1行目 {body.splitlines()[1][:120] if len(body.splitlines()) > 1 else ''}")
            continue
        kap = [i for i, h in enumerate(head) if h == "a-p"]
        tg = SPLIT_T.get(k, a)                                 # 版2：A6M の注入は 前・後ろ・間 に分ける
        for r_ in rd:
            if len(r_) <= max(cs[0], kl[0]):
                continue
            src, lam = r_[cs[0]].strip(), r_[kl[0]].strip()
            if src in ("V1", "A8b") and kap:
                try:
                    APREF[src].append(float(r_[kap[0]]))
                except ValueError:
                    pass
            if src == "A6M" and kap:
                try:
                    ap = float(r_[kap[0]])
                    AP6M.append(ap)
                    CELLS6M.append((tg, ap, lam))
                except ValueError:
                    pass
            j = 0 if lam in ("1", "+1") else 1 if lam == "-1" else 2
            for key in {(a, src), (tg, src)}:                  # A6M の注入は「A6M」と「A6M前・後・間」の両方に数える
                cnt[key][j] += 1
            if j < 2:
                loc[src][j] += 1
        for src, (s, i) in loc.items():
            if s + i >= 10:
                per[(a, src)].append(s / (s + i))


def line(src, tgt):
    s, i, u = cnt[(tgt, src)]
    if s + i == 0:
        return f"  {src + ' → ' + tgt:<16}  {'―':>6}  （的の注入 {len(by_area.get(tgt, []))}・細胞 0）"
    p = per[(tgt, src)]
    rng = f"{min(p):.2f}〜{max(p):.2f}" if p else "―"
    return f"  {src + ' → ' + tgt:<16}  {s / (s + i):6.2f}  上 {s:5d}・下 {i:5d}・決まらない {u:4d} ／ 注入ごと {rng}（{len(p)}回）"


for title, pairs in (("確かめ：上下が分かっている組（上の行が上り。上りなら SLN ＞ 0.5 のはず）", PAIRS_CHECK),
                     ("補足運動野まわり", PAIRS)):
    print(f"\n══ {title}")
    for a, b in pairs:
        print(line(a, b)); print(line(b, a)); print()


# ── 版2：A6M を前・後ろに分けて ──
print("\n══ A6M を受け手として注入の場所で分ける（前 ＝ 前補足・後ろ ＝ 補足・間）")
for src in ("A8b", "A8aD", "A46D", "A9", "A4ab", "A6DR", "A6DC", "A6Va", "A6Vb", "A24c", "A24d", "A32"):
    for tg in ("A6M前", "A6M間", "A6M後"):
        s_, i_, u_ = cnt[(tg, src)]
        v = f"{s_ / (s_ + i_):5.2f}" if s_ + i_ else "  ―  "
        print(f"  {src + ' → ' + tg:<16} {v}  上 {s_:5d}・下 {i_:5d}")
    print()
if AP6M:
    lo, hi = min(AP6M), max(AP6M)
    w = (hi - lo) / 3 or 1.0
    import statistics as _st
    mv1 = _st.median(APREF["V1"]) if APREF["V1"] else None
    m8b = _st.median(APREF["A8b"]) if APREF["A8b"] else None
    front_big = (m8b > mv1) if (mv1 is not None and m8b is not None) else True
    print(f"\n座標の向き：V1 の細胞の A-P 中央 {mv1} ／ 8b {m8b} ⇒ {'大きいほど前' if front_big else '小さいほど前'}"
          + ("" if (mv1 is not None and m8b is not None) else "（⚠️ 決められない。大きいほど前と見た）"))

    def part(ap):
        j = min(2, int((ap - lo) / w))
        return (("後ろ", "中", "前") if front_big else ("前", "中", "後ろ"))[j]
    print(f"══ A6M を送り手として前後の座標で3つに（A-P {lo:.2f}〜{hi:.2f}）")
    agg = collections.defaultdict(lambda: [0, 0])
    for tg, ap, lam in CELLS6M:
        tg0 = tg.replace("A6M前", "A6M").replace("A6M後", "A6M").replace("A6M間", "A6M")
        if lam in ("1", "+1"):
            agg[(tg0, part(ap))][0] += 1
        elif lam == "-1":
            agg[(tg0, part(ap))][1] += 1
    for tg0 in ("A4ab", "A8b", "A8aD", "A46D", "A6DR", "A6DC", "A6Va", "A24d", "A9"):
        for pt in ("前", "中", "後ろ"):
            s_, i_ = agg[(tg0, pt)]
            v = f"{s_ / (s_ + i_):5.2f}" if s_ + i_ else "  ―  "
            print(f"  {'A6M' + pt + ' → ' + tg0:<16} {v}  上 {s_:5d}・下 {i_:5d}")
        print()
