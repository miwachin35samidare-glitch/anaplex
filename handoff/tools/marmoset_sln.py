"""マーモセットの公開データ（Majka 2020 Nat Commun、marmosetbrain.org。登録なし）から、区画の組の両向きの SLN を数える（arkhe で。網は使わない）
  python3 marmoset_sln.py
【み】10-23 2:57「今手に入らなかったところは降りていいよ」⇒ 7（上下）を霊長類で。マカクの層のデータ（core-nets）は登録制 ⇒ マーモセット
SLN ＝ 送り手の区画で、層4より上（Laminar position +1）の細胞の数 ÷（上 ＋ 下 −1）。0（決まらない）は数えない
  SLN ＞ 0.5 ＝ 上り（送り手が下）／ ＜ 0.5 ＝ 下り（送り手が上）／ 0.5 前後 ＝ 横（Markov 2014 の読み方）
読み方を先に確かめる：上下が分かっている組（V1 → V2 → V4、A1 → ベルト）で SLN が上りの側で ＞ 0.5 になるか
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
        for r_ in rd:
            if len(r_) <= max(cs[0], kl[0]):
                continue
            src, lam = r_[cs[0]].strip(), r_[kl[0]].strip()
            c = cnt[(a, src)]
            if lam in ("1", "+1"):
                c[0] += 1; loc[src][0] += 1
            elif lam == "-1":
                c[1] += 1; loc[src][1] += 1
            else:
                c[2] += 1
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
