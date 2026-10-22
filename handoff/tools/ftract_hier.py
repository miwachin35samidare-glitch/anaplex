"""F-TRACT（人の CCEP、780人）で、補足運動野まわりの区画の組の「両向き」を並べる（arkhe で。zip は展開しない。網は使わない）
  python3 ftract_hier.py ~/f-tract_v2112.zip
【み】10-23 2:45「1、3、7は多角的に調べ直して」⇒ 7（上下）を人の値で見る1つの角度
【ヒト】Lemaréchal 2022 Brain：780人。表の向き：刺激した区画が行（送り手）、記録した区画が列（受け手）
読み方を先に確かめる：上下が分かっている組（V1 → V2 → V4、A1 → ベルト）で、上りと下りの確率・振幅・遅れがどう違うかを並べる
  ⇒ その違い方が見えたときだけ、補足運動野まわりの組に同じ読み方を当てる（見えなければ、CCEP の向きは上下の根拠にしない）
区画（HCP-MMP1）【C】当てはめ
  前頭前野 46・9-46d・p9-46v・a9-46v ／ 前補足運動野 6ma ／ 補足運動野 6mp ／ SCEF（6ma と 6mp の間）／ 一次運動 4 ／ 運動（模型）6v ／ 44
  確かめの組：V1 ／ V2 ／ V4 ／ A1 ／ ベルト MBelt・LBelt・PBelt
"""
import sys, re, zipfile, math, statistics

AGE = "ages_15_100"
PARC = "MNI-HCP-MMP1"
FEATS = ["probability", "amplitude", "onset_latency", "peak_latency", "axonal_conduction_delay"]
GROUPS = {"前頭前野": ["46", "9-46d", "p9-46v", "a9-46v"], "前補足": ["6ma"], "補足": ["6mp"], "SCEF": ["SCEF"],
          "一次運動": ["4"], "運動6v": ["6v"], "44": ["44"],
          "V1": ["V1"], "V2": ["V2"], "V4": ["V4"], "A1": ["A1"], "ベルト": ["MBelt", "LBelt", "PBelt"]}
PAIRS_CHECK = [("V1", "V2"), ("V2", "V4"), ("V1", "V4"), ("A1", "ベルト")]          # 左が下（上りは左 → 右）
PAIRS = [("前頭前野", "前補足"), ("前頭前野", "補足"), ("前補足", "補足"), ("前補足", "SCEF"), ("SCEF", "補足"),
         ("補足", "一次運動"), ("補足", "運動6v"), ("前補足", "運動6v"), ("前補足", "一次運動"),
         ("前頭前野", "運動6v"), ("44", "前補足"), ("44", "補足"), ("44", "運動6v"), ("一次運動", "運動6v")]

z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if not n.startswith("__MACOSX")]


def find(feat):
    c = [n for n in names if AGE in n and f"/{PARC}/" in n and n.rstrip("/").split("/")[-1].startswith(feat + ".")]
    return c[0] if c else None


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def read_matrix(path):
    rows, header = [], None
    with z.open(path) as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith("#"):
                continue
            cells = [c.strip().strip('"') for c in re.split(r"[,;\t]", line)]
            if header is None and all(num(c) is None for c in cells[1:] if c):
                header = cells
                continue
            rows.append(cells)
    return header, rows


def label_key(s):
    s = s.strip()
    m = re.match(r"^([LR])_(.+?)(?:_ROI)?$", s) or re.match(r"^(lh|rh)[._](.+)$", s)
    if m:
        return m.group(1)[0].upper(), m.group(2)
    m = re.match(r"^(.+?)[_ ]([LR])$", s)
    if m:
        return m.group(2), m.group(1)
    return "?", s


p0 = find("probability")
if p0 is None:
    sys.exit("⚠️ 年齢 15〜100 ／ HCP-MMP1 の probability が見つからない")
header, rows = read_matrix(p0)
row_lab = [r[0] for r in rows]
col_lab = header[1:]
print(f"表：{p0}（行 {len(rows)}）")
mats = {}
for feat in FEATS:
    p = find(feat)
    if p is None:
        print(f"  （{feat} の表が無い）")
        continue
    _, rr = read_matrix(p)
    mats[feat] = [[num(c) for c in r[1:]] for r in rr]
print("  使う表：" + "・".join(mats))
allp = sorted({label_key(l)[1] for l in row_lab})
for g, ps in GROUPS.items():
    miss = [p for p in ps if p not in allp]
    if miss:
        print(f"  ⚠️ {g} の区画 {miss} が表に無い")


def idx(labels, hemi, parcels):
    return [i for i, l in enumerate(labels) if label_key(l)[1] in parcels and label_key(l)[0] in (hemi, "?")]


def cell(feat, hemi, ga, gb):
    M = mats[feat]
    ia, ib = idx(row_lab, hemi, GROUPS[ga]), idx(col_lab, hemi, GROUPS[gb])
    v = [M[i][j] for i in ia for j in ib if i < len(M) and j < len(M[i]) and M[i][j] is not None and not math.isnan(M[i][j])]
    return (statistics.median(v), len(v)) if v else (None, 0)


def show(pairs, title):
    for hemi in ("L", "R"):
        print(f"\n══ {title}（{'左' if hemi == 'L' else '右'}）区画の組の中央値〔値のある組の数〕")
        print(f"  {'組':<18}" + "".join(f"{f[:13]:>22}" for f in mats))
        for a, b in pairs:
            for s, d in ((a, b), (b, a)):
                cs = []
                for f in mats:
                    v, n = cell(f, hemi, s, d)
                    cs.append(f"{v:12.3f}〔{n:3d}〕" if v is not None else f"{'―':>17}")
                print(f"  {s + ' → ' + d:<18}" + "".join(f"{c:>22}" for c in cs))
            print()


show(PAIRS_CHECK, "確かめ：上下が分かっている組（上の行が上り）")
show(PAIRS, "補足運動野まわり")
