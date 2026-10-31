"""F-TRACT（人の CCEP、780人）で、統語の部位と上側頭溝のまわりが繋がるか（刺激すると応える確率）を見る（arkhe で。zip は展開しない）
  python3 ftract_syntax.py ~/f-tract_v2112.zip
11-01 3:53【み】「進めて」：上側頭溝（STS1・STS2）へ下から入る線が人の束で引けない ⇒ 人の CCEP（繋がりのあるなし）で見る
⚠️ CCEP の向き（どちらが上か）は 10-23 に確かめの組で外れた（10 §7）⇒ ここでは繋がりのあるなし（確率）だけを見る。向きは型の表で読む
模型の領野 ⇒ HCP-MMP1 の区画【C】当てはめ：
  聴覚 A1 ／ 連合 MBelt・LBelt・PBelt ／ 高次 A4・A5 ／ STS1（上の岸）STSda・STSdp ／ STS2（下の岸）STSva・STSvp ／
  中側頭回の後ろ TE1p ／ TPJ PSL・STV・TPOJ1 ／ 44 ／ 45 ／ 運動 6v
"""
import sys, re, zipfile, math, statistics

AGE = "ages_15_100"
PARC = "MNI-HCP-MMP1"
FEATS = ["probability"]

z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if not n.startswith("__MACOSX")]


def find(feat):
    c = [n for n in names if AGE in n and f"/{PARC}/" in n and n.rstrip("/").split("/")[-1].startswith(feat + ".")]
    return c[0] if c else None


def num(x):
    try:
        v = float(x)
        return v
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
    """'L_A1_ROI' ／ 'lh.A1' ／ 'A1_L' などから（半球, 区画名）"""
    s = s.strip()
    m = re.match(r"^([LR])_(.+?)(?:_ROI)?$", s) or re.match(r"^(lh|rh)[._](.+)$", s)
    if m:
        h = m.group(1)[0].upper()
        return h, m.group(2)
    m = re.match(r"^(.+?)[_ ]([LR])$", s)
    if m:
        return m.group(2), m.group(1)
    return "?", s


GROUPS = {"聴覚": ["A1"], "連合": ["MBelt", "LBelt", "PBelt"], "高次": ["A4", "A5"],
          "STS1": ["STSda", "STSdp"], "STS2": ["STSva", "STSvp"], "中側頭後": ["TE1p"],
          "TPJ": ["PSL", "STV", "TPOJ1"], "44": ["44"], "45": ["45"], "運動": ["6v"]}

path0 = find("probability")
if path0 is None:
    sys.exit("⚠️ 年齢 15〜100 ／ HCP-MMP1 の probability が見つからない")
header, rows = read_matrix(path0)
print(f"表：{path0}")
print(f"見出し {('あり ' + str(len(header))) if header else 'なし'} ／ 行 {len(rows)} ／ 頭の見出し {header[:6] if header else None}")
print(f"行の頭 {[r[0] for r in rows[:4]]} ／ 1行の長さ {len(rows[0])}")
row_lab = [r[0] for r in rows] if num(rows[0][0]) is None else None
col_lab = header[1:] if header and row_lab else (header if header else None)
if row_lab is None or col_lab is None:
    print("⚠️ 区画の名前の並びが読めない。表の頭を出して止める")
    with z.open(path0) as f:
        for i, raw in enumerate(f):
            if i >= 8:
                break
            print("   " + raw.decode("utf-8", "replace").rstrip()[:400])
    sys.exit(1)
print(f"区画の名前の例 {row_lab[:3]} → {[label_key(x) for x in row_lab[:3]]}")


def idx(labels, hemi, parcels):
    out = []
    for i, l in enumerate(labels):
        h, p = label_key(l)
        if p in parcels and (h == hemi or h == "?"):
            out.append(i)
    return out



_, rr = read_matrix(path0)
M = [[num(c) for c in r[1:]] for r in rr]
names_g = list(GROUPS)
for hemi in ("L", "R"):
    print(f"\n══ {'左' if hemi == 'L' else '右'}：刺激（行）→ 記録（列）で応える確率の中央値〔値のある組の数〕", flush=True)
    miss = [g for g, ps in GROUPS.items() if not idx(row_lab, hemi, ps)]
    if miss:
        print(f"  ⚠️ 区画が見つからない：{miss}", flush=True)
    print("  " + " " * 8 + "".join(f"{g:>13}" for g in names_g), flush=True)
    for ga in names_g:
        ia = idx(row_lab, hemi, GROUPS[ga])
        cells = []
        for gb in names_g:
            ib = idx(col_lab, hemi, GROUPS[gb])
            vals = [M[i][j] for i in ia for j in ib if i != j and i < len(M) and j < len(M[i])
                    and M[i][j] is not None and not math.isnan(M[i][j])]
            cells.append(f"{statistics.median(vals):7.2f}〔{len(vals):2d}〕" if vals else f"{'―':>13}")
        print(f"  {ga:>8}" + "".join(f"{c:>13}" for c in cells), flush=True)
print("\n終わり", flush=True)
