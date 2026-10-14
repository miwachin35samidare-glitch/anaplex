"""F-TRACT（人の CCEP、780人）から、模型の領野の組ごとの遅れを抜き出す（arkhe で。zip は展開しない。網は使わない）
  python3 ftract_delays.py ~/f-tract_v2112.zip
【ヒト】Lemaréchal 2022 Brain（15歳超の軸索の遅れ 中央 10.2ms）。年齢 15〜100 の表・HCP-MMP1（360区画）を使う
表の向き：刺激した区画が縦（行）、記録した区画が横（列）⇒ 行 ＝ 送り手、列 ＝ 受け手
模型の領野 ⇒ HCP-MMP1 の区画【C】当てはめ（左の半球。右も並べる）
  聴覚 A1 ／ 連合 MBelt・LBelt・PBelt ／ 高次 A4・A5 ／ 運動 6v ／ 前頭前野 46・9-46d・a9-46v・p9-46v ／ 嗅内 EC ／ 海馬 H
出すもの（組ごとに、区画の組の中央値と、値のある組の数）
  繋がる確率 ／ 始まりの遅れ onset_latency ／ 山の遅れ peak_latency ／ 軸索の遅れ axonal_conduction_delay ／ onset_delay
  区画ごとの興奮・抑制のシナプスの遅れ（あれば）
"""
import sys, re, zipfile, math, statistics

AGE = "ages_15_100"
PARC = "MNI-HCP-MMP1"
FEATS = ["probability", "onset_latency", "peak_latency", "axonal_conduction_delay", "onset_delay"]
GROUPS = {"聴覚": ["A1"], "連合": ["MBelt", "LBelt", "PBelt"], "高次": ["A4", "A5"], "運動": ["6v"],
          "前頭前野": ["46", "9-46d", "a9-46v", "p9-46v"], "嗅内": ["EC"], "海馬": ["H"]}

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


mats = {}
for feat in FEATS:
    p = find(feat)
    if p is None:
        print(f"  （{feat} の表が無い）")
        continue
    _, rr = read_matrix(p)
    mats[feat] = [[num(c) for c in r[1:]] for r in rr]

for hemi in ("L", "R"):
    print(f"\n══ {'左' if hemi == 'L' else '右'}の半球（行：送り手 → 受け手。区画の組の中央値〔値のある組の数〕）")
    miss = [g for g, ps in GROUPS.items() if not idx(row_lab, hemi, ps)]
    if miss:
        print(f"  ⚠️ 区画が見つからない領野：{miss}")
    print("  送り手 → 受け手      " + "  ".join(f"{f[:14]:>16}" for f in mats))
    for ga, pa in GROUPS.items():
        ia = idx(row_lab, hemi, pa)
        for gb, pb in GROUPS.items():
            if ga == gb:
                continue
            ib = idx(col_lab, hemi, pb)
            cells = []
            for feat, M in mats.items():
                vals = [M[i][j] for i in ia for j in ib if i < len(M) and j < len(M[i])
                        and M[i][j] is not None and not math.isnan(M[i][j])]
                cells.append(f"{statistics.median(vals):9.2f}〔{len(vals):2d}〕" if vals else f"{'―':>16}")
            print(f"  {ga:>5} → {gb:<5}  " + "  ".join(cells))

print("\n══ 区画ごとのシナプスの遅れ（あれば）")
for feat in ("synaptic_excitatory_delay", "synaptic_inhibitory_delay"):
    c = [n for n in names if AGE in n and f"/{PARC}/" in n and feat in n.split("/")[-1]]
    if not c:
        print(f"  {feat}：HCP-MMP1 には無い")
        continue
    with z.open(c[0]) as f:
        txt = f.read().decode("utf-8", "replace").splitlines()
    print(f"  {c[0]}（{len(txt)}行）")
    for line in txt:
        for g, ps in GROUPS.items():
            for p in ps:
                if re.search(rf"(^|[^\w-]){re.escape(p)}([^\w-]|$)", line):
                    print(f"    {g} {p}: {line.strip()[:120]}")
