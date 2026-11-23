"""F-TRACT の確率が見分けに使えるかを確かめる（arkhe で）
  python3 ftract_check.py ~/f-tract_v2112.zip
11-01：ftract_syntax で聴覚・統語のまわりがほぼ全部 0.8〜1.0 ⇒ 表全体でもそうなら「繋がりのあるなし」は読めない
 1) 同じ置き場にある表の名前（刺激の数・人数の表があるか）
 2) 確率の表全体の値の散らばり
 3) 遠い組（V1 ⇄ 45、V1 ⇄ A1、45 ⇄ 4 など）の値＝比べの組
"""
import sys, re, zipfile, math, statistics
z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
base = "ages_15_100/sr_8.40/seg_None_None/pl_200/MNI-HCP-MMP1/export/"
print("── 置き場の表")
for n in sorted(set("/".join(x.split("/")[:8]) for x in names if x.startswith(base))):
    print("  ", n)
def num(x):
    try: return float(x)
    except ValueError: return None
def read(path):
    with z.open(path) as f:
        L = [re.split(r"[,;\t]", raw.decode("utf-8", "replace").strip()) for raw in f if raw.strip()]
    head = [c.strip().strip('"') for c in L[0]]
    rows = [[c.strip().strip('"') for c in r] for r in L[1:]]
    return head[1:], [r[0] for r in rows], [[num(c) for c in r[1:]] for r in rows]
cols, rows, M = read(base + "probability/probability.csv")
v = [x for r in M for x in r if x is not None and not math.isnan(x)]
print(f"\n── 確率の表全体：値のある組 {len(v)} ／ {len(rows)*len(cols)}")
q = statistics.quantiles(v, n=10)
print("  十分位：", " ".join(f"{x:.2f}" for x in q))
for t in (0.5, 0.8, 0.9, 0.95):
    print(f"  {t} 以上の割合 {sum(x >= t for x in v)/len(v):.2f}")
def get(a, b):
    try: x = M[rows.index(a)][cols.index(b)]
    except ValueError: return "名前なし"
    return "―" if x is None or math.isnan(x) else f"{x:.2f}"
print("\n── 遠い組（比べ）と近い組")
for a, b in [("L_V1","L_45"),("L_45","L_V1"),("L_V1","L_A1"),("L_A1","L_V1"),("L_45","L_4"),("L_4","L_45"),
             ("L_V1","L_V2"),("L_A1","L_MBelt"),("L_10pp","L_V1"),("L_V1","L_10pp"),("L_45","L_STSva"),("L_A1","L_STSdp")]:
    print(f"  {a:>8} → {b:<8} {get(a, b)}")
# 他に表があれば（人数・刺激の数など）同じ組を出す
others = sorted(set(x for x in names if x.startswith(base) and x.endswith(".csv") and "probability" not in x))
for o in others[:6]:
    try:
        c2, r2, M2 = read(o)
    except Exception as e:
        print(f"\n── {o}：読めない {e}"); continue
    print(f"\n── {o}")
    for a, b in [("L_V1","L_45"),("L_A1","L_STSdp"),("L_45","L_STSva"),("L_A1","L_MBelt")]:
        try: x = M2[r2.index(a)][c2.index(b)]
        except ValueError: x = "名前なし"
        print(f"  {a:>8} → {b:<8} {x}")
print("\n終わり")
