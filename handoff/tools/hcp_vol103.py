"""版103：中側頭回の後ろの体数を、1つの地図の中の比で出し直す（地図をまたぐ換算を消す）
  python3 hcp_vol103.py        （arkhe で。ネットから表を1つ取るだけ。網には触らない）
  表：HCP-MMP1_UniqueRegionList.csv（Glasser 2016 の 360区画を MNI に写した体積 mm³。neuroimaging-core-docs が配る表）
      https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv
  出すもの
    1 表の頭の行と、使う区画の行（そのまま）
    2 区画の体積（左＋右）と、高次（A4・A5、rosen_syntax と同じ当てはめ）との比 × 224 体
      ⇒ 今の Julich の体数（44 180・45 241・STS1 180・STS2 321・中側頭後 436・TPJ 98）と並べる
  ⚠️ 区画の当てはめ（高次 A4・A5／STS1 STSda・STSdp／STS2 STSva・STSvp／中側頭後 TE1p／TPJ PSL・STV・TPOJ1）は rosen_syntax と同じ【C】
  ⚠️ 体積で比べるのは同じ密度を当てるとき（今の決まり：新しい領野とも高次の Tpt の密度）
"""
import csv, io, sys, urllib.request

URL = "https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv"
OUT = "data_hcp_vol103.txt"
MAP = {"高次": ("A4", "A5"), "44": ("44",), "45": ("45",), "STS1": ("STSda", "STSdp"), "STS2": ("STSva", "STSvp"),
       "中側頭後": ("TE1p",), "TPJ": ("PSL", "STV", "TPOJ1")}
NOW = {"44": 180, "45": 241, "STS1": 180, "STS2": 321, "中側頭後": 436, "TPJ": 98}
N_HIGH = 224

try:
    raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8", "replace")
except Exception as e:
    sys.exit(f"⚠️ 表を取れない：{e}\n   ブラウザで {URL} を開いて HCP-MMP1_UniqueRegionList.csv として ~/anaplex に置き、もう一度")
open("HCP-MMP1_UniqueRegionList.csv", "w").write(raw)
rows = list(csv.reader(io.StringIO(raw)))
head = rows[0]
lines = [f"1 表の頭：{head}（{len(rows) - 1} 行）"]
low = [h.lower() for h in head]
vcol = next((i for i, h in enumerate(low) if "vol" in h), None)
ncol = next((i for i, h in enumerate(low) if h in ("region", "regionname", "name", "label")), None)
if ncol is None:
    ncol = next((i for i, h in enumerate(low) if "name" in h or "region" in h), 0)
if vcol is None:
    print("\n".join(lines)); sys.exit("⚠️ 体積の列が見つからない（頭の行を貼ってください）")
lines.append(f"  名前の列：{head[ncol]} ／ 体積の列：{head[vcol]}")
want = {r for v in MAP.values() for r in v}
vol = {}
for r in rows[1:]:
    if len(r) <= max(vcol, ncol):
        continue
    nm = r[ncol].strip()
    base = nm
    for pre in ("L_", "R_"):
        if base.startswith(pre):
            base = base[2:]
    for suf in ("_ROI", "_L", "_R"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    if base in want:
        lines.append("  " + ",".join(r))
        try:
            vol[base] = vol.get(base, 0.0) + float(r[vcol])
        except ValueError:
            pass
lines.append("2 体積（左＋右、mm³）と体数（高次 224 × 区画の体積 ÷ 高次の体積）")
vh = sum(vol.get(k, 0.0) for k in MAP["高次"])
lines.append(f"  高次（{'・'.join(MAP['高次'])}）{vh:,.0f}")
for k, regs in MAP.items():
    if k == "高次":
        continue
    v = sum(vol.get(x, 0.0) for x in regs)
    miss = [x for x in regs if x not in vol]
    n = N_HIGH * v / vh if vh else float("nan")
    lines.append(f"  {k}（{'・'.join(regs)}）{v:,.0f} ⇒ {n:.0f}体（今の Julich から {NOW[k]}）" + (f" ⚠️ 見つからない区画 {miss}" if miss else ""))
txt = "\n".join(lines)
print(txt)
open(OUT, "w").write(txt + "\n")
print(f"（{OUT} に書いた）")
