"""人の拡散 MRI の繋がり（Rosen & Halgren 2021 eNeuro、HCP 1065人、確率的トラクトグラフィ、HCP-MMP1 の 360 区画）を取ってくる（arkhe で）
  python3 rosen_get.py
11-01：F-TRACT の確率は表全体で中央値 0.95・0.5 以上が 97% ⇒ 見分けが無い。STS へ入る線は人の拡散の表で見る
置き場：Zenodo 4060485（新しい版があればそちら）。10MB 以下の表だけ ~/anaplex/rosen に落とし、頭を出す
"""
import json, os, urllib.request
OUT = os.path.expanduser("~/anaplex/rosen")
os.makedirs(OUT, exist_ok=True)
def get(u):
    with urllib.request.urlopen(u, timeout=60) as r:
        return json.load(r)
rec = get("https://zenodo.org/api/records/4060485")
lat = rec.get("links", {}).get("latest")
if lat and lat.rstrip("/") != rec["links"]["self"].rstrip("/"):
    rec = get(lat)
print("版：", rec["metadata"].get("version"), rec["links"]["self"])
for f in rec["files"]:
    k, s = f["key"], f["size"]
    print(f"  {s/1e6:9.2f} MB  {k}")
for f in rec["files"]:
    k, s = f["key"], f["size"]
    if s > 10e6 or not k.lower().endswith((".csv", ".txt", ".tsv")):
        continue
    p = os.path.join(OUT, k)
    if not os.path.exists(p):
        urllib.request.urlretrieve(f["links"]["self"], p)
    with open(p, encoding="utf-8", errors="replace") as fh:
        lines = [next(fh, "") for _ in range(3)]
    print(f"\n── {k}")
    for l in lines:
        print("   ", l.rstrip()[:300])
print("\n終わり")
