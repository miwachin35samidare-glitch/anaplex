"""Julich v3.1 の「Area TPJ (STG, SMG)」の出どころ（資料の説明・論文）を siibra から出す（arkhe で。何も書き換えない）
  ~/siibra-venv/bin/python tpj_info.py
11-01 2:30【み】「TPJも」：TPJ の層の記載を人で引きたい。論文が検索で見つからないので、地図に付いた資料の説明・論文を見る
"""
import sys
try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
ju = siibra.parcellations.get("julich 3.1")
for name in ("Area TPJ (STG, SMG) left", "Area TPJ (STG, SMG)"):
    try:
        r = ju.get_region(name)
        break
    except Exception as e:
        r = None
        print(f"  {name}：取れない（{type(e).__name__}）", flush=True)
if r is None:
    sys.exit("⚠️ 区画が取れない")
print(f"区画：{r.name}", flush=True)
for attr in ("description", "publications", "urls", "doi", "datasets"):
    try:
        v = getattr(r, attr)
    except Exception as e:
        print(f"[{attr}] 取れない（{type(e).__name__}：{e}）", flush=True)
        continue
    print(f"[{attr}]", flush=True)
    if isinstance(v, (list, tuple, set)):
        for x in v:
            print("  ", x, flush=True)
            for a2 in ("name", "description", "urls", "publications", "doi"):
                if hasattr(x, a2):
                    try:
                        print(f"     {a2}: {getattr(x, a2)}", flush=True)
                    except Exception:
                        pass
    else:
        print("  ", v, flush=True)
print("終わり", flush=True)
