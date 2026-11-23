"""前頭前野の切り口 y0 を出どころ（Ribeiro 2013 ＝ Gabi 2016 の人の脳）に合わせて出し、眼窩前頭・前帯状の体数を計算する
（arkhe で。網は使わない。何も書き換えない）
  ~/siibra-venv/bin/python julich_pfc_y0.py
出どころ
  Gabi 2016 PNAS の人の値は Ribeiro 2013 Front Neuroanat の脳（65歳女性の右半球 1つ）
  - 前頭前野 ＝「脳梁より前の全部の切片」：2mm の冠状切片 101 枚のうち 1〜17 枚目
  - 前頭前野は皮質の灰白質の体積の 10.0%（Gabi 2016 表S1 も 10.0%）・ニューロンの 7.6%（Gabi 7.8%）
  - 海馬は全部の切片から外した
  ⇒ 地図の中で「y0 より前の皮質が皮質全体の 10.0%」になる y0 と、「前後の長さの 17/101」の y0 を出す
前の julich_pfc_share との違い
  分母を大脳皮質だけにする（v3.1 の地図には視床・線条体・淡蒼球・扁桃体・小脳核なども入っている）。海馬（CA・DG・海馬台・HATA・TrS）と嗅結節は外す
数え方【み選】10-25 ア（同じ地図の中の体積の割合で 13億 ＝ 今の 946 を分ける）
  眼窩前頭 ＝ 946 × 「y0 より前の眼窩前頭 ÷ y0 より前の皮質」＋ 946 × 「y0 より後ろの眼窩前頭 ÷ y0 より前の皮質」（後ろは同じ密度で）
"""
import sys
import numpy as np

PFC = 946                                    # 今の前頭前野（約13億 × 0.728/100万、Gabi 2016）
SHARE = 0.100                                # Ribeiro 2013・Gabi 2016 表S1：前頭前野 ＝ 皮質の灰白質の 10.0%
SEC = (17, 101)                              # Ribeiro 2013：101 枚のうち 1〜17 枚目（2mm）
REF = (25.0, 30.0, 35.0)                     # 前に出した y0（くらべる用）
OFC = ("fo1", "fo2", "fo3", "fo4", "fo5", "fo6", "fo7")
ACC = ("s24", "p24ab", "p24c", "s32", "p32", "area 33", "area 25")
HIPPO = ("(hippocampus",)                  # CA1〜3・DG・海馬台（Subc）・HATA・TrS（嗅内皮質 EC は残す）
NOTCTX = ("tuberculum", "terminal islands")
BINS = np.arange(-130.0, 101.0, 1.0)         # y の 1mm の箱

try:
    import siibra
except ImportError:
    sys.exit("⚠️ siibra が無い：~/siibra-venv/bin/python で回す")
parc = None
for name in ("julich 3.1", "julich 3.0.3", "julich 3", "julich"):
    try:
        parc = siibra.parcellations.get(name)
        break
    except Exception as e:
        last = e
if parc is None:
    sys.exit(f"⚠️ 地図が取れない：{last}")
print(f"地図：{parc.name}", flush=True)
space = "mni152"


def in_cortex(r):
    p = r
    while p is not None:
        if p.name.lower().startswith("cerebral cortex"):
            return True
        p = p.parent
    return False


def hist(region):
    img = region.get_regional_mask(space, maptype="labelled").fetch()
    d = np.asarray(img.get_fdata()) > 0
    vox = abs(np.linalg.det(img.affine[:3, :3]))
    ijk = np.argwhere(d)
    y = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T)[1] if len(ijk) else np.zeros(0)
    h, _ = np.histogram(y, bins=BINS)
    return h * vox


done, miss, rows = set(), [], []
todo = [r for r in parc if not r.children]
while todo:
    r = todo.pop(0)
    if r.name in done:
        continue
    done.add(r.name)
    try:
        rows.append((r.name, in_cortex(r), hist(r)))
    except Exception:
        p = r.parent
        if p is not None and p.name not in done and all(c.name in done or c in todo for c in p.children):
            todo.append(p)                      # 下の葉に印が無ければ親で
        else:
            miss.append(r.name)
print(f"数えた領域 {len(rows)}・印なし {len(miss)}", flush=True)
if miss:
    print("印なし：" + "／".join(miss[:40]), flush=True)


def has(n, keys):
    return any(k in n.lower() for k in keys)


ctx = [(n, h) for n, c, h in rows if c and not has(n, HIPPO) and not has(n, NOTCTX)]
out = [n for n, c, h in rows if not (c and not has(n, HIPPO) and not has(n, NOTCTX))]
print(f"皮質に数えた領域 {len(ctx)}・外した領域 {len(out)}（視床・基底核・扁桃体・海馬・小脳核など）", flush=True)
print("外した領域の例：" + "／".join(out[:12]) + " …", flush=True)
yc = (BINS[:-1] + BINS[1:]) / 2


def report(label, sel):
    H = sum(h for _, h in sel)
    tot = H.sum()
    nz = np.nonzero(H)[0]
    yf, yb = BINS[nz[-1] + 1], BINS[nz[0]]          # 前の端・後ろの端
    ant = np.cumsum(H[::-1])[::-1]                   # y の箱より前（その箱を含む）の体積
    k = int(np.argmin(np.abs(ant / tot - SHARE)))
    y_share = float(BINS[k])
    y_sec = float(yf - (yf - yb) * SEC[0] / SEC[1])
    y_34 = float(yf - 2.0 * SEC[0])
    print(f"\n【{label}】皮質 {tot:,.0f}mm³・前の端 y={yf:.0f}・後ろの端 y={yb:.0f}（前後 {yf - yb:.0f}mm）", flush=True)
    print(f"  y0（前が皮質の 10.0%）＝ {y_share:.0f}  ／ y0（長さの 17/101）＝ {y_sec:.1f}  ／ y0（前の端から 34mm）＝ {y_34:.0f}", flush=True)
    for y0 in REF:
        print(f"  くらべる：y＞{y0:.0f} の皮質は {(H[yc > y0].sum()) / tot:.1%}", flush=True)
    o = sum(h for n, h in sel if has(n, OFC))
    a = sum(h for n, h in sel if has(n, ACC))
    for tag, y0 in (("10.0%", y_share), ("17/101", y_sec), ("34mm", y_34)) + tuple(("くらべる", r) for r in REF):
        m = yc > y0
        C = H[m].sum()
        oa, op, aa, ap = o[m].sum(), o[~m].sum(), a[m].sum(), a[~m].sum()
        no_a, no_p = PFC * oa / C, PFC * op / C
        na_a, na_p = PFC * aa / C, PFC * ap / C
        print(f"  [{tag} y0={y0:.0f}] 眼窩前頭 前 {oa / o.sum():.0%}：{no_a:.0f} ＋ 後ろ {no_p:.0f} ＝ {no_a + no_p:.0f}"
              f" ／ 前帯状 前 {aa / a.sum():.0%}：{na_a:.0f} ＋ 後ろ {na_p:.0f} ＝ {na_a + na_p:.0f}"
              f" ／ 前頭前野の残り（眼窩前頭を引く）{PFC - no_a:.0f}・（前帯状も）{PFC - no_a - na_a:.0f}", flush=True)


report("左右込み", ctx)
report("右だけ（Ribeiro は右半球）", [(n, h) for n, h in ctx if "right" in n.lower()])
