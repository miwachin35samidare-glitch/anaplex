#!/usr/bin/env python3
# h01_density.py — H01（人の側頭葉、Shapson-Coe 2024）で、抑制の細胞の樹状突起 1µm あたりに入る興奮のシナプスを数える
#   組み表_版116 §28〜§30。h01_probe.py の読み方（11-26 22:0x に arkhe で本物が読めた）を使う
#   ① 札（~/data/h01/segment_properties.json、下見で保存済み）から層ごとの NSe・NSi（全部の細胞、新たに落とさない）
#   ② L2・L3 の抑制の細胞を N 体（番号の順を種 0 で混ぜて先頭から）：骨格・区画の点（H01 の機械の分類）・入るシナプス
#      樹状突起の長さ ＝ 両端が「樹状突起」の辺の和 ／ シナプスは一番近い骨格の点の区画に入れる
#      細胞体からの道のりごとの密度（板で切れた木は根元が多く残るので、道のりで分けて見る）
#   型：シナプスの type 1 ＝ 抑制・2 ＝ 興奮（H01 の見せ方の状態：type 1 を赤・incoming_inhibitory も赤、
#       下見の 3体で type 2 の数 ≒ 札の NSe）
#   網・GPU は触らない。落とす量は 1体 数百KB の見込み。上限 300MB で止まる。保存は ~/data/h01/density_cells.json
# 使い方：python3 h01_density.py 40        （出力を貼ってください）
#         python3 h01_density.py selftest  （容器の確かめ。網に出ない）
import sys, os, json, gzip, struct, time
import numpy as np
import h01_probe as P

SUBC_ANN = "h01-release/data/20210601/c3/subcompartments/annotations"
OUT = P.OUT
P.CAP = 300 * 1000**2
Z_LO, Z_HI = 63 * 33.0, 5231 * 33.0          # シナプスの注釈の z の範囲（nm）＝ 板の厚さの見当
CLS = {0: "軸索", 1: "樹状突起", 2: "アストロ", 3: "細胞体", 4: "繊毛", 5: "起始部", 1000: "髄鞘の軸索", 1001: "髄鞘の軸索"}
BINS = [0, 25, 50, 100, 150, 250, 1e9]

def pop_stats(props):
    inl = props["inline"]; tags = None; NSe = NSi = None
    for pr in inl["properties"]:
        if pr["id"] == "tags": tags = pr
        if pr["id"] == "NSe": NSe = np.array(pr["values"], float)
        if pr["id"] == "NSi": NSi = np.array(pr["values"], float)
    names = tags["tags"]; tv = tags["values"]
    has = lambda name: np.array([names.index(name) in v for v in tv]) if name in names else np.zeros(len(tv), bool)
    print("== ① 札の NSe・NSi（全部の細胞。何の版のシナプスで数えたかは札に書いていない）")
    print("  %-14s %-4s %6s %8s %8s %10s %10s" % ("型", "層", "体", "NSe中央", "NSi中央", "E割合中央", "E割合(和)"))
    for typ in ("interneuron", "pyramidal"):
        for L in ("L1", "L2", "L3", "L4", "L5", "L6"):
            m = has(typ) & has(L) & has("neuron")
            if m.sum() < 5: continue
            e, i = NSe[m], NSi[m]; tot = e + i; ok = tot > 0
            print("  %-14s %-4s %6d %8.0f %8.0f %10.2f %10.2f" % (typ, L, m.sum(), np.median(e), np.median(i),
                  np.median(e[ok] / tot[ok]), e.sum() / max(tot.sum(), 1)))
    return inl, names, tv

def pick_cells(inl, names, tv, n):
    ids = np.array([int(x) for x in inl["ids"]], np.uint64)
    ii, nn = names.index("interneuron"), names.index("neuron")
    L2, L3 = names.index("L2"), names.index("L3")
    sel = [k for k, v in enumerate(tv) if ii in v and nn in v and (L2 in v or L3 in v)]
    rng = np.random.default_rng(0); rng.shuffle(sel)
    return [(int(ids[k]), "L2" if L2 in tv[k] else "L3") for k in sel[:n]], len(sel)

def nearest(a, b, chunk=2048):
    # a の各点に一番近い b の番号と距離（nm）
    out_i = np.empty(len(a), np.int64); out_d = np.empty(len(a))
    bb = (b ** 2).sum(1)
    for s in range(0, len(a), chunk):
        x = a[s:s + chunk]
        d2 = (x ** 2).sum(1)[:, None] - 2 * x @ b.T + bb[None, :]
        j = d2.argmin(1); out_i[s:s + chunk] = j
        out_d[s:s + chunk] = np.sqrt(np.maximum(d2[np.arange(len(x)), j], 0))
    return out_i, out_d

def path_from(v, e, src):
    # 木（森）の上の src からの道のり（nm）。届かない点は inf
    n = len(v); adj = [[] for _ in range(n)]
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    for (a, b), ww in zip(e, w):
        adj[a].append((b, ww)); adj[b].append((a, ww))
    dist = np.full(n, np.inf); dist[src] = 0.0; st = [src]
    while st:
        a = st.pop()
        for b, ww in adj[a]:
            if dist[b] == np.inf:
                dist[b] = dist[a] + ww; st.append(b)
    return dist

def analyze_cell(seg, skel, sub, syn):
    # skel：(v nm, e)、sub：(点 nm, 区画)、syn：(前の点 nm, 後の点 nm, type)
    v, e = skel; sp, sc = sub; pa, pb, ty = syn
    r = {"seg": seg, "n_vert": len(v), "n_sub": len(sp), "n_syn": len(ty)}
    if len(v) < 2 or len(sp) == 0:
        r["skip"] = "骨格か区画の点が無い"; return r
    j, dsub = nearest(v, sp); vc = sc[j]
    r["sub_dist_med_nm"] = float(np.median(dsub))
    soma = np.nonzero(vc == 3)[0]
    if len(soma) == 0:
        r["skip"] = "細胞体の点が無い（板の外か、別の番号）"; return r
    c = v[soma].mean(0); src = soma[np.argmin(np.linalg.norm(v[soma] - c, axis=1))]
    pd = path_from(v, e, src)
    r["soma_z_nm"] = float(c[2]); r["soma_to_face_um"] = float(min(c[2] - Z_LO, Z_HI - c[2]) / 1000)
    L = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    both = lambda k: (vc[e[:, 0]] == k) & (vc[e[:, 1]] == k)
    r["dend_um"] = float(L[both(1)].sum() / 1000)
    r["axon_um"] = float(L[np.isin(vc[e[:, 0]], [0, 5, 1000, 1001]) & np.isin(vc[e[:, 1]], [0, 5, 1000, 1001])].sum() / 1000)
    r["mixed_um"] = float(L[vc[e[:, 0]] != vc[e[:, 1]]].sum() / 1000)
    r["all_um"] = float(L.sum() / 1000)
    r["unreached_um"] = float(L[~np.isfinite(pd[e[:, 0]])].sum() / 1000)
    if len(ty) == 0:
        r["skip"] = "入るシナプスが無い"; return r
    ja, da = nearest(pa, v); jb, db = nearest(pb, v)
    useb = db <= da; jj = np.where(useb, jb, ja); dd = np.minimum(da, db)
    r["syn_post_is_b"] = float(useb.mean()); r["syn_dist_med_nm"] = float(np.median(dd))
    far = dd > 2000; r["syn_far_frac"] = float(far.mean())
    sc_ = vc[jj]; cnt = {}
    for k, name in ((1, "樹状突起"), (3, "細胞体"), (-1, "軸索ほか")):
        m = (sc_ == k) if k > 0 else ~np.isin(sc_, [1, 3])
        m = m & ~far
        cnt[name] = {"E": int(((ty == 2) & m).sum()), "I": int(((ty == 1) & m).sum())}
    r["cnt"] = cnt
    # 道のりごと（樹状突起の辺・樹状突起のシナプス）
    em = both(1); epd = 0.5 * (pd[e[:, 0]] + pd[e[:, 1]]) / 1000
    spd = pd[jj] / 1000; dm = (sc_ == 1) & ~far
    rows = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        lm = em & (epd >= a) & (epd < b)
        sm = dm & (spd >= a) & (spd < b)
        rows.append([a, float(L[lm].sum() / 1000), int(((ty == 2) & sm).sum()), int(((ty == 1) & sm).sum())])
    r["by_path"] = rows
    r["dend_max_path_um"] = float(np.nanmax(np.where(np.isfinite(pd), pd, np.nan)[vc == 1]) / 1000) if (vc == 1).any() else 0.0
    return r

def fetch_cell(seg, si, ki, ai):
    sk = P.sharded_get(P.SKEL, seg, ki["sharding"])
    if sk is None: return None
    v, e, _ = P.decode_skel(sk, ki)
    rel = {x["id"]: x for x in ai["relationships"]}["skeleton_id"]
    if rel.get("sharding"):
        b = P.sharded_get(SUBC_ANN + "/" + rel["key"], seg, rel["sharding"])
    else:
        try:
            b = P.get(SUBC_ANN + "/" + rel["key"] + "/" + str(seg))
        except Exception:
            b = None
    if b is None: return None
    xyz, pr, _ = P.decode_multi(b, ai)
    scale = to_nm(ai)
    ck = "class_label" if "class_label" in pr else [k for k in pr if "class" in k][0]
    sp = xyz[:, :3] * scale; sc = pr[ck].astype(np.int64)
    rs = {x["id"]: x for x in si["relationships"]}["post_synaptic_cell"]
    bs = P.sharded_get(P.SYN + "/" + rs["key"], seg, rs["sharding"])
    s8 = to_nm(si)
    if bs is None:
        syn = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int))
    else:
        g, ps, _ = P.decode_multi(bs, si)
        syn = (g[:, :3] * s8, g[:, 3:6] * s8, ps["type"].astype(int))
    return (v.astype(float), e.astype(np.int64)), (sp, sc), syn

import warnings
warnings.filterwarnings("ignore")

UNIT_NM = {"m": 1e9, "mm": 1e6, "um": 1e3, "µm": 1e3, "nm": 1.0}

def to_nm(info):
    # info の dimensions（[値, 単位]）から nm への倍率。⟲ 11-26 22:3x までは単位を見ずに m とみなしていた
    #   （シナプスは m、区画の点は nm ⇒ 区画の点が 10^9 倍にずれ、骨格の点に細胞体の札が付かなかった）
    return np.array([info["dimensions"][k][0] * UNIT_NM[info["dimensions"][k][1]] for k in ("x", "y", "z")])

def diag_print(seg, skel, sub, syn):
    v, e = skel; sp, sc = sub; pa, pb, ty = syn
    print("  -- 番号", seg)
    print("    骨格 点 %d・辺 %d・辺の和 %.0fµm・範囲(µm) %s〜%s" % (len(v), len(e), P.cable_um(v, e),
          np.round(v.min(0) / 1000).tolist(), np.round(v.max(0) / 1000).tolist()))
    if len(sp) == 0:
        print("    区画の点 0"); return
    u, c = np.unique(sc, return_counts=True)
    print("    区画の点 %d・範囲(µm) %s〜%s" % (len(sp), np.round(sp.min(0) / 1000).tolist(), np.round(sp.max(0) / 1000).tolist()))
    print("    区画の点の札と数", {CLS.get(int(a), str(int(a))): int(b) for a, b in zip(u, c)})
    j, d = nearest(v, sp); _, d2 = nearest(sp, v)
    print("    骨格の点 → 一番近い区画の点 中央 %.0fnm・90%% %.0fnm ／ 区画の点 → 骨格 中央 %.0fnm・90%% %.0fnm" % (
          np.median(d), np.percentile(d, 90), np.median(d2), np.percentile(d2, 90)))
    u, c = np.unique(sc[j], return_counts=True)
    print("    骨格の点に付いた札と数", {CLS.get(int(a), str(int(a))): int(b) for a, b in zip(u, c)})
    same = len(sp) == len(v) and np.allclose(np.sort(sp, 0), np.sort(v, 0), atol=1)
    print("    区画の点は骨格の点と同じか", bool(same))
    if len(ty):
        _, da = nearest(pa, v); _, db = nearest(pb, v)
        print("    入るシナプス %d（興奮 %d・抑制 %d）・骨格まで 中央 %.0fnm" % (len(ty), int((ty == 2).sum()), int((ty == 1).sum()),
              np.median(np.minimum(da, db))))

def diag(ids):
    si = json.loads(P.get(P.SYN + "/info")); ki = json.loads(P.get(P.SKEL + "/info"))
    ai = json.loads(P.maybe_gunzip(P.get(SUBC_ANN + "/info")))
    for seg in ids:
        d = fetch_cell(seg, si, ki, ai)
        if d is None:
            print("  -- 番号", seg, "：骨格か区画の索引に無い"); continue
        diag_print(seg, *d)
    print("== 落とした量 %.1f MB" % (P._got[0] / 1e6))

def summarize(rows):
    ok = [r for r in rows if "skip" not in r]
    print("== ③ まとめ（使えた %d 体 ／ 引いた %d 体）" % (len(ok), len(rows)))
    if not ok: return
    def q(x):
        x = np.array(x, float)
        if len(x) == 0: return "（0体）"
        return "中央 %.3g（%.3g〜%.3g、%d体）" % (np.median(x), np.percentile(x, 25), np.percentile(x, 75), len(x))
    den = [r for r in ok if r["dend_um"] > 100]
    dE = [r["cnt"]["樹状突起"]["E"] / r["dend_um"] for r in den]
    dI = [r["cnt"]["樹状突起"]["I"] / r["dend_um"] for r in den]
    fr = [r["cnt"]["樹状突起"]["E"] / max(1, r["cnt"]["樹状突起"]["E"] + r["cnt"]["樹状突起"]["I"]) for r in den]
    print("  樹状突起 1µm あたり 興奮", q(dE)); print("  樹状突起 1µm あたり 抑制", q(dI))
    print("  樹状突起に入る興奮の割合", q(fr))
    print("  細胞体の 興奮・抑制", q([r["cnt"]["細胞体"]["E"] for r in ok]), "／", q([r["cnt"]["細胞体"]["I"] for r in ok]))
    print("  板の中の樹状突起の長さ", q([r["dend_um"] for r in ok]), "µm")
    print("  樹状突起の一番遠い道のり", q([r["dend_max_path_um"] for r in ok]), "µm（人の切片 道のり平均 248µm、Wilbers 2023）")
    print("  細胞体から板の面まで", q([r["soma_to_face_um"] for r in ok]), "µm")
    print("  道のりごと（全部の細胞を足した）：区間µm 樹状突起µm 興奮 抑制 興奮/µm")
    tot = np.zeros((len(BINS) - 1, 3))
    for r in den:
        tot += np.array([x[1:] for x in r["by_path"]], float)
    for k, a in enumerate(BINS[:-1]):
        b = BINS[k + 1]; Lk, E_, I_ = tot[k]
        print("    %4.0f〜%-5s %9.0f %7.0f %7.0f %7.3f" % (a, ("%.0f" % b) if b < 1e8 else "", Lk, E_, I_, E_ / Lk if Lk > 0 else float("nan")))
    print("  確かめ：区画の点から骨格の点まで 中央", q([r["sub_dist_med_nm"] for r in ok]), "nm ／ シナプスの後ろ側が2つ目の点の割合", q([r["syn_post_is_b"] for r in ok]),
          "／ シナプスから骨格まで 中央", q([r["syn_dist_med_nm"] for r in ok]), "nm ／ 2µm より遠く外したシナプス", q([r["syn_far_frac"] for r in ok]))
    print("  確かめ：辺の内訳 µm 樹状突起", q([r["dend_um"] for r in ok]), "・軸索", q([r["axon_um"] for r in ok]),
          "・区画をまたぐ辺", q([r["mixed_um"] for r in ok]), "・細胞体から届かない", q([r["unreached_um"] for r in ok]))

def main(n):
    props = json.load(open(os.path.join(OUT, "segment_properties.json")))
    inl, names, tv = pop_stats(props)
    si = json.loads(P.get(P.SYN + "/info")); ki = json.loads(P.get(P.SKEL + "/info"))
    ai = json.loads(P.maybe_gunzip(P.get(SUBC_ANN + "/info")))
    print("== 区画の点の info：", P.short({k: ai[k] for k in ai if k not in ("spatial",)}, 1500))
    cells, nall = pick_cells(inl, names, tv, n)
    print("== ② L2・L3 の抑制の細胞 %d 体のうち %d 体を引く" % (nall, len(cells)))
    rows = []
    for k, (seg, L) in enumerate(cells):
        t0 = time.time()
        try:
            d = fetch_cell(seg, si, ki, ai)
            r = {"seg": seg, "skip": "骨格か区画の索引に無い"} if d is None else analyze_cell(seg, *d)
        except Exception as ex:
            r = {"seg": seg, "skip": "読めない %s" % type(ex).__name__}
        r["layer"] = L; rows.append(r)
        if "skip" in r:
            print("  %2d %s %s：%s" % (k, L, seg, r["skip"]))
        else:
            c = r["cnt"]
            print("  %2d %s %s：樹状突起 %.0fµm 興奮 %d 抑制 %d（%.3f/µm）・細胞体 興奮 %d 抑制 %d・軸索ほか %d・軸索 %.0fµm・板の面まで %.0fµm・%.1fs・計 %.1fMB" % (
                k, L, seg, r["dend_um"], c["樹状突起"]["E"], c["樹状突起"]["I"], c["樹状突起"]["E"] / max(r["dend_um"], 1e-9),
                c["細胞体"]["E"], c["細胞体"]["I"], c["軸索ほか"]["E"] + c["軸索ほか"]["I"], r["axon_um"], r["soma_to_face_um"],
                time.time() - t0, P._got[0] / 1e6), flush=True)
    with open(os.path.join(OUT, "density_cells.json"), "w") as f:
        json.dump(rows, f)
    summarize(rows)
    print("== 落とした量 %.1f MB" % (P._got[0] / 1e6))

def selftest():
    import tempfile, neuroglancer
    from neuroglancer import write_annotations as wa
    rng = np.random.default_rng(1)
    # 細胞体（原点）から 樹状突起 2本（各 200µm、0.5µm 刻み）と軸索 1本（300µm）
    v = [np.zeros(3)]; e = []; cls = [3]
    def arm(d, length, c):
        prev = 0
        for s in range(1, int(length / 0.5) + 1):
            v.append(d * s * 500.0); e.append([prev, len(v) - 1]); cls.append(c); prev = len(v) - 1
    arm(np.array([1, 0, 0.]), 200, 1); arm(np.array([0, 1, 0.]), 200, 1); arm(np.array([-1, 0, 0.]), 300, 0)
    v = np.array(v); e = np.array(e); cls = np.array(cls)
    # 区画の点：骨格の点そのまま＋少しずらす
    sp = v + rng.normal(0, 50, v.shape)
    # シナプス：樹状突起に 1µm あたり 興奮 1.2・抑制 0.4、細胞体に 興奮 5・抑制 20、軸索に 3
    pa, pb, ty = [], [], []
    def add(pt, t):
        q = pt + rng.normal(0, 100, 3); pa.append(q + np.array([0, 0, 400])); pb.append(q); ty.append(t)  # 前の点は骨格の面から離す
    dend = np.nonzero(cls == 1)[0]
    for _ in range(int(400 * 1.2)): add(v[rng.choice(dend)], 2)
    for _ in range(int(400 * 0.4)): add(v[rng.choice(dend)], 1)
    for _ in range(5): add(v[0], 2)
    for _ in range(20): add(v[0], 1)
    for _ in range(3): add(v[rng.choice(np.nonzero(cls == 0)[0])], 2)
    r = analyze_cell(7, (v, e), (sp, cls), (np.array(pa), np.array(pb), np.array(ty)))
    assert abs(r["dend_um"] - 399) < 1, r["dend_um"]  # 細胞体とつながる 0.5µm の辺 2本は区画をまたぐので外れる
    assert abs(r["axon_um"] - 299.5) < 1, r["axon_um"]
    c = r["cnt"]
    assert 478 <= c["樹状突起"]["E"] <= 486 and 158 <= c["樹状突起"]["I"] <= 166, c  # 細胞体のすぐ横の数個は隣の樹状突起の点に入る（作り物の細胞体が1点のため）
    assert c["細胞体"]["I"] >= 15 and c["軸索ほか"]["E"] >= 2, c
    assert r["syn_post_is_b"] > 0.9
    print("✓ 作り物の細胞：樹状突起 %.1fµm（399）・軸索 %.1fµm・樹状突起の 興奮 %d 抑制 %d・細胞体 抑制 %d" % (
        r["dend_um"], r["axon_um"], c["樹状突起"]["E"], c["樹状突起"]["I"], c["細胞体"]["I"]))
    tot = np.array([x[1:] for x in r["by_path"]], float)
    assert abs(tot[:, 0].sum() - 399) < 1 and tot[:, 1].sum() == c["樹状突起"]["E"]
    print("✓ 道のりごとの和が全体と合う：", [[round(x[0]), round(x[1]), x[2], x[3]] for x in r["by_path"]])
    # 区画の点を neuroglancer の書き手で書き、skeleton_id の索引から読めるか（形の確かめ）
    cs = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=[32, 32, 33])
    w = wa.AnnotationWriter(coordinate_space=cs, annotation_type="point",
                            properties=[neuroglancer.AnnotationPropertySpec(id="class_label", type="uint32")],
                            relationships=["skeleton_id"])
    for p_, c_ in zip(sp, cls):
        w.add_point(p_ / np.array([32, 32, 33]), class_label=int(c_), skeleton_id=[7])
    d = tempfile.mkdtemp(); w.write(d); info = json.load(open(os.path.join(d, "info")))
    b = open(os.path.join(d, info["relationships"][0]["key"], "7"), "rb").read()
    xyz, pr, _ = P.decode_multi(b, info)
    scale = to_nm(info)
    assert np.allclose(xyz[:, :3] * scale, sp, atol=1) and (pr["class_label"] == cls).all()
    print("✓ neuroglancer が書いた区画の点（点・class_label・skeleton_id の索引）を nm に直して読めた")
    real_sub = {"dimensions": {"x": [1, "nm"], "y": [1, "nm"], "z": [1, "nm"]}}
    real_syn = {"dimensions": {"x": [8e-09, "m"], "y": [8e-09, "m"], "z": [3.3e-08, "m"]}}
    assert np.allclose(to_nm(real_sub), [1, 1, 1]) and np.allclose(to_nm(real_syn), [8, 8, 33])
    print("✓ arkhe の info の単位（区画の点 nm・シナプス m）で倍率が 1・1・1 と 8・8・33")
    props = {"inline": {"ids": ["1", "2", "3", "4", "5", "6"], "properties": [
        {"id": "tags", "type": "tags", "tags": ["neuron", "interneuron", "pyramidal", "L2", "L3"],
         "values": [[0, 1, 3]] * 3 + [[0, 2, 3]] * 3},
        {"id": "NSe", "type": "number", "values": [10, 20, 30, 100, 200, 300]},
        {"id": "NSi", "type": "number", "values": [10, 20, 30, 50, 50, 50]}]}}
    inl, names, tv = pop_stats(props)
    cells, nall = pick_cells(inl, names, tv, 2)
    assert nall == 3 and len(cells) == 2 and all(c[0] in (1, 2, 3) for c in cells)
    print("✓ 札から L2・L3 の抑制の細胞だけを選べた")
    diag_print(7, (v, e), (sp, cls), (np.array(pa), np.array(pb), np.array(ty)))
    summarize([{"seg": 1, "skip": "x"}])
    print("✓ 確かめの表示と、使える細胞が 0 体のまとめが落ちない")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "diag":
        # 下見で通った L2 の例 3体 ＋ 11-26 22:3x の回で「細胞体の点が無い」だった先頭 3体 ＋ 細胞体だけだった 2体
        diag(P.L2_IN + [40670855127, 6139546370, 1204043735, 28916800359, 42402355619])
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
