#!/usr/bin/env python3
# h01_probe.py — H01（人の側頭葉の電子顕微鏡、Shapson-Coe 2024）の公開データの形を確かめる下見
#   組み表_版116 §28：抑制の細胞1つの興奮の相手の総数を人で数える前に、
#   ① 細胞の札（segment_properties）に何が入っているか
#   ② 受け手の番号で入るシナプスを引けるか（post_synaptic_cell の索引・型の値）
#   ③ 骨格（c3 skeletons）が引けるか・座標の単位がシナプスと合うか
#   を数個の細胞で見る。網は触らない・GPU を使わない・落とすのは数MB（上限 50MB で止まる）
# 使い方：python3 h01_probe.py            （出力を貼ってください）
#         python3 h01_probe.py selftest   （読み方の確かめだけ。網に出ない）
import sys, os, json, gzip, struct, urllib.request, time
import numpy as np

BASE = "https://storage.googleapis.com/"
SYN = "h01-release/data/20210729/c3/synapses/precomputed"
SKEL = "h01-release/data/20210601/c3/skeletons"
SUBC = "h01-release/data/20210601/c3/subcompartments"
PROPS = "lichtman-h01-49eee972005c8846803ef58fbd36e049/goog14r0s5c3_new_props/segment_properties"
# gallery/l2_interneurons.json（H01 の見せ方の状態、L2 の抑制の細胞の例）の先頭3つ
L2_IN = [1100054524, 1115430292, 12237931142]
OUT = os.path.join(os.path.expanduser("~"), "data", "h01")
CAP = 50 * 1000**2
_got = [0]

def get(path, rng=None):
    req = urllib.request.Request(BASE + path)
    if rng is not None:
        req.add_header("Range", "bytes=%d-%d" % (rng[0], rng[1] - 1))
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
            break
        except Exception as e:
            if k == 2:
                raise
            time.sleep(2)
    _got[0] += len(b)
    if _got[0] > CAP:
        sys.exit("止めた：落とした量が %.1fMB を越えた" % (CAP / 1e6))
    return b

def maybe_gunzip(b):
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b

# ---- murmurhash3_x86_128（seed 0）の下 64bit ----
def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF

def _fmix(h):
    h ^= h >> 16; h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13; h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16
    return h

def murmur3_x86_128_lo64(data, seed=0):
    c1, c2, c3, c4 = 0x239b961b, 0xab0e9789, 0x38b34ae5, 0xa1e38b93
    h1 = h2 = h3 = h4 = seed
    n = len(data); nb = n // 16
    for i in range(nb):
        k1, k2, k3, k4 = struct.unpack_from("<4I", data, i * 16)
        k1 = (k1 * c1) & 0xFFFFFFFF; k1 = _rotl(k1, 15); k1 = (k1 * c2) & 0xFFFFFFFF; h1 ^= k1
        h1 = _rotl(h1, 19); h1 = (h1 + h2) & 0xFFFFFFFF; h1 = (h1 * 5 + 0x561ccd1b) & 0xFFFFFFFF
        k2 = (k2 * c2) & 0xFFFFFFFF; k2 = _rotl(k2, 16); k2 = (k2 * c3) & 0xFFFFFFFF; h2 ^= k2
        h2 = _rotl(h2, 17); h2 = (h2 + h3) & 0xFFFFFFFF; h2 = (h2 * 5 + 0x0bcaa747) & 0xFFFFFFFF
        k3 = (k3 * c3) & 0xFFFFFFFF; k3 = _rotl(k3, 17); k3 = (k3 * c4) & 0xFFFFFFFF; h3 ^= k3
        h3 = _rotl(h3, 15); h3 = (h3 + h4) & 0xFFFFFFFF; h3 = (h3 * 5 + 0x96cd1c35) & 0xFFFFFFFF
        k4 = (k4 * c4) & 0xFFFFFFFF; k4 = _rotl(k4, 18); k4 = (k4 * c1) & 0xFFFFFFFF; h4 ^= k4
        h4 = _rotl(h4, 13); h4 = (h4 + h1) & 0xFFFFFFFF; h4 = (h4 * 5 + 0x32ac3b17) & 0xFFFFFFFF
    tail = data[nb * 16:]
    k = [0, 0, 0, 0]
    for i, byte in enumerate(tail):
        k[i // 4] |= byte << (8 * (i % 4))
    if len(tail) > 12:
        k4 = (k[3] * c4) & 0xFFFFFFFF; k4 = _rotl(k4, 18); k4 = (k4 * c1) & 0xFFFFFFFF; h4 ^= k4
    if len(tail) > 8:
        k3 = (k[2] * c3) & 0xFFFFFFFF; k3 = _rotl(k3, 17); k3 = (k3 * c4) & 0xFFFFFFFF; h3 ^= k3
    if len(tail) > 4:
        k2 = (k[1] * c2) & 0xFFFFFFFF; k2 = _rotl(k2, 16); k2 = (k2 * c3) & 0xFFFFFFFF; h2 ^= k2
    if len(tail) > 0:
        k1 = (k[0] * c1) & 0xFFFFFFFF; k1 = _rotl(k1, 15); k1 = (k1 * c2) & 0xFFFFFFFF; h1 ^= k1
    h1 ^= n; h2 ^= n; h3 ^= n; h4 ^= n
    h1 = (h1 + h2 + h3 + h4) & 0xFFFFFFFF
    h2 = (h2 + h1) & 0xFFFFFFFF; h3 = (h3 + h1) & 0xFFFFFFFF; h4 = (h4 + h1) & 0xFFFFFFFF
    h1, h2, h3, h4 = _fmix(h1), _fmix(h2), _fmix(h3), _fmix(h4)
    h1 = (h1 + h2 + h3 + h4) & 0xFFFFFFFF
    h2 = (h2 + h1) & 0xFFFFFFFF
    return h1 | (h2 << 32)

# ---- neuroglancer_uint64_sharded_v1 ----
def shard_loc(key, sh):
    pre = sh.get("preshift_bits", 0); mb = sh["minishard_bits"]; sb = sh["shard_bits"]
    x = key >> pre
    if sh.get("hash", "identity") == "murmurhash3_x86_128":
        x = murmur3_x86_128_lo64(struct.pack("<Q", x))
    mini = x & ((1 << mb) - 1)
    shard = (x >> mb) & ((1 << sb) - 1)
    name = ("%0" + str((sb + 3) // 4) + "x") % shard if sb > 0 else "0"
    return name + ".shard", mini

def decode_minishard(b, n):
    a = np.frombuffer(b, dtype="<u8").reshape(3, n).copy()
    ids = np.cumsum(a[0]).astype(np.uint64)
    starts = np.zeros(n, np.uint64); sizes = a[2]
    end = np.uint64(0)
    for i in range(n):
        starts[i] = end + a[1][i]; end = starts[i] + sizes[i]
    return ids, starts, sizes

def sharded_get(prefix, key, sh, fetch=get):
    fname, mini = shard_loc(key, sh)
    mb = sh["minishard_bits"]; idx_len = 16 * (1 << mb)
    ent = fetch(prefix + "/" + fname, (16 * mini, 16 * mini + 16))
    s, e = struct.unpack("<QQ", ent)
    if e <= s:
        return None
    mbytes = fetch(prefix + "/" + fname, (idx_len + s, idx_len + e))
    if sh.get("minishard_index_encoding", "raw") == "gzip":
        mbytes = gzip.decompress(mbytes)
    n = len(mbytes) // 24
    ids, starts, sizes = decode_minishard(mbytes, n)
    hit = np.nonzero(ids == np.uint64(key))[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    d = fetch(prefix + "/" + fname, (idx_len + int(starts[i]), idx_len + int(starts[i] + sizes[i])))
    if sh.get("data_encoding", "raw") == "gzip":
        d = gzip.decompress(d)
    return d

# ---- 注釈（複数の注釈の形）：count、geometry＋properties × count、id × count ----
GEOM = {"point": 3, "line": 6, "axis_aligned_bounding_box": 6, "ellipsoid": 6}
PSIZE = {"uint32": 4, "int32": 4, "float32": 4, "uint16": 2, "int16": 2, "uint8": 1, "int8": 1}
PFMT = {"uint32": "<u4", "int32": "<i4", "float32": "<f4", "uint16": "<u2", "int16": "<i2", "uint8": "u1", "int8": "i1"}

def decode_multi(b, info):
    rank = len(info["dimensions"]); g = GEOM[info["annotation_type"].lower()] // 3 * rank
    props = info.get("properties", [])
    order = sorted(props, key=lambda p: -PSIZE[p["type"]])  # 4 → 2 → 1 バイトの順
    psz = sum(PSIZE[p["type"]] for p in props); rec = 4 * g + psz
    rec = (rec + 3) // 4 * 4
    n = struct.unpack_from("<Q", b, 0)[0]
    body = b[8:8 + n * rec]
    geom = np.frombuffer(body, np.uint8).reshape(n, rec)
    xyz = geom[:, :4 * g].copy().view("<f4").reshape(n, g)
    out = {}
    off = 4 * g
    for p in order:
        w = PSIZE[p["type"]]
        out[p["id"]] = geom[:, off:off + w].copy().view(PFMT[p["type"]]).reshape(n)
        off += w
    ids = np.frombuffer(b[8 + n * rec:8 + n * rec + 8 * n], "<u8")
    return xyz, out, ids

# ---- 骨格：nv, ne, 座標 nv×3, 辺 ne×2, 頂点の属性 ----
def decode_skel(b, info):
    nv, ne = struct.unpack_from("<II", b, 0)
    v = np.frombuffer(b, "<f4", nv * 3, 8).reshape(nv, 3)
    e = np.frombuffer(b, "<u4", ne * 2, 8 + 12 * nv).reshape(ne, 2)
    off = 8 + 12 * nv + 8 * ne; attrs = {}
    for a in info.get("vertex_attributes", []):
        k = a.get("num_components", 1)
        attrs[a["id"]] = np.frombuffer(b, PFMT[a["data_type"]], nv * k, off).reshape(nv, k)
        off += 4 * nv * k
    return v, e, attrs

def cable_um(v, e):
    if len(e) == 0:
        return 0.0
    return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).sum()) / 1000.0

def short(o, n=600):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " …"

def main():
    os.makedirs(OUT, exist_ok=True)
    print("== ① 細胞の札（segment_properties）")
    p = json.loads(maybe_gunzip(get(PROPS + "/info")))
    with open(os.path.join(OUT, "segment_properties.json"), "w") as f:
        json.dump(p, f)
    inl = p.get("inline", {}); ids = inl.get("ids", [])
    print("  @type", p.get("@type"), "・番号の数", len(ids))
    for pr in inl.get("properties", []):
        vals = pr.get("values", [])
        print("  性質 id=%s type=%s 値の数 %d" % (pr.get("id"), pr.get("type"), len(vals)))
        if pr.get("type") == "tags":
            tags = pr.get("tags", []); cnt = {}
            for vv in vals:
                for t in vv:
                    cnt[tags[t]] = cnt.get(tags[t], 0) + 1
            print("    札と数", short(sorted(cnt.items(), key=lambda x: -x[1]), 1500))
        else:
            if pr.get("type") in ("label", "description", "string"):
                cnt = {}
                for vv in vals:
                    cnt[vv] = cnt.get(vv, 0) + 1
                top = sorted(cnt.items(), key=lambda x: -x[1])[:40]
                print("    多い値", short(top, 1500))
            print("    初めの値", short(vals[:5]))
    print("  初めの番号", ids[:5])
    s_ids = set(str(i) for i in ids)
    print("  L2 の抑制の細胞の例のうち札にある数", sum(str(i) in s_ids for i in L2_IN), "/", len(L2_IN))
    for i in L2_IN:
        if str(i) in s_ids:
            k = [str(x) for x in ids].index(str(i))
            row = {}
            for pr in inl.get("properties", []):
                vv = pr["values"][k]
                row[pr["id"]] = [pr["tags"][t] for t in vv] if pr.get("type") == "tags" else vv
            print("   ", i, short(row))

    print("== ② シナプスの注釈の info")
    si = json.loads(get(SYN + "/info"))
    print("  ", short({k: si[k] for k in si if k not in ("spatial",)}, 2500))
    rel = {r["id"]: r for r in si.get("relationships", [])}
    print("  関係の索引", list(rel))
    print("== ③ 骨格の info")
    ki = json.loads(get(SKEL + "/info"))
    print("  ", short(ki, 800))
    try:
        ci = json.loads(get(SUBC + "/info"))
        print("== 区画の info", short(ci, 1200))
    except Exception as ex:
        print("== 区画の info：取れない", type(ex).__name__)

    print("== ④ 細胞ごと（L2 の抑制の細胞の例 3つ）")
    pr_ = rel.get("post_synaptic_cell")
    for seg in L2_IN:
        print("  -- 番号", seg)
        if pr_ is not None:
            b = sharded_get(SYN + "/" + pr_["key"], seg, pr_["sharding"])
            if b is None:
                print("    入るシナプス：索引に無い")
            else:
                xyz, props, aid = decode_multi(b, si)
                print("    入るシナプス", len(aid), "個・座標の範囲（注釈の単位）",
                      np.round(xyz.min(0)).tolist(), "〜", np.round(xyz.max(0)).tolist())
                for k, v in props.items():
                    u, c = np.unique(v, return_counts=True)
                    print("    性質 %s の値と数" % k, dict(zip(u.tolist(), c.tolist())))
        sk = sharded_get(SKEL, seg, ki["sharding"]) if "sharding" in ki else None
        if sk is None:
            print("    骨格：無い")
        else:
            v, e, at = decode_skel(sk, ki)
            print("    骨格 点 %d・辺 %d・座標の範囲" % (len(v), len(e)),
                  np.round(v.min(0)).tolist(), "〜", np.round(v.max(0)).tolist())
            print("    辺の長さの和（座標を nm とすると）%.0f µm" % cable_um(v, e),
                  "・半径の中央", float(np.median(at["radius"])) if "radius" in at else "-")
    print("== 落とした量 %.2f MB（保存したのは ~/data/h01/segment_properties.json だけ）" % (_got[0] / 1e6))

def selftest():
    import mmh3
    from cloudvolume.datasource.precomputed.sharding import ShardingSpecification, synthesize_shard_files
    from cloudvolume import Skeleton
    rng = np.random.default_rng(0)
    for _ in range(200):
        key = int(rng.integers(0, 2**62))
        want = mmh3.hash64(struct.pack("<Q", key), seed=0, x64arch=False, signed=False)[0]
        assert murmur3_x86_128_lo64(struct.pack("<Q", key)) == want
    print("✓ murmurhash3_x86_128 が mmh3 と 200 回同じ")
    spec = ShardingSpecification("neuroglancer_uint64_sharded_v1", preshift_bits=1, hash="murmurhash3_x86_128",
                                 minishard_bits=3, shard_bits=2, minishard_index_encoding="gzip", data_encoding="gzip")
    data = {int(k): os.urandom(int(rng.integers(5, 300))) for k in rng.integers(1, 2**40, 300)}
    files = synthesize_shard_files(spec, data)
    def fetch(path, rg):
        return files[path.split("/")[-1]][rg[0]:rg[1]]
    sh = {"preshift_bits": 1, "hash": "murmurhash3_x86_128", "minishard_bits": 3, "shard_bits": 2,
          "minishard_index_encoding": "gzip", "data_encoding": "gzip"}
    for k, v in data.items():
        assert sharded_get("x", k, sh, fetch) == v
    print("✓ cloud-volume が作った分割ファイル %d 個から %d 個を正しく引けた" % (len(files), len(data)))
    sk = Skeleton(vertices=rng.random((50, 3)).astype(np.float32) * 1000,
                  edges=np.array([[i, i + 1] for i in range(49)], np.uint32),
                  radii=rng.random(50).astype(np.float32))
    v, e, at = decode_skel(sk.to_precomputed(), {"vertex_attributes": [{"id": "radius", "data_type": "float32", "num_components": 1}]})
    assert np.allclose(v, sk.vertices) and (e == sk.edges).all() and np.allclose(at["radius"][:, 0], sk.radius)
    print("✓ cloud-volume が書いた骨格を読めた（点・辺・半径）")
    import neuroglancer, tempfile
    from neuroglancer import write_annotations as wa
    cs = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=[8, 8, 33])
    w = wa.AnnotationWriter(coordinate_space=cs, annotation_type="line",
                            properties=[neuroglancer.AnnotationPropertySpec(id="type", type="uint32")],
                            relationships=["post_synaptic_cell"])
    pts = []
    for i in range(40):
        a = rng.random(3) * 1000; b = a + rng.random(3) * 10; t = int(rng.integers(1, 3))
        w.add_line(a, b, type=t, post_synaptic_cell=[7 if i < 25 else 9]); pts.append((a, b, t))
    d = tempfile.mkdtemp(); w.write(d)
    info = json.load(open(os.path.join(d, "info")))
    relkey = info["relationships"][0]["key"]
    rd = os.path.join(d, relkey)
    if info["relationships"][0].get("sharding"):
        print("  （この neuroglancer の書き手は関係の索引を分割で書いた）")
        rs = info["relationships"][0]["sharding"]
        def f2(path, rg):
            return open(os.path.join(rd, path.split("/")[-1]), "rb").read()[rg[0]:rg[1]]
        b7 = sharded_get(relkey, 7, rs, f2)
    else:
        b7 = open(os.path.join(rd, "7"), "rb").read()
    xyz, props, aid = decode_multi(b7, info)
    assert len(aid) == 25 and np.allclose(xyz[:, :3], np.array([p[0] for p in pts[:25]]), atol=1e-3)
    assert (props["type"] == np.array([p[2] for p in pts[:25]])).all()
    print("✓ neuroglancer が書いた注釈（線・型・受け手の索引）を読めた：受け手 7 に 25 個")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
