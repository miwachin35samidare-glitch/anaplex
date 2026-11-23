"""F-TRACT の人の CCEP の図（780人）の中身の並びを見る（arkhe で。網は使わない）
  python3 ftract_look.py              ⇒ Zenodo の zip を全部は落とさず、目録と要る表の頭だけを部分的に読む（数 MB）
  python3 ftract_look.py ~/f-tract_v2112.zip   ⇒ 落とし終わった zip があればそれを読む
【み】10-14 Zenodo からの wget が 50〜170kB/s・残り 3時間半 ⇒ zip の目録（末尾）と要る所だけを範囲指定で読む
【ヒト】Lemaréchal 2022 Brain：780人・34,354回の刺激・774,445本の CCEP。15歳超の軸索の遅れ 中央 10.2ms（16% が 20ms 超）・伝わる速さ 中央 3.9m/s
出すもの
  1 中のフォルダの並び（上の3段）
  2 HCP-MMP1 の .csv の名前と、頭の3行（300字まで）
  3 読んだ量
⇒ これを見て、模型の領野の組の値を抜き出す道具を作る
"""
import sys, io, zipfile, collections, urllib.request

URL = "https://zenodo.org/records/7015415/files/f-tract_v2112.zip?download=1"
CHUNK = 256 * 1024


class RangeFile(io.RawIOBase):
    """HTTP の範囲指定で読む、読み専用のファイル（zipfile が目録と中身を必要な所だけ読む）"""

    def __init__(self, url):
        self.url, self.pos, self.got, self.cache = url, 0, 0, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range")
            if not cr:
                sys.exit("⚠️ 範囲指定が効かない。wget で全部落としてから python3 ftract_look.py ~/f-tract_v2112.zip")
            self.size = int(cr.split("/")[-1])
            self.url = r.geturl()                      # 行き先（転送先）を覚える

    def _chunk(self, i):
        if i not in self.cache:
            a = i * CHUNK
            b = min(a + CHUNK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.cache[i] = r.read()
            self.got += len(self.cache[i])
        return self.cache[i]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, p, end = bytearray(), self.pos, self.pos + n
        while p < end:
            i = p // CHUNK
            c = self._chunk(i)
            s = p - i * CHUNK
            take = c[s:s + (end - p)]
            out += take
            p += len(take)
        b[:n] = out
        self.pos = end
        return n


if len(sys.argv) > 1:
    src = sys.argv[1]
    z = zipfile.ZipFile(src)
    rf = None
else:
    rf = RangeFile(URL)
    print(f"zip の大きさ {rf.size/2**20:.0f}MB（全部は落とさない）", flush=True)
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=CHUNK))
names = z.namelist()
print(f"中のファイル {len(names):,}", flush=True)
dirs = collections.Counter("/".join(n.split("/")[:-1][:3]) for n in names)
print("── フォルダ（上の3段まで。ファイル数）")
for d, c in sorted(dirs.items())[:80]:
    print(f"  {c:6d}  {d}")
hcp = [n for n in names if "HCP" in n.upper() and n.endswith(".csv")]
print(f"\n── HCP-MMP1 の .csv {len(hcp):,}本（名前の種類ごとに1つ見せる）")
seen = set()
for n in hcp:
    base = n.split("/")[-1]
    key = (base, "15" in n and "100" in n)
    if key in seen:
        continue
    seen.add(key)
    print(f"\n  {n}  （{z.getinfo(n).file_size/1024:.0f}kB）")
    with z.open(n) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print("    " + line.decode("utf-8", "replace").rstrip()[:300])
    if len(seen) >= 30:
        break
other = [n for n in names if n.lower().endswith((".txt", ".md", ".pdf", ".m"))][:20]
print("\n── 説明・読むためのファイル", *("  " + n for n in other), sep="\n")
if rf is not None:
    print(f"\n読んだ量 {rf.got/2**20:.1f}MB", flush=True)
