"""口語の句（アクセント句）が何モーラあるかを数える（arkhe。網は使わない）
  wget https://mmsrv.ninjal.ac.jp/nucc/nucc.zip        （名大会話コーパス、CC BY-NC-ND 4.0、国語研）
  python3 count_ap.py nucc.zip [数える発話の数=5000]
【み】09-30「口語調のもの」⇒ 名大会話コーパス（129会話・約100時間の雑談の文字起こし）
句の区切りは VOICEVOX（OpenJTalk）が決まりで自動に切ったもの。人が聞いて付けた区切りではない【C】
出すもの
  1 句ごとのモーラ数の分布（全部の句）
  2 発話の最初の句のモーラ数の分布（【ヒト】Allum・Wheeldon 2007：日本語でも準備の範囲は文頭の句1つ）
  3 発話全体のモーラ数の分布（参考）
"""
import re
import sys
import zipfile
import numpy as np
from voicevox import Voicevox

ZIP = sys.argv[1] if len(sys.argv) > 1 else "nucc.zip"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

# 文字起こしの印を外す：＜笑い＞などの注記、【】（）の中身、＊（名前を伏せた所）、話者の記号
DROP = re.compile(r"＜[^＞]*＞|<[^>]*>|【[^】]*】|\[[^\]]*\]|（[^）]*）|\([^)]*\)|[＊*]+")
KEEP = re.compile(r"[ぁ-んァ-ヶー一-龥々〆ヵヶ、。？！・]")


def lines_of(zf):
    for name in sorted(zf.namelist()):
        if not name.lower().endswith(".txt"):
            continue
        raw = zf.read(name)
        for enc in ("utf-8", "cp932"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            continue
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(("＠", "@", "％", "%")):
                continue
            m = re.match(r"^[A-Za-zＡ-Ｚ0-9０-９]+[：:](.*)$", ln)   # 例「F107：うん。」
            if not m:
                continue
            s = DROP.sub("", m.group(1))
            s = "".join(ch for ch in s if KEEP.match(ch))
            if sum(1 for ch in s if ch not in "、。？！・ー") >= 1:
                yield s


def show(name, xs):
    xs = np.array(xs)
    print(f"\n[{name}] {len(xs)}個  平均 {xs.mean():.2f}  中央 {np.median(xs):.0f}  "
          f"90% {np.percentile(xs, 90):.0f}  95% {np.percentile(xs, 95):.0f}  最大 {xs.max()}")
    for k in range(1, 13):
        print(f"  {k:2d}モーラ {np.mean(xs == k):6.1%}   {k}以下 {np.mean(xs <= k):6.1%}")


with zipfile.ZipFile(ZIP) as zf:
    utts = list(lines_of(zf))
print(f"発話 {len(utts)}（印を外したあと）。例：{utts[:5]}")
rng = np.random.default_rng(0)
pick = rng.choice(len(utts), size=min(N, len(utts)), replace=False)
vv = Voicevox().connect()
ap_len, first_len, utt_len, fail = [], [], [], 0
for j, i in enumerate(pick):
    try:
        q = vv._audio_query(utts[i])
    except Exception:
        fail += 1
        continue
    lens = [len(ap["moras"]) for ap in q["accent_phrases"] if ap["moras"]]
    if not lens:
        continue
    ap_len += lens
    first_len.append(lens[0])
    utt_len.append(sum(lens))
    if (j + 1) % 1000 == 0:
        print(f"  {j + 1}発話", flush=True)
print(f"VOICEVOX が通らなかった発話 {fail}")
show("句ごとのモーラ数（全部の句）", ap_len)
show("発話の最初の句のモーラ数", first_len)
show("発話全体のモーラ数（参考）", utt_len)
