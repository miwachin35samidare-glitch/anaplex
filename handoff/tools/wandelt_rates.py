"""Wandelt 2024（Zenodo 10697024）縁上回の体ごとの率を相ごとに（11-14）
  使い方：~/dandi-venv/bin/python wandelt_rates.py <DatabaseInternalSpeech を展開した所>
  データ：Table_<s>_Speech.mat の Go_data（1行 ＝ 1試行）。SMG_Go は [157 区切り × 体]、区切り 50ms、値は Hz（論文の図の軸・
    回帰は「50ms の区切りの率」）。ならしてある（核の形は論文の抜き出しから分からない）
  相の札 time_phase_labels：1 ITI（試行のあいだ）・2 Cue（合図）・3 D1・4 Internal（頭の中で言う）・5 D2・6 Speech（声に出して言う）
  体は記録の日ごとに分けてある（論文：日ごとに並べ分けた単一の体）⇒ 「日 × 体」を1体として数える
  各相の端の 2区切り（100ms）は除く【C】：ならしで隣の相の撃ちが混ざるのを避ける（核の幅は不明）
  出すもの：相ごとの体の率（平均・中央・四分位）・Speech ÷ ITI の体ごとの比・ITI より上がる／下がる体の割合
  パス・機械の名は出さない（リポジトリは公開）
"""
import sys
import pathlib
import numpy as np

try:
    from matio import load_from_mat
except ImportError:
    sys.exit("⚠️ mat-io が無い：~/dandi-venv/bin/pip install mat-io")

PH = {1: "ITI", 2: "Cue", 3: "D1", 4: "Internal", 5: "D2", 6: "Speech"}
EDGE = 2


def unwrap(x):
    while isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.dtype == object):
        x = x[0]
    return np.asarray(x)


def q(a):
    a = np.asarray(a)
    return f"平均 {a.mean():6.2f}  中央 {np.median(a):6.2f}  四分位 {np.percentile(a, 25):5.2f}〜{np.percentile(a, 75):5.2f}"


root = pathlib.Path(sys.argv[1]).expanduser()
for s in ("s2", "s3"):
    hits = sorted(root.rglob(f"Table_{s}_Speech.mat"))
    if not hits:
        print(f"{s}：無い")
        continue
    df = load_from_mat(str(hits[0]))["Go_data"]
    dates = np.array([str(unwrap(x).ravel()[0]) for x in df["session_date"]])
    cues = np.array([str(unwrap(x).ravel()[0]) for x in df["cueType"]])
    print(f"\n==== {s}：試行 {len(df)}・記録の日 {len(set(dates))}・合図 {dict(zip(*np.unique(cues, return_counts=True)))}")
    per_unit = {p: [] for p in PH}          # 「日 × 体」ごとの相の平均の率
    for d in sorted(set(dates)):
        rows = np.where(dates == d)[0]
        lab = unwrap(df["time_phase_labels"].iloc[rows[0]]).ravel().astype(int)
        keep = {}
        for p in PH:
            idx = np.where(lab == p)[0]
            keep[p] = idx[EDGE:len(idx) - EDGE] if len(idx) > 2 * EDGE + 2 else idx
        X = np.stack([unwrap(df["SMG_Go"].iloc[r]).astype(float) for r in rows])   # [試行, 157, 体]
        for p in PH:
            per_unit[p].append(X[:, keep[p], :].mean(axis=(0, 1)))
        if d == sorted(set(dates))[0]:
            print("  1試行の相の区切りの数：" + "・".join(f"{PH[p]} {int((lab == p).sum())}（{(lab == p).sum() * 50} ms）" for p in PH))
    U = {p: np.concatenate(v) for p, v in per_unit.items()}
    n = len(U[1])
    print(f"  日 × 体 ＝ {n}（端 {EDGE} 区切りを除いた相の平均、Hz）")
    for p in PH:
        print(f"  {PH[p]:9s} {q(U[p])}")
    it, sp, inn = U[1], U[6], U[4]
    ok = it > 0.05
    r = sp[ok] / it[ok]
    print(f"  Speech ÷ ITI（ITI 0.05Hz 超の {ok.sum()} 体）：中央 {np.median(r):.2f}・四分位 {np.percentile(r, 25):.2f}〜{np.percentile(r, 75):.2f}")
    print(f"  Internal ÷ ITI：中央 {np.median(inn[ok] / it[ok]):.2f}")
    print(f"  全体の平均の比 Speech ÷ ITI {sp.mean() / it.mean():.2f}・Internal ÷ ITI {inn.mean() / it.mean():.2f}")
    for nm_, th in (("1.5倍超に上がる", 1.5), ("2/3 未満に下がる", 2 / 3)):
        frac = np.mean(r > th) if th > 1 else np.mean(r < th)
        print(f"  Speech が ITI の {nm_} 体：{frac * 100:.0f}%")
    print(f"  ITI の率が 1Hz 未満の体：{np.mean(it < 1) * 100:.0f}%")
