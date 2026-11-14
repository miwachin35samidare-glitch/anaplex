"""Wandelt 2024（Zenodo 10697024、縁上回・体性感覚野、四肢まひの2人）のデータの形を見る（11-14）
  使い方：~/dandi-venv/bin/python wandelt_peek.py <DatabaseInternalSpeech を展開した所>
  読むのは Database/<s2|s3>/Data/Table_<s>_Speech.mat（MATLAB の table。mat-io で読む）
  出すもの：列の名前と型・1行の中身の形・相の札（ITI・Cue・D1・Internal・D2・Speech）・率の値の範囲（数か Hz か）
  パス・機械の名は出さない（リポジトリは公開）
"""
import sys
import pathlib
import numpy as np

try:
    from matio import load_from_mat
except ImportError:
    sys.exit("⚠️ mat-io が無い：~/dandi-venv/bin/pip install mat-io")

root = pathlib.Path(sys.argv[1]).expanduser()
mats = sorted(root.rglob("*.mat"))
print(f"mat の数 {len(mats)}")
for m in mats[:60]:
    print(f"  {m.relative_to(root)}  {m.stat().st_size / 1e6:.1f} MB")

for s in ("s2", "s3"):
    hits = [m for m in mats if m.name == f"Table_{s}_Speech.mat"]
    if not hits:
        print(f"\n{s}：Table_{s}_Speech.mat が無い")
        continue
    print(f"\n==== {s}：{hits[0].relative_to(root)}")
    d = load_from_mat(str(hits[0]))
    for k, v in d.items():
        if k.startswith("__"):
            continue
        print(f"変数 {k}：{type(v).__name__}")
        if hasattr(v, "columns"):
            print(f"  行 {len(v)}・列 {list(v.columns)}")
            for c in v.columns:
                x = v[c].iloc[0]
                desc = type(x).__name__
                if isinstance(x, np.ndarray):
                    desc += f" {x.shape} {x.dtype}"
                elif isinstance(x, (list, tuple)):
                    desc += f" 長さ {len(x)}"
                    if len(x) and isinstance(x[0], np.ndarray):
                        desc += f"・中 {x[0].shape} {x[0].dtype}"
                print(f"    {c}：{desc}  例 {str(x)[:80]!r}")
            for reg in ("SMG_Go", "S1X_Go"):
                if reg not in v.columns:
                    continue
                arrs = []
                for x in v[reg].iloc[:200]:
                    while isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.dtype == object):
                        x = x[0]
                    arrs.append(np.asarray(x, dtype=float))
                a = np.concatenate([q.ravel() for q in arrs])
                ints = np.mean(np.abs(a - np.round(a)) < 1e-9)
                print(f"  {reg}：1行の形 {arrs[0].shape}・値 最小 {a.min():.3g} 中央 {np.median(a):.3g} 平均 {a.mean():.3g} 最大 {a.max():.3g}・整数の割合 {ints:.2f}")
            if "time_phase_labels" in v.columns:
                t = v["time_phase_labels"].iloc[0]
                while isinstance(t, (list, tuple)) or (isinstance(t, np.ndarray) and t.dtype == object):
                    t = t[0]
                t = np.asarray(t).ravel()
                u, c = np.unique(t, return_counts=True)
                print(f"  相の札：{dict(zip(u.tolist(), c.tolist()))}（1試行 {len(t)} 区切り）")
            if "session_date" in v.columns:
                print(f"  記録の日の数 {v['session_date'].nunique()}")
