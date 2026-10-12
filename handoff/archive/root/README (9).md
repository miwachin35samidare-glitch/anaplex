# numpy版（参照用）

torch版は、これらを移したもの。**移すときに落としたもの・変えたものがある。**
突き合わせは `デフォルメ版_引き継ぎ_移植の点検.md` を見る。

⚠️ ここのファイルは動かすためのものではない。torch版と中身を比べるための原本。

| numpy | torch | 中身 |
|---|---|---|
| area2.py | torch_area.py / torch_group2.py | 資源・順位・予備・束の成長。**抑制の振り方と資源の使い方の原本** |
| area3.py | 同上 | 表現のニューロンも学ぶ／抑制を体にする／傾きの細胞 |
| area4.py | 同上 | 送り手ごとに連ねる持ち方（撃った分だけ触る）／周回／時間細胞の有無 |
| group.py | torch_group2.py | 領野をまとめて1回で進める |
| hippocampus.py | torch_brain.py | 歯状回 → CA3 → CA1、苔状線維・シャッファー側枝・引き込み |
| thalamus.py | torch_brain.py | 視床。運動視床の蹴り |
| brain.py | torch_brain.py | 段の組み方 |
| chains.py | torch_area.py（TimeCells） | 時間細胞（対数圧縮）・傾きの細胞 |
| stp.py | torch_area.py（STP） | 短期可塑性。触ったときにまとめて減衰 |
| element1.py | torch_area.py（Prior） | 焼いた素子 |
| timing.py | torch_area.py の定数 | 歩と ms の対応 |
| sparse.py | torch_area.py | 送り手ごとに連ねる持ち方の部品 |
| dpca.py | （なし） | 人と同じ測り方（dPCA） |
| mouth_drive.py | voice.py / mouth.py | 補助輪を外した口（作り物の声の版） |
