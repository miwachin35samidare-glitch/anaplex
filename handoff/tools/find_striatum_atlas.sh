#!/bin/bash
# arkhe に FSL と Tziortzi 2014 の線条体の地図（Oxford-GSK-Imanova Striatal Connectivity Atlas）があるかを見る（何も書き換えない）
#   bash find_striatum_atlas.sh
# 11-01 3:26【み】「人にできるなら人にして」：皮質 → 線条体の本数を人の比（核ごと）に置き直すため
echo "FSLDIR=${FSLDIR:-（無し）}"
which fsl flirt fslstats 2>/dev/null || echo "fsl のコマンドは見つからない"
for d in "$FSLDIR" /usr/local/fsl /opt/fsl "$HOME/fsl" /usr/share/fsl /usr/share/fsl-*; do
  [ -n "$d" ] && [ -d "$d" ] && echo "あり：$d"
done
echo "--- 線条体の地図のファイル（striatum-con-*）を探す（1分ほどかかることがある）"
find / -xdev \( -iname 'striatum-con-*' -o -iname 'striatum-structural*' \) 2>/dev/null | head -20
echo "--- HarvardOxford の皮質下の地図（尾状核・被殻・側坐核の切り分けに使う）"
find / -xdev -iname 'HarvardOxford-sub-maxprob-thr25*' 2>/dev/null | head -5
echo "終わり"
