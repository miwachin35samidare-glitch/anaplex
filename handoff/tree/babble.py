"""喃語を回す（arkhe 用）—— 保存して、続きから再開できる

保存するのは学習で変わるものだけ
  シナプスの重み ／ 辺の並び（刈り込みで変わる）／ 恒常性の状態 ／
  素子の中のゆっくりした状態 ／ 短期可塑性の状態 ／ 時間細胞の age
焼いた素子（prior_v1.npz）は変わらないので保存しない

使い方
  python3 babble.py 100            100回まわして保存（state.pt が無ければ0から）
  python3 babble.py 100            もう100回（続きから）
  python3 babble.py 0              回さずに測るだけ
  python3 babble.py 100 --fresh    保存を無視して0から

⚠️ 声は作り物（VOICEVOX 未接続）。本物の声は mouth.audio_query() を voicevox.py に渡す
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import config as C
import torch_area as T
import brainstem as BS
from torch_brain import Brain, NAMES, to_dev

STATE = "state.pt"
VERSION = 115         # コードの版。網の作りを変えたら上げる
SAME_NET = (113, 114)     # 版115：駆動 x の組み立てを番号の表1回で（意味は同じ・網の形は同じ ⇒ 版113・114 の state.pt は読める）／ 版114：網の中の興奮の辺の上限を受け手ごとに 6mV（接触で割る）。網の形は版113 と同じ ⇒ 版113 の state.pt は読める ／ 版113：皮質の 表現 → 表現 を人の本数・重さ・下からの線と同じ学びに（網の形が変わる）⇒ state.pt は 0 から ／ 版112：CL 256・視床網様核の抑制の体なし・橋核 5%（網の形が変わる）⇒ state.pt は 0 から ／ 版111：運動の表現・誤差に恒常性の目標 9.6Hz・滑る境目（網の形は版110 と同じ ⇒ 版109・110 の state.pt は読める）／ 版110：下からの線・CL の「受け手が先」の項にも滑る境目（学びの式だけ。網の形は版109 と同じ ⇒ 版109 の state.pt は読める）／ 版109：皮質の型の領野を全部 切片の見方の抑制に（配線が変わる）⇒ state.pt は 0 から ／ 版107：落とした 13部位・ストリオソームを人の値・門 2つ（網の形が変わる）⇒ state.pt は 0 から（版104・105・105b は VERSION を 103 のまま当てていた）／ 版103：統語の部位（44・45・STS1・STS2・中側頭後・TPJ・視床枕）と皮質 → 線条体を人の比に（網の形が変わる）⇒ state.pt は 0 から ／ 版102：縮める側の体数 × 0.85（網の形が変わる）⇒ state.pt は 0 から ／ 版101：来ないことに撃つ体を足した（網の形が変わる）⇒ state.pt は 0 から ／ 版100：前帯状・PAG を足した（網の形が変わる）⇒ state.pt は 0 から ／ 版99：扁桃体・眼窩前頭 ／ 版98：小脳
_SAME_NET_93 = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92)   # 版57 で海馬の抑制の体の配線を引き直した ⇒ 版56 までの state.pt は読めない（版58 は自発の確率だけ）
#   1  最初の torch 版
#   2  周回のあいだも発火が伝わるように直した
#   3  歯状回の閾値が、入力ゼロのとき全部通っていたのを直した
#   4  表現のニューロンも合計で撃つ（人：134±28本）／ まとめた側に刈り込みを移した ／
#      恒常性の目標を人の 135本/歩 に（前は前の作りの 0.855 を意味を確かめずに置いていた）
#   5  刈り込みが一度も呼ばれていなかったのを直した（学習する周と歩を進める周が別だった）／
#      VOICEVOX を繋いだ（声が変わるので、これ以前の state は使えない）
#   6  海馬 → 皮質 の戻りを繋いだ（CA1 の出力が、どこにも渡っていなかった）／
#      睡眠（徐波・紡錘波・リップル）を入れた
#   7  視床（MGB）を領野として入れた。耳 → 視床 → 聴覚。
#      耳は聴神経らしく密に、視床が受けて疎にする
#      ⚠️ 蝸牛神経核と下丘は入っていない。1kHz超の時刻を扱う段で、
#         こちらの耳は10.67msごとなので変換する元が無い（要るなら耳の刻みから作り直し）
#   8  指令の写しを、時刻に置くのをやめて発話のあいだ流す形にした
#      （時刻に置くと supp 0.38。前のモーラの長さが後ろ全部をずらすので、
#       指令から時刻を計算しても音と合わない。前の作りも丸ごと渡す形で動いていた）
#   9  疎化の段を人の並びに直した。視床・一次は密、連合から疎（前は視床で疎にしていた）
#      【ヒト】ヘシュル回は話し言葉に 39〜40% が反応／上側頭回で 45% が特定の言葉にだけ撃つ
#  10  耳を人の形に太くした（ERB 40個 × 聴神経10本 ＝ 400体。閾値が繊維ごとに違う）
#  11  視床の自発を 8.7Hz（マーモセット）から 0.3Hz に戻した。
#      人の MGB の発火率は取れない（DBS の通り道ではない）ので、0.3Hz はノブ
#  12  段と段の繋ぎを切り詰め（_fit）から、配り切る投射に直した
#      聴覚400 → 連合40 が先頭40本だけ＝メル0〜3chしか上へ行っていなかった
#      （耳を40→400に太くした版10で、40→40の何もしない行が10:1の切り詰めに変わっていた）
#      運動1812 → 高次の樹状突起も先頭40本だけだった ⇒ 丸ごと渡す（配線側が相手を選ぶ）
#  13  段ごとの体数を人の比に直した（連合 40→530 ／ 高次 40→580）
#      【ヒト】Julich-Brain の同じ地図で Te1 5,917 ／ Te2 9,682 ／ Te3 10,591 mm³、
#      密度 A1 42,069 ／ Tpt 34,147 個/mm³ ⇒ 1 : 1.33 : 1.45
#      台帳の「Te1 704」は Te1 ではなく TeI（側頭・島の領域）の読み違いだった
#      作り物の声の encode が 40 のままだったのを、耳と同じ 400 に直した
#  14  段と段の投射を周波数で組み直した（版12の _deal は上の段が大きいと1対1になっていた）
#      一次【ヒト】1帯（Bitterman 2008：1/12 oct。模型の1帯より狭い）
#      連合【マカク】外側ベルト 1/3〜2 oct を14段（Rauschecker & Tian 2004）
#      高次【ヒト】上側頭回 帯域 4.03±1.57 oct・峰 7.72±2.14 個（Leonard ほか 2024）
#      駆動は届いた本数を足す（割らない。09-11 に割り算を周転円として落としている）
#  15  抑制の割合をシナプス数で【ヒト】16% に（前は「抑制の体 ÷ 送り元」で 2.4〜6.6% まで薄まっていた）
#      刈り込みで生やす先を、受け手と同じ領野・切った辺と同じ種類に
#      （Group2 にまとめたとき、全領野の送り元から引くようになっていた）
#  16  連合の帯域幅を【ヒト】後側ベルト（平面側頭 ＝ Te2）の推定範囲 10〜27 ERB に（Besle ほか 2022）
#      （前はマカク外側ベルトの 1/3〜2 oct。みわの判断で人の推定範囲へ）
#  17  連合・高次へ渡す駆動を「足した本数」から「その細胞が撃つか（0か1）」に
#      予測はシグモイドで1を超えないので、本数のままだと誤差が永久に正で重みが伸び続けた
#      【ヒト】錐体細胞の出力は集まった入力を閾値と比べた撃つ／撃たない。海馬の貫通路と同じ形
#  18  numpy版（原本）との突き合わせで見つかった移植後の変更を、人の側へ戻した
#      scale：本数→135 を 強さ→G_FIRE 0.855 に（0.855 は受け手1体の総量。1本あたり 0.00633 と取り違えていた）
#      貫通路：12.5% → 2%（【ヒト】・霊長類の疎な投射。代用の 1/8 だけが残っていた）
#      視床：耳の素通し → 1体が耳の 1/8 を束ねる中継。樹状突起に感覚そのもの
#      蹴り：32体 → 出力の 75%（⟲ 75% は人の値ではなかった。原本の置き値）
#  19  視床を 耳1本 → 1体 に戻した。【ヒト】MGB はトノトピーで並ぶ。numpy 版の「1/8 をランダム」は
#      人の値ではなく、耳400体では視床が約80% 撃って聴覚の誤差が耳の2倍に溢れた（版18）
#  20  シナプススケーリングの目標を、細胞自身の発火率【ヒト】0.45%/歩 に
#      【ヒト】iPS 由来の皮質細胞：活動を止めると興奮性シナプス全体が強まり活動が戻る（Cordella 2022）
#      G_FIRE 0.855 はコンダクタンスの素子の値で、このラインの素子には持ち越せない（落-039 と同じ形）
#  21  撃つ閾値を人の1/1（繋がり 14本ぶん ＝ 16mV ÷ 1.12mV【ヒト】）に。入り切らない本数は x_gain のノブで補う
#      誤差の的を「下からの入力（本数 × 重さ）が閾値を越えるか」に一般化（版17 の _fire を置き換え）
#      運動が一度も撃たなかったのは、蹴り 1 × 重さ 1 が閾値 1.0 に並ぶだけで越えなかったため
#  22  CA3 を人の1/1 に：閾値 約29本（(66.5−43.6)mV ÷ 0.80mV【ヒト】）、相手 16,100（【ヒト】Watson 2025）
#      苔状線維は4本のまま、ノブと明記（人は約280本。歯状回300体では疎さと両立しない）
#  23  興奮性の接触の数を領野ごとに。CA3 は【ヒト】ほぼ1つ（皮質は 3.3 のまま）
#  24  蹴りを話し始めに出す（go() がどこからも呼ばれていなかった）／ scale の下限 0.05 を外す（numpy 原本に無い）
#  25  話し始めの準備 103歩（【ヒト】合図→声 約1.1秒）を発話の頭に置き、そのあいだ蹴りを入れ続ける
#      写しは音の5歩前から（前と同じ関係）。版24 の蹴り（発話の頭で3歩）は人の形でなかった
#  26  TAU を【ヒト】積分の窓の比 1.0／1.63／3.4 に（Norman-Haignere 2025。前はマカクの自己相関の比）
#  27  聴覚は自分の帯の視床10本の合計を受け、10本そろって越える（重さ 1.50、的が人の発火率の範囲に入る値）
#  28  耳の繊維の閾値の重なりを直した（2群の境目の値が2本あり、10本で9段だった）
#  29  聴覚は帯の9本そろって越える（重さ 1.68）。版27 の10本では実際の的が 0.20% で人の範囲を割った
#  30  蹴りを「毎歩 1」から、運動の視床の人の発火率（18.8Hz ＝ 1歩 20%）のスパイクの列に
#  31  誤差の的に、表現と同じ抑制を入れた（的＝下からの入力で抑制込みで撃つか。横の興奮は入れない）
#  32  版31 を戻した。学びが抑制を育てて的を消し、皮質が止まった。的は学ぶ量に依存させない
#  33  抑制の繋がりを人の値（1本 漏れの 0.07倍）で始め、誤差では学ばない。恒常性の scale は抑制に掛けない。
#      刈り込みの順位から抑制を外す。興奮性の繋がりは予測誤差で学び続ける
#  34  刈り込み：興奮性を強さで切り、切った分は興奮性で生やす（【ヒト】興奮と抑制の割合は領野・個人でほぼ一定）
#  35  版33・34 を取り下げて版32 に戻した（根拠の MRS はシナプスの抑制ではなかった。学ばない抑制は根幹を外す）
#  36  興奮の重みも 0 以上（【ヒト】符号は送り手の型で決まる。【み】09-26）／
#      表現の的を「自分がこの歩に撃ったか」に（09-11 の形。版4 でずれていた。【み選】09-26）
#      ⚠️ 脳幹（覚醒系・調整）はまだ無い。値はそれを加味して読む
#  37  上の段から下へ送るものを、上の表現の予測 p から発火に（【ヒト】別の領野へ届くのは撃ったスパイクだけ。
#      09-11「段の出力は表現の発火」の形。上→下 だけ p のまま残っていた）
#  38  脳幹（覚醒系）を入れた（brainstem.py）。状態 起きている ／ N3 ／ REM を持って全領野へ配る
#      眠りは N3 と REM をくり返す（前は N3 だけ。REM を落とす承認は無かった）
#      眠りで上からの戻りを落とす（N3 0.18 倍 ／ REM 0.33 倍。【ヒト】Hayat 2022【み選】）
#      起きているあいだ、驚き（皮質の誤差がいつもより多い）で学ぶ速さを上げる（量はノブ【C】）
#      眠りの長さを【ヒト】乳児の比に（約10万歩、1周 約6万歩の半分が REM【み選】）
#  39  表現の的を下からの入力に戻した（版36 の「自分の発火」は、学んだシナプスが駆動に入る今の撃ち方では
#      「いつも撃つ」が誤差0の解になり撃ちっぱなしになった。的は人から決まらない【C】）／
#      傾きの細胞を【ヒト】皮質の発火率 0.3Hz に（前は 30%/歩【C】）／
#      上からの戻り3本と指令の写しは表現を撃たせない（予測 p には入る）／ 驚きの基準が 0 から始まる不具合を直した
#  39b 恒常性の目標を受け手の種類ごとの【ヒト】のふだんの率に（皮質の抑制 2Hz・視床の表現 18.8Hz。値の無い
#      視床の誤差・抑制と運動は下げない）。起きているあいだは変わらないので VERSION は 39 のまま
#  40  学びの門：その歩の学びは辺ごとの跡（時定数 1秒【ラット】）にため、門の分だけ重みを変える。
#      門 ＝ ふだんの 1 ＋ 驚きに比例する上乗せ（【ヒト】黒質 150〜375ms 後、上限【マカク】ふだんの 6.25倍）／
#      返し手（春日部つむぎ）が喃語の36%に、【ヒト】家庭録音の遅れで、まね返しか短い言葉を返す【み】／
#      驚きは学ぶ速さから外した（覚醒の側へ）
#  41  覚醒 A（青斑核のふだんの高さ ＝ 驚きの短い平均 ÷ 長い平均。【リスザル】0.20〜6.12）で
#      表現の閾値を最大 21% 下げる（【ラット】3.4mV ÷【ヒト】16mV）／ 眠るのは 眠りの圧 ≥ A のとき
#  42  驚きでその歩の学ぶ速さを上げる（青斑核、版38〜39 の形）を戻した。版40 で外したのは誤り
#  43  a 前頭前野・背内側視床・嗅内・側坐核 ／ b 上丘・下丘・中脳の核（黒質の門）／
#      d 口が運動の領野の発火を読んで、VOICEVOX（疑核と筋の代わり【み】）が声にする。
#        写しは運動の領野の表現から。蹴りは語の指令の前で終える（config.CMD_LEAD）
#  44  口の連続値の段数を聞き分けで決めた（高さ30・母音33・子音15・大きさ16。運動の領野 1,812 → 840体）／
#      喃語の高さを中心 300Hz・発話ごと 3.3半音・発話の中 4.47半音に（設計案_脳幹.md §12-63〜73）
#  45  抑制の繋がりを人の強さで固定し、学ばない【み】10-01
#  46  （戻した）抑制の体を閾値で撃たせる形。相手 1,080 が多すぎて言えなくなった
#  47  形A の直し：錐体 → 抑制の体 を人の分布で始め、予測誤差では学ばず、大きい繋がりだけ連発で弱まる／
#      抑制の体の興奮性の相手 180（【ヒト】覚醒の率から【C】。海馬・嗅内 194）【み選】B・【み】10-02
#  48  形B の直し：抑制の体は届いた興奮（上からの戻りを除く）÷ (1＋抑制) ＞ 20.9mV ぶんで撃つ。焼いた見込みを使わない
#      （【ヒト】Wilbers 2023・Lee・Dalley 2023。版46 の撃ち方。【み】10-02・10-03）
#  49  語の指令を運動前野の送り手 115本（ノブ）の発火の数で入れる（1本 ＝ 人の繋がり1本。指令 35Hz・ふだん 3.3Hz【ヒト】）／
#      口は発火を筋の形（【ヒト】Ito 2004 6.1Hz・ζ 0.69・15ms）でならした力 ≥ 0.20【C】で読む【み】10-03
#  50  Group2 の履歴 tb を行と同じ並びで足す。版49 まで先頭の領野の誤差の行のほかは別の体の発火を履歴にしていた
#      （Group2 にまとめたときのずれ【C】。1領野の Area は合っていた。【測】chk_tb 10-04）
#  51  視床の中継の細胞が自分の領野の中から受ける興奮を 約6,400本 → 64本（ノブ【C】）【み選】10-05 B「疎にして残す」
#      （【マカク】核の中の側枝は記録した中で1本 Wilson 1989 ／【ネコ】側枝の約2割が中継の細胞へ Bickford 2008）
#  52  網の作りは変えない（版51 の state.pt を読める）。1歩を縮める（Group2 の待ち合わせ A・B、撃った辺の一覧 C）【み】10-07 ／
#      脳の中の乱数を走らせるたびに OS の乱れから引き直す（種はログ。--seed N で再生）【み】10-07「脳ってそうだったか？」
#  53  刈り方を細胞の撃ち具合で（ふだん 弱い興奮→興奮 ／ 静か 抑制→興奮 ／ 撃ちすぎ 弱い興奮→抑制）。辺の寿命 14日【ヒト神経】／
#      恒常性の目標を人の起きているときの率に（皮質 3.30・抑制 4.70Hz、海馬・嗅内 3.1・5.1Hz）／ 起きているあいだは上げるだけ【ラット】・眠りは下げるだけ
#      【測】measure_v52a・v52b（刈り込みの片道が下がりの大元）【み】10-08・10-09
#  54  意味を変えずに1歩を縮める：周回の使い回し ／ 学びを触ったときにまとめる ／ 焼いた網を1歩1回（【み】10-10）
#  54b 意味を変えずに CPU の待ちを減らす（印で選ぶ → 番号で選ぶ ほか。measure_sync 10-10）
#  55  焼いた網を型ごとに焼き直したもの（皮質・視床・海馬・側坐核・上丘）に。床 0.121 → 0.003〜0.010【み】10-10
#  56  海馬の時間細胞を調整する側に（予測に入り、撃たせない）。v55b で海馬の表現の 70〜99% を自分の時間細胞が撃たせていた【み】10-10
#  57  海馬の抑制の体の興奮の相手 194 を錐体（遅れの列）から引く（194 の前提どおり）。版56 で抑制の体が止まった【み】10-10
#  58  自発のくじを周ごとのまま1周ぶんの確率に（1周 ＝ 人のシナプス1本の遅れ）。前は届く自発が約10倍【み】10-10
#  59  不応期 3周（【ヒト】Ison 2011：3ms 以内 0.10%）。履歴・時間細胞・返す発火・恒常性は「この歩のどこかで撃ったか」【み】10-10
#  60  表現の体に f-I（撃ったあと 1/f(r) 待つ。【ヒト】Barzó・Szöts 2024 0.15Hz/pA・230pA）。自発は不応期だけ【み】10-11
#  ⟲ 10-11 版60 の f-I を外して版59 に戻した（切片に電流を流した値の当てはめで、人の自然な反応の根拠が無い【み】）
#  61  視床の中継細胞どうしの興奮を外す（RELAY_EXC 64 → 0、刈り込みでも生やさない）【マカク】Wilson 1989・Carandini 2007【み】10-11
#  62  上丘の時間細胞を調整する側へ・抑制の体の相手を錐体から（海馬の版56・57 と同じ）。v59d で上丘の横の 98% が時間細胞【み】10-11
#  63  周回の中で撃った抑制の体の発火も次の周で届ける（配線は版62 のまま）【ヒト】Szegedi 2017。v62b で運動の抑制が10周とも同じ【み】10-11
#  64  運動の表現・誤差の抑制の相手を、抑制の体そのもの（1歩前の列）から 体の数 × 23% を1回ずつ【ヒト】Jiang 2012【み】10-11「Aで様子を見よう」
#  65  聴覚も版64 と同じ抑制の相手に（INH_SLICE に聴覚）【み】10-11「まぁいいよ、やってみて」
#  66  指令の写しが聴覚の抑制の体を撃たせる（配線は版65 のまま）【マーモセット】Eliades & Wang 2003【み】10-12「まぁアでいいよ」「人じゃなくて霊長類」
#  67  上行の覚醒の道（脳幹の橋の被蓋 → CL → 皮質の表現の体）。CL → 皮質 は学ぶシナプス（重み・短期の増減・三つの因子）。
#      皮質の自発のくじを外す【ヒト】設計案_自発・Frontiers Cell Neurosci 2021・Verhoog 2013【み】10-12「シナプスにしてよ」「人の形は一つ」
#  68  K_CL 78 → 40（対数正規の重さで数え直し）・K_PM 115 → 27（今の抑制と CL の上で 35Hz）。版67 は運動の表現が撃ちすぎて言えた 0/43
#      【測】measure_v67b・c・d【み】10-12「a」「イ」「いいよ」
#  69  下から上への道（皮質・皮質下の11本）を学ぶシナプスに。今の本数のまま1本 ＝ 人の1本（対数正規・短期の増減・三つの因子）。
#      gain は入口に残す（ハードの制約のノブ）【み】10-12「全体をシナプスにしよう」「1本は1本」「アでいいよ」
#  70  残りの道（海馬の中3本・下丘 → 視床・運動前野 → 運動・PPN → CL）も学ぶシナプスに。蹴りは保留。
#      海馬の短期の増減・学びは人の値なし ⇒ 皮質の値を当てた【C】【み】10-12「全部適用してずれた所を直していくほうがよくないか」
#  71  シナプスの短期の増減でずれた3つのノブを定義で出し直し：K_PM 27 → 53（35Hz）・M_PC 4 → 2.7（CL 11Hz）・
#      視床の閾値 0.3 → 0.07（中継の効き 50%【マカク】Sincich 2007・Lee 1985）【み】10-12「ウで駄目ならイ」「もっと探してみて」
#  72  時間の足し合わせ（表現 15ms・海馬 30ms・抑制 6.8ms【ヒト】、視床ほか 15ms【C】）・学びの弱まる窓（−80〜−130ms −55%【ヒト】海馬）・
#      K_PM 45【測】・視床の閾値 0.09【計算】【み】10-12「2発重なると撃つも要るんじゃないの」「一旦アでやって」
#  73  下からのシナプスの学びを人の組の窓に（皮質 Verhoog 2013・海馬 Testa-Silva 2010）。組は一番近い発火どうし【C】。
#      ⟲ 版72 の皮質の弱まる窓（海馬の値を当てた誤り）【み】10-12「どうして根幹に関わるシナプスの学びを外そうとする？」
#  74  下からのシナプスの受け手が先の窓・海馬の窓にもドーパミンの門（開き 2d − 1）【ヒト】Ziemann 2004・Louth 2021【C】当て方。
#      【み】10-13「起きてる時にどうか調べて」「一旦暴走してるとこに入れよう」
#  75  下からのシナプス17本を1つの表で回す（速さだけ。線・重さ・式は同じ。放出のくじの並びだけ変わる）
#      【測】版74 1歩 38.3ms ／ シナプスの分 8.6ms（17回/歩 CPU が待つ）【み】10-14「それならいいよ」
#  76  外すスイッチ（ANAPLEX_BU・CL・SUM の 0）と前の網の道を消した【み】10-14「外していいよ」
#      網の辺の計算（足し込み・学び・まとめて足す）を torch.compile で1つに（式・足す順は同じ）【測】版75 1歩 33.0ms
#      ⟲ GPU で起動できなかった（driver error 401）⇒ 版76b で止めた（Claude の誤り：逃げ道が GPU では役に立たない作り）
#  77  torch.compile を戻した：核の中の番号の assert を外す【測】probe_compile3 で9つとも動く ／ 学びの値は fill_ で（CPU が待たない）
#  78  周回の中の体ごとの計算（駆動・見込み・撃つか／誤差・抑制の体・不応期）・放出・時間細胞を torch.compile で（式・くじの順は同じ）
#  79  2周目から最後の周までを HIP グラフで（撃った辺は決まった大きさの箱に。周をまたぐ値は決まった置き場所。式・くじの順は同じ）
#      【測】版78 1歩 23.2ms（2周目から 1.19ms/周）ANAPLEX_GRAPH=0 で版78 と同じ周ごと
#  79b 箱の余りの辺を1か所に集めない（版79 1歩 21.5ms。余りが先頭の辺の受け手にぶつかっていた）。値は同じ
#  80  1周目と2周目の頭で1歩ぶんの辺を読む回数 5 → 3（学びの分を足す＋足し込み／学ぶ＋2周目の頭）。式・くじの順は同じ
#      【測】版79b 1歩 18.3ms ／ GPU 13.7ms（1歩ぶんの辺の足し込み 2.18ms × 2回）
#  81  1周目の足し込みを Triton の核で：調整する側でない辺は 0 を足さない・辺の数は整数で数える（値は同じ）
#      【測】版80 1歩 15.8ms ／ probe_atomic 小数の足し込み 0.67ms・整数 0.158ms（230万本）
#  82  1周目の足し込みの下書きを 64枚に分けて最後に足し合わせる（292万本が受け手 9,736 に集まり順番待ちだった）
#      【み】10-14「その仕組みならいいよ」【測】probe_scatter_real2 2.37 → 1.10ms
#  83  2周目から最後の周までを1つのグラフに（辺の数は GPU の中で読む。CPU が待つ 9 → 0回/歩）。式・くじの順は同じ
#      【み】10-14「まだ詰めれるなら速さを」【測】版82 1歩 13.4ms ／ GPU 9.4ms
#  84  歩の終わりの入口の並べ直し・buf のずらしを番号の表1つで ／ 1周目の撃った辺を累積和から（CPU の待ち 2 → 1回）。値は同じ
#      【測】版83 1歩 11.6ms ／ GPU 8.9ms
#  85  下からのシナプスの表の drive を torch.compile で（くじの前と後。くじは同じ所）【測】版84 1歩 11.2ms
#  86  焼いた網の見込みを1回に・グラフに渡す値のうち置き場所の変わらないものは写さない【測】版85 1歩 10.9ms
#  87  耳の入力はピン留めした置き場所から待たずに送る・誤差の和は GPU の上で足して最後に1回読む【測】版86 1歩 10.8ms
#  88  恒常性・恒常性の観測・下からの表の学び・CL の学びを compile で（まとめて4つ）【測】版87 1歩 10.0ms
#  89  恒常性の観測で誤差と表現の行が入れ違っていた（版59 から）のを直す【み】10-15「バグは直して」⚠️ 値が変わる
#  90  1周目から最後の周までを1つのグラフに（1周目の辺は Triton の核。くじ・放出・歩の終わりは外）【み】10-15「稼げるならやっておいて」
#  91  刈り込みの受け手ごとの一番小さい辺を並べ替えなしで（選ぶ辺は同じ。GPU の山 14.9GB → 下げる）【み】10-15「予算12gb」
#  92  組む途中と刈り込みの並べ直しで、番号を 32bit・写しを1本ずつ（値は同じ）【み】10-15「いいよ」【測】版91 山 10.16GB
#  93  中脳の核を撃ち方 B・周ごと・学ぶ形に（各 256体）・PPN は周ごと・領野の撃った周のまま渡す・来なかった誤差は渡さない【み】10-18
#  94d CL を素子に（網の領野、型「視床」・閾値 M_PC_SYN のまま）【み】10-20「イ」
#  102 12GB に収める：縮める側（連合・高次・前頭前野・眼窩前頭・前帯状）× 0.85（係数 0.619/100万。measure_size101 で 11.84GB）
#  103g 皮質の時間細胞・傾きの細胞も調整する側に（海馬・上丘と同じ形。ANAPLEX_CX_TC=drive で 103f）
#  103f 速さ：抑制の体の的（領野の入口の平均）を密な行列の積から累積和の差に（式は同じ・丸めだけ・VRAM も減る）
#  103e 速さ：受け手の順に並んだ線（平行線維など）の足し込みを累積和の差で（版95b と同じ手。網は同じ・丸めだけ）
#  109 皮質の型の領野を全部、版64・65 の切片の見方の抑制に（抑制の体そのもの × 23% を1回ずつ【ヒト】Jiang 2012）【み】11-11「むしろ切片になってないのがいけない」
#      （版108 は 11-10 に外した。番号は飛ばす）
#  103d 速さ：放出のくじ（二項分布）を同じ分布の別の引き方に（GPU のとき。出目は変わる。網は同じ）
#  103c 速さ（意味を変えない）：下書き 64 → 16 枚・焼いた素子の見込みを Triton の核1つで・下からの表の数えを searchsorted で（網は同じ・state.pt は読める）
#  103b 12GB に収める：縮める側（連合・高次・前頭前野・眼窩前頭・前帯状・統語の 6つ）× 0.55（係数 0.340/100万。measure_size103 で 11.29GB）
#  103 統語の部位（44・45・STS1・STS2・中側頭後・TPJ・視床枕、組み表_版103）・44 と 45 に来ない体・皮質 → 線条体を人の比に（Tziortzi 2014）
#  101 来ないことに撃つ体（誤差の体と別の組・皮質の型・撃つ ＝ (p − 的) ＞ THETA。聴覚・連合・高次・視覚・前頭前野・眼窩前頭・前帯状。上へは誤差の体の線に半々・前頭前野・眼窩前頭・前帯状 → 視床下核・驚きの和にも）【み】10-29 12:35・22:52・10-30 0:36
#  98  小脳の切片（橋核・顆粒＋ゴルジ・分子層・プルキンエ・小脳核2型・下オリーブ・赤核・VLp）・平行線維の登上線維の学び（組み表_版98）【み】10-25「それで組んで」
#  97  続けて流す暮らし（life.py）：口は毎歩 運動の領野を読む・時は止まらない・返し手と録音は口が出したどの声にも。網は版96 と同じ【み】10-24
#  96  補足・前補足・VA・VLa を足す・蹴りを消して合図を補足へ・CM 19・Pf 4・CL 5（設計案_版96の組み表）【み】10-24
#  95d 速さ：下からの表は撃った送り手の線だけ・NucNet を並べた形で・核 → 領野の抑制を表で1回【み】10-22 22:12
#  95c CL 4,096 → 64（人の決め 20/100万・下限 64）【み】10-22
#  95b 足し込みの渋滞を直す（下からのシナプスの表・NucNet を受け手の順に並べて累積和の差で。意味は同じ）
#  95  基底核（線条体の型・外節・内節・網様部・腹側淡蒼球）・CM/Pf・PPN を素子に・核どうしを周の遅れに（NucNet）【み】10-21 23:37
#  94c 前のスレッドの核の値を戻す（64本・69.3%・側坐核の表現＋誤差・τm 40ms・i0・NMDA の尾）【み】10-20「ア」
#  94b 核ごとの素子の型・視床下核の抑制の体 7.5%【ヒト】・人から外れた列で焼かない【み】10-20
#  94  中脳の5核を素子に（網の領野、型「中脳」）。入る線は下からのシナプス・抑制の線は学ばない【み】10-19「まずニューロンを素子に直して」
#      【測】版77 1歩 27.1ms ／ コンテナで数えた命令 1歩 2,321（周回の残り 924）【み】10-14「実時間を下回る必要があるね」
N_MEL, N_SYL, LATENCY = 40, 8, 5
DEV = "cuda"
N_EAR = 40 * 10        # 【ヒト】ERB 40個 × 1チャネル10本の聴神経


# ── 作り物の声（VOICEVOX が繋がるまでの代役）──
def make_inventory(rng):
    inv = []
    for _ in range(N_SYL):
        ch = np.arange(N_MEL)
        cons = np.clip(rng.normal(0.5, 0.15, N_MEL), 0, 1) * (ch > rng.integers(15, 30)) * 0.7
        f1, f2 = rng.integers(3, 15), rng.integers(15, 35)
        vow = np.clip(np.exp(-0.5 * ((ch - f1) / 2.0) ** 2)
                      + 0.8 * np.exp(-0.5 * ((ch - f2) / 2.5) ** 2), 0, 1)
        inv.append((cons, vow))
    return inv


def utter(inv, rng, n_cmd, syl=None):
    """1回の発話 → 耳に届くメル [歩, 40] と、指令の写し [歩, n_cmd]
    指令には子音の始まり・母音の始まり・強さの段を乗せる（声の揺れが写しに乗る）
    syl を渡すと、その音節の並びで言う（長さや強さの揺れは毎回ちがう）"""
    syl = rng.integers(0, N_SYL, rng.integers(0, 5)) if syl is None else np.asarray(syl)
    segs, ev, t = [], [], 0
    for s in syl:
        cons, vow = inv[s]
        nc, nv = max(2, int(rng.normal(7, 3))), max(4, int(rng.normal(16, 5)))
        vol = max(0.3, rng.normal(1.0, 0.15))
        ev += [(t, s), (t + nc, N_SYL + s), (t, 2 * N_SYL + int(np.clip(vol * 2, 0, 3)))]
        segs += [cons * vol] * nc + [vow * vol] * nv
        t += nc + nv
    lead = C.GO_LEAD if len(syl) else LATENCY      # 準備（声を出すときだけ）
    n = t + lead + 20
    cmd = np.zeros((n, n_cmd), np.float32)
    for tt, k in ev:
        cmd[tt + lead - LATENCY, k % n_cmd] = 1
    mel = np.clip(rng.normal(0.05, 0.03, (n, N_MEL)), 0, 1)
    if segs:
        mel[lead:lead + len(segs)] = np.clip(np.array(segs) + mel[lead:lead + len(segs)], 0, 1)
    return mel.astype(np.float32), cmd, list(map(int, syl))


N_FIBER = 10                 # voice.py と同じ（【ヒト】1つの内有毛細胞に聴神経 10〜20本）
# ⚠️ 閾値は作り物の声の目盛り（0〜1）に合わせた代役の値。
#    voice.py の閾値（-11〜-2）は対数メルの目盛りなので、そのままは使えない。
#    形（繊維ごとに閾値が違う・二峰性・越えているあいだ撃つ）だけ揃えてある
_FAKE_TH = np.concatenate([np.linspace(0.15, 0.45, N_FIBER - 4),
                           np.linspace(0.45, 0.80, 4)])


def encode(mel):
    """メル [歩, 40] → 耳の発火 [歩, 40 × N_FIBER]。voice.Voice.encode と同じ形

    ⚠️ 版10 で耳を 40 → 400 に太くしたとき、この作り物の声の側だけ 40 のまま残っていた。
       疎にするのは皮質の上の段（【ヒト】上側頭回）で、耳ではない
    """
    out = np.zeros((len(mel), mel.shape[1] * N_FIBER), np.float32)
    for i, th in enumerate(_FAKE_TH):
        out[:, i::N_FIBER] = (mel > th).astype(np.float32)
    return out


# ── 保存と再開 ──
def save(b, done, path=STATE, log=None):
    g = b.g
    g.flush_all()                          # 版54：学びの分を全部の辺へ足してから書く
    d = {"done": done, "version": VERSION, "log": log or [],
         "w": g.w, "e_recv": g.e_recv, "e_src": g.e_src, "e_inh": g.e_inh, "start": g.start,
         "scale": g.scale, "drive": g.drive, "recv_ema": g.recv_ema, "rate_ema": g.rate_ema, "rate_long": g.rate_long, "scale_up": g.scale_up, "tb": g.tb,
         "bufs": g.bufs, "age": g.ch.age, "t": g.t,
         "up": b.up, "om": b.om, "pred": b.pred, "ca3_prev": b.ca3_prev, "bs": b.bs.state_dict()}   # 版101：om
    if g.stp is not None:
        d["stp"] = {k: getattr(g.stp, k) for k in ("row", "state", "fac", "last", "deplete")}
    if getattr(b, "bu_on", False):     # 版69：下からのシナプス
        b.bu_unpack()                   # 版75：表から道へ写してから保存
        d["bu"] = {k: {"w": s.w, "t": s.t, "stp": {kk: getattr(s.stp, kk) for kk in ("row", "state", "fac", "last", "deplete")}}
                   for k, s in b.bu.items()}
    if getattr(b.g, "nucnet", None) is not None:   # 版95：核どうしの線
        d["nucnet"] = b.g.nucnet.state_dict()
    if getattr(b, "cl_on", False):     # 版67：CL → 皮質 のシナプス
        d["cl"] = {"w": b.cl_w, "src": b.cl_src, "t": b.cl_t,
                   "stp": {k: getattr(b.cl_stp, k) for k in ("row", "state", "fac", "last", "deplete")}}
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)          # 書きかけで壊さない


# 版52：脳の中の乱数を引き直す
#   【み】10-07「とても致命的に見える　君たちでさえ毎回回答が変わるのに」
#   配線は組み立てのときの seed=0 のまま（保存した重みと合わせる）。走らせるたびに、脳の中の雑音の並びを変える
#   ⟲ 版51 まで、走らせ直すたびに脳の中の雑音が毎回同じ並びで始まっていた
_OURS = ("torch_brain", "torch_group2", "torch_area", "midbrain", "brainstem", "sleep")


def _generators(b):
    """脳が持っている torch の乱数（重なりなし。見つけた順は毎回同じ）"""
    gens, seen, todo = [], set(), [b]
    while todo:
        o = todo.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Generator):
            gens.append(o)
        elif isinstance(o, (list, tuple)):
            todo += list(o)
        elif isinstance(o, dict):
            todo += list(o.values())
        elif type(o).__module__ in _OURS:
            todo += list(vars(o).values())
    return gens


def reseed(b, argv=None):
    """--seed N があればその種（同じ1日の再生。GPU の足し算の順も決める）。無ければ OS の乱れから。返り値は種"""
    argv = sys.argv if argv is None else argv
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1])
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        seed = int.from_bytes(os.urandom(4), "little")
    gens = _generators(b)
    for k, g in enumerate(gens):
        g.manual_seed(seed + 7919 * k)
    print(f"脳の中の乱数の種 {seed}（乱数 {len(gens)} 個）{'  再生：足し算の順を決めた' if '--seed' in argv else ''}", flush=True)
    return seed


def load(b, path=STATE):
    """返り値: (これまでの回数, 履歴)。版が違う state は読まない"""
    if not os.path.exists(path):
        return 0, []
    d = torch.load(path, map_location=DEV, weights_only=False)
    v = int(d.get("version", 0))
    if v != VERSION and v not in SAME_NET:
        raise SystemExit(
            f"⚠️ {path} は版 {v} で育てたもの。いまのコードは版 {VERSION}。\n"
            f"   網の作りが変わっているので、そのまま続けると歪んだ重みが残る。\n"
            f"   --fresh で0から育て直すか、古いコードで読むこと。")
    g = b.g
    # 版が同じでも体数が違えば、黙って歪んだ重みが入る。形も見る
    for k, now in (("w", len(g.w)), ("scale", len(g.scale)), ("tb", len(g.tb))):
        if len(d[k]) != now:
            raise SystemExit(
                f"⚠️ {path} は {k} が {len(d[k]):,} 本／体。いまの網は {now:,}。\n"
                f"   体数の違う網には読み込めない。--fresh で0から育て直すこと。")
    for k in ("w", "e_recv", "e_src", "e_inh", "start", "scale", "drive", "recv_ema", "tb"):
        setattr(g, k, d[k])
    if "rate_ema" in d:                    # 版20 から（それより前の state には無い）
        g.rate_ema = d["rate_ema"]
    if "rate_long" in d:                   # 版53 から（それより前は目標から始める）
        g.rate_long, g.scale_up = d["rate_long"], d["scale_up"]
    g.bufs, g.ch.age, g.t = d["bufs"], d["age"], d["t"]
    g._prior_l, g._amp_buf, g._amp_prev = None, None, None   # 版54：読み込んだ tb・辺で作り直す
    b.up, b.pred, b.ca3_prev = d["up"], d["pred"], d["ca3_prev"]
    if "om" in d:                                   # 版101
        b.om = d["om"]
    if "bs" in d:                          # 版38 から
        b.bs.load_state_dict(d["bs"])
    if g.stp is not None and "stp" in d:
        for k, vv in d["stp"].items():
            setattr(g.stp, k, vv)
    if getattr(b, "bu_on", False) and "bu" in d:   # 版69（版68 までの state には無い ⇒ 初めの重さ）
        for k, s in b.bu.items():
            v = d["bu"].get(k)
            if v is not None and len(v["w"]) == len(s.w):
                s.w, s.t = v["w"], v["t"]
                for kk, vv in v["stp"].items():
                    setattr(s.stp, kk, vv)
        b.bu_pack()                     # 版75：読み込んだ道から表を作り直す
    if getattr(b, "cl_on", False) and "cl" in d and len(d["cl"]["w"]) == len(b.cl_w):   # 版67（版66 の state には無い ⇒ 初めの重さ）
        b.cl_w, b.cl_src, b.cl_t = d["cl"]["w"], d["cl"]["src"], d["cl"]["t"]
        for k, vv in d["cl"]["stp"].items():
            setattr(b.cl_stp, k, vv)
    if getattr(b.g, "nucnet", None) is not None and "nucnet" in d:   # 版95
        b.g.nucnet.load_state_dict(d["nucnet"])
    return int(d["done"]), list(d.get("log", []))


# ── 回す・測る ──
def run(b, cmd, ear, own, learn):
    n = 0.0
    zero = torch.zeros(b.n["運動"], device=DEV)
    b.new_utterance(own and bool(np.any(cmd)))
    for t in range(len(ear)):
        c = to_dev(cmd[t], DEV) if own else zero            # 版87：待たずに送る
        _, e, _, _ = b.step(to_dev(ear[t], DEV), c, learn=learn)
        n = n + e.sum().double()                              # 版87：GPU の上で float64 に足す（CPU の float で足すのと同じ値）
    return float(n)


def own_voice(b, vo, mo, rng, learn):
    """自分で言う（版43d）。返り値 (耳の発火, 運動の領野に入れた指令, 誤差の和, 選んだモーラ, 言えたモーラ)
    合図 → 蹴り → 語の指令（補助輪：何を言うかは乱数が選ぶ）→ 口が運動の領野の発火を読む
    → VOICEVOX が声にする → 自分の耳で聞く
    口の読み方：指令の窓（CMD_LEAD 歩）のうち半分以上の歩で撃った体を「撃った」とみる
      ⚠️【C】筋が運動ニューロンの発火をならして動く、の代わり。窓の長さでならすのは人の値ではない"""
    morae, vol = vo.random_morae(rng)
    if not morae:                                       # 黙る（合図も蹴りも無し）
        ear = vo.encode(vo.say(morae, vol))
        cmd = np.zeros((len(ear), mo.n_cmd), np.float32)
        return ear, cmd, run(b, cmd, ear, True, learn), [], []
    one_np = mo.encode(morae, vol)[0]
    one = torch.tensor(one_np, device=DEV)
    zero = torch.zeros_like(one)
    quiet_np = vo.encode(np.full((1, N_MEL), vo.floor, np.float32))[0]
    quiet = torch.tensor(quiet_np, device=DEV)
    w0 = C.GO_LEAD - C.CMD_LEAD
    b.new_utterance(True)
    n, reps = 0.0, []
    for t in range(C.GO_LEAD):                          # 準備（声はまだ出ない）
        _, e, _, _ = b.step(quiet, one if t >= w0 else zero, learn=learn)
        n = n + e.sum().double()                              # 版87
        if t >= w0:
            reps.append(b.mouth_in.clone())                # 版100：口が読むのは運動の領野 ＋ PAG
    # 版49：口は発火を筋の形でならした力で読む（【ヒト】Ito 2004）。⟲ 版48 まで「窓の半分以上の歩」【C】
    import mouth as MO
    force = MO.muscle_force(torch.stack(reps).cpu().numpy())
    said, v = mo.decode((force >= MO.READ_TH).astype(np.float32)[None])
    snd = vo.encode(vo.sound(said, v))                  # 言えたものが last_morae に入る（返し手はそれを返す）
    for t in range(len(snd)):                           # 声のあいだも指令は運動の領野に入り続ける
        _, e, _, _ = b.step(to_dev(snd[t], DEV), one, learn=learn)     # 版87
        n = n + e.sum().double()
    ear = np.concatenate([np.repeat(quiet_np[None], C.GO_LEAD, 0), snd])
    cmd = np.zeros((len(ear), mo.n_cmd), np.float32)
    cmd[w0:] = one_np
    return ear, cmd, float(n), [m for m, *_ in morae], [m for m, *_ in said]


def voices(b, fake):
    """声の用意。返り値 (乱数を進めるだけの関数, 言う関数 say(乱数, learn), 返し手)
    say の返り値は own_voice と同じ形"""
    if fake:
        # ⚠️ 作り物の声には口が無い（指令がそのまま音になる）。版43d の「口が運動の領野を読む」は通らない
        inv = make_inventory(np.random.default_rng(0))

        def say(r, learn):
            mel, cmd, syl = utter(inv, r, b.n["運動"])
            ear = encode(mel)
            return ear, cmd, run(b, cmd, ear, True, learn), syl, syl
        return (lambda r: utter(inv, r, b.n["運動"])), say, None   # 返し手（版40）は作り物の声では返さない
    from voice import Voice, Replier
    from mouth import Mouth
    vo, mo = Voice(), Mouth()
    return vo.random_morae, (lambda r, learn: own_voice(b, vo, mo, r, learn)), Replier(vo)


def life_of(b, fake, rng, rng_r):
    """版97：続けて流す暮らし（life.Life）を作る。fake なら作り物の声（VOICEVOX を使わない。口は本物の mouth.Mouth）"""
    import life as L
    from mouth import Mouth
    if fake:
        vo = L.FakeVoice()
        return L.Life(b, vo, Mouth(), L.FakeReplier(vo), rng, rng_r, DEV)
    from voice import Voice, Replier
    vo = Voice()
    return L.Life(b, vo, Mouth(), Replier(vo), rng, rng_r, DEV)


def reply(b, rp, rng):
    """返し手が返すなら、それを聞く（写しなし。学ぶ。版40）"""
    mel = rp.answer(rng)
    if mel is not None:
        run(b, None, rp.encode(mel), False, True)


def measure(b, say, n_test=20):
    """自分で言う（写しあり）と 録音で聞く（写しなし）で、聴覚の誤差を比べる
    版43d：録音は、自分で言ったときに耳に届いたものをそのまま流す（口が読み違えた声も含む）
    返り値に「選んだとおりに言えた回数 ／ 声を出そうとした回数」を足した"""
    own = rec = sp = 0.0
    ok = tried = 0
    for i in range(n_test):
        ear, cmd, n, want, got = say(np.random.default_rng(10000 + i), False)
        if want:
            tried += 1; ok += int(want == got)
        if ear.sum() == 0:
            continue
        own += n
        rec += run(b, cmd, ear, False, False)
        sp += float(ear.sum())
    return own / max(sp, 1), rec / max(sp, 1), rec / max(own, 1), ok, tried


def main():
    n_add = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    fresh = "--fresh" in sys.argv
    fake = "--fake" in sys.argv          # 作り物の声（VOICEVOX を使わない）
    prior = T.Prior.load("prior_v1.npz", dev=DEV)
    b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=N_EAR, dev=DEV)
    done, log = (0, []) if fresh else load(b)
    seed = reseed(b)                            # 版52
    print(f"体 {b.n_cells:,}  辺 {len(b.g.w):,}  これまで喃語 {done} 回（版 {VERSION}）", flush=True)
    for e in log[-5:]:
        print(f"  これまで: {e}", flush=True)
    b.bs.set_state(BS.WAKE)     # 起きている（版38：恒常性・上からの戻りは脳幹が決める。day.py と同じ）
    #   ⟲ 09-26 まで babble.py だけ止めていなかった（版30 までの babble の結果は、起きているあいだも下げていた）
    rng = np.random.default_rng(0)
    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（喃語の列とは別）
    life = life_of(b, fake, rng, rng_r)          # 版97：時間を続けて流す（⟲ 版96 まで 言う → 返事 の台本）
    for _ in range(done):                       # 声の乱数を進めて、同じ列を繰り返さない
        life.vo.random_morae(rng)
    t0 = time.time()
    for i in range(n_add):                      # 1回 ＝ 番1つ（補助輪がモーラを選ぶ。2割は黙る）。時間は流れ続け、自発の声も出うる
        life.run_plans(1, learn=True)
        done += 1
        if (i + 1) % 20 == 0:
            save(b, done, log=log)
            el = time.time() - t0
            st = life.stats
            print(f"  番 {done} 回（合図 {st['plans']}）声 {st['voices']}（自発 {st['spont']}）返し手 {st['replies']}  門 {b.bs.gate_mean():.3f}  "
                  f"覚醒 {float(b.bs.arousal):.2f}  経過 {el/60:.1f}分  残り見込み {el/(i+1)*(n_add-i-1)/60:.1f}分", flush=True)
    h = life.health()
    s = h["supp"]
    print(f"番 {done} 回  聴覚の誤差（1歩あたり）自分 {h['自分']:.2f} ／ 録音 {h['録音']:.2f} ／ 返し手 {h['返し手']:.2f}  supp {s:.2f}"
          f"（返し手 {h['supp_rep']:.2f}）  言えた {h['言えた']}/{h['試した']}  自発 {h['自発']}/{h['声']}", flush=True)
    # VRAM は2つある。rocm-smi が見ているのは「確保」の方
    #   使った山 ＝ こちらが実際に要る量
    #   確保     ＝ PyTorch が抱えたまま返さない量（断片化で膨らむ。足りなくなるまで返さない）
    GB = 2 ** 30
    print(f"VRAM  使った山 {torch.cuda.max_memory_allocated()/GB:.2f} GB"
          f" ／ 確保 {torch.cuda.memory_reserved()/GB:.2f} GB", flush=True)
    if n_add:
        log.append(f"{datetime.now():%m-%d %H:%M} 喃語 {done}回（+{n_add}）種 {seed}"
                   f" 相手{b.g.areas[0].n_partner} 周回{b.g.recur}"
                   f" 声={'作り物' if fake else 'VOICEVOX'}  supp {s:.2f}")
        save(b, done, log=log)


if __name__ == "__main__":
    main()
