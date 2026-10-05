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
from torch_brain import Brain, NAMES

STATE = "state.pt"
VERSION = 51          # コードの版。網の作りを変えたら上げる
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
    d = {"done": done, "version": VERSION, "log": log or [],
         "w": g.w, "e_recv": g.e_recv, "e_src": g.e_src, "e_inh": g.e_inh, "start": g.start,
         "scale": g.scale, "drive": g.drive, "recv_ema": g.recv_ema, "rate_ema": g.rate_ema, "tb": g.tb,
         "bufs": g.bufs, "age": g.ch.age, "t": g.t,
         "up": b.up, "pred": b.pred, "ca3_prev": b.ca3_prev, "bs": b.bs.state_dict()}
    if g.stp is not None:
        d["stp"] = {k: getattr(g.stp, k) for k in ("row", "state", "fac", "last", "deplete")}
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)          # 書きかけで壊さない


def load(b, path=STATE):
    """返り値: (これまでの回数, 履歴)。版が違う state は読まない"""
    if not os.path.exists(path):
        return 0, []
    d = torch.load(path, map_location=DEV, weights_only=False)
    v = int(d.get("version", 0))
    if v != VERSION:
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
    g.bufs, g.ch.age, g.t = d["bufs"], d["age"], d["t"]
    b.up, b.pred, b.ca3_prev = d["up"], d["pred"], d["ca3_prev"]
    if "bs" in d:                          # 版38 から
        b.bs.load_state_dict(d["bs"])
    if g.stp is not None and "stp" in d:
        for k, vv in d["stp"].items():
            setattr(g.stp, k, vv)
    return int(d["done"]), list(d.get("log", []))


# ── 回す・測る ──
def run(b, cmd, ear, own, learn):
    n = 0.0
    zero = torch.zeros(b.n["運動"], device=DEV)
    b.new_utterance(own and bool(np.any(cmd)))
    for t in range(len(ear)):
        c = torch.tensor(cmd[t], device=DEV) if own else zero
        _, e, _, _ = b.step(torch.tensor(ear[t], device=DEV), c, learn=learn)
        n += float(e.sum())
    return n


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
        n += float(e.sum())
        if t >= w0:
            reps.append(b.motor_rep.clone())
    # 版49：口は発火を筋の形でならした力で読む（【ヒト】Ito 2004）。⟲ 版48 まで「窓の半分以上の歩」【C】
    import mouth as MO
    force = MO.muscle_force(torch.stack(reps).cpu().numpy())
    said, v = mo.decode((force >= MO.READ_TH).astype(np.float32)[None])
    snd = vo.encode(vo.sound(said, v))                  # 言えたものが last_morae に入る（返し手はそれを返す）
    for t in range(len(snd)):                           # 声のあいだも指令は運動の領野に入り続ける
        _, e, _, _ = b.step(torch.tensor(snd[t], device=DEV), one, learn=learn)
        n += float(e.sum())
    ear = np.concatenate([np.repeat(quiet_np[None], C.GO_LEAD, 0), snd])
    cmd = np.zeros((len(ear), mo.n_cmd), np.float32)
    cmd[w0:] = one_np
    return ear, cmd, n, [m for m, *_ in morae], [m for m, *_ in said]


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
    print(f"体 {b.n_cells:,}  辺 {len(b.g.w):,}  これまで喃語 {done} 回（版 {VERSION}）", flush=True)
    for e in log[-5:]:
        print(f"  これまで: {e}", flush=True)
    b.bs.set_state(BS.WAKE)     # 起きている（版38：恒常性・上からの戻りは脳幹が決める。day.py と同じ）
    #   ⟲ 09-26 まで babble.py だけ止めていなかった（版30 までの babble の結果は、起きているあいだも下げていた）
    rng = np.random.default_rng(0)
    draw, say, rp = voices(b, fake)
    for _ in range(done):                       # 声の乱数を進めて、同じ列を繰り返さない
        draw(rng)
    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（喃語の列とは別）
    t0 = time.time()
    for i in range(n_add):
        say(rng, True)
        if rp is not None:
            reply(b, rp, rng_r)
        done += 1
        if (i + 1) % 20 == 0:
            save(b, done, log=log)
            el = time.time() - t0
            print(f"  喃語 {done} 回  門 {b.bs.gate_mean():.3f}  覚醒 {float(b.bs.arousal):.2f}  経過 {el/60:.1f}分  残り見込み "
                  f"{el/(i+1)*(n_add-i-1)/60:.1f}分", flush=True)
    o, r, s, ok, tried = measure(b, say)
    print(f"喃語 {done} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  選んだとおりに言えた {ok}/{tried}", flush=True)
    # VRAM は2つある。rocm-smi が見ているのは「確保」の方
    #   使った山 ＝ こちらが実際に要る量
    #   確保     ＝ PyTorch が抱えたまま返さない量（断片化で膨らむ。足りなくなるまで返さない）
    GB = 2 ** 30
    print(f"VRAM  使った山 {torch.cuda.max_memory_allocated()/GB:.2f} GB"
          f" ／ 確保 {torch.cuda.memory_reserved()/GB:.2f} GB", flush=True)
    if n_add:
        log.append(f"{datetime.now():%m-%d %H:%M} 喃語 {done}回（+{n_add}）"
                   f" 相手{b.g.areas[0].n_partner} 周回{b.g.recur}"
                   f" 声={'作り物' if fake else 'VOICEVOX'}  supp {s:.2f}")
        save(b, done, log=log)


if __name__ == "__main__":
    main()
