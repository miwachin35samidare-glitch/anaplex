"""torch 版の網（arkhe 用）—— 段・海馬・視床をまとめて1つの Group2 で進める

並べ方（並べ方は脳を模す。すべて【ヒト】の実測から）
  一次   聴覚（耳を受ける）／ 運動（口へ出す）／ 視覚（まぶた閉で場所取り）
  連合   聴覚の上に1段（時定数 ×1.9）
  高次   聴覚と運動をまたぐ（時定数 ×3.2）
  海馬   歯状回 → CA3 → CA1（比 15 : 2.7 : 16）
  視床   感覚を中継し、運動を蹴り込む（駆動する入力）
  版43a（09-29【み】）前頭前野・背内側視床・嗅内皮質（浅い層／深い層）・側坐核を足した。大きさは 設計案_脳幹.md §12-13
    海馬と報酬の輪は人の片側 100万体あたり 20 体（下限）／ 連合・高次・前頭前野は人の比で 12GB に収まるよう縮めた
    線は §12-10・§12-13（種は線ごと）

繋ぎ方
  下 → 上   誤差の発火（予測符号化で上へ送るのは誤差）
  上 → 下   表現のニューロンの発火（調整する入力。樹状突起へ。撃たせない）（版37。前は予測 p）
  横        運動 → 聴覚 の指令の写し（随伴放電）
  視床 → 皮質  駆動する入力（細胞体へ。撃たせる）
  海馬 → 皮質  CA1 の出力を、次の歩に連合の樹状突起へ（調整する入力）
            【ヒト】睡眠では 皮質の徐波と紡錘波が合図し、海馬のリップルが内容を返す
  領野をまたぐのは1歩に1回。すべての領野が「1歩前の他の領野」を見て同時に進む

⚠️ 領野は全部ひとつの Group2 に入れて、1回で進める（呼ぶ回数の値段を避ける）
"""
import numpy as np
import torch
from torch_area import Prior, Area, steps, THETA
from torch_area import INH_N_EXC, INH_N_EXC_HIPPO   # 版47
from torch_group2 import Group2
import brainstem as BS

# 段ごとの時間の比（各細胞の「見込み」の元になる、入力の履歴の痕跡の時定数に掛かる）
#   ＝「どれだけ前までの入力が今の反応を変えるか」＝ 積分の窓
#   【ヒト】Norman-Haignere 2025（Nat Neurosci、頭蓋内電極 15人・108電極）：一次の中心（TE1.1）からの距離で
#     0〜10mm 80ms ／ 10〜20mm 130ms ／ 20〜30mm 272ms（中央値）⇒ 1.0 ／ 1.63 ／ 3.4
#   ⚠️ 距離の輪と Te1／Te2／Te3 の対応は近似（論文は解剖の区分 HG・PT・STG でも同様と書く）
#   ⟲ 前は【マカク】MT の自発活動の自己相関の比 1.0／1.9／3.2（量が違う）
TAU = {"一次": 1.0, "連合": 130.0 / 80.0, "高次": 272.0 / 80.0}
# 段ごとの体数の比 1 : 1.33 : 1.45（【ヒト】体積 × 密度）
#   体積は Julich-Brain の最大確率地図（死後脳10体、1mm等方、左右込み、mm³）
#     一次 Te1（＝41野・コニオ皮質・視床から受ける）  Te1.0 1,822 ＋ Te1.1 2,687 ＋ Te1.2 1,408 ＝ 5,917
#     連合 Te2（＝42野・平面側頭）                    Te2.1 2,604 ＋ Te2.2 7,078 ＝ 9,682
#     高次 Te3（＝22野・上側頭回の外側）              10,591
#   密度は【ヒト】A1 42,069個/mm³ ／ Tpt 34,147個/mm³（対照19名、ステレオロジー）
#     Tpt は Te2.1・Te3 に当たる（Zachlod ほか 2020 表1）
#   ⇒ 一次 400（耳で決まる）なら 連合 531 ／ 高次 581
NAMES = ("視床", "聴覚", "運動", "視覚", "連合", "高次", "歯状回", "CA3", "CA1",
         "前頭前野", "背内側視床", "嗅内浅", "嗅内深", "側坐核",   # 版43a：後ろの5つを足した（前の9つの並びは変えない）
         "上丘")                                                    # 版43b
# 版43a の大きさ（設計案_脳幹.md §12-13）
#   人の片側 100万体あたり 20 体（海馬の比。下限として置く【み】09-29「下限は置いて」）
#     嗅内皮質【ヒト】Wegiel 2014 表3：層 II（島）102万・III 607万 ⇒ 浅 20＋121 ／ V 201万・VI 381万 ⇒ 深 40＋76
#     側坐核【ヒト】Wegiel 2014：729万 ⇒ 146 ／ 背内側視床【ヒト】Abitz 2007：643万 ⇒ 129
#   人の両半球 100万体あたり 0.728 体（比で縮める側。12GB に収める係数【C】。組んだら measure_size で直す）
#     連合 Te2 3.30億 ⇒ 240 ／ 高次 Te3 3.62億 ⇒ 263 ／ 前頭前野【ヒト】Gabi 2016 約13億 ⇒ 946
EC_II = 20              # 嗅内浅 の頭 20 体が層 II（→ 歯状回・CA3）。残り 121 体が層 III（→ CA1）
N_NEW = {"前頭前野": 946, "背内側視床": 129, "嗅内浅": 141, "嗅内深": 116, "側坐核": 146}
# 版43b（設計案_脳幹.md §12-26〜§12-43）
#   上丘の深層：人・霊長類の体数なし ⇒ 動く最低数【み】09-29「動く最低数でやって」
#   前頭前野の「来なかった誤差」の細胞：【マカク】眼窩前頭の 3.6〜7.1%（Thorpe 1983）／【ヒト】外側眼窩前頭
#     ⇒ 前頭前野の頭 N_OMIT 体に置く（動く最低数）。予測したのに下からの入力が無い歩に撃つ（誤差の向きが逆）
#   黒質・視床下核の辺縁・境の細胞・外側手綱核・吻内側被蓋核は midbrain.py（学ばない小さな核）
N_SC = 64
N_OMIT = 64
N_MB = {"黒質": 56, "視床下核": 64, "境": 64, "外側手綱核": 64, "吻内側被蓋核": 64}
N_NEW["上丘"] = N_SC
# 新しい領野の細胞体への投射：受け手1体が送り手から K_NEW 本を拾い、3本そろって閾値を越える
#   ⚠️ ノブ【C】。高次（【ヒト】上側頭回の峰 約8本・3本で越える）の置き方をそのまま使う。人の本数は未取得
K_NEW = 8
X_NEW = 16.0 / 1.12 / 2.0
HIPPO_RATIO = {"歯状回": 15.0, "CA3": 2.7, "CA1": 16.0}     # 【ヒト】片側の体数の比
SPARSE = 0.025          # 【ヒト】海馬で1つの項目に反応する割合（2.5%未満）
# 撃つ閾値。段で違う（小さいほど撃ちやすい）。視床・一次は密、連合から疎
# 撃つ閾値（人の繋がり何本ぶんか）と、模型の入力1本の重さ（人の繋がり何本ぶんか）
#   【ヒト】L2/3 錐体細胞：静止から閾値まで約 16mV、繋がり1本で 1.12mV（Hunt 2023、MTG 切除切片 32組）
#     ⇒ 同時に約 14本の繋がりで撃つ（抑制なし・静止から）。皮質の段は人の1/1 でこの値
#   X_GAIN は「模型の1本が人の繋がり何本ぶんか」。人の1/1 は 1。本数が入り切らない段だけ、動くためのノブ
#     連合   1     窓に 50〜360本 来るので人の1/1 で動く
#     聴覚  28.6   ⚠️ ノブ。視床から1本しか来ない。1本で閾値の2倍（前の 0.5 と同じ比）
#     高次   7.15  ⚠️ ノブ。峰 約8本。3本で越える（前の 2.0 と同じ）
#     運動  28.6   ⚠️ ノブ。蹴り1本で越える
#       ⟲ 前は閾値 1.0・蹴り 1・重さ 1 で「ちょうど並ぶだけで越えない」。運動が一度も撃たなかった理由
#   視床・海馬は人の閾値を引いていない（⚠️ 前の値のまま）
HUMAN_TH = 16.0 / 1.12
FIRE_TH = {"視床": 0.3, "聴覚": HUMAN_TH, "運動": HUMAN_TH, "視覚": HUMAN_TH,
           "連合": HUMAN_TH, "高次": HUMAN_TH}
# 【ヒト】CA3 錐体細胞（Watson ほか 2025, Cell 188、切除切片、硬化なし）
#   静止 −66.5mV（187体）・閾値 −43.6mV（184体）⇒ 22.9mV。CA3→CA3 の1本 0.80mV（9組）
#   ⇒ 同時に約 29本で撃つ（抑制なし・静止から）。人の皮質 L2/3 の約 14本より多く、長く積分して撃つ
HUMAN_TH_CA3 = (66.5 - 43.6) / 0.80
FIRE_TH["CA3"] = HUMAN_TH_CA3
# 聴覚：帯の10本のうち9本そろって閾値を越える重さ（14.3 ÷ 8.5 ＝ 1.68）⚠️ ノブ（人の発火率に合わせる値）
#   calib（VOICEVOX 40発話、版28 の耳）：9本 1.60% ／ 10本 0.82%。どちらも【ヒト】皮質の細胞の発火率
#   （1歩 0.53〜1.80%、Chung 2022）の範囲
#   版27 は10本で、実際の聴覚の的は 0.20%（範囲の下を割った）。視床が下から来た歩の半分ほどしか通さないため
#   ⇒ 減る側を見込んで9本
#   ⟲ 前は視床1本で閾値の2倍（28.6）。視床の発火がそのまま聴覚の的になり、誤差が人の10倍前後になっていた
X_GAIN = {"聴覚": HUMAN_TH / 8.5, "運動": 1.0, "高次": HUMAN_TH / 2.0,
          "CA3": 2 * HUMAN_TH_CA3}    # ⚠️ ノブ。苔状線維1本で越える（下の苔状線維を参照）
TGT_BY_FIRE = ("聴覚", "運動", "視覚", "連合", "高次", "CA3",
               "前頭前野", "背内側視床", "嗅内浅", "嗅内深", "側坐核", "上丘")   # 版43a：足した領野も下からの入力で撃つか   # 誤差の的 ＝ 下からの入力で撃つか
# 受け手1体の相手の数。既定は人の皮質（30,000シナプス ÷ 接触3.3 ＝ 9,100）
#   【ヒト】CA3 は CA3 から約 16,100入力（棘 ≒ 送り手、接触ほぼ1つ。Watson 2025）
N_PARTNER = {"CA3": 16100}
# 1つの繋がりの興奮性の接触の数。既定は【ヒト】皮質 3.3
#   【ヒト】CA3 は棘 ≒ 送り手で、接触ほぼ1つ（Watson 2025。CA3→CA1 の結合解析でも棘1つに入力1つ）
#   ⚠️ 抑制の 4.0 は人の皮質の値のまま（CA3 の抑制の接触数は見つけていない）
CONTACTS_EXC = {"CA3": 1.0}
# 蹴りを送る運動の視床の細胞が1歩に撃つ確率
#   【ヒト】運動の視床 Vim の単一細胞 18.8 ± 9.8Hz（手術中、27体、振戦の患者）⇒ 18.8 × 10.67ms ≈ 0.20
#   【ヒト】話すときの運動野の細胞は、ふだん 10〜15Hz、話すあいだの山で 30〜40Hz（Stavisky 2019 図1D）
#   ⟲ 前は準備の 103歩のあいだ毎歩 1（約94Hz 相当）。蹴りの入った細胞が毎歩撃ち、外では撃たなかった
KICK_P = 18.8 * 10.67 / 1000.0
KICK_FRAC = 0.75        # 運動視床の蹴りが及ぶ出力の割合。⚠️ 人の値ではない（numpy 原本の置き値）
# 版49：語の指令は運動前野から、皮質どうしの繋がりで来る（【ヒト】Greenlee 2004 ／ M1・6v に第4層なし Bakken 2021・Comm Biol 2025）
#   運動の領野の1体に運動前野の送り手 K_PM 本。1本 ＝ 人の繋がり1本（重さ 1）。届いた数がそのまま入口
#   K_PM ⚠️ ノブ【C】：版48 の指令の歩の ihn 約 2 で、指令の体が【ヒト】話すあいだの 35Hz で撃つ本数
K_PM = 115
R_PM_ON = 35.0 * 10.67 / 1000.0     # 指令が入っている体の送り手【ヒト】話すあいだの運動野の細胞の山 30〜40Hz（Stavisky 2019）
R_PM_OFF = 3.30 * 10.67 / 1000.0    # 入っていない体の送り手【ヒト】起きている錐体（Verma 2024）
KICK_W = 2 * HUMAN_TH               # 蹴りの1発の重さ ⚠️ ノブ（前の X_GAIN 運動。視床の本数は人に無い）
#   【ヒト】BrainGate で分かっているのは「語によらない大きな信号が話し始めに出る」ことまで。割合は出ていない
CA3_SUCCESS = 0.90      # 【ヒト】CA3 のシナプスの成功率（皮質の放出確率 0.33 と対比）


class Brain:
    def __init__(self, prior, n_ear=40, n_cmd=None, n_eye=40, n_assoc=240, n_high=263,
                 hippo_scale=4.0, n_partner=9100, n_cell=128, eta=0.3,
                 stp_path="synapse_model_human.json", dev="cuda", seed=0, eye_open=False):
        if n_cmd is None:                    # 版44：運動の領野の体数は口の指令の数（mouth.N_CMD ＝ 840）
            from mouth import N_CMD as n_cmd  # ⟲ 版43 までは 1,812（連続値1つに 100体）
        h = {k: max(4, int(round(v * hippo_scale))) for k, v in HIPPO_RATIO.items()}
        self.h = h
        self.n = dict(視床=n_ear, 聴覚=n_ear, 運動=n_cmd, 視覚=n_eye,
                      連合=n_assoc, 高次=n_high,
                      歯状回=h["歯状回"], CA3=h["CA3"], CA1=h["CA1"], **N_NEW)
        nn = self.n
        # ext（樹状突起に入る調整する入力）の数
        self.n_ext = dict(視床=0,                    # 中継。樹状突起には何も来ない
                          聴覚=n_cmd + n_assoc,      # 指令の写し ＋ 連合からの予測
                          運動=n_high + nn["前頭前野"],   # 高次からの予測 ＋ 前頭前野から（版43a）
                          視覚=0,
                          連合=n_high + nn["嗅内深"],  # 高次からの予測 ＋ 海馬の戻り（版43a：CA1 → 嗅内深 を通して）
                          高次=n_cmd + nn["前頭前野"],  # 運動の誤差 ＋ 前頭前野から（版43a）
                          歯状回=EC_II,               # 嗅内浅 層 II の誤差（版43a。前は高次の誤差）
                          CA3=h["歯状回"] + h["CA3"] + EC_II,   # 歯状回 ＋ 1歩前の自分 ＋ 嗅内浅 層 II（版43a）
                          CA1=nn["嗅内浅"] - EC_II,   # 嗅内浅 層 III（版43a。前は 1歩前の CA3）
                          前頭前野=nn["嗅内深"],       # 嗅内深 から
                          背内側視床=nn["前頭前野"],   # 前頭前野から（調整する側。版43a の組み直し下）
                          嗅内浅=nn["嗅内深"],         # 深い層 → 浅い層（層内）
                          嗅内深=0,
                          側坐核=nn["前頭前野"],       # 前頭前野から（側坐核の予測）
                          上丘=0)                      # 版43b：下丘から細胞体へだけ
        # 上からの戻り（眠りで落とす線。脳幹が組むときに読むので、脳幹より先に置く）版43a で 前頭前野 → 高次・運動 を足した
        self.topdown_rows = (("聴覚", nn["聴覚"] + nn["運動"], nn["連合"]),     # 連合から
                             ("運動", nn["運動"], nn["高次"]),                  # 高次から
                             ("運動", nn["運動"] + nn["高次"], nn["前頭前野"]),  # 前頭前野から（版43a）
                             ("連合", nn["連合"], nn["高次"]),                  # 高次から
                             ("高次", nn["高次"] + nn["運動"], nn["前頭前野"]))  # 前頭前野から（版43a）
        mk = lambda nm, i: Area(
            n_ch=self.n[nm], prior=prior, n_ext=self.n_ext[nm], eta=eta,
            n_partner=N_PARTNER.get(nm, n_partner), n_cell=n_cell, stp_path=stp_path, dev=dev, seed=seed + i,
            # 版43a：前頭前野・嗅内・側坐核は高次の時定数（⚠️ ノブ【C】。人の値は未取得）
            tau_scale=(TAU["連合"] if nm == "連合"
                       else TAU["高次"] if nm in ("高次", "前頭前野", "嗅内浅", "嗅内深", "側坐核")
                       else 55.9 / 100.0 if nm in ("歯状回", "CA3", "CA1") else TAU["一次"]),
            gate=(nm != "視覚" or eye_open),
            motor=(nm in ("運動", "視床", "背内側視床")),   # 視床は貯めずに中継する（時間細胞を持たない）
            inh_frac=(0.275 if nm in ("視床", "背内側視床") else 0.16),   # 【ヒト】視床の抑制性 25〜30%
            complete=(nm in ("CA3", "CA1")),
            # 疎さは段で違う（【ヒト】）
            #   視床（MGB）      密。皮質へ密な入力を送る
            #   聴覚（ヘシュル回）  まだ密。話し言葉に反応する電極が 39〜40%（8名、頭蓋内脳波）
            #   連合（上側頭回）   疎。背景の発火は低く、45%が特定の言葉にだけ強く撃つ
            #                   （単一ニューロン 140体超、Chan ほか 2014）
            #   ⚠️ 前は視床で疎にしていた（2.4% → 0.7%）。疎化の段を2つ手前に置いていた
            fire_th=FIRE_TH.get(nm, HUMAN_TH if nm in N_NEW else 1.0),
            x_gain=X_GAIN.get(nm, X_NEW if nm in N_NEW else 1.0),
            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),
            # 版47：抑制の体の興奮性の相手 180（【ヒト】覚醒の率から【C】）。海馬・嗅内は【ヒト】の率（錐体 3.1Hz・抑制 5.1Hz）から 194
            inh_n_exc=(INH_N_EXC_HIPPO if nm in ("歯状回", "CA3", "CA1", "嗅内浅", "嗅内深") else INH_N_EXC),
            # ⚠️ 視床の自発は【ヒト】の値が無い。人の視床に電極を入れるのは DBS 手術のときだけで、
            #    通すのは運動視床（Vim 19.8Hz）と視床下核。聴覚の視床（MGB）は通り道ではない。
            #    人の MGB で分かっているのは fMRI の活動の増減だけで、発火率は出ていない
            #    一度【マーモセット】の 8.7Hz を入れたら、音が背景に埋もれて supp が 1.43 → 1.04 に落ちた
            #    ⇒ 皮質と同じ 0.3Hz に置く。これは人の値ではない。ノブ
            spont_hz=(0.5 if nm in ("歯状回", "CA3", "CA1") else 0.3))
        self.areas = {nm: mk(nm, i) for i, nm in enumerate(NAMES)}
        self.g = Group2([self.areas[nm] for nm in NAMES])
        self.dev = dev
        g = torch.Generator(device=dev); g.manual_seed(seed + 99)
        # 運動視床：合図のあと、語によらない大きな型を数歩だけ出す（【ヒト】BrainGate の「蹴り」）
        #   ⟲ 移したとき「出力の75%」が「32体」に変わっていた
        self.kick_to = torch.randperm(n_cmd, device=dev, generator=g)[:max(1, int(n_cmd * KICK_FRAC))]
        self.kick_left = 0
        # 感覚の視床（MGB）：耳の1本 → 視床の1体（帯をまたがない）
        #   【ヒト】MGB はトノトピーで並ぶ（7T fMRI、背内側→腹外側へ 低・高・低）⇒ 中継は自分の帯から受ける
        #   ✗ numpy 版の「1体が耳の 1/8 をランダムに拾う」は人の値ではなく、トノトピーとも逆。
        #      耳40体のときに書かれた式で、耳400体だと 50本を束ね、視床が約80% 撃っていた（版18）
        # 海馬の投射（【ヒト】貫通路は疎、苔状線維は起爆、シャッファー側枝）
        # 嗅内皮質 → 歯状回：1体が入口の約2%を拾う（【ヒト】・霊長類の疎な投射）
        #   ⟲ 移したとき、体数が少ないときの代用（1/8）だけが残って 2% が落ちていた
        #   版43a：送り手は嗅内浅の層 II（20体）。前は高次（皮質から直に）
        #   ⚠️ 20体では 2% が1本を割るので下限の2本（今の式のまま）
        k_perf = max(2, int(round(EC_II * 0.02))) if EC_II >= 100 else max(2, EC_II // 8)
        self.perforant = torch.stack([torch.randperm(EC_II, device=dev, generator=g)[:k_perf]
                                      for _ in range(h["歯状回"])])
        # 苔状線維：CA3 の1体に歯状回から4本
        #   ⚠️ ノブ（みわ 09-23）。【ヒト】は1体に約 280本（282±16）、顆粒細胞の 0.002% が繋がる（Watson 2025）
        #      模型の歯状回は300体なので、人の疎さに合わせると1本を下回り、本数と疎さは両立しない
        #      動き方によっては本数と疎さの両方をノブにして釣り合いを取る
        self.mossy = torch.stack([torch.randperm(h["歯状回"], device=dev, generator=g)[:4]
                                  for _ in range(h["CA3"])])
        self.schaffer = torch.stack([torch.randperm(h["CA3"], device=dev, generator=g)[:4]
                                     for _ in range(h["CA1"])])
        # 段と段の投射（下 → 上 の駆動する入力）。周波数で繋ぐ
        #   一次 Te1   視床から1帯ずつ（いじらない）
        #              【ヒト】Bitterman ほか 2008：最も鋭い細胞で約 1/12 オクターブ
        #              ⇒ 模型の1帯（最も狭くて 1/3 オクターブで1帯）より狭い。1帯が模型の底
        #   連合 Te2   【ヒト】Besle ほか 2022（7T fMRI 順応＋模型、11人、プレプリント）
        #              後側ベルト（平面側頭 ＝ Te2）の細胞1体の同調幅の推定範囲 10〜27 ERB（半値全幅）
        #              ⚠️ 人の Te2 で「細胞ごとの幅の分布」を直接測ったものは無い。これは平均の推定範囲で、
        #                 著者自身が直接記録より広く出ると書いている（人の核の直接記録は 1 ERB 未満）
        #              ⚠️ 範囲の中を ERB で等間隔に刻むのは Claude の置き方（出典の模型は領野内で幅を1つに揃えている）
        #              1つの中心周波数に幅の段を並べる（中心周波数 × 幅の2軸は【マカク】ベルトの形）
        #   高次 Te3   【ヒト】上側頭回（Leonard ほか 2024, Nature 626）単一ニューロン n=217
        #              帯域 4.03 ± 1.57 オクターブ、峰 7.72 ± 2.14 個
        #              ⇒ 連続した窓ではなく、帯域の中の飛び飛びの峰から受ける
        #              ⚠️ 同調の中央値 326.5Hz（低い側に寄る）は分布の形が無いので入れていない
        #              ⚠️ 刺激は話し言葉の文だけ。峰が倍音の並びを拾っている可能性は残る
        rng = np.random.default_rng(seed + 7)
        self.a2c = self._belt(n_ear, self.n["連合"], rng)
        self.c2h = self._stg(self.a2c_center, self.n["高次"], rng)
        # 版43a：足した領野の細胞体への投射（受け手1体が送り手から K_NEW 本。⚠️ ノブ）
        #   高次 → 前頭前野       【ヒト】Saur 2008（側頭 → 腹外側前頭、最外包）／【マカク】Romanski 1999
        #   背内側視床 → 前頭前野  【ヒト】Behrens 2003（拡散 MRI、視床の区分と皮質）⚠️ 記憶の書誌
        #   前頭前野 → 背内側視床  ⚠️ 細胞体に入れると 前頭前野 ⇄ 背内側視床 の興奮の輪で両方が撃ちっぱなしになった
        #     （【測】コンテナの小さい網、60歩で 前頭前野・背内側視床 とも 100%）。人では視床網様核の抑制と
        #     腹側淡蒼球の抑制がこの輪に掛かる（模型に無い）⇒ 43a では樹状突起（調整する側）に入れ、駆動は 43b で決める
        #   高次 → 嗅内浅         【マカク】上側頭回 → 嗅内皮質（Brain Res 1983）
        #   前頭前野 → 嗅内浅     【マカク】眼窩・内側前頭 → 嗅内皮質（Insausti 1987）
        #   CA1 → 嗅内深          【マカク】J Comp Neurol 2020 ／【ヒト】Maass 2014
        #   CA1・嗅内深 → 側坐核   【マカク】Friedman 2002 ／ 層 Va → 側坐核（総説 2021）
        rp = lambda ns, nt: self._rand_proj(ns, nt, K_NEW, g)
        self.p_h2pfc = rp(self.n["高次"], nn["前頭前野"])
        self.p_md2pfc = rp(nn["背内側視床"], nn["前頭前野"])
        self.p_h2ecs = rp(self.n["高次"], nn["嗅内浅"])
        self.p_pfc2ecs = rp(nn["前頭前野"], nn["嗅内浅"])
        self.p_ca12ecd = rp(h["CA1"], nn["嗅内深"])
        self.p_ca12nac = rp(h["CA1"], nn["側坐核"])
        self.p_ecd2nac = rp(nn["嗅内深"], nn["側坐核"])
        # 版43b：下丘（耳と視床のあいだの中継。耳の1本 → 下丘の1体、1歩遅れ）
        #   【ヒト】上る音の道は下丘を通る（教科書の段 ⚠️）／【ヒト】中心核 42万体超（Mansour 2019）。模型は耳の本数（下限）
        #   ⚠️ 下丘の中の変換（周波数の鋭さ・時間の形）は入れていない。中継だけ【C】
        self.ic_out = torch.zeros(n_ear, device=dev)
        # 下丘 → 上丘の深層（⚠️ 霊長類の線の論文はまだ読んでいない。受け手1体が K_NEW 本）
        self.p_ic2sc = rp(n_ear, nn["上丘"])
        # 中脳の小さな核（学ばない）
        import midbrain as MB
        self.mb = MB.Midbrain(N_MB, dev=dev, gen=g)
        self.mb.wire(nn["上丘"], nn["側坐核"], N_OMIT)
        self.omit = torch.zeros(N_OMIT, device=dev)
        self.gate = torch.ones((), device=dev)
        self.gen = g
        self.up = {nm: torch.zeros(self.n[nm], device=dev) for nm in NAMES}
        # 視床は誤差ではなく表現を上へ送る（中継なので）
        self.thal_out = torch.zeros(n_ear, device=dev)
        self.pred = {nm: torch.zeros(self.n[nm], device=dev) for nm in ("連合", "高次", "前頭前野", "嗅内深")}
        self.ca3_prev = torch.zeros(h["CA3"], device=dev)
        self.ca1_out = torch.zeros(h["CA1"], device=dev)
        self.md_out = torch.zeros(nn["背内側視床"], device=dev)   # 版43a
        # 版43d：運動の領野の表現（1歩前）。口の指令の写しとして聴覚へ送る
        self.motor_rep = torch.zeros(n_cmd, device=dev)
        # 海馬の受け手に印をつける（眠っているあいだ、皮質だけ下げるため）
        hip = torch.zeros(self.g.n_recv_all, dtype=torch.bool, device=dev)
        for i, nm in enumerate(NAMES):
            if nm in ('歯状回', 'CA3', 'CA1'):
                hip[int(self.g.off_recv[i]):int(self.g.off_recv[i + 1])] = True
        self.g.is_hippo = hip
        hip_ch = torch.zeros(self.g.n_ch_all, dtype=torch.bool, device=dev)
        for i, nm in enumerate(NAMES):
            if nm in ('歯状回', 'CA3', 'CA1'):
                hip_ch[int(self.g.off_ch[i]):int(self.g.off_ch[i + 1])] = True
        self.g.is_hippo_ch = hip_ch
        self.off_ch = np.cumsum([0] + [self.n[nm] for nm in NAMES])
        self.off_ext = np.cumsum([0] + [self.n_ext[nm] for nm in NAMES])
        # 毎歩おなじ場所に おなじ順で書くだけなので、番号表を先に作って1回で済ませる
        self.cut_len = [self.n[nm] for nm in NAMES]
        self.kick_abs = int(self.off_ch[NAMES.index("運動")]) + self.kick_to
        # 脳幹（版38）：いまの状態（起きている ／ N3 ／ REM）を持って全領野へ配る。起きているから始める
        self.bs = BS.Brainstem(self)
        # 調整する側の経路（版39）：予測 p には入るが、表現を撃たせない
        #   【ヒト】遠い樹状突起は細胞体をほとんど興奮させない（Beaulieu-Laroche 2018）／ 上からは調整する入力（09-11）
        #   【ヒト】話すと上側頭回に来る入力が減る（Chan 2014）⇒ 指令の写しも調整する側
        #   ⚠️ ほかの経路（海馬・CA3 の再帰・運動の誤差→高次・皮質→歯状回・CA1 の戻り）は未決のまま撃たせる側
        n = self.n
        for area, first, cnt in (("聴覚", n["聴覚"], n["運動"]),) + self.topdown_rows + (
                ("嗅内浅", n["嗅内浅"], n["嗅内深"]),          # 深い層 → 浅い層（版43a）
                ("背内側視床", n["背内側視床"], n["前頭前野"]),  # 前頭前野 → 背内側視床（版43a）
                ("側坐核", n["側坐核"], n["前頭前野"])):        # 前頭前野 → 側坐核（予測。版43a）
            # ⚠️ 海馬の戻り（嗅内深 → 連合・前頭前野）は前と同じく撃たせる側のまま（未決）
            self.g.mod_send[self._send_rows(area, first, cnt)] = True
        # 恒常性の目標を受け手の種類ごとに（版39b）【み】09-27「普段の率を目標とみなす」「Bで様子見」
        #   皮質  表現・誤差 0.45%（【ヒト】Peyrache 2012 RS 0.5Hz 未満。誤差は表現と同じ【C 当てはめ】）／ 抑制 2Hz（FS）
        #   視床  表現 18.8Hz（【ヒト】運動視床 Vim。MGB は人・霊長類とも数値なし）。誤差・抑制は値なし ⇒ 下げない
        #   運動  話す運動野の値なし ⇒ 領野ごと下げない ／ 海馬は眠りでは sleep_mode で下げない（今のまま）
        from torch_area import DT_MS as _DT
        hz = lambda v: v * _DT / 1000.0
        gg = self.g
        for i, nm in enumerate(NAMES):
            lo, hi, nc = int(gg.off_recv[i]), int(gg.off_recv[i + 1]), self.areas[nm].n_ch
            er, rp, ih = slice(lo, lo + nc), slice(lo + nc, lo + 2 * nc), slice(lo + 2 * nc, hi)
            if nm == "視床":
                gg.target_recv[rp] = hz(18.8)
                gg.scale_skip[er] = True
                gg.scale_skip[ih] = True
            elif nm == "運動":
                gg.scale_skip[lo:hi] = True
            elif nm in ("聴覚", "視覚", "連合", "高次", "前頭前野", "嗅内浅", "嗅内深"):
                gg.target_recv[ih] = hz(2.0)
            elif nm in ("背内側視床", "側坐核", "上丘"):      # 版43a・43b：人の値なし ⇒ 下げない
                gg.scale_skip[lo:hi] = True
        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）

    def _send_rows(self, area, first, n):
        """その領野の送り元の行 first〜first+n が持つ送り手の番号（時間細胞・傾きの細胞の分と、短い遅れの分）"""
        g = self.g
        i = NAMES.index(area)
        buf_off = sum(self.areas[nm].n_src * self.areas[nm].D for nm in NAMES[:i])
        a = self.areas[area]
        per = a.ch.n_cell + a.ch.n_ramp
        rows = torch.arange(first, first + n, device=self.dev)
        tc = int(g.off_send[i]) + rows.unsqueeze(1) * per + torch.arange(per, device=self.dev)
        bf = int(g.off_send[-1]) + buf_off + rows.unsqueeze(1) * a.D + torch.arange(a.D, device=self.dev)
        return torch.cat([tc.reshape(-1), bf.reshape(-1)])

    @property
    def n_cells(self):
        return self.g.n_cells

    def go(self, n=3):
        """合図（話し始める）。運動視床が蹴りを数歩だけ出す"""
        self.kick_left = n

    def new_utterance(self, speaking=True):
        """発話の頭。話すなら、準備のはじめに蹴りを入れる。聞くだけなら出さない
        版43d：蹴りは発話によらない同じ長さで、語の指令が入り始める前に終える。その後の状態は網に任せる【み】09-29
        【ヒト】Stavisky 2019：語によらない成分は合図の直後から立ち上がり、声はその約1.1秒後
        ⚠️ 人の成分は「上がり続ける」形。模型の蹴りは入れる／入れないの2値で、坂は作れない
        ⟲ 版43c までは声まで入れ続けた（マカクの腕 Kaufman 2016 から。発話に霊長類は使わない【み】09-29）
        ⟲ 版23 まで go() をどこからも呼んでいなかった。版24 は発話の頭で3歩だけだった"""
        import config as C
        if speaking:
            self.go(C.GO_LEAD - C.CMD_LEAD)     # 版43d：語の指令が入る前に終える（config.CMD_LEAD）

    @staticmethod
    def _octave():
        """耳の40帯の中心周波数（オクターブ）。ear.py と同じメルの並び"""
        import config as C
        from ear import _hz_to_mel, _mel_to_hz
        m = np.linspace(_hz_to_mel(C.MEL_FMIN), _hz_to_mel(C.MEL_FMAX), C.N_MELS + 2)
        return np.log2(_mel_to_hz(m)[1:-1])

    def _pack(self, src, tgt, n_tgt):
        """(送り手, 受け手) の組を持ち、受け手ごとの本数で割る表を作る"""
        src = torch.tensor(np.asarray(src), dtype=torch.long, device=self.dev)
        tgt = torch.tensor(np.asarray(tgt), dtype=torch.long, device=self.dev)
        cnt = torch.zeros(n_tgt, device=self.dev).index_add_(0, tgt, torch.ones_like(tgt, dtype=torch.float))
        assert bool((cnt > 0).all()), "入力が1本も来ない受け手がいる"
        return (src, tgt, cnt)

    @staticmethod
    def _erb():
        """耳の40帯の中心周波数（ERB 番号）。ERB 番号 ＝ 21.4 log10(1 + 0.00437 f)"""
        import config as C
        from ear import _hz_to_mel, _mel_to_hz
        m = np.linspace(_hz_to_mel(C.MEL_FMIN), _hz_to_mel(C.MEL_FMAX), C.N_MELS + 2)
        return 21.4 * np.log10(1.0 + 0.00437 * _mel_to_hz(m)[1:-1])

    def _belt(self, n_ear, n_tgt, rng):
        """聴覚 → 連合【ヒト】後側ベルトの同調幅 10〜27 ERB（Besle ほか 2022）"""
        erb = self._erb()
        n_mel = len(erb)
        n_fib = n_ear // n_mel
        assert n_fib * n_mel == n_ear, "耳の並びが 帯 × 繊維 になっていない"
        n_bw = -(-n_tgt // n_mel)                      # 1つの中心周波数に並ぶ段の数
        center = np.arange(n_tgt) % n_mel
        step = np.arange(n_tgt) // n_mel
        # ⚠️ 10〜27 ERB を等間隔に刻むのは Claude の置き方（未確認）
        bw = 10.0 + (27.0 - 10.0) * step / max(n_bw - 1, 1)
        src, tgt = [], []
        for j in range(n_tgt):
            chs = np.where(np.abs(erb - erb[center[j]]) <= bw[j] / 2)[0]
            for ch in chs:                              # その帯の繊維を全部（閾値の違う10本）
                for f in range(n_fib):
                    src.append(ch * n_fib + f); tgt.append(j)
        self.a2c_center, self.a2c_bw = center, bw
        return self._pack(src, tgt, n_tgt)

    def _stg(self, belt_center, n_tgt, rng):
        """連合 → 高次【ヒト】上側頭回。帯域 4.03±1.57 oct の中の飛び飛びの峰 7.72±2.14 個"""
        octv = self._octave()
        n_mel = len(octv)
        by_ch = [np.where(belt_center == c)[0] for c in range(n_mel)]
        src, tgt = [], []
        self.c2h_bw, self.c2h_peaks = np.zeros(n_tgt), np.zeros(n_tgt, int)
        for j in range(n_tgt):
            c = rng.integers(n_mel)
            bw = float(np.clip(rng.normal(4.03, 1.57), 1 / 3, octv[-1] - octv[0]))
            k = int(np.clip(round(rng.normal(7.72, 2.14)), 1, None))
            chs = np.where(np.abs(octv - octv[c]) <= bw / 2)[0]
            picks = rng.choice(chs, size=k, replace=len(chs) < k)
            for ch in picks:                             # 峰1つ ＝ その帯を中心に持つ連合の1体
                src.append(int(rng.choice(by_ch[ch]))); tgt.append(j)
            self.c2h_bw[j], self.c2h_peaks[j] = bw, k
        return self._pack(src, tgt, n_tgt)

    def _fire(self, total, nm):
        """集まった入力を、受ける細胞が撃つかどうか（0か1）に直す
        【ヒト】皮質の錐体細胞の出力は「集まった興奮性の入力を閾値と比べた結果の撃つか撃たないか」
          （L2/3 は同時 134±28本で 50%、切除切片）。入力の本数そのものを出す細胞はいない
        ⇒ 誤差のニューロンの的（x）も 0か1。足した本数のままだと、予測（シグモイド、1を超えない）が
          追いつく先が無く、誤差が永久に正のまま重みが伸び続けた（版12〜16 の連合・高次）
        海馬の貫通路（足す → 上位 2.5% だけ撃つ）と同じ形
        ⚠️ 閾値は受ける領野の FIRE_TH（ノブ）。人の関係「平均的な入力がちょうど閾値」はまだ入れていない"""
        return (total > self.areas[nm].fire_th).float()

    def _band_sum(self, thal):
        """聴覚の細胞は、自分の帯の視床の本数（繊維の段）の合計を受ける
        【ヒト】一次聴覚野は柱の中で周波数の同調がそろう（Bitterman 2008）⇒ 同じ帯の細胞は同じ入力を受ける
        帯の並びは 帯 × 繊維（耳の encode と同じ）"""
        import config as C
        n_fib = len(thal) // C.N_MELS
        return thal.view(C.N_MELS, n_fib).sum(1).repeat_interleave(n_fib)

    def _proj(self, v, proj):
        """届いた本数を足す（x_gain は「何本ぶんの入力か」）
        🔴 本数で割らない。09-11 に「撃った数で割る」を周転円として落としている
           （割り算は脳の形として人にも霊長類にも出てこない）。
           広い窓の体に本数が多く届くのは、測ってから原因を見る"""
        src, tgt, cnt = proj
        return torch.zeros(len(cnt), device=self.dev).index_add_(0, tgt, v[src])

    def _rand_proj(self, n_src, n_tgt, k, g):
        """版43a：受け手1体が送り手から k 本を一様に拾う（重複なし）。⚠️ 人の本数は未取得（ノブ）"""
        k = min(k, n_src)
        src = torch.stack([torch.randperm(n_src, device=self.dev, generator=g)[:k] for _ in range(n_tgt)])
        tgt = torch.arange(n_tgt, device=self.dev).repeat_interleave(k)
        cnt = torch.full((n_tgt,), float(k), device=self.dev)
        return (src.reshape(-1), tgt, cnt)

    def _premotor(self, cmd):
        """版49：運動前野の送り手 K_PM 本のうち、この歩に撃った数（指令が入っている体は R_PM_ON、ほかは R_PM_OFF）"""
        p = torch.where(cmd > 0, torch.full_like(cmd, R_PM_ON), torch.full_like(cmd, R_PM_OFF))
        return torch.binomial(torch.full_like(p, float(K_PM)), p, generator=self.gen)

    def _fit(self, v, n):
        if len(v) == n:
            return v
        out = torch.zeros(n, device=self.dev)
        m = min(n, len(v))
        out[:m] = v[:m]
        return out

    @torch.no_grad()
    def step(self, ear, cmd, eye=None, learn=True):
        """ear [n_ear]（耳のメル）  cmd [n_cmd]（口の指令の写し）  eye [n_eye] または None"""
        # 海馬：嗅内浅 層 II の誤差 → 貫通路 → 歯状回（上位 2.5% だけ撃つ）（版43a。前は高次の誤差）
        ec2 = self.up["嗅内浅"][:EC_II]
        ec3 = self.up["嗅内浅"][EC_II:]
        drive = ec2[self.perforant].sum(1)
        k = max(1, int(round(self.n["歯状回"] * SPARSE)))
        # 上位 2.5% だけ撃つ。⚠️ 何も来ていないとき（drive が全部ゼロ）は誰も撃たない。
        #   閾値だけで判定すると、ゼロ同士が並んで全部が通ってしまう
        # 版43a：同じ値が並んだら乱数で選び、ちょうど k 体にする（【ヒト】2.5% 未満 ＝ 撃つ数の上限）
        #   ⟲ 嗅内浅 層 II が 20 体になり、1体が拾うのは2本 ⇒ 値が 0〜2 しかなく同点だらけで、
        #      「閾値以上ぜんぶ」だと歯状回が平均 19 体・多い歩は 300 体すべて撃った（【測】コンテナ）
        #   ⚠️ 入力が粗い（2本）こと自体は、嗅内浅 層 II の体数の下限の問題として残る
        rank = drive + 0.5 * torch.rand(len(drive), device=self.dev, generator=self.gen)
        top = torch.topk(rank, k).indices
        dg = torch.zeros_like(drive)
        dg[top] = 1.0
        dg = dg * (drive > 0).float()

        # ── 細胞体に入るもの（駆動する入力。撃たせる）──番号表の順に繋いで1回で作る
        ic = self.ic_out                                       # 版43b：1歩前の下丘（耳 → 下丘 → 視床）
        self.ic_out = ear
        x = torch.cat([
            ic,                                                # 下丘 → 視床（1本 → 1体、帯をまたがない）
            self._band_sum(self.up["視床"]),                   # 視床 → 聴覚（自分の帯の10本の合計）
            self._premotor(cmd),                                               # 口の指令 → 運動の領野（版43d で戻した） ／ 版49：運動前野の送り手の数
            #   ⟲ numpy 版（brain.py）は運動の領野が cmd を受けていた。torch 版では 0 になっていた（記録なし。Claude の落とし）
            #     運動の領野が「何を言ったか」を持たず、写しは乱数の指令から聴覚へ直に入っていた
            #   補助輪：何を言うかは乱数が選ぶ。選んだ指令は運動の領野を通して口へ出す
            eye if eye is not None else torch.zeros(self.n["視覚"], device=self.dev),
            self._proj(self.up["聴覚"], self.a2c),               # 本数のまま（撃つかは領野の側で判定）
            self._proj(self.up["連合"], self.c2h),
            dg,
            (dg[self.mossy].max(1).values > 0).float(),        # 苔状線維（起爆）
            (self.ca3_prev[self.schaffer].max(1).values > 0).float(),
            # 版43a
            self._proj(self.up["高次"], self.p_h2pfc) + self._proj(self.md_out, self.p_md2pfc),   # 前頭前野
            torch.zeros(self.n["背内側視床"], device=self.dev),                                  # 背内側視床（駆動は 43b）
            self._proj(self.up["高次"], self.p_h2ecs) + self._proj(self.up["前頭前野"], self.p_pfc2ecs),  # 嗅内浅
            self._proj(self.ca1_out, self.p_ca12ecd),                                             # 嗅内深
            self._proj(self.ca1_out, self.p_ca12nac) + self._proj(self.pred["嗅内深"], self.p_ecd2nac),  # 側坐核
            self._proj(ic, self.p_ic2sc)])                                                        # 上丘（版43b）
        if self.kick_left > 0:                                 # 運動視床の蹴り
            # 視床の細胞が人の速さで撃つスパイクの列（毎歩 1 ではない）
            x[self.kick_abs] = x[self.kick_abs] + KICK_W * (torch.rand(len(self.kick_abs), device=self.dev,
                                           generator=self.gen) < KICK_P).float()
            self.kick_left -= 1

        # ── 樹状突起に入るもの（調整する入力。撃たせない）──
        pc = (self.pred["連合"] > 0.5).float()
        ph = (self.pred["高次"] > 0.5).float()
        pf = self.pred["前頭前野"]
        ed = self.pred["嗅内深"]
        e = torch.cat([self.motor_rep, pc,                     # 聴覚（視床は ext を持たない）
                       #   版43d：写しは運動の領野が撃った表現（1歩前）【ヒト】腹側中心前回 → 上側頭回、発話の 107.5ms 前にピーク
                       #   ⟲ 前は乱数の指令そのもの（口に渡した答え）を直に入れていた
                       ph, pf,                                 # 運動（＋ 前頭前野。版43a）
                       ph, ed,                                 # 連合（＋ 海馬の戻り。版43a：嗅内深 から）
                       self.up["運動"], pf,                     # 高次（運動の誤差を丸ごと ＋ 前頭前野。版43a）
                       ec2,                                    # 歯状回（版43a：嗅内浅 層 II）
                       dg, self.ca3_prev, ec2,                  # CA3（＋ 嗅内浅 層 II。版43a）
                       ec3,                                    # CA1（版43a：嗅内浅 層 III）
                       ed,                                     # 前頭前野（嗅内深 から）
                       pf,                                     # 背内側視床（前頭前野から。調整する側）
                       ed,                                     # 嗅内浅（深い層から）
                       pf])                                    # 側坐核（前頭前野から）。上丘は ext なし

        outs = self.g.step(x, e, learn)
        rep, err = outs[0], outs[1]
        cut = lambda nm, v: v[int(self.off_ch[NAMES.index(nm)]):
                              int(self.off_ch[NAMES.index(nm)]) + self.n[nm]]
        for nm, v in zip(NAMES, torch.split(err, self.cut_len)):
            self.up[nm] = v
        self.up["視床"] = cut("視床", rep)        # 視床は表現を送る（中継。誤差ではない）
        # 上から下へ送るのは、上の表現の発火（版37）。【ヒト】別の領野へ届くのは撃ったスパイクだけ
        #   ⟲ 版36 までは上の表現の予測 p を 0.5 で切って送っていた
        self.pred["連合"] = cut("連合", rep)
        self.pred["高次"] = cut("高次", rep)
        self.pred["前頭前野"] = cut("前頭前野", rep)      # 版43a
        self.pred["嗅内深"] = cut("嗅内深", rep)
        self.md_out = cut("背内側視床", rep)              # 視床は表現を送る（中継）
        # CA3 のシナプスは確実に伝わる（【ヒト】0.90。皮質の 0.33 との差が速く覚える理由）
        c3 = cut("CA3", rep)
        self.ca3_prev = c3 * (torch.rand(len(c3), device=self.dev, generator=self.gen) < CA3_SUCCESS)
        self.ca1_out = cut("CA1", rep)
        self.motor_rep = cut("運動", rep)                 # 版43d
        # 版43b：前頭前野の「来なかった誤差」（頭 N_OMIT 体）＝ 予測 − 下からの入力 が THETA を越えた
        #   【ヒト】上側頭回の後ろ・外側眼窩前頭（別の群れ）／【マカク】眼窩前頭 3.6〜7.1%
        #   誤差のニューロンの予測 p と的（下からの入力で撃つか）は、音の誤差と同じものを使う（向きだけ逆）
        pe, xt = outs[2], self.g.last_xt
        lo = int(self.off_ch[NAMES.index("前頭前野")])
        self.omit = ((pe[lo:lo + N_OMIT] - xt[lo:lo + N_OMIT]) > THETA).float()
        # 中脳：1歩前の 上丘の誤差・側坐核の誤差・来なかった誤差 を受けて進む。門は脳幹が使う
        self.gate = self.mb.step(self.up["上丘"], self.up["側坐核"], self.omit)
        self.bs.observe(self)                  # 脳幹：驚き・起こすか（版38）
        return cut("聴覚", rep), cut("聴覚", err), cut("運動", rep), cut("運動", err)
