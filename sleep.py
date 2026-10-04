"""睡眠（arkhe 用）—— 脳幹の状態に合わせて N3 と REM をくり返す（版38）

人で分かっていること（【ヒト】頭蓋内記録＋単一ユニット。てんかん患者）
  徐波（1Hz未満）     膜電位が上下する。上の状態（up）と下の状態（down）を行き来する
  紡錘波（12〜16Hz）  徐波の上の状態が支配して起きる。視床と皮質のやり取りで作られる
  リップル（80〜120Hz）紡錘波が谷でリップルを束ねる。そこで発火が跳ね上がる
  順番              徐波の上 → 紡錘波 → リップル → 発火の急増
  方向              皮質から海馬へ（紡錘波の結びつきの向き）
  ⇒ 海馬が勝手に再生するのではない。皮質の徐波と紡錘波が「いま渡していい」と合図する
  再生【ヒト】起きているあいだの発火の並びが、そのあとの睡眠で前より多く現れる
  下方調整【ヒト】TMS の応答は起きている時間とともに大きくなり、眠ると元に戻る
        ⚠️ 下げるのは皮質だけ。海馬は眠っているあいだ強くなるという報告がある【げっ歯類】
  REM【ヒト】徐波は無く、皮質の時間の窓は起きているときに近い（覚醒から +16ms、N3 は +105ms）
        上からの戻りは落ちる（脳幹が決める）。REM でも再生が出て、一晩の上達と相関（eLife 2023）

1歩 10.67ms に直すと
  徐波 94〜187歩 ／ 紡錘波 6〜8歩 ／ リップル 0.8〜1.2歩（1歩より速い ＝ 周回の側で表す）

段  N3（徐波。海馬の再生はここ）と REM をくり返す。長さ・周期は脳幹（brainstem.py）が持つ
  ⟲ 版37 までは N3 だけだった。REM を入れない承認は取っていなかった（【C】の省き。09-26 夜）
  ⚠️ N1・N2 はまだ入れていない（N1 が何をしているかは調べていない）
  ⚠️ REM の短い出来事（眼球運動・ぴくつき）は入れていない（目・体の段と一緒【み】）
"""
import numpy as np
import torch
import brainstem as BS

SO_STEPS = 140          # 徐波の1周期（歩）。【ヒト】1Hz未満 ⇒ 94〜187歩
SO_UP = 0.45            # そのうち「上の状態」の割合
SPINDLE_STEPS = 7       # 紡錘波の1周期（歩）。【ヒト】12〜16Hz ⇒ 6〜8歩
# リップルのとき、CA3 の再帰の興奮を底上げする
#   【ヒト】海馬の単一ニューロン：錐体細胞 3.1Hz → 16Hz（5.2倍）
#                            抑制性   5.1Hz → 19.2Hz（3.8倍。一緒に増える）
#   ⇒ 抑制は外れない。興奮の塊が来て、それに抑制が伴う。抑制性の方が先に立ち上がる
#   ⇒ こちらでは興奮の入力だけ底上げすれば、抑制性の体は領野の発火を受けて勝手についてくる
RIPPLE_EXC = 5.2        # 【ヒト】リップルのときの錐体細胞の発火の倍率
#   ⚠️ 一部だけが撃つ（人：錐体細胞の14%、抑制性の17%は逆に下がる）のは、
#      育ったシナプスの強さから自然に出るはず。どの体を撃たせるかは選ばない
CORTEX = ("聴覚", "運動", "視覚", "連合", "高次")
HIPPO = ("歯状回", "CA3", "CA1")


def rhythm(t):
    """いまの歩が、徐波のどこか・紡錘波のどこか。返り値: (上の状態か, 紡錘波の谷か)"""
    phase = (t % SO_STEPS) / SO_STEPS
    up = phase < SO_UP                                   # 徐波の上の状態
    trough = up and (t % SPINDLE_STEPS) == 0             # 紡錘波の谷（上の状態のあいだだけ）
    return up, trough


@torch.no_grad()
def sleep(b, n_step=BS.SLEEP_STEPS, report_every=None, verbose=True):
    """眠る。入力を止め、脳幹の状態に合わせて N3（徐波 → 紡錘波 → リップル）と REM をくり返す

    返り値: (海馬の発火の歩ごとの列, 皮質の発火の列)（前と同じ形。眠った歩数は b.bs.last_slept）
    脳幹が「起こす」と決めたら途中で起きる
    """
    dev = b.dev
    report_every = report_every or max(1, n_step // 10)
    zero_ear = torch.zeros(b.n["聴覚"], device=dev)
    zero_cmd = torch.zeros(b.n["運動"], device=dev)
    g, bs = b.g, b.bs
    base_spont = g.spont_p.clone()
    hip, ctx, n_ripple, n_rem = [], [], 0, 0
    bs.k = 0
    k = 0
    for k in range(n_step):
        st = bs.sleep_state(k)
        bs.set_state(st)
        if st == BS.N3:
            up, trough = rhythm(g.t)
            # 徐波：皮質は上の状態でだけ撃ちやすい（下の状態は静まる）
            g.spont_p = torch.where(g.is_hippo_ch, base_spont * 0.5,
                                    base_spont * (1.5 if up else 0.05))
            # リップル：紡錘波の谷で、CA3 の再帰の興奮を底上げする（皮質 → 海馬 の向き）
            g.exc_gain = torch.where(g.is_hippo, RIPPLE_EXC if trough else 1.0, 1.0)
            n_ripple += int(trough)
        else:
            # REM：徐波もリップルも無い。自発は起きているときと同じ
            g.spont_p = base_spont
            g.exc_gain = None
            n_rem += 1
        b.step(zero_ear, zero_cmd, learn=True)
        hip.append(float(b.ca3_prev.sum()))
        ctx.append(float(b.up["聴覚"].sum()))
        if verbose and (k + 1) % report_every == 0:
            print(f"    眠り {k+1:6d}歩（{st}）  CA3 {np.mean(hip[-report_every:]):5.2f}体/歩  "
                  f"聴覚 {np.mean(ctx[-report_every:]):5.2f}体/歩  "
                  f"scale {float(g.scale.mean()):.3f}  リップル {n_ripple}回  REM {n_rem}歩", flush=True)
        if bs.wake_request:
            if verbose:
                print(f"    脳幹が起こした（{k+1}歩目、{st}）", flush=True)
            break
    g.spont_p = base_spont
    g.exc_gain = None
    bs.set_state(BS.WAKE)
    bs.last_slept = k + 1
    return np.array(hip), np.array(ctx)


def replay_score(b, sig, n_step=500):
    """眠っているあいだ（N3）の海馬の発火が、起きているときの型とどれだけ似るか
    sig: 起きているあいだに控えた型 [発話, CA3]"""
    dev = b.dev
    zero_ear = torch.zeros(b.n["聴覚"], device=dev)
    zero_cmd = torch.zeros(b.n["運動"], device=dev)
    g = b.g
    base_spont = g.spont_p.clone()
    b.bs.set_state(BS.N3)
    hits = []
    for t in range(n_step):
        up, trough = rhythm(g.t)
        g.spont_p = torch.where(g.is_hippo_ch, base_spont * 0.5,
                                base_spont * (1.5 if up else 0.05))
        g.exc_gain = torch.where(g.is_hippo, RIPPLE_EXC if trough else 1.0, 1.0)
        b.step(zero_ear, zero_cmd, learn=False)
        v = b.ca3_prev.detach().cpu().numpy()
        if v.sum() == 0:
            continue
        d = np.linalg.norm(v) * np.linalg.norm(sig, axis=1)
        hits.append(float(np.max((sig @ v) / np.maximum(d, 1e-9))))
    g.spont_p = base_spont
    g.exc_gain = None
    b.bs.set_state(BS.WAKE)
    return float(np.mean(hits)) if hits else 0.0
