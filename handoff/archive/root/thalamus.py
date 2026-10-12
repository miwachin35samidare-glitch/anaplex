"""視床 —— 感覚の中継と、運動を蹴り込む駆動の入力

人から取れたもの
  【ヒト】3系統  聴覚 MGB ／ 視覚 LGN ／ 体性感覚 VP。感覚は視床を通して皮質の L4 へ
  【ヒト】体数    核ごとに 10⁶ の桁（前腹側＋内側核 約130万 ／ 背内側核 180万〜643万）
                ⚠️ LGN は方法で6倍ばらつく（0.57〜3.48百万）⇒ 桁として使う
  【ヒト】型      視床興奮性（中継細胞）70〜75% ／ 中脳由来抑制性 ／ Splatter
  【ヒト】抑制性   25〜30%（皮質の 16% より多い）
                ⚠️ 皮質と別系統（中脳由来）。皮質の型で代用できない
  【ヒト】様式    低閾値スパイクのバーストを出すことは確認済み（DBS 手術中の細胞外記録）
  【サル】随伴放電は皮質下に無い ⇒ 視床は予測の輪に入らない。駆動する側に専念する

人に無いもの（この作りでは要らない）
  細胞内の3量（τ・sag・リセット深さ）  モルモット・ラット止まり
  ⇒ こちらの素子は焼いた網で τ を直接持たない。ゆっくりした状態と段の時定数だけ

駆動する入力と調整する入力（Sherman・Guillery）
  駆動   信号の中身を運ぶ。閾値を越える活動を起こす
  調整   撃たせずに、反応しやすさを変えるだけ
  ⇒ 視床 → 皮質（下から上）は駆動。皮質 → 皮質（上から下）は調整
  ⇒ 運動を始める「蹴り」は、視床から来る駆動の入力
"""
import numpy as np
from element1 import Prior
from area4 import Area4

INH_FRAC = 0.275          # 【ヒト・霊長類】視床の抑制性 25〜30%
RELAY_FRAC = 0.725        # 【ヒト】中継細胞 70〜75%


class Thalamus:
    """1つの核。感覚を受けて皮質へ駆動する入力を出す"""

    def __init__(self, prior, n_in, n_ch=None, n_partner=200, n_cell=128, eta=0.3,
                 seed=0, name="MGB"):
        self.name = name
        self.n_ch = n_ch or n_in
        self.n_in = n_in
        # 中継細胞は時間細胞を持たない（貯めずに中継する）。抑制性は皮質より多い
        self.relay = Area4(n_ch=self.n_ch, prior=prior, n_ext=n_in, eta=eta,
                           n_partner=n_partner, n_cell=n_cell, stp=False,
                           inh_frac=INH_FRAC, timecells=False, spont_hz=0.3,
                           motor=True, rewire_every=0, seed=seed)
        rng = np.random.default_rng(seed + 11)
        # 感覚 → 中継細胞の投射（1体が入口の一部を拾う）
        k = max(1, n_in // 8)
        self.fan = np.stack([rng.choice(n_in, k, replace=False) for _ in range(self.n_ch)])

    @property
    def n_cells(self):
        return self.relay.n_cells

    def step(self, sense, learn=True):
        """sense [1, n_in] → 皮質の L4 へ出す駆動の入力 [1, n_ch]"""
        drive = (sense[0][self.fan].max(1) > 0).astype(np.float32)[None, :]
        rep, _, _ = self.relay.step(drive, sense, learn)
        return rep


class MotorThalamus:
    """運動を始める合図を出す核（人の運動視床 Voa/Vop/Vim にあたる）

    【ヒト】BrainGate：合図のあと、どの語でも同じ大きな「蹴り」が起き、
            そこから回転する動きに従って進む
    ⇒ 合図を受けたら、語によらない大きな型を数歩だけ出す
    """

    def __init__(self, n_out=64, n_burst=3, frac=0.75, seed=0):
        """n_out は「駆動する先の皮質のチャネル数」。出したものは細胞体に入る（＝撃たせる）"""
        rng = np.random.default_rng(seed)
        self.n_out = n_out
        self.n_burst = n_burst                       # バーストの長さ（歩）
        # 語によらない型（どの語でも同じ）。人：語ごとの成分とほぼ直交
        self.pattern = np.zeros((1, n_out), np.float32)
        self.pattern[0, rng.choice(n_out, int(n_out * frac), replace=False)] = 1.0
        self.left = 0

    def go(self):
        """合図（話し始める）"""
        self.left = self.n_burst

    def step(self):
        if self.left > 0:
            self.left -= 1
            return self.pattern
        return np.zeros((1, self.n_out), np.float32)
