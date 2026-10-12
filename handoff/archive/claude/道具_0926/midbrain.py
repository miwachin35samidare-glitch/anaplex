"""中脳と基底核の小さな核（版43b）—— ふだん撃ち続ける細胞の群れ。学ばない
設計の経緯・出どころは claude/設計案_脳幹.md §12-24〜§12-41

置くもの（ふだんの率）
  黒質（ドーパミン）  4.56Hz【ヒト】Ramayya 2014（DBS 手術中、13個）
  視床下核の辺縁      36Hz  【ヒト】Rossi 2017（視床下核 100個）
  境の細胞（GPb）     33Hz  【マカク】Hong・Hikosaka 2008（外側手綱核へ送る 74個）
  外側手綱核          30Hz  【マカク】Bromberg-Martin 2010
  吻内側被蓋核        17.8Hz【マカク】Hong 2011（82個）
線（1歩前の発火が次の歩に届く。受け手1体が送り手から K 本を一様に拾う ⚠️ 本数は【C】）
  上丘の誤差        →（興奮）黒質              【マカク】May 2009
  側坐核の誤差      →（抑制）境の細胞           【マカク】Haber 1990（Hong 2008 の引用）【マウス】Stephenson-Jones 2016
  前頭前野の「来なかった誤差」→（興奮）視床下核の辺縁  【マカク】Haynes・Haber 2013（解剖）
  視床下核の辺縁    →（興奮）境の細胞           【マウス】Stephenson-Jones 2016
  境の細胞          →（興奮）外側手綱核         【マカク】Hong 2008
  外側手綱核        →（興奮）吻内側被蓋核       【マカク】Hong 2011
  吻内側被蓋核      →（抑制）黒質              【マカク】Hong 2011（刺激でドーパミン 16/17 が止まる）
腹側淡蒼球は置かない（§12-43）
  【マカク】見込みの得を「今どれだけか」として得まで持ち続ける（Tachibana 2012）。予測誤差ではない
  【マカク】得の大きさを線条体より先に表す（115ms 対 225ms、J Neurosci 2019）⇒ 元は線条体ではない（扁桃体・外側視床下部・前頭前野と著者）
  ⟲ 最初は「側坐核の誤差 →（抑制）腹側淡蒼球 → 黒質・外側手綱核」（【ラット】Lisman・Grace）で組んだ
     側坐核の誤差で外側手綱核が上がり、霊長類（良い知らせで下がる）と逆になった【測】
  上げる元が模型に無いので、置いても一定の率で撃つだけ ⇒ この版では置かない
撃ち方（1歩に撃つ確率）
  p ＝ ふだんの確率 × exp( G ×（興奮の届いた数 − ふだん届く数）− G ×（直近 PAUSE 歩の抑制の届いた数 − ふだん届く数）)
  指数の形【マカク】網膜の神経節細胞の発火の当てはめ（Pillow ほか 2008 Nature、入力の和 → 指数 → 発火）⚠️ 別の場所の細胞の形を借りた
  ⟲ 足し算（p ＝ ふだん ＋ G × ずれ）にしたら、ふだんの確率が小さい黒質（1歩 0.049）で 0 に切られる分だけ上に偏り、
     ふだん 14Hz になった（人は 4.56Hz）
  ふだんの入力のもとでは、ふだんの率で撃つ（送り手の率 × 拾う本数 が「ふだん届く数」）
  皮質から来る誤差（上丘・側坐核・来なかった）は、ふだん届く数を 0 とみる（たまにしか撃たない）⚠️【C】
  抑制は引き算【理論】Holt・Koch 1997 ／【ラット】Mitchell・Silver 2003（ずれの分だけ下がる）
  抑制の効く長さ PAUSE 歩【ヒト】Dostrovsky 2000（淡蒼球内節、GABA の放出で 10〜25ms 黙る）
  ⟲ 最初は「抑制が1つ届いたら PAUSE 歩黙る」にした。ふだん 26.6Hz の腹側淡蒼球から毎歩のように届き、黒質が止まったまま
     （Dostrovsky は多数の終末を一度に刺激した値。1本ずつの効きではない）
  G ＝ log(25/4) ÷ 拾う本数：拾った本数がそろって撃つと、ふだんの 6.25 倍
     【マカク】予期しない報酬でドーパミン細胞が 3〜5Hz → 20〜30Hz（Glimcher 2011。版40 の DA_CAP、【み】09-27 許可）
     ⚠️ 黒質の値を、ほかの核にもそのまま使う【C】
⚠️ この核は学ばない【C】。人の基底核にも可塑性はあるが、この版では皮質・側坐核の学びだけで足りるかを先に見る
"""
import math
import torch
from torch_area import DT_MS

HZ = {"黒質": 4.56, "視床下核": 36.0, "境": 33.0, "外側手綱核": 30.0, "吻内側被蓋核": 17.8}
PAUSE = 2          # 【ヒト】Dostrovsky 2000：10〜25ms ÷ 10.67ms
K_IN = 8           # ⚠️【C】受け手1体が拾う本数（torch_brain.K_NEW と同じ）
# 門：黒質の率を、ドーパミンが残る長さでならして、ふだんの率で割る【み選】割り算（§12-24）
#   長さ：【ヒト】Zaghloul 2009 黒質は 150〜375ms 上がる ⇒ 225ms ÷ 10.67ms ≈ 21歩（版40 の DA_DUR と同じ）
DA_TAU = int(round(225.0 / DT_MS))


EXC = {("上丘", "黒質"), ("来なかった", "視床下核"), ("視床下核", "境"), ("境", "外側手綱核"),
       ("外側手綱核", "吻内側被蓋核")}


class Midbrain:
    def __init__(self, n, dev="cpu", gen=None):
        """n: 核ごとの体数 {名前: 数}（黒質・視床下核・境・外側手綱核・吻内側被蓋核）"""
        self.n, self.dev, self.gen = dict(n), dev, gen
        self.p0 = {k: HZ[k] * DT_MS / 1000.0 for k in HZ}
        self.spk = {k: torch.zeros(self.n[k], device=dev) for k in HZ}
        self.inh_hist = {k: torch.zeros(PAUSE, self.n[k], device=dev) for k in HZ}
        self.G = math.log(25.0 / 4.0) / K_IN
        self.proj = {}
        self.da = torch.tensor(self.p0["黒質"], device=dev)   # ならした黒質の率（1歩あたり）

    def _pick(self, n_src, n_tgt):
        k = min(K_IN, n_src)
        return torch.stack([torch.randperm(n_src, device=self.dev, generator=self.gen)[:k] for _ in range(n_tgt)])

    def wire(self, n_sc_err, n_nac_err, n_omit):
        """外から来る送り手の数を受けて、線を張る（組むときに1回）"""
        p, n = self._pick, self.n
        self.proj = {
            ("上丘", "黒質"): p(n_sc_err, n["黒質"]),
            ("側坐核", "境"): p(n_nac_err, n["境"]),
            ("来なかった", "視床下核"): p(n_omit, n["視床下核"]),
            ("視床下核", "境"): p(n["視床下核"], n["境"]),
            ("境", "外側手綱核"): p(n["境"], n["外側手綱核"]),
            ("外側手綱核", "吻内側被蓋核"): p(n["外側手綱核"], n["吻内側被蓋核"]),
            ("吻内側被蓋核", "黒質"): p(n["吻内側被蓋核"], n["黒質"]),
        }
        # ふだん届く数：核から来る線は 送り手のふだんの確率 × 拾った本数。皮質から来る誤差は 0
        self.base_exc = {k: 0.0 for k in HZ}
        self.base_inh = {k: 0.0 for k in HZ}
        for (a, b), idx in self.proj.items():
            if a in HZ:
                (self.base_exc if (a, b) in EXC else self.base_inh)[b] += self.p0[a] * idx.shape[1]

    def _hits(self, src_spk, key):
        """受け手ごとに、拾った送り手のうち撃った数"""
        return src_spk[self.proj[key]].sum(1)

    @torch.no_grad()
    def step(self, sc_err, nac_err, omit):
        """1歩進める。入力は1歩前の外の発火。返り値：門（黒質のならした率 ÷ ふだんの率）"""
        s = self.spk
        src = {"上丘": sc_err, "側坐核": nac_err, "来なかった": omit, **s}
        exc = {k: torch.zeros(self.n[k], device=self.dev) for k in HZ}
        inh = {k: torch.zeros(self.n[k], device=self.dev) for k in HZ}
        for (a, b) in self.proj:
            h = self._hits(src[a], (a, b))
            (exc if (a, b) in EXC else inh)[b] += h
        new = {}
        for k in HZ:
            self.inh_hist[k] = torch.cat([inh[k].unsqueeze(0), self.inh_hist[k][:-1]])
            dev_ = self.G * (exc[k] - self.base_exc[k]) - self.G * (self.inh_hist[k].sum(0) - PAUSE * self.base_inh[k])
            prob = (self.p0[k] * torch.exp(dev_)).clamp(0.0, 1.0)
            new[k] = (torch.rand(self.n[k], device=self.dev, generator=self.gen) < prob).float()
        self.spk = new
        self.da = self.da + (new["黒質"].mean() - self.da) / DA_TAU
        return self.da / self.p0["黒質"]

    def rates_hz(self):
        return {k: float(v.mean()) * 1000.0 / DT_MS for k, v in self.spk.items()}
