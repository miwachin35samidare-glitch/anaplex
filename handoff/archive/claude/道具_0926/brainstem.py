"""脳幹（覚醒系）—— いまの状態を持って、全領野へ配る（版38）

デフォルメ：物質（ノルアドレナリン・アセチルコリン）では分けない。状態と、状態ごとの働きだけを配る
  【み】09-26「その判断に物質の種類とか要る？」「脳幹にある機能なら驚きは入れて」
設計の経緯・出どころは claude/設計案_脳幹.md

状態    起きている ／ N3 ／ REM。眠りは N3 と REM をくり返す（N3 から入る）
        いつ眠るかは day.py が決めて、ここへ伝える（自分で眠くなる形は後）
状態ごとの働き
  上からの戻り   起きている 1 ／ N3 0.18 倍 ／ REM 0.33 倍（上の段から下の樹状突起への線の辺の強さに掛ける）
                 【ヒト】眠ると音への上からの戻りの印（α・β帯の下がり）が N3 −82% ／ REM −67%
                 （頭蓋内、13人。Hayat ほか 2022 Nat Neurosci）。印を線の重さに当てるのは【C】【み選】09-26
                 海馬（CA1）→ 連合 の戻りには掛けない（【ヒト】N3 のリップルと紡錘波で内容を渡す）
  恒常性の下げ   眠りのあいだだけ（海馬は下げない ＝ sleep_mode）。【ヒト】起きていると強まり、眠ると戻る（TMS）
  徐波           N3 だけ（sleep.py）。REM は起きているときに近い（【ヒト】時間の窓 N3 +105ms ／ REM +16ms）
  学ぶ速さ       状態では変えない（【ヒト】どの状態でも学び・再生が記憶や上達と結びつく）
  驚き           起きているあいだ、全領野（皮質）の誤差の発火が いつもより多い歩で、全領野の学ぶ速さを上げる
                 【ヒト】驚くと新しい情報で考えを大きく改める（瞳孔、Nassar 2012、行動の段）
                 【マカク】青斑核は予期しない合図に 約90ms で短く撃つ（J Neurosci 1994）⇒ 8歩遅れて効かせる
                 ⚠️ 入口を「誤差の発火の合計」にするのは【C】。上げる量 SURPRISE_K・基準の長さ SURPRISE_TAU はノブ
  起こす         眠っているあいだ、約1秒（94歩）の聴覚の誤差の平均が、起きているときの WAKE_TH 倍を越えたら起こす
                 ⟲ 最初は「94歩の中の最大」で見ていたが、小さな網で揺らぎだけで起きた（09-26 夜の試し）
                 【ネズミ】大きな音で起きるのは青斑核を通る（Sci Adv 2020、題名まで）⚠️ 形と閾値 WAKE_TH は【C】のノブ
入れていないもの
  REM で抑制側に寄せる仕組み（結果として寄るかを測る【み選】）／ REM の短い出来事（目・体の段と一緒【み】）
記録として持つもの
  青斑核の段ごとの比 1 ／ 0.10 ／ 0.01（【ネズミ】Aston-Jones・Bloom 1981【み選】）。いまは何にも掛けていない
"""
import torch

WAKE, N3, REM = "起きている", "N3", "REM"
TOPDOWN_GAIN = {WAKE: 1.0, N3: 0.18, REM: 0.33}
LC_RATE = {WAKE: 1.0, N3: 0.10, REM: 0.01}          # 記録として（いまは使っていない）
# 眠りの長さ【み選】09-26：起きる:眠る を【ヒト】乳児の比（約 2.5:1）に。喃語800回（約25万歩）に対し
SLEEP_STEPS = 100_000
CYCLE_STEPS = 60_000         # 【ヒト】乳児の眠りの1周 40〜50分 を、起きている方と同じく約1/4 に縮めた
REM_FRAC = 0.5               # 【ヒト】新生児・乳児は眠りの半分が REM
SURPRISE_LAT = 8             # 【マカク】青斑核の短い発火まで 90.7ms（多細胞）÷ 10.67ms
SURPRISE_K = 1.0             # ⚠️ ノブ：いつもの何倍の誤差で、学ぶ速さを何倍にするか
SURPRISE_TAU = 2000.0        # ⚠️ ノブ：「いつも」を測る長さ（歩）
WAKE_TH = 3.0                # ⚠️ ノブ：眠っているとき、聴覚の誤差が起きているときの何倍で起こすか
WAKE_CHECK = 94              # 起こすかを見るのは約1秒ごと（毎歩 GPU から値を出すと遅くなるため）
CORTEX = ("聴覚", "運動", "連合", "高次")
# 上からの戻りの線：受け手の領野 → ext の中の位置（torch_brain.py の e の並び）
#   聴覚 ext ＝ [指令の写し, 連合から]  ／ 運動 ext ＝ [高次から] ／ 連合 ext ＝ [高次から, CA1 の戻り]
TOPDOWN = {"聴覚": ("連合", "after_cmd"), "運動": ("高次", "head"), "連合": ("高次", "head")}


class Brainstem:
    def __init__(self, brain):
        self.b, self.g, self.dev = brain, brain.g, brain.dev
        self.td_send = self._td_send(brain)
        self.state = None
        self.base = None             # 起きているときの、皮質の誤差の発火の率（移動平均）
        self.base_aud = None         # 同じく聴覚だけ
        self.hist = torch.zeros(SURPRISE_LAT + 1, device=self.dev)
        self.k = 0
        self.wake_request = False
        self._aud_sum = torch.zeros((), device=self.dev)
        self.set_state(WAKE)

    def _td_send(self, b):
        """上からの戻りを運ぶ送り手の番号（時間細胞の分と、短い遅れの分の両方）"""
        g = b.g
        names = list(b.areas.keys())
        ch_start = g.off_send[:-1]
        buf_base = int(g.off_send[-1])
        idx, b_off = [], 0
        for i, nm in enumerate(names):
            a = b.areas[nm]
            per = a.ch.n_cell + a.ch.n_ramp
            if nm in TOPDOWN:
                src, where = TOPDOWN[nm]
                n = b.n[src]
                first = a.n_ch + (b.n["運動"] if where == "after_cmd" else 0)
                rows = torch.arange(first, first + n, device=self.dev)
                idx.append((int(ch_start[i]) + rows.unsqueeze(1) * per
                            + torch.arange(per, device=self.dev)).reshape(-1))
                idx.append((buf_base + b_off + rows.unsqueeze(1) * a.D
                            + torch.arange(a.D, device=self.dev)).reshape(-1))
            b_off += a.n_src * a.D
        return torch.cat(idx)

    def set_state(self, s):
        """状態を変える。恒常性・海馬の扱い・上からの戻り・学ぶ速さをここだけで決める"""
        if s == self.state:
            return
        self.state = s
        g = self.g
        g.scaling_on = (s != WAKE)       # 下方調整は眠りのあいだだけ
        g.sleep_mode = (s != WAKE)       # 眠りでは海馬を下げない
        g.send_gain[self.td_send] = TOPDOWN_GAIN[s]
        g.eta_gain = torch.ones((), device=self.dev)
        self.wake_request = False
        self._aud_sum.zero_()

    def sleep_state(self, k):
        """眠りに入ってから k 歩目の状態（N3 から入り、1周の後半が REM）"""
        return N3 if (k % CYCLE_STEPS) < CYCLE_STEPS * (1 - REM_FRAC) else REM

    @torch.no_grad()
    def observe(self, b):
        """毎歩の終わりに呼ぶ。驚きと、起こすかを決める"""
        s = torch.cat([b.up[nm] for nm in CORTEX]).mean()
        aud = b.up["聴覚"].mean()
        self.hist = torch.cat([self.hist[1:], s.reshape(1)])
        if self.state == WAKE:
            if self.base is None:
                self.base, self.base_aud = s.clone(), aud.clone()
            a = 1.0 / SURPRISE_TAU
            self.base = self.base + a * (s - self.base)
            self.base_aud = self.base_aud + a * (aud - self.base_aud)
            ratio = self.hist[0] / self.base.clamp(min=1e-6)     # SURPRISE_LAT 歩前の誤差
            self.g.eta_gain = 1.0 + SURPRISE_K * (ratio - 1.0).clamp(min=0.0)
        else:
            self._aud_sum = self._aud_sum + aud
            self.k += 1
            if self.k % WAKE_CHECK == 0:
                if self.base_aud is not None and \
                        float(self._aud_sum) / WAKE_CHECK > WAKE_TH * float(self.base_aud.clamp(min=1e-6)):
                    self.wake_request = True
                self._aud_sum.zero_()

    def state_dict(self):
        return {"state": self.state, "base": self.base, "base_aud": self.base_aud}

    def load_state_dict(self, d):
        self.base, self.base_aud = d.get("base"), d.get("base_aud")
        self.state = None
        self.set_state(d.get("state") or WAKE)
