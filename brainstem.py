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
from torch_area import DT_MS

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
# 学びの門（版40）設計案 §8-6〜8-8。門 ＝ ふだんの分 1 ＋ 驚きに比例する上乗せ（学んだ跡に掛かる）
#   【ヒト】黒質のドーパミン細胞は予期しない得の 150〜375ms 後に上がる（Zaghloul 2009）⇒ 遅れ 14歩・長さ 21歩
#   【マカク】予期しない報酬で 3〜5Hz → 20〜30Hz、予測誤差に比例（Glimcher 2011）⇒ 上限はふだんの 25/4 倍【み】09-27 許可
DA_LAT = int(round(150.0 / DT_MS))       # 14歩
DA_DUR = int(round(225.0 / DT_MS))       # 21歩
DA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）
# 驚きで学ぶ速さを上げる（青斑核。版38 の形を版42 で戻した）【ヒト】Nassar 2012 ⚠️ 量は【C】のノブ
LC_K = 1.0
# 覚醒（版41）設計案 §9
#   【リスザル】青斑核は起きているあいだ ぼんやり 0.5Hz 〜 好物 15Hz、ふだん 2.45Hz（Foote 1980）【み】09-27 許可
LC_MIN, LC_MAX = 0.5 / 2.45, 15.0 / 2.45
#   【ラット】NA で 3.4mV 脱分極（Grzelka 2017）÷【ヒト】静止から閾値 16mV（Hunt 2023）【み】09-27 許可
NA_DEPOL = 3.4 / 16.0
AROUSAL_TAU = SURPRISE_TAU               # ⚠️ ノブ：覚醒の短い平均の長さ（歩）
# 眠りの圧（版39c）【み選】09-27 C（時間 ＋ 疲れ）・年齢は脳幹が持つ・疲れは深さと眠くなる早さの両方
COMPRESS = 4.0              # 起きる・眠るを人の約1/4 に縮めた（【み選】09-26。版38 の眠りの長さと同じ）
AGE0_MONTH = 6.0            # 喃語が出るころ（【ヒト】6〜9か月）
# (月齢から, まで, 起きている時間, 1回の眠りの時間)
#   【ヒト】6〜9か月 起きている 2.5〜3.5時間 ／ 昼寝 1〜1.5時間（生物台帳）。ほかの月齢は未取得
AGE_TABLE = ((6.0, 9.0, 3.0, 1.25),)
FATIGUE_AREAS = ("聴覚", "連合", "高次", "前頭前野")   # 版43a：前頭前野を足した   # 疲れを読む皮質（運動は目標なし・視覚は目を閉じている）
_STEP_DAYS = DT_MS * COMPRESS / 1000.0 / 86400.0


def _hours_to_steps(h):
    return int(h * 3600.0 * 1000.0 / DT_MS / COMPRESS)
CORTEX = ("聴覚", "運動", "連合", "高次", "前頭前野")   # 版43a：前頭前野を足した
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
        self.n_wake = 0              # 起きていた歩数（基準ができたかを見る。版39）
        self.age_days = AGE0_MONTH * 30.44   # 年齢（版39c）。起きている・眠っている歩ごとに進む
        self.pressure = 0.0          # 眠りの圧。1 に届いたら眠る
        self.fatigue = 0.0           # 疲れ（皮質の 率÷目標 が起きてから増えた割合）
        self.r0 = None               # 疲れの基準（起きて homeo_tau 歩たってからの 率÷目標）
        self.k_wake = 0              # 起きてからの歩数
        self._fmask = None
        self.hist = torch.zeros(max(SURPRISE_LAT, DA_LAT) + 1, device=self.dev)   # 版40：中脳の遅れの分まで持つ
        self.da_buf = torch.zeros(DA_DUR, device=self.dev)   # 中脳の短い発火が続く長さ（版40）
        self.surprise = torch.zeros((), device=self.dev)
        self.lc_s = torch.ones((), device=self.dev)       # 青斑核の短い平均（版41）
        self.lc_l = torch.ones((), device=self.dev)       # 起きているあいだの長い平均
        self.arousal = torch.ones((), device=self.dev)    # 覚醒 A（ふだん 1）
        self.lc_on = False
        self.gate_sum = torch.zeros((), device=self.dev)     # 門の開き（測る用）
        self.gate_n = 0
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
        # 版43a：線の表は torch_brain.topdown_rows（受け手, 最初の行, 行の数）から読む
        td = {}
        for nm, first, n in getattr(b, "topdown_rows", ()):
            td.setdefault(nm, []).append((first, n))
        if not td:
            for nm, (src, where) in TOPDOWN.items():
                td[nm] = [(b.n[nm] + (b.n["運動"] if where == "after_cmd" else 0), b.n[src])]
        for i, nm in enumerate(names):
            a = b.areas[nm]
            per = a.ch.n_cell + a.ch.n_ramp
            for first, n in td.get(nm, ()):
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
        if s == WAKE and self.state in (N3, REM):   # 眠りから起きた（版39c）：眠りの圧を下ろし、疲れを測り直す
            self.pressure, self.fatigue, self.r0, self.k_wake = 0.0, 0.0, None, 0
        self.state = s
        g = self.g
        g.scaling_on = (s != WAKE)       # 下方調整は眠りのあいだだけ
        g.sleep_mode = (s != WAKE)       # 眠りでは海馬を下げない
        g.send_gain[self.td_send] = TOPDOWN_GAIN[s]
        g.eta_gain = torch.ones((), device=self.dev)
        g.third = torch.ones((), device=self.dev)   # 学びの門（版40）：ふだんの分。眠りは上乗せなし（学ぶ速さは状態で変えない）
        self.da_buf.zero_()
        # 覚醒（版41）：眠りは段の比（【ネズミ】【み選】09-26）。起きたら observe が決める
        self.arousal = torch.tensor(max(LC_RATE[s], LC_MIN), device=self.dev)
        self._set_gain()
        self.wake_request = False
        self._aud_sum.zero_()

    def _row(self):
        m = self.age_days / 30.44
        for lo, hi, w, sl in AGE_TABLE:
            if lo <= m < hi:
                return w, sl
        row = AGE_TABLE[0] if m < AGE_TABLE[0][0] else AGE_TABLE[-1]   # ⚠️ 表の外は端の値
        return row[2], row[3]

    def wake_steps(self):
        """いまの年齢で、疲れが無いときに起きていられる歩数"""
        return _hours_to_steps(self._row()[0])

    def sleep_steps(self):
        """いまの年齢の1回の眠りの歩数"""
        return _hours_to_steps(self._row()[1])

    @property
    def sleep_due(self):
        """眠りの圧が覚醒を上回ったら眠る（版41。落ち着く・退屈で A が下がると早く、驚きが続くと遅く）"""
        return self.pressure >= float(self.arousal)

    def _ratio(self):
        g = self.g
        if self._fmask is None:
            m = torch.zeros(g.n_recv_all, dtype=torch.bool, device=self.dev)
            for i, nm in enumerate(self.b.areas.keys()):
                if nm in FATIGUE_AREAS:
                    m[int(g.off_recv[i]):int(g.off_recv[i + 1])] = True
            self._fmask = m & ~g.scale_skip
        return float((g.rate_ema[self._fmask] / g.target_recv[self._fmask]).mean())

    def _press(self, n):
        """起きているあいだの眠りの圧。(1 ＋ 疲れ) ÷ 起きていられる歩数 ずつたまる"""
        if self.k_wake >= self.g.homeo_tau:     # 率は約2000歩の移動平均。起きた直後は眠りの値を引きずる
            r = self._ratio()
            if self.r0 is None:
                self.r0 = r
            self.fatigue = max(0.0, r / max(self.r0, 1e-9) - 1.0)
        self.pressure += n * (1.0 + self.fatigue) / self.wake_steps()

    def _set_gain(self):
        """覚醒 A に応じて表現の閾値を下げる（最大で【ラット】3.4mV ÷【ヒト】16mV。青斑核の率に比例【C】）"""
        a = (self.arousal - 1.0) / (LC_MAX - 1.0)
        self.g.th_gain = 1.0 - NA_DEPOL * a

    def gate_mean(self):
        """前に呼んでから起きていたあいだの、門の開きの平均（測る用。呼ぶと 0 に戻す。ふだんは 1）"""
        m = float(self.gate_sum) / max(self.gate_n, 1)
        self.gate_sum.zero_()
        self.gate_n = 0
        return m

    def sleep_state(self, k):
        """眠りに入ってから k 歩目の状態（N3 から入り、1周の後半が REM）"""
        return N3 if (k % CYCLE_STEPS) < CYCLE_STEPS * (1 - REM_FRAC) else REM

    @torch.no_grad()
    def observe(self, b):
        """毎歩の終わりに呼ぶ。驚きと、起こすかを決める"""
        self.age_days += _STEP_DAYS
        if self.state == WAKE:                  # 眠りの圧（版39c）。値を出すのは約1秒ごと
            self.k_wake += 1
            if self.k_wake % WAKE_CHECK == 0:
                self._press(WAKE_CHECK)
        s = torch.cat([b.up[nm] for nm in CORTEX]).mean()
        aud = b.up["聴覚"].mean()
        self.hist = torch.cat([self.hist[1:], s.reshape(1)])
        if self.state == WAKE:
            # ⟲ 版38 は最初の歩の誤差（組んだ直後は 0）を基準にしたので、学ぶ速さが最初の数千歩 数十〜280倍になった
            #   ⇒ SURPRISE_TAU 歩までは起きてからの平均で基準を作り、驚きは効かせない【C】（版39）
            if self.base is None:
                self.base, self.base_aud = torch.zeros_like(s), torch.zeros_like(aud)
            self.n_wake += 1
            a = max(1.0 / SURPRISE_TAU, 1.0 / self.n_wake)
            self.base = self.base + a * (s - self.base)
            self.base_aud = self.base_aud + a * (aud - self.base_aud)
            ratio = self.hist[-1 - SURPRISE_LAT] / self.base.clamp(min=1e-6)     # SURPRISE_LAT 歩前の誤差（青斑核）
            ready = self.n_wake >= SURPRISE_TAU        # 基準ができるまでは驚きを出さない（版39）
            # 驚き（青斑核）：覚醒（感度・眠気の覆い）に使う（版41）。学ぶ速さにも掛ける（版42 で戻した）
            self.surprise = (ratio - 1.0).clamp(min=0.0) if ready else torch.zeros_like(ratio)
            # 驚きで、その歩の学ぶ速さを上げる（青斑核・8歩遅れ。版42 で戻した。版40 で外したのは【C】の誤り）
            self.g.eta_gain = 1.0 + LC_K * self.surprise
            # 覚醒（版41）：青斑核のふだんの高さ ＝ 皮質の誤差（予期しない入力の量。8歩遅れ）の短い平均 ÷ 長い平均
            #   何も起きない・聞き慣れた音ばかりだと下がり、予期しない音が続くと上がる（入口を誤差にするのは【C】）
            if ready:
                lc = self.hist[-1 - SURPRISE_LAT]
                if not self.lc_on:                       # 基準ができた歩で、どちらも「いつも」から始める
                    self.lc_s, self.lc_l, self.lc_on = self.base.clone(), self.base.clone(), True
                self.lc_s = self.lc_s + (lc - self.lc_s) / AROUSAL_TAU
                self.lc_l = self.lc_l + (lc - self.lc_l) / max(self.wake_steps(), 1)
                self.arousal = (self.lc_s / self.lc_l.clamp(min=1e-6)).clamp(LC_MIN, LC_MAX)
            else:
                self.arousal = torch.ones((), device=self.dev)
            self._set_gain()
            # 学びの門（版43b）：黒質のドーパミン細胞の率（ならしたもの）÷ ふだんの率（midbrain.py）【み選】割り算
            #   ⟲ 版40〜42 は「皮質の誤差 ÷ いつも」を入口にし、21歩の最大で保った。門はほぼずっと 2〜3 倍に開いていた（【C】の誤り）
            gate = getattr(b, "gate", None)
            self.g.third = gate.reshape(()) if gate is not None else torch.ones((), device=self.dev)
            self.gate_sum = self.gate_sum + self.g.third
            self.gate_n += 1
        else:
            self._aud_sum = self._aud_sum + aud
            self.k += 1
            if self.k % WAKE_CHECK == 0:
                if self.base_aud is not None and \
                        float(self._aud_sum) / WAKE_CHECK > WAKE_TH * float(self.base_aud.clamp(min=1e-6)):
                    self.wake_request = True
                self._aud_sum.zero_()

    def state_dict(self):
        return {"state": self.state, "base": self.base, "base_aud": self.base_aud, "n_wake": self.n_wake,
                "age_days": self.age_days, "pressure": self.pressure, "fatigue": self.fatigue,
                "r0": self.r0, "k_wake": self.k_wake, "lc_s": self.lc_s, "lc_l": self.lc_l, "lc_on": self.lc_on}

    def load_state_dict(self, d):
        self.base, self.base_aud = d.get("base"), d.get("base_aud")
        self.n_wake = d.get("n_wake", 0)
        self.age_days = d.get("age_days", self.age_days)
        self.pressure, self.fatigue = d.get("pressure", 0.0), d.get("fatigue", 0.0)
        self.r0, self.k_wake = d.get("r0"), d.get("k_wake", 0)
        if "lc_s" in d:
            self.lc_s, self.lc_l, self.lc_on = d["lc_s"], d["lc_l"], d.get("lc_on", True)
        self.state = None
        self.set_state(d.get("state") or WAKE)
