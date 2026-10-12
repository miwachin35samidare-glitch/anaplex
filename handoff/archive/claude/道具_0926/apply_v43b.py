"""版43a（＋持ち方 43m）→ 版43b：腹側淡蒼球と新しい門（arkhe の ~/anaplex で  python3 apply_v43b.py ）

設計と出どころは claude/設計案_脳幹.md §12-13・§12-14・§12-17
  腹側淡蒼球 20 体（⚠️ 人の体数 未取得・仮）。ふだん 34.8 回/秒で自分で撃つ（【マカク】J Neurosci 2019）。時間細胞なし【C】
  抑制で送り出す線（細胞体へ）：側坐核の誤差 → 腹側淡蒼球 ／ 腹側淡蒼球 → 背内側視床
    駆動と自発を 1 ＋ 届いた本数 で割る（短絡。⚠️ 形は【C】）
  側坐核の細胞体の入力：CA1 の表現 → CA1 の誤差（合う／合わないの食い違い。【ヒト】Duncan 2012）
  門（中脳）：1 ＋ 0.6 × 腹側淡蒼球の落ち込み（ふだんの率からの減り ÷ ふだんの率、18歩前）
    0.6 ＝【ヒト】Kamiński 2018（新しさで黒質がふだんの 0.6 倍ぶん多く撃つ）／ 遅れ 計20歩 ＝【ヒト】Kamiński 2018
    ⚠️ 腹側淡蒼球 → 中脳（抑制が外れて上がる）の形は【ラット】Lisman・Grace 2005。人・霊長類では未確認
    ⟲ 前の門（皮質の誤差 ÷ いつも、21歩の最大で保つ）は外した
【測】コンテナ（CPU、相手200本、耳に乱数 150歩）：腹側淡蒼球 36.7%（ふだん 37%）／ 門 平均 1.03〜1.11・最大 1.52
先に apply_v43a.py と apply_v43m.py を当てておくこと。印が1つでも見つからなければ何も書かずに止まる
state.pt は 43a から作っていないので VERSION は 43 のまま。回すなら --fresh
"""
import ast
import json
import shutil
import sys

HUNKS = json.loads(r'''{
"torch_group2.py": [
[
"            self.mean_map[c:c + a.n_ch, i] = 1.0 / a.n_ch\n        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])\n        self.complete = torch.cat([torch.full((a.n_ch,), float(a.complete), device=dev)\n                                   for a in self.areas])\n",
"            self.mean_map[c:c + a.n_ch, i] = 1.0 / a.n_ch\n        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])\n        # 細胞体に来る抑制の投射（版43b。側坐核 → 腹側淡蒼球、腹側淡蒼球 → 背内側視床）。届いた本数を持つ\n        #   駆動と自発の両方を 1 ＋ 本数 で割って弱める（短絡。⚠️ 形は【C】、学ぶ抑制と同じ割り算の形に揃えた）\n        self.x_inh = torch.zeros(self.n_ch_all, device=dev)\n        self.complete = torch.cat([torch.full((a.n_ch,), float(a.complete), device=dev)\n                                   for a in self.areas])\n"
],
[
"        #   抑制（短絡）はこの合計に掛かるので、ここで初めて表現の発火に効く\n        # 調整する側（上からの戻り・指令の写し）は予測 p には入るが、表現を撃たせない（版39）\n        drive = ((exc - exc_mod)[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])\n        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n        if self.complete.any():           # 入力が無いところは予測が足しになる（撃たせはしない）\n",
"        #   抑制（短絡）はこの合計に掛かるので、ここで初めて表現の発火に効く\n        # 調整する側（上からの戻り・指令の写し）は予測 p には入るが、表現を撃たせない（版39）\n        drive = ((exc - exc_mod)[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep] + self.x_inh)\n        rep = (drive > self.fire_th * self.th_gain).float()   # 版41：覚醒で閾値が下がる（感度）\n        if self.complete.any():           # 入力が無いところは予測が足しになる（撃たせはしない）\n"
],
[
"            rep = torch.where(fill, torch.ones_like(rep), rep)\n        rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,\n                                             generator=self.gen) < self.spont_p).float())\n\n        # 的：誤差と表現は自分の入口、抑制はその領野の入口の平均（ループなしで一度に）\n",
"            rep = torch.where(fill, torch.ones_like(rep), rep)\n        rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,\n                                             generator=self.gen) < self.spont_p / (1.0 + self.x_inh)).float())\n\n        # 的：誤差と表現は自分の入口、抑制はその領野の入口の平均（ループなしで一度に）\n"
]
],
"torch_brain.py": [
[
"#   ⇒ 一次 400（耳で決まる）なら 連合 531 ／ 高次 581\nNAMES = (\"視床\", \"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"歯状回\", \"CA3\", \"CA1\",\n         \"前頭前野\", \"背内側視床\", \"嗅内浅\", \"嗅内深\", \"側坐核\")   # 版43a：後ろの5つを足した（前の9つの並びは変えない）\n# 版43a の大きさ（設計案_脳幹.md §12-13）\n#   人の片側 100万体あたり 20 体（海馬の比。下限として置く【み】09-29「下限は置いて」）\n",
"#   ⇒ 一次 400（耳で決まる）なら 連合 531 ／ 高次 581\nNAMES = (\"視床\", \"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"歯状回\", \"CA3\", \"CA1\",\n         \"前頭前野\", \"背内側視床\", \"嗅内浅\", \"嗅内深\", \"側坐核\",   # 版43a：後ろの5つを足した（前の9つの並びは変えない）\n         \"腹側淡蒼球\")                                              # 版43b\n# 版43a の大きさ（設計案_脳幹.md §12-13）\n#   人の片側 100万体あたり 20 体（海馬の比。下限として置く【み】09-29「下限は置いて」）\n"
],
[
"#     連合 Te2 3.30億 ⇒ 240 ／ 高次 Te3 3.62億 ⇒ 263 ／ 前頭前野【ヒト】Gabi 2016 約13億 ⇒ 946\nEC_II = 20              # 嗅内浅 の頭 20 体が層 II（→ 歯状回・CA3）。残り 121 体が層 III（→ CA1）\nN_NEW = {\"前頭前野\": 946, \"背内側視床\": 129, \"嗅内浅\": 141, \"嗅内深\": 116, \"側坐核\": 146}\n# 新しい領野の細胞体への投射：受け手1体が送り手から K_NEW 本を拾い、3本そろって閾値を越える\n#   ⚠️ ノブ【C】。高次（【ヒト】上側頭回の峰 約8本・3本で越える）の置き方をそのまま使う。人の本数は未取得\n",
"#     連合 Te2 3.30億 ⇒ 240 ／ 高次 Te3 3.62億 ⇒ 263 ／ 前頭前野【ヒト】Gabi 2016 約13億 ⇒ 946\nEC_II = 20              # 嗅内浅 の頭 20 体が層 II（→ 歯状回・CA3）。残り 121 体が層 III（→ CA1）\nN_NEW = {\"前頭前野\": 946, \"背内側視床\": 129, \"嗅内浅\": 141, \"嗅内深\": 116, \"側坐核\": 146,\n         \"腹側淡蒼球\": 20}   # 版43b ⚠️ 人の体数 未取得（Pakkenberg 1990 の本文にある）。仮に 20\n# 腹側淡蒼球のふだんの発火（版43b）【マカク】J Neurosci 2019（2頭 102細胞）34.8 ± 1.8 回/秒。人の値は未取得\nVP_HZ = 34.8\n# 新しい領野の細胞体への投射：受け手1体が送り手から K_NEW 本を拾い、3本そろって閾値を越える\n#   ⚠️ ノブ【C】。高次（【ヒト】上側頭回の峰 約8本・3本で越える）の置き方をそのまま使う。人の本数は未取得\n"
],
[
"          \"CA3\": 2 * HUMAN_TH_CA3}    # ⚠️ ノブ。苔状線維1本で越える（下の苔状線維を参照）\nTGT_BY_FIRE = (\"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"CA3\",\n               \"前頭前野\", \"背内側視床\", \"嗅内浅\", \"嗅内深\", \"側坐核\")   # 版43a：足した領野も下からの入力で撃つか   # 誤差の的 ＝ 下からの入力で撃つか\n# 受け手1体の相手の数。既定は人の皮質（30,000シナプス ÷ 接触3.3 ＝ 9,100）\n#   【ヒト】CA3 は CA3 から約 16,100入力（棘 ≒ 送り手、接触ほぼ1つ。Watson 2025）\n",
"          \"CA3\": 2 * HUMAN_TH_CA3}    # ⚠️ ノブ。苔状線維1本で越える（下の苔状線維を参照）\nTGT_BY_FIRE = (\"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"CA3\",\n               \"前頭前野\", \"背内側視床\", \"嗅内浅\", \"嗅内深\", \"側坐核\", \"腹側淡蒼球\")   # 版43a：足した領野も下からの入力で撃つか   # 誤差の的 ＝ 下からの入力で撃つか\n# 受け手1体の相手の数。既定は人の皮質（30,000シナプス ÷ 接触3.3 ＝ 9,100）\n#   【ヒト】CA3 は CA3 から約 16,100入力（棘 ≒ 送り手、接触ほぼ1つ。Watson 2025）\n"
],
[
"                          嗅内浅=nn[\"嗅内深\"],         # 深い層 → 浅い層（層内）\n                          嗅内深=0,\n                          側坐核=nn[\"前頭前野\"])       # 前頭前野から（側坐核の予測）\n        # 上からの戻り（眠りで落とす線。脳幹が組むときに読むので、脳幹より先に置く）版43a で 前頭前野 → 高次・運動 を足した\n        self.topdown_rows = ((\"聴覚\", nn[\"聴覚\"] + nn[\"運動\"], nn[\"連合\"]),     # 連合から\n",
"                          嗅内浅=nn[\"嗅内深\"],         # 深い層 → 浅い層（層内）\n                          嗅内深=0,\n                          側坐核=nn[\"前頭前野\"],       # 前頭前野から（側坐核の予測）\n                          腹側淡蒼球=0)                # 版43b：樹状突起の入力なし\n        # 上からの戻り（眠りで落とす線。脳幹が組むときに読むので、脳幹より先に置く）版43a で 前頭前野 → 高次・運動 を足した\n        self.topdown_rows = ((\"聴覚\", nn[\"聴覚\"] + nn[\"運動\"], nn[\"連合\"]),     # 連合から\n"
],
[
"                       else 55.9 / 100.0 if nm in (\"歯状回\", \"CA3\", \"CA1\") else TAU[\"一次\"]),\n            gate=(nm != \"視覚\" or eye_open),\n            motor=(nm in (\"運動\", \"視床\", \"背内側視床\")),   # 視床は貯めずに中継する（時間細胞を持たない）\n            inh_frac=(0.275 if nm in (\"視床\", \"背内側視床\") else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n            complete=(nm in (\"CA3\", \"CA1\")),\n",
"                       else 55.9 / 100.0 if nm in (\"歯状回\", \"CA3\", \"CA1\") else TAU[\"一次\"]),\n            gate=(nm != \"視覚\" or eye_open),\n            motor=(nm in (\"運動\", \"視床\", \"背内側視床\", \"腹側淡蒼球\")),   # 視床は貯めずに中継する（時間細胞を持たない）\n            #   ⚠️ 腹側淡蒼球も時間細胞を持たせない（ふだん自分で撃つ細胞。【C】）\n            inh_frac=(0.275 if nm in (\"視床\", \"背内側視床\") else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n            complete=(nm in (\"CA3\", \"CA1\")),\n"
],
[
"            #    一度【マーモセット】の 8.7Hz を入れたら、音が背景に埋もれて supp が 1.43 → 1.04 に落ちた\n            #    ⇒ 皮質と同じ 0.3Hz に置く。これは人の値ではない。ノブ\n            spont_hz=(0.5 if nm in (\"歯状回\", \"CA3\", \"CA1\") else 0.3))\n        self.areas = {nm: mk(nm, i) for i, nm in enumerate(NAMES)}\n        self.g = Group2([self.areas[nm] for nm in NAMES])\n",
"            #    一度【マーモセット】の 8.7Hz を入れたら、音が背景に埋もれて supp が 1.43 → 1.04 に落ちた\n            #    ⇒ 皮質と同じ 0.3Hz に置く。これは人の値ではない。ノブ\n            spont_hz=(0.5 if nm in (\"歯状回\", \"CA3\", \"CA1\") else VP_HZ if nm == \"腹側淡蒼球\" else 0.3))\n        self.areas = {nm: mk(nm, i) for i, nm in enumerate(NAMES)}\n        self.g = Group2([self.areas[nm] for nm in NAMES])\n"
],
[
"        self.p_ca12nac = rp(h[\"CA1\"], nn[\"側坐核\"])\n        self.p_ecd2nac = rp(nn[\"嗅内深\"], nn[\"側坐核\"])\n        self.gen = g\n        self.up = {nm: torch.zeros(self.n[nm], device=dev) for nm in NAMES}\n",
"        self.p_ca12nac = rp(h[\"CA1\"], nn[\"側坐核\"])\n        self.p_ecd2nac = rp(nn[\"嗅内深\"], nn[\"側坐核\"])\n        # 版43b：抑制で送り出す線（細胞体へ。g.x_inh に本数を入れる）\n        #   側坐核 →（抑制）腹側淡蒼球  【ヒト】Pakkenberg 1990 が腹側淡蒼球を側坐核とともに数える ／ 送り出しは GABA（総説）\n        #     送るのは側坐核の誤差（予測されなかった分）。【ヒト】D'Ardenne 2008：腹側線条体は予測との差に沿う\n        #   腹側淡蒼球 →（抑制）背内側視床  （総説 2009。⚠️ 霊長類の線ごとの確かめはまだ）\n        self.p_nac2vp = rp(nn[\"側坐核\"], nn[\"腹側淡蒼球\"])\n        self.p_vp2md = rp(nn[\"腹側淡蒼球\"], nn[\"背内側視床\"])\n        self.vp_out = torch.zeros(nn[\"腹側淡蒼球\"], device=dev)\n        self._inh_md = slice(int(np.cumsum([0] + [nn[k] for k in NAMES])[NAMES.index(\"背内側視床\")]),\n                             int(np.cumsum([0] + [nn[k] for k in NAMES])[NAMES.index(\"背内側視床\") + 1]))\n        self._inh_vp = slice(int(np.cumsum([0] + [nn[k] for k in NAMES])[NAMES.index(\"腹側淡蒼球\")]),\n                             int(np.cumsum([0] + [nn[k] for k in NAMES])[NAMES.index(\"腹側淡蒼球\") + 1]))\n        self.gen = g\n        self.up = {nm: torch.zeros(self.n[nm], device=dev) for nm in NAMES}\n"
],
[
"            elif nm in (\"聴覚\", \"視覚\", \"連合\", \"高次\", \"前頭前野\", \"嗅内浅\", \"嗅内深\"):\n                gg.target_recv[ih] = hz(2.0)\n            elif nm in (\"背内側視床\", \"側坐核\"):      # 版43a：人の値なし ⇒ 下げない\n                gg.scale_skip[lo:hi] = True\n        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n",
"            elif nm in (\"聴覚\", \"視覚\", \"連合\", \"高次\", \"前頭前野\", \"嗅内浅\", \"嗅内深\"):\n                gg.target_recv[ih] = hz(2.0)\n            elif nm in (\"背内側視床\", \"側坐核\", \"腹側淡蒼球\"):      # 版43a・43b：人の値なし ⇒ 下げない\n                gg.scale_skip[lo:hi] = True\n        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n"
],
[
"            self._proj(self.up[\"高次\"], self.p_h2ecs) + self._proj(self.up[\"前頭前野\"], self.p_pfc2ecs),  # 嗅内浅\n            self._proj(self.ca1_out, self.p_ca12ecd),                                             # 嗅内深\n            self._proj(self.ca1_out, self.p_ca12nac) + self._proj(self.pred[\"嗅内深\"], self.p_ecd2nac)])  # 側坐核\n        if self.kick_left > 0:                                 # 運動視床の蹴り\n            # 視床の細胞が人の速さで撃つスパイクの列（毎歩 1 ではない）\n",
"            self._proj(self.up[\"高次\"], self.p_h2ecs) + self._proj(self.up[\"前頭前野\"], self.p_pfc2ecs),  # 嗅内浅\n            self._proj(self.ca1_out, self.p_ca12ecd),                                             # 嗅内深\n            # 側坐核：CA1 の誤差（合う／合わないの食い違い。【ヒト】Duncan 2012）＋ 嗅内深（版43b：CA1 は表現から誤差へ）\n            self._proj(self.up[\"CA1\"], self.p_ca12nac) + self._proj(self.pred[\"嗅内深\"], self.p_ecd2nac),\n            torch.zeros(self.n[\"腹側淡蒼球\"], device=self.dev)])                                  # 腹側淡蒼球（自分で撃つ）\n        # 抑制の投射（版43b）：1歩前の側坐核の誤差 → 腹側淡蒼球 ／ 1歩前の腹側淡蒼球 → 背内側視床\n        xi = self.g.x_inh\n        xi[self._inh_vp] = self._proj(self.up[\"側坐核\"], self.p_nac2vp)\n        xi[self._inh_md] = self._proj(self.vp_out, self.p_vp2md)\n        if self.kick_left > 0:                                 # 運動視床の蹴り\n            # 視床の細胞が人の速さで撃つスパイクの列（毎歩 1 ではない）\n"
],
[
"                       pf,                                     # 背内側視床（前頭前野から。調整する側）\n                       ed,                                     # 嗅内浅（深い層から）\n                       pf])                                    # 側坐核（前頭前野から）\n\n        outs = self.g.step(x, e, learn)\n",
"                       pf,                                     # 背内側視床（前頭前野から。調整する側）\n                       ed,                                     # 嗅内浅（深い層から）\n                       pf])                                    # 側坐核（前頭前野から）。腹側淡蒼球は ext なし\n\n        outs = self.g.step(x, e, learn)\n"
],
[
"        self.pred[\"嗅内深\"] = cut(\"嗅内深\", rep)\n        self.md_out = cut(\"背内側視床\", rep)              # 視床は表現を送る（中継）\n        # CA3 のシナプスは確実に伝わる（【ヒト】0.90。皮質の 0.33 との差が速く覚える理由）\n        c3 = cut(\"CA3\", rep)\n",
"        self.pred[\"嗅内深\"] = cut(\"嗅内深\", rep)\n        self.md_out = cut(\"背内側視床\", rep)              # 視床は表現を送る（中継）\n        self.vp_out = cut(\"腹側淡蒼球\", rep)              # 版43b\n        # CA3 のシナプスは確実に伝わる（【ヒト】0.90。皮質の 0.33 との差が速く覚える理由）\n        c3 = cut(\"CA3\", rep)\n"
]
],
"brainstem.py": [
[
"DA_DUR = int(round(225.0 / DT_MS))       # 21歩\nDA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）\n# 驚きで学ぶ速さを上げる（青斑核。版38 の形を版42 で戻した）【ヒト】Nassar 2012 ⚠️ 量は【C】のノブ\nLC_K = 1.0\n",
"DA_DUR = int(round(225.0 / DT_MS))       # 21歩\nDA_CAP = 25.0 / 4.0                      # 門の上限（ふだんの 6.25 倍）\n# 版43b：門の入口を腹側淡蒼球に（設計案_脳幹.md §12-13・§12-14）\n#   中脳は腹側淡蒼球の抑制が外れると上がる（⚠️【ラット】Lisman・Grace 2005 の形。人・霊長類ではこの線を見つけていない）\n#   門 ＝ 1 ＋ DA_NOV × 落ち込み、落ち込み ＝（ふだんの率 − 今の率）÷ ふだんの率（0〜1。ふだんの率は定数【マカク】34.8Hz）\n#   DA_NOV：【ヒト】Kamiński 2018 黒質の新しさの細胞は、新しい絵でふだんの 0.60 倍ぶん多く撃つ ⇒ 腹側淡蒼球が黙りきったとき 1.6\n#   遅れ：【ヒト】Kamiński 2018 内側側頭葉 311ms → 黒質 527ms（約20歩）。CA1 → 側坐核 → 腹側淡蒼球 で2歩 ⇒ 残り 18歩\nDA_NOV = 0.60\nVP_LAT = 18\n# 驚きで学ぶ速さを上げる（青斑核。版38 の形を版42 で戻した）【ヒト】Nassar 2012 ⚠️ 量は【C】のノブ\nLC_K = 1.0\n"
],
[
"        self._fmask = None\n        self.hist = torch.zeros(max(SURPRISE_LAT, DA_LAT) + 1, device=self.dev)   # 版40：中脳の遅れの分まで持つ\n        self.da_buf = torch.zeros(DA_DUR, device=self.dev)   # 中脳の短い発火が続く長さ（版40）\n        self.surprise = torch.zeros((), device=self.dev)\n        self.lc_s = torch.ones((), device=self.dev)       # 青斑核の短い平均（版41）\n",
"        self._fmask = None\n        self.hist = torch.zeros(max(SURPRISE_LAT, DA_LAT) + 1, device=self.dev)   # 版40：中脳の遅れの分まで持つ\n        self.da_buf = torch.zeros(DA_DUR, device=self.dev)   # 中脳の短い発火が続く長さ（版40。版43b から使わない）\n        self.vp_hist = None                                   # 腹側淡蒼球の率の列（版43b）\n        self.surprise = torch.zeros((), device=self.dev)\n        self.lc_s = torch.ones((), device=self.dev)       # 青斑核の短い平均（版41）\n"
],
[
"            # 学びの門（版40）：中脳は DA_LAT 歩前の驚きに比例して撃ち、DA_DUR 歩続く。門 ＝ 1 ＋ その分（上限 DA_CAP）\n            #   ⚠️ 驚きの入口（皮質の誤差の平均）と、比例の係数 SURPRISE_K は【C】のノブのまま\n            r_da = self.hist[0] / self.base.clamp(min=1e-6)\n            da = SURPRISE_K * (r_da - 1.0).clamp(min=0.0) if ready else torch.zeros_like(r_da)\n            self.da_buf = torch.cat([self.da_buf[1:], da.reshape(1)])\n            self.g.third = (1.0 + self.da_buf.max()).clamp(max=DA_CAP)\n            self.gate_sum = self.gate_sum + self.g.third\n            self.gate_n += 1\n",
"            # 学びの門（版40）：中脳は DA_LAT 歩前の驚きに比例して撃ち、DA_DUR 歩続く。門 ＝ 1 ＋ その分（上限 DA_CAP）\n            #   ⚠️ 驚きの入口（皮質の誤差の平均）と、比例の係数 SURPRISE_K は【C】のノブのまま\n            #   ⟲ 版43b：入口を「皮質の誤差 ÷ いつも」から 腹側淡蒼球の落ち込み に替えた。21歩の最大で保つ作りも外した\n            #      （版40〜42 の門はほぼずっと 2〜3 倍に開いていた。Claude の作りの誤り）\n            if getattr(b, \"vp_out\", None) is not None:\n                vp0 = float(b.areas[\"腹側淡蒼球\"].spont_p)\n                v = b.vp_out.mean().reshape(1)\n                if self.vp_hist is None:\n                    self.vp_hist = torch.full((VP_LAT + 1,), vp0, device=self.dev)\n                self.vp_hist = torch.cat([self.vp_hist[1:], v])\n                drop = ((vp0 - self.vp_hist[0]) / vp0).clamp(0.0, 1.0)\n                self.g.third = 1.0 + DA_NOV * drop\n            else:\n                self.g.third = torch.ones((), device=self.dev)\n            self.gate_sum = self.gate_sum + self.g.third\n            self.gate_n += 1\n"
]
]
}''')
out = {}
for f, hs in HUNKS.items():
    s = open(f, encoding="utf-8").read()
    for i, (old, new) in enumerate(hs):
        c = s.count(old)
        if c != 1:
            sys.exit(f"⚠️ {f} の {i+1} 番目の印が {c} 回。何も書いていない（もう当ててあるか、43a・43m がまだ）\n--- 探した行 ---\n{old}")
        s = s.replace(old, new)
    ast.parse(s)
    out[f] = s
for f, s in out.items():
    shutil.copy(f, f + ".v43m")
    open(f, "w", encoding="utf-8").write(s)
print("版43b にした（元は *.v43m に残した）")
print("次：python3 measure_size.py（腹側淡蒼球の分）→ python3 measure_v42.py full 100")
