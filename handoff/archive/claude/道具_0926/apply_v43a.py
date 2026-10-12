"""版42 → 版43a：前頭前野・背内側視床・嗅内皮質（浅い層／深い層）・側坐核を足す（arkhe の ~/anaplex で  python3 apply_v43a.py ）

設計と出どころは claude/設計案_脳幹.md §12（特に §12-10・§12-13・§12-14）
  大きさ  海馬と報酬の輪は人の片側 100万体あたり 20（下限【み】09-29）／ 連合 240・高次 263・前頭前野 946（比で縮める）
  線      高次 → 前頭前野 ／ 背内側視床 → 前頭前野 ／ 高次・前頭前野 → 嗅内浅 ／ 嗅内浅 II → 歯状回・CA3 ／
          嗅内浅 III → CA1（樹状突起。前の「1歩前の CA3」を置き換え）／ CA1 → 嗅内深 ／ 嗅内深 → 連合・前頭前野・嗅内浅 ／
          CA1・嗅内深 → 側坐核 ／ 前頭前野 → 側坐核・背内側視床・高次・運動（樹状突起）
  上からの戻り（眠りで落とす線）に 前頭前野 → 高次・運動 を足した（脳幹は torch_brain.topdown_rows を読む）
  歯状回：同点を乱数で分けて、ちょうど上位 2.5% だけ撃つ（【測】同点で最大 300 体すべて撃った）
  ⚠️ 門（中脳）はまだ今のまま（皮質の誤差）。腹側淡蒼球と新しい門は 43b
  ⚠️ 前頭前野 → 背内側視床 は樹状突起に入れた（細胞体だと輪で両方が撃ちっぱなし【測】）。背内側視床の駆動は 43b で決める
  ⚠️ ノブ【C】：新しい領野の細胞体の入力は 1体 8本・3本で閾値（高次の置き方）／ 時定数は高次と同じ
網の作りが変わるので VERSION 43。state.pt は読めない ⇒ --fresh
印が1つでも見つからなければ何も書かずに止まる（コンテナの写しと arkhe のファイルが違うときも止まる）
"""
import ast
import json
import shutil
import sys

HUNKS = json.loads(r'''{
"torch_brain.py": [
[
"  海馬   歯状回 → CA3 → CA1（比 15 : 2.7 : 16）\n  視床   感覚を中継し、運動を蹴り込む（駆動する入力）\n\n繋ぎ方\n",
"  海馬   歯状回 → CA3 → CA1（比 15 : 2.7 : 16）\n  視床   感覚を中継し、運動を蹴り込む（駆動する入力）\n  版43a（09-29【み】）前頭前野・背内側視床・嗅内皮質（浅い層／深い層）・側坐核を足した。大きさは 設計案_脳幹.md §12-13\n    海馬と報酬の輪は人の片側 100万体あたり 20 体（下限）／ 連合・高次・前頭前野は人の比で 12GB に収まるよう縮めた\n    線は §12-10・§12-13（種は線ごと）\n\n繋ぎ方\n"
],
[
"#     Tpt は Te2.1・Te3 に当たる（Zachlod ほか 2020 表1）\n#   ⇒ 一次 400（耳で決まる）なら 連合 531 ／ 高次 581\nNAMES = (\"視床\", \"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"歯状回\", \"CA3\", \"CA1\")\nHIPPO_RATIO = {\"歯状回\": 15.0, \"CA3\": 2.7, \"CA1\": 16.0}     # 【ヒト】片側の体数の比\nSPARSE = 0.025          # 【ヒト】海馬で1つの項目に反応する割合（2.5%未満）\n",
"#     Tpt は Te2.1・Te3 に当たる（Zachlod ほか 2020 表1）\n#   ⇒ 一次 400（耳で決まる）なら 連合 531 ／ 高次 581\nNAMES = (\"視床\", \"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"歯状回\", \"CA3\", \"CA1\",\n         \"前頭前野\", \"背内側視床\", \"嗅内浅\", \"嗅内深\", \"側坐核\")   # 版43a：後ろの5つを足した（前の9つの並びは変えない）\n# 版43a の大きさ（設計案_脳幹.md §12-13）\n#   人の片側 100万体あたり 20 体（海馬の比。下限として置く【み】09-29「下限は置いて」）\n#     嗅内皮質【ヒト】Wegiel 2014 表3：層 II（島）102万・III 607万 ⇒ 浅 20＋121 ／ V 201万・VI 381万 ⇒ 深 40＋76\n#     側坐核【ヒト】Wegiel 2014：729万 ⇒ 146 ／ 背内側視床【ヒト】Abitz 2007：643万 ⇒ 129\n#   人の両半球 100万体あたり 0.728 体（比で縮める側。12GB に収める係数【C】。組んだら measure_size で直す）\n#     連合 Te2 3.30億 ⇒ 240 ／ 高次 Te3 3.62億 ⇒ 263 ／ 前頭前野【ヒト】Gabi 2016 約13億 ⇒ 946\nEC_II = 20              # 嗅内浅 の頭 20 体が層 II（→ 歯状回・CA3）。残り 121 体が層 III（→ CA1）\nN_NEW = {\"前頭前野\": 946, \"背内側視床\": 129, \"嗅内浅\": 141, \"嗅内深\": 116, \"側坐核\": 146}\n# 新しい領野の細胞体への投射：受け手1体が送り手から K_NEW 本を拾い、3本そろって閾値を越える\n#   ⚠️ ノブ【C】。高次（【ヒト】上側頭回の峰 約8本・3本で越える）の置き方をそのまま使う。人の本数は未取得\nK_NEW = 8\nX_NEW = 16.0 / 1.12 / 2.0\nHIPPO_RATIO = {\"歯状回\": 15.0, \"CA3\": 2.7, \"CA1\": 16.0}     # 【ヒト】片側の体数の比\nSPARSE = 0.025          # 【ヒト】海馬で1つの項目に反応する割合（2.5%未満）\n"
],
[
"X_GAIN = {\"聴覚\": HUMAN_TH / 8.5, \"運動\": 2 * HUMAN_TH, \"高次\": HUMAN_TH / 2.0,\n          \"CA3\": 2 * HUMAN_TH_CA3}    # ⚠️ ノブ。苔状線維1本で越える（下の苔状線維を参照）\nTGT_BY_FIRE = (\"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"CA3\")   # 誤差の的 ＝ 下からの入力で撃つか\n# 受け手1体の相手の数。既定は人の皮質（30,000シナプス ÷ 接触3.3 ＝ 9,100）\n#   【ヒト】CA3 は CA3 から約 16,100入力（棘 ≒ 送り手、接触ほぼ1つ。Watson 2025）\n",
"X_GAIN = {\"聴覚\": HUMAN_TH / 8.5, \"運動\": 2 * HUMAN_TH, \"高次\": HUMAN_TH / 2.0,\n          \"CA3\": 2 * HUMAN_TH_CA3}    # ⚠️ ノブ。苔状線維1本で越える（下の苔状線維を参照）\nTGT_BY_FIRE = (\"聴覚\", \"運動\", \"視覚\", \"連合\", \"高次\", \"CA3\",\n               \"前頭前野\", \"背内側視床\", \"嗅内浅\", \"嗅内深\", \"側坐核\")   # 版43a：足した領野も下からの入力で撃つか   # 誤差の的 ＝ 下からの入力で撃つか\n# 受け手1体の相手の数。既定は人の皮質（30,000シナプス ÷ 接触3.3 ＝ 9,100）\n#   【ヒト】CA3 は CA3 から約 16,100入力（棘 ≒ 送り手、接触ほぼ1つ。Watson 2025）\n"
],
[
"\nclass Brain:\n    def __init__(self, prior, n_ear=40, n_cmd=1812, n_eye=40, n_assoc=530, n_high=580,\n                 hippo_scale=4.0, n_partner=9100, n_cell=128, eta=0.3,\n                 stp_path=\"synapse_model_human.json\", dev=\"cuda\", seed=0, eye_open=False):\n",
"\nclass Brain:\n    def __init__(self, prior, n_ear=40, n_cmd=1812, n_eye=40, n_assoc=240, n_high=263,\n                 hippo_scale=4.0, n_partner=9100, n_cell=128, eta=0.3,\n                 stp_path=\"synapse_model_human.json\", dev=\"cuda\", seed=0, eye_open=False):\n"
],
[
"        self.n = dict(視床=n_ear, 聴覚=n_ear, 運動=n_cmd, 視覚=n_eye,\n                      連合=n_assoc, 高次=n_high,\n                      歯状回=h[\"歯状回\"], CA3=h[\"CA3\"], CA1=h[\"CA1\"])\n        # ext（樹状突起に入る調整する入力）の数\n        self.n_ext = dict(視床=0,                    # 中継。樹状突起には何も来ない\n                          聴覚=n_cmd + n_assoc,      # 指令の写し ＋ 連合からの予測\n                          運動=n_high,                # 高次からの予測\n                          視覚=0,\n                          連合=n_high + h[\"CA1\"],     # 高次からの予測 ＋ 海馬（CA1）の戻り\n                          高次=n_cmd,                 # 運動の誤差（丸ごと渡す。樹状突起側は配線を通る）\n                          歯状回=n_high,              # 皮質から\n                          CA3=h[\"歯状回\"] + h[\"CA3\"],  # 歯状回 ＋ 1歩前の自分\n                          CA1=h[\"CA3\"])\n        mk = lambda nm, i: Area(\n            n_ch=self.n[nm], prior=prior, n_ext=self.n_ext[nm], eta=eta,\n            n_partner=N_PARTNER.get(nm, n_partner), n_cell=n_cell, stp_path=stp_path, dev=dev, seed=seed + i,\n            tau_scale=(TAU[\"連合\"] if nm == \"連合\" else TAU[\"高次\"] if nm == \"高次\"\n                       else 55.9 / 100.0 if nm in (\"歯状回\", \"CA3\", \"CA1\") else TAU[\"一次\"]),\n            gate=(nm != \"視覚\" or eye_open),\n            motor=(nm in (\"運動\", \"視床\")),         # 視床は貯めずに中継する（時間細胞を持たない）\n            inh_frac=(0.275 if nm == \"視床\" else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n            complete=(nm in (\"CA3\", \"CA1\")),\n            # 疎さは段で違う（【ヒト】）\n",
"        self.n = dict(視床=n_ear, 聴覚=n_ear, 運動=n_cmd, 視覚=n_eye,\n                      連合=n_assoc, 高次=n_high,\n                      歯状回=h[\"歯状回\"], CA3=h[\"CA3\"], CA1=h[\"CA1\"], **N_NEW)\n        nn = self.n\n        # ext（樹状突起に入る調整する入力）の数\n        self.n_ext = dict(視床=0,                    # 中継。樹状突起には何も来ない\n                          聴覚=n_cmd + n_assoc,      # 指令の写し ＋ 連合からの予測\n                          運動=n_high + nn[\"前頭前野\"],   # 高次からの予測 ＋ 前頭前野から（版43a）\n                          視覚=0,\n                          連合=n_high + nn[\"嗅内深\"],  # 高次からの予測 ＋ 海馬の戻り（版43a：CA1 → 嗅内深 を通して）\n                          高次=n_cmd + nn[\"前頭前野\"],  # 運動の誤差 ＋ 前頭前野から（版43a）\n                          歯状回=EC_II,               # 嗅内浅 層 II の誤差（版43a。前は高次の誤差）\n                          CA3=h[\"歯状回\"] + h[\"CA3\"] + EC_II,   # 歯状回 ＋ 1歩前の自分 ＋ 嗅内浅 層 II（版43a）\n                          CA1=nn[\"嗅内浅\"] - EC_II,   # 嗅内浅 層 III（版43a。前は 1歩前の CA3）\n                          前頭前野=nn[\"嗅内深\"],       # 嗅内深 から\n                          背内側視床=nn[\"前頭前野\"],   # 前頭前野から（調整する側。版43a の組み直し下）\n                          嗅内浅=nn[\"嗅内深\"],         # 深い層 → 浅い層（層内）\n                          嗅内深=0,\n                          側坐核=nn[\"前頭前野\"])       # 前頭前野から（側坐核の予測）\n        # 上からの戻り（眠りで落とす線。脳幹が組むときに読むので、脳幹より先に置く）版43a で 前頭前野 → 高次・運動 を足した\n        self.topdown_rows = ((\"聴覚\", nn[\"聴覚\"] + nn[\"運動\"], nn[\"連合\"]),     # 連合から\n                             (\"運動\", nn[\"運動\"], nn[\"高次\"]),                  # 高次から\n                             (\"運動\", nn[\"運動\"] + nn[\"高次\"], nn[\"前頭前野\"]),  # 前頭前野から（版43a）\n                             (\"連合\", nn[\"連合\"], nn[\"高次\"]),                  # 高次から\n                             (\"高次\", nn[\"高次\"] + nn[\"運動\"], nn[\"前頭前野\"]))  # 前頭前野から（版43a）\n        mk = lambda nm, i: Area(\n            n_ch=self.n[nm], prior=prior, n_ext=self.n_ext[nm], eta=eta,\n            n_partner=N_PARTNER.get(nm, n_partner), n_cell=n_cell, stp_path=stp_path, dev=dev, seed=seed + i,\n            # 版43a：前頭前野・嗅内・側坐核は高次の時定数（⚠️ ノブ【C】。人の値は未取得）\n            tau_scale=(TAU[\"連合\"] if nm == \"連合\"\n                       else TAU[\"高次\"] if nm in (\"高次\", \"前頭前野\", \"嗅内浅\", \"嗅内深\", \"側坐核\")\n                       else 55.9 / 100.0 if nm in (\"歯状回\", \"CA3\", \"CA1\") else TAU[\"一次\"]),\n            gate=(nm != \"視覚\" or eye_open),\n            motor=(nm in (\"運動\", \"視床\", \"背内側視床\")),   # 視床は貯めずに中継する（時間細胞を持たない）\n            inh_frac=(0.275 if nm in (\"視床\", \"背内側視床\") else 0.16),   # 【ヒト】視床の抑制性 25〜30%\n            complete=(nm in (\"CA3\", \"CA1\")),\n            # 疎さは段で違う（【ヒト】）\n"
],
[
"            #                   （単一ニューロン 140体超、Chan ほか 2014）\n            #   ⚠️ 前は視床で疎にしていた（2.4% → 0.7%）。疎化の段を2つ手前に置いていた\n            fire_th=FIRE_TH.get(nm, 1.0), x_gain=X_GAIN.get(nm, 1.0),\n            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),\n            # ⚠️ 視床の自発は【ヒト】の値が無い。人の視床に電極を入れるのは DBS 手術のときだけで、\n",
"            #                   （単一ニューロン 140体超、Chan ほか 2014）\n            #   ⚠️ 前は視床で疎にしていた（2.4% → 0.7%）。疎化の段を2つ手前に置いていた\n            fire_th=FIRE_TH.get(nm, HUMAN_TH if nm in N_NEW else 1.0),\n            x_gain=X_GAIN.get(nm, X_NEW if nm in N_NEW else 1.0),\n            tgt_by_fire=(nm in TGT_BY_FIRE), contacts_exc=CONTACTS_EXC.get(nm, 3.3),\n            # ⚠️ 視床の自発は【ヒト】の値が無い。人の視床に電極を入れるのは DBS 手術のときだけで、\n"
],
[
"        # 嗅内皮質 → 歯状回：1体が入口の約2%を拾う（【ヒト】・霊長類の疎な投射）\n        #   ⟲ 移したとき、体数が少ないときの代用（1/8）だけが残って 2% が落ちていた\n        k_perf = max(2, int(round(n_high * 0.02))) if n_high >= 100 else max(2, n_high // 8)\n        self.perforant = torch.stack([torch.randperm(n_high, device=dev, generator=g)[:k_perf]\n                                      for _ in range(h[\"歯状回\"])])\n        # 苔状線維：CA3 の1体に歯状回から4本\n",
"        # 嗅内皮質 → 歯状回：1体が入口の約2%を拾う（【ヒト】・霊長類の疎な投射）\n        #   ⟲ 移したとき、体数が少ないときの代用（1/8）だけが残って 2% が落ちていた\n        #   版43a：送り手は嗅内浅の層 II（20体）。前は高次（皮質から直に）\n        #   ⚠️ 20体では 2% が1本を割るので下限の2本（今の式のまま）\n        k_perf = max(2, int(round(EC_II * 0.02))) if EC_II >= 100 else max(2, EC_II // 8)\n        self.perforant = torch.stack([torch.randperm(EC_II, device=dev, generator=g)[:k_perf]\n                                      for _ in range(h[\"歯状回\"])])\n        # 苔状線維：CA3 の1体に歯状回から4本\n"
],
[
"        self.a2c = self._belt(n_ear, self.n[\"連合\"], rng)\n        self.c2h = self._stg(self.a2c_center, self.n[\"高次\"], rng)\n        self.gen = g\n        self.up = {nm: torch.zeros(self.n[nm], device=dev) for nm in NAMES}\n        # 視床は誤差ではなく表現を上へ送る（中継なので）\n        self.thal_out = torch.zeros(n_ear, device=dev)\n        self.pred = {nm: torch.zeros(self.n[nm], device=dev) for nm in (\"連合\", \"高次\")}\n        self.ca3_prev = torch.zeros(h[\"CA3\"], device=dev)\n        self.ca1_out = torch.zeros(h[\"CA1\"], device=dev)\n        # 海馬の受け手に印をつける（眠っているあいだ、皮質だけ下げるため）\n        hip = torch.zeros(self.g.n_recv_all, dtype=torch.bool, device=dev)\n",
"        self.a2c = self._belt(n_ear, self.n[\"連合\"], rng)\n        self.c2h = self._stg(self.a2c_center, self.n[\"高次\"], rng)\n        # 版43a：足した領野の細胞体への投射（受け手1体が送り手から K_NEW 本。⚠️ ノブ）\n        #   高次 → 前頭前野       【ヒト】Saur 2008（側頭 → 腹外側前頭、最外包）／【マカク】Romanski 1999\n        #   背内側視床 → 前頭前野  【ヒト】Behrens 2003（拡散 MRI、視床の区分と皮質）⚠️ 記憶の書誌\n        #   前頭前野 → 背内側視床  ⚠️ 細胞体に入れると 前頭前野 ⇄ 背内側視床 の興奮の輪で両方が撃ちっぱなしになった\n        #     （【測】コンテナの小さい網、60歩で 前頭前野・背内側視床 とも 100%）。人では視床網様核の抑制と\n        #     腹側淡蒼球の抑制がこの輪に掛かる（模型に無い）⇒ 43a では樹状突起（調整する側）に入れ、駆動は 43b で決める\n        #   高次 → 嗅内浅         【マカク】上側頭回 → 嗅内皮質（Brain Res 1983）\n        #   前頭前野 → 嗅内浅     【マカク】眼窩・内側前頭 → 嗅内皮質（Insausti 1987）\n        #   CA1 → 嗅内深          【マカク】J Comp Neurol 2020 ／【ヒト】Maass 2014\n        #   CA1・嗅内深 → 側坐核   【マカク】Friedman 2002 ／ 層 Va → 側坐核（総説 2021）\n        rp = lambda ns, nt: self._rand_proj(ns, nt, K_NEW, g)\n        self.p_h2pfc = rp(self.n[\"高次\"], nn[\"前頭前野\"])\n        self.p_md2pfc = rp(nn[\"背内側視床\"], nn[\"前頭前野\"])\n        self.p_h2ecs = rp(self.n[\"高次\"], nn[\"嗅内浅\"])\n        self.p_pfc2ecs = rp(nn[\"前頭前野\"], nn[\"嗅内浅\"])\n        self.p_ca12ecd = rp(h[\"CA1\"], nn[\"嗅内深\"])\n        self.p_ca12nac = rp(h[\"CA1\"], nn[\"側坐核\"])\n        self.p_ecd2nac = rp(nn[\"嗅内深\"], nn[\"側坐核\"])\n        self.gen = g\n        self.up = {nm: torch.zeros(self.n[nm], device=dev) for nm in NAMES}\n        # 視床は誤差ではなく表現を上へ送る（中継なので）\n        self.thal_out = torch.zeros(n_ear, device=dev)\n        self.pred = {nm: torch.zeros(self.n[nm], device=dev) for nm in (\"連合\", \"高次\", \"前頭前野\", \"嗅内深\")}\n        self.ca3_prev = torch.zeros(h[\"CA3\"], device=dev)\n        self.ca1_out = torch.zeros(h[\"CA1\"], device=dev)\n        self.md_out = torch.zeros(nn[\"背内側視床\"], device=dev)   # 版43a\n        # 海馬の受け手に印をつける（眠っているあいだ、皮質だけ下げるため）\n        hip = torch.zeros(self.g.n_recv_all, dtype=torch.bool, device=dev)\n"
],
[
"        #   ⚠️ ほかの経路（海馬・CA3 の再帰・運動の誤差→高次・皮質→歯状回・CA1 の戻り）は未決のまま撃たせる側\n        n = self.n\n        for area, first, cnt in ((\"聴覚\", n[\"聴覚\"], n[\"運動\"]),                  # 指令の写し\n                                 (\"聴覚\", n[\"聴覚\"] + n[\"運動\"], n[\"連合\"]),      # 連合から\n                                 (\"運動\", n[\"運動\"], n[\"高次\"]),                  # 高次から\n                                 (\"連合\", n[\"連合\"], n[\"高次\"])):                 # 高次から（CA1 の戻りは含めない）\n            self.g.mod_send[self._send_rows(area, first, cnt)] = True\n        # 恒常性の目標を受け手の種類ごとに（版39b）【み】09-27「普段の率を目標とみなす」「Bで様子見」\n",
"        #   ⚠️ ほかの経路（海馬・CA3 の再帰・運動の誤差→高次・皮質→歯状回・CA1 の戻り）は未決のまま撃たせる側\n        n = self.n\n        for area, first, cnt in ((\"聴覚\", n[\"聴覚\"], n[\"運動\"]),) + self.topdown_rows + (\n                (\"嗅内浅\", n[\"嗅内浅\"], n[\"嗅内深\"]),          # 深い層 → 浅い層（版43a）\n                (\"背内側視床\", n[\"背内側視床\"], n[\"前頭前野\"]),  # 前頭前野 → 背内側視床（版43a）\n                (\"側坐核\", n[\"側坐核\"], n[\"前頭前野\"])):        # 前頭前野 → 側坐核（予測。版43a）\n            # ⚠️ 海馬の戻り（嗅内深 → 連合・前頭前野）は前と同じく撃たせる側のまま（未決）\n            self.g.mod_send[self._send_rows(area, first, cnt)] = True\n        # 恒常性の目標を受け手の種類ごとに（版39b）【み】09-27「普段の率を目標とみなす」「Bで様子見」\n"
],
[
"            elif nm == \"運動\":\n                gg.scale_skip[lo:hi] = True\n            elif nm in (\"聴覚\", \"視覚\", \"連合\", \"高次\"):\n                gg.target_recv[ih] = hz(2.0)\n        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n\n",
"            elif nm == \"運動\":\n                gg.scale_skip[lo:hi] = True\n            elif nm in (\"聴覚\", \"視覚\", \"連合\", \"高次\", \"前頭前野\", \"嗅内浅\", \"嗅内深\"):\n                gg.target_recv[ih] = hz(2.0)\n            elif nm in (\"背内側視床\", \"側坐核\"):      # 版43a：人の値なし ⇒ 下げない\n                gg.scale_skip[lo:hi] = True\n        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n\n"
],
[
"        return torch.zeros(len(cnt), device=self.dev).index_add_(0, tgt, v[src])\n\n    def _fit(self, v, n):\n        if len(v) == n:\n",
"        return torch.zeros(len(cnt), device=self.dev).index_add_(0, tgt, v[src])\n\n    def _rand_proj(self, n_src, n_tgt, k, g):\n        \"\"\"版43a：受け手1体が送り手から k 本を一様に拾う（重複なし）。⚠️ 人の本数は未取得（ノブ）\"\"\"\n        k = min(k, n_src)\n        src = torch.stack([torch.randperm(n_src, device=self.dev, generator=g)[:k] for _ in range(n_tgt)])\n        tgt = torch.arange(n_tgt, device=self.dev).repeat_interleave(k)\n        cnt = torch.full((n_tgt,), float(k), device=self.dev)\n        return (src.reshape(-1), tgt, cnt)\n\n    def _fit(self, v, n):\n        if len(v) == n:\n"
],
[
"    def step(self, ear, cmd, eye=None, learn=True):\n        \"\"\"ear [n_ear]（耳のメル）  cmd [n_cmd]（口の指令の写し）  eye [n_eye] または None\"\"\"\n        # 海馬：皮質（高次）の誤差 → 貫通路 → 歯状回（上位 2.5% だけ撃つ）\n        drive = self.up[\"高次\"][self.perforant].sum(1)\n        k = max(1, int(round(self.n[\"歯状回\"] * SPARSE)))\n        # 上位 2.5% だけ撃つ。⚠️ 何も来ていないとき（drive が全部ゼロ）は誰も撃たない。\n        #   閾値だけで判定すると、ゼロ同士が並んで全部が通ってしまう\n        th = torch.topk(drive, k).values[-1]\n        dg = ((drive >= th) & (drive > 0)).float()\n\n        # ── 細胞体に入るもの（駆動する入力。撃たせる）──番号表の順に繋いで1回で作る\n",
"    def step(self, ear, cmd, eye=None, learn=True):\n        \"\"\"ear [n_ear]（耳のメル）  cmd [n_cmd]（口の指令の写し）  eye [n_eye] または None\"\"\"\n        # 海馬：嗅内浅 層 II の誤差 → 貫通路 → 歯状回（上位 2.5% だけ撃つ）（版43a。前は高次の誤差）\n        ec2 = self.up[\"嗅内浅\"][:EC_II]\n        ec3 = self.up[\"嗅内浅\"][EC_II:]\n        drive = ec2[self.perforant].sum(1)\n        k = max(1, int(round(self.n[\"歯状回\"] * SPARSE)))\n        # 上位 2.5% だけ撃つ。⚠️ 何も来ていないとき（drive が全部ゼロ）は誰も撃たない。\n        #   閾値だけで判定すると、ゼロ同士が並んで全部が通ってしまう\n        # 版43a：同じ値が並んだら乱数で選び、ちょうど k 体にする（【ヒト】2.5% 未満 ＝ 撃つ数の上限）\n        #   ⟲ 嗅内浅 層 II が 20 体になり、1体が拾うのは2本 ⇒ 値が 0〜2 しかなく同点だらけで、\n        #      「閾値以上ぜんぶ」だと歯状回が平均 19 体・多い歩は 300 体すべて撃った（【測】コンテナ）\n        #   ⚠️ 入力が粗い（2本）こと自体は、嗅内浅 層 II の体数の下限の問題として残る\n        rank = drive + 0.5 * torch.rand(len(drive), device=self.dev, generator=self.gen)\n        top = torch.topk(rank, k).indices\n        dg = torch.zeros_like(drive)\n        dg[top] = 1.0\n        dg = dg * (drive > 0).float()\n\n        # ── 細胞体に入るもの（駆動する入力。撃たせる）──番号表の順に繋いで1回で作る\n"
],
[
"            dg,\n            (dg[self.mossy].max(1).values > 0).float(),        # 苔状線維（起爆）\n            (self.ca3_prev[self.schaffer].max(1).values > 0).float()])\n        if self.kick_left > 0:                                 # 運動視床の蹴り\n            # 視床の細胞が人の速さで撃つスパイクの列（毎歩 1 ではない）\n",
"            dg,\n            (dg[self.mossy].max(1).values > 0).float(),        # 苔状線維（起爆）\n            (self.ca3_prev[self.schaffer].max(1).values > 0).float(),\n            # 版43a\n            self._proj(self.up[\"高次\"], self.p_h2pfc) + self._proj(self.md_out, self.p_md2pfc),   # 前頭前野\n            torch.zeros(self.n[\"背内側視床\"], device=self.dev),                                  # 背内側視床（駆動は 43b）\n            self._proj(self.up[\"高次\"], self.p_h2ecs) + self._proj(self.up[\"前頭前野\"], self.p_pfc2ecs),  # 嗅内浅\n            self._proj(self.ca1_out, self.p_ca12ecd),                                             # 嗅内深\n            self._proj(self.ca1_out, self.p_ca12nac) + self._proj(self.pred[\"嗅内深\"], self.p_ecd2nac)])  # 側坐核\n        if self.kick_left > 0:                                 # 運動視床の蹴り\n            # 視床の細胞が人の速さで撃つスパイクの列（毎歩 1 ではない）\n"
],
[
"        pc = (self.pred[\"連合\"] > 0.5).float()\n        ph = (self.pred[\"高次\"] > 0.5).float()\n        e = torch.cat([cmd, pc,                                # 聴覚（視床は ext を持たない）\n                       ph,                                     # 運動\n                       ph, self.ca1_out,                       # 連合（＋ 海馬の戻り）\n                       self.up[\"運動\"],                         # 高次（運動の誤差を丸ごと）\n                       self.up[\"高次\"],                          # 歯状回\n                       dg, self.ca3_prev,                        # CA3\n                       self.ca3_prev])                           # CA1\n\n        outs = self.g.step(x, e, learn)\n",
"        pc = (self.pred[\"連合\"] > 0.5).float()\n        ph = (self.pred[\"高次\"] > 0.5).float()\n        pf = self.pred[\"前頭前野\"]\n        ed = self.pred[\"嗅内深\"]\n        e = torch.cat([cmd, pc,                                # 聴覚（視床は ext を持たない）\n                       ph, pf,                                 # 運動（＋ 前頭前野。版43a）\n                       ph, ed,                                 # 連合（＋ 海馬の戻り。版43a：嗅内深 から）\n                       self.up[\"運動\"], pf,                     # 高次（運動の誤差を丸ごと ＋ 前頭前野。版43a）\n                       ec2,                                    # 歯状回（版43a：嗅内浅 層 II）\n                       dg, self.ca3_prev, ec2,                  # CA3（＋ 嗅内浅 層 II。版43a）\n                       ec3,                                    # CA1（版43a：嗅内浅 層 III）\n                       ed,                                     # 前頭前野（嗅内深 から）\n                       pf,                                     # 背内側視床（前頭前野から。調整する側）\n                       ed,                                     # 嗅内浅（深い層から）\n                       pf])                                    # 側坐核（前頭前野から）\n\n        outs = self.g.step(x, e, learn)\n"
],
[
"        self.pred[\"連合\"] = cut(\"連合\", rep)\n        self.pred[\"高次\"] = cut(\"高次\", rep)\n        # CA3 のシナプスは確実に伝わる（【ヒト】0.90。皮質の 0.33 との差が速く覚える理由）\n        c3 = cut(\"CA3\", rep)\n",
"        self.pred[\"連合\"] = cut(\"連合\", rep)\n        self.pred[\"高次\"] = cut(\"高次\", rep)\n        self.pred[\"前頭前野\"] = cut(\"前頭前野\", rep)      # 版43a\n        self.pred[\"嗅内深\"] = cut(\"嗅内深\", rep)\n        self.md_out = cut(\"背内側視床\", rep)              # 視床は表現を送る（中継）\n        # CA3 のシナプスは確実に伝わる（【ヒト】0.90。皮質の 0.33 との差が速く覚える理由）\n        c3 = cut(\"CA3\", rep)\n"
]
],
"brainstem.py": [
[
"#   【ヒト】6〜9か月 起きている 2.5〜3.5時間 ／ 昼寝 1〜1.5時間（生物台帳）。ほかの月齢は未取得\nAGE_TABLE = ((6.0, 9.0, 3.0, 1.25),)\nFATIGUE_AREAS = (\"聴覚\", \"連合\", \"高次\")   # 疲れを読む皮質（運動は目標なし・視覚は目を閉じている）\n_STEP_DAYS = DT_MS * COMPRESS / 1000.0 / 86400.0\n\n",
"#   【ヒト】6〜9か月 起きている 2.5〜3.5時間 ／ 昼寝 1〜1.5時間（生物台帳）。ほかの月齢は未取得\nAGE_TABLE = ((6.0, 9.0, 3.0, 1.25),)\nFATIGUE_AREAS = (\"聴覚\", \"連合\", \"高次\", \"前頭前野\")   # 版43a：前頭前野を足した   # 疲れを読む皮質（運動は目標なし・視覚は目を閉じている）\n_STEP_DAYS = DT_MS * COMPRESS / 1000.0 / 86400.0\n\n"
],
[
"def _hours_to_steps(h):\n    return int(h * 3600.0 * 1000.0 / DT_MS / COMPRESS)\nCORTEX = (\"聴覚\", \"運動\", \"連合\", \"高次\")\n# 上からの戻りの線：受け手の領野 → ext の中の位置（torch_brain.py の e の並び）\n#   聴覚 ext ＝ [指令の写し, 連合から]  ／ 運動 ext ＝ [高次から] ／ 連合 ext ＝ [高次から, CA1 の戻り]\n",
"def _hours_to_steps(h):\n    return int(h * 3600.0 * 1000.0 / DT_MS / COMPRESS)\nCORTEX = (\"聴覚\", \"運動\", \"連合\", \"高次\", \"前頭前野\")   # 版43a：前頭前野を足した\n# 上からの戻りの線：受け手の領野 → ext の中の位置（torch_brain.py の e の並び）\n#   聴覚 ext ＝ [指令の写し, 連合から]  ／ 運動 ext ＝ [高次から] ／ 連合 ext ＝ [高次から, CA1 の戻り]\n"
],
[
"        buf_base = int(g.off_send[-1])\n        idx, b_off = [], 0\n        for i, nm in enumerate(names):\n            a = b.areas[nm]\n            per = a.ch.n_cell + a.ch.n_ramp\n            if nm in TOPDOWN:\n                src, where = TOPDOWN[nm]\n                n = b.n[src]\n                first = a.n_ch + (b.n[\"運動\"] if where == \"after_cmd\" else 0)\n                rows = torch.arange(first, first + n, device=self.dev)\n                idx.append((int(ch_start[i]) + rows.unsqueeze(1) * per\n",
"        buf_base = int(g.off_send[-1])\n        idx, b_off = [], 0\n        # 版43a：線の表は torch_brain.topdown_rows（受け手, 最初の行, 行の数）から読む\n        td = {}\n        for nm, first, n in getattr(b, \"topdown_rows\", ()):\n            td.setdefault(nm, []).append((first, n))\n        if not td:\n            for nm, (src, where) in TOPDOWN.items():\n                td[nm] = [(b.n[nm] + (b.n[\"運動\"] if where == \"after_cmd\" else 0), b.n[src])]\n        for i, nm in enumerate(names):\n            a = b.areas[nm]\n            per = a.ch.n_cell + a.ch.n_ramp\n            for first, n in td.get(nm, ()):\n                rows = torch.arange(first, first + n, device=self.dev)\n                idx.append((int(ch_start[i]) + rows.unsqueeze(1) * per\n"
]
],
"babble.py": [
[
"\nSTATE = \"state.pt\"\nVERSION = 42          # コードの版。網の作りを変えたら上げる\n#   1  最初の torch 版\n#   2  周回のあいだも発火が伝わるように直した\n",
"\nSTATE = \"state.pt\"\nVERSION = 43          # コードの版。網の作りを変えたら上げる\n#   1  最初の torch 版\n#   2  周回のあいだも発火が伝わるように直した\n"
]
]
}''')

out = {}
for f, hs in HUNKS.items():
    s = open(f, encoding="utf-8").read()
    for i, (old, new) in enumerate(hs):
        c = s.count(old)
        if c != 1:
            sys.exit(f"⚠️ {f} の {i+1} 番目の印が {c} 回（arkhe のファイルがコンテナの写しと違う）。何も書いていない\n--- 探した行 ---\n{old}")
        s = s.replace(old, new)
    ast.parse(s)
    out[f] = s
if "VERSION = 43 " not in out["babble.py"]:
    sys.exit("⚠️ babble.py が版42 ではない。何も書いていない")
for f, s in out.items():
    shutil.copy(f, f + ".v42")
    open(f, "w", encoding="utf-8").write(s)
print("版43a にした（元は *.v42 に残した）")
print("まず  python3 measure_size.py  で 12GB に収まるかを見る（組むだけ・保存しない）")
