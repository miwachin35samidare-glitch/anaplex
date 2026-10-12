"""版43b → 版43d：口が運動の領野の発火を読んで言う ／ 蹴りは語の指令の前で終える
  arkhe の ~/anaplex で  python3 apply_v43d.py

設計と出どころは claude/設計案_脳幹.md §12-57〜
  運動の領野に口の指令（補助輪：何を言うかは乱数が選ぶ）を入れる。torch 版では 0 になっていた
  写し（→ 聴覚）は、運動の領野が撃った表現（1歩前）。前は乱数の指令を直に入れていた
  口（mouth.decode）は、語の指令の窓（声の前 14歩）で半分以上撃った体を読む【C】→ VOICEVOX が声にする
    疑核と筋は置かない。VOICEVOX がその代わり【み】09-29
    返し手は、言えたモーラをまね返す
  蹴り：合図の後、発話によらない同じ長さ（103 − 14 ＝ 89歩）。語の指令が入る前に終える【み】09-29
    config.CMD_LEAD ＝ 14歩（【ヒト】Stavisky 2019：語ごとの成分は声の約 150ms 前から）
  babble.py・day.py：自分で言うのは babble.own_voice に（口が読み違えたら、その声を聞く）
    測る（measure）に「選んだとおりに言えた回数」を足した
  measure_v43d.py（新しいファイル）：学ぶにつれて言えるか
先に apply_v43b2.py を当てておくこと。印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import json
import shutil
import sys

HUNKS = json.loads(r'''{
"config.py": [
[
"#   ⚠️ 合図を聞き取る時間も入っている。喃語（合図なし）の準備の長さは違う可能性がある\nGO_LEAD = 103\nN_MELS = 40\nMEL_FMIN = 40.0\n",
"#   ⚠️ 合図を聞き取る時間も入っている。喃語（合図なし）の準備の長さは違う可能性がある\nGO_LEAD = 103\n# 語の指令が運動の領野に入り始めるのは、声の何歩前か（版43d）\n#   【ヒト】Stavisky 2019：語ごとに違う成分は声の約 150ms 前から\n#   ⇒ 150ms ÷ 10.67ms ≈ 14歩\n# 蹴りはその前で終える（合図 → 蹴り GO_LEAD − CMD_LEAD 歩 → 語の指令 CMD_LEAD 歩 → 声）【み】09-29\n#   【ヒト】補足運動野の発話前の活動は、声の 300ms 前に山。終わりが腹側運動前野・一次運動野の立ち上がりと重なる（115人、iEEG）\n#   【ヒト】StartReact：準備した音節は、短い大きな音ひとつで丸ごと出る（Stevenson 2014）\n#   ⟲ 版43c までは蹴りが声まで入り続け、指令の窓に重なっていた。口が運動の領野を読むと 0/6 しか指令どおりに言えない【測】\nCMD_LEAD = 14\nN_MELS = 40\nMEL_FMIN = 40.0\n"
]
],
"torch_brain.py": [
[
"        self.ca1_out = torch.zeros(h[\"CA1\"], device=dev)\n        self.md_out = torch.zeros(nn[\"背内側視床\"], device=dev)   # 版43a\n        # 海馬の受け手に印をつける（眠っているあいだ、皮質だけ下げるため）\n        hip = torch.zeros(self.g.n_recv_all, dtype=torch.bool, device=dev)\n",
"        self.ca1_out = torch.zeros(h[\"CA1\"], device=dev)\n        self.md_out = torch.zeros(nn[\"背内側視床\"], device=dev)   # 版43a\n        # 版43d：運動の領野の表現（1歩前）。口の指令の写しとして聴覚へ送る\n        self.motor_rep = torch.zeros(n_cmd, device=dev)\n        # 海馬の受け手に印をつける（眠っているあいだ、皮質だけ下げるため）\n        hip = torch.zeros(self.g.n_recv_all, dtype=torch.bool, device=dev)\n"
],
[
"\n    def new_utterance(self, speaking=True):\n        \"\"\"発話の頭。話すなら、準備のあいだ（声が出るまで）蹴りを入れ続ける。聞くだけなら出さない\n        【ヒト】Stavisky 2019：語によらない成分は合図の直後から立ち上がり、声はその約1.1秒後\n        ⚠️ 人の成分は「上がり続ける」形。模型の蹴りは入れる／入れないの2値で、坂は作れない\n        ⚠️ 400ms より先も続くかは図の窓の外（マカク Kaufman 2016：動きのあいだも続く、からの見込み）\n        ⟲ 版23 まで go() をどこからも呼んでいなかった。版24 は発話の頭で3歩だけだった\"\"\"\n        import config as C\n        if speaking:\n            self.go(C.GO_LEAD)\n\n    @staticmethod\n",
"\n    def new_utterance(self, speaking=True):\n        \"\"\"発話の頭。話すなら、準備のはじめに蹴りを入れる。聞くだけなら出さない\n        版43d：蹴りは発話によらない同じ長さで、語の指令が入り始める前に終える。その後の状態は網に任せる【み】09-29\n        【ヒト】Stavisky 2019：語によらない成分は合図の直後から立ち上がり、声はその約1.1秒後\n        ⚠️ 人の成分は「上がり続ける」形。模型の蹴りは入れる／入れないの2値で、坂は作れない\n        ⟲ 版43c までは声まで入れ続けた（マカクの腕 Kaufman 2016 から。発話に霊長類は使わない【み】09-29）\n        ⟲ 版23 まで go() をどこからも呼んでいなかった。版24 は発話の頭で3歩だけだった\"\"\"\n        import config as C\n        if speaking:\n            self.go(C.GO_LEAD - C.CMD_LEAD)     # 版43d：語の指令が入る前に終える（config.CMD_LEAD）\n\n    @staticmethod\n"
],
[
"            ic,                                                # 下丘 → 視床（1本 → 1体、帯をまたがない）\n            self._band_sum(self.up[\"視床\"]),                   # 視床 → 聴覚（自分の帯の10本の合計）\n            torch.zeros(self.n[\"運動\"], device=self.dev),\n            eye if eye is not None else torch.zeros(self.n[\"視覚\"], device=self.dev),\n            self._proj(self.up[\"聴覚\"], self.a2c),               # 本数のまま（撃つかは領野の側で判定）\n",
"            ic,                                                # 下丘 → 視床（1本 → 1体、帯をまたがない）\n            self._band_sum(self.up[\"視床\"]),                   # 視床 → 聴覚（自分の帯の10本の合計）\n            cmd,                                               # 口の指令 → 運動の領野（版43d で戻した）\n            #   ⟲ numpy 版（brain.py）は運動の領野が cmd を受けていた。torch 版では 0 になっていた（記録なし。Claude の落とし）\n            #     運動の領野が「何を言ったか」を持たず、写しは乱数の指令から聴覚へ直に入っていた\n            #   補助輪：何を言うかは乱数が選ぶ。選んだ指令は運動の領野を通して口へ出す\n            eye if eye is not None else torch.zeros(self.n[\"視覚\"], device=self.dev),\n            self._proj(self.up[\"聴覚\"], self.a2c),               # 本数のまま（撃つかは領野の側で判定）\n"
],
[
"        pf = self.pred[\"前頭前野\"]\n        ed = self.pred[\"嗅内深\"]\n        e = torch.cat([cmd, pc,                                # 聴覚（視床は ext を持たない）\n                       ph, pf,                                 # 運動（＋ 前頭前野。版43a）\n                       ph, ed,                                 # 連合（＋ 海馬の戻り。版43a：嗅内深 から）\n",
"        pf = self.pred[\"前頭前野\"]\n        ed = self.pred[\"嗅内深\"]\n        e = torch.cat([self.motor_rep, pc,                     # 聴覚（視床は ext を持たない）\n                       #   版43d：写しは運動の領野が撃った表現（1歩前）【ヒト】腹側中心前回 → 上側頭回、発話の 107.5ms 前にピーク\n                       #   ⟲ 前は乱数の指令そのもの（口に渡した答え）を直に入れていた\n                       ph, pf,                                 # 運動（＋ 前頭前野。版43a）\n                       ph, ed,                                 # 連合（＋ 海馬の戻り。版43a：嗅内深 から）\n"
],
[
"        self.ca3_prev = c3 * (torch.rand(len(c3), device=self.dev, generator=self.gen) < CA3_SUCCESS)\n        self.ca1_out = cut(\"CA1\", rep)\n        # 版43b：前頭前野の「来なかった誤差」（頭 N_OMIT 体）＝ 予測 − 下からの入力 が THETA を越えた\n        #   【ヒト】上側頭回の後ろ・外側眼窩前頭（別の群れ）／【マカク】眼窩前頭 3.6〜7.1%\n",
"        self.ca3_prev = c3 * (torch.rand(len(c3), device=self.dev, generator=self.gen) < CA3_SUCCESS)\n        self.ca1_out = cut(\"CA1\", rep)\n        self.motor_rep = cut(\"運動\", rep)                 # 版43d\n        # 版43b：前頭前野の「来なかった誤差」（頭 N_OMIT 体）＝ 予測 − 下からの入力 が THETA を越えた\n        #   【ヒト】上側頭回の後ろ・外側眼窩前頭（別の群れ）／【マカク】眼窩前頭 3.6〜7.1%\n"
]
],
"voice.py": [
[
"            return v\n        one = mouth.encode(morae, volume)[0]\n        v[C.GO_LEAD - LATENCY:] = one  # 音の LATENCY 歩前から流す（準備のあいだは流さない）\n        return v\n\n    def say(self, morae, volume):\n        \"\"\"指令 → 波形 → メル [歩, 40]。0モーラなら沈黙（床値のまま）\"\"\"\n        self.last_morae, self.last_volume = morae, volume     # 返し手がまね返す（版40）\n        if not morae:\n            n = LATENCY + 20\n            return np.full((n, N_MEL), self.floor, np.float32)\n        q = self.vv.build_query(morae)\n        q[\"volumeScale\"] = float(volume)\n",
"            return v\n        one = mouth.encode(morae, volume)[0]\n        v[C.GO_LEAD - C.CMD_LEAD:] = one  # 版43d：声の CMD_LEAD 歩前から（蹴りが終わってから）。運動の領野に入る\n        #   ⟲ 版43c までは音の LATENCY 歩前から、聴覚へ直に入る写しだった\n        return v\n\n    def say(self, morae, volume):\n        \"\"\"指令 → 波形 → メル [歩, 40]。0モーラなら沈黙（床値のまま）\"\"\"\n        if not morae:\n            self.last_morae, self.last_volume = morae, volume\n            n = LATENCY + 20\n            return np.full((n, N_MEL), self.floor, np.float32)\n        pad = np.full((C.GO_LEAD, N_MEL), self.floor, np.float32)   # 準備（蹴りのあいだ、声は出ない）\n        return np.concatenate([pad, self.sound(morae, volume)])\n\n    def sound(self, morae, volume):\n        \"\"\"口が読んだモーラ → 声のメル [歩, 40]（準備の静けさを付けない。後ろに 20歩の静けさ）\n        版43d：疑核と筋は置かない。VOICEVOX がその代わり【み】09-29\"\"\"\n        self.last_morae, self.last_volume = morae, volume     # 返し手がまね返す（版40）。言えたものを返す\n        tail = np.full((20, N_MEL), self.floor, np.float32)\n        if not morae:\n            return tail\n        q = self.vv.build_query(morae)\n        q[\"volumeScale\"] = float(volume)\n"
],
[
"        wav = self.vv.synthesize(q)\n        mel = self.ear.hear(wav)\n        pad = np.full((C.GO_LEAD, N_MEL), self.floor, np.float32)   # 準備（蹴りのあいだ、声は出ない）\n        tail = np.full((20, N_MEL), self.floor, np.float32)\n        return np.concatenate([pad, mel.astype(np.float32), tail])\n\n    def encode(self, mel):\n",
"        wav = self.vv.synthesize(q)\n        mel = self.ear.hear(wav)\n        return np.concatenate([mel.astype(np.float32), tail])\n\n    def encode(self, mel):\n"
]
],
"babble.py": [
[
"#      表現の閾値を最大 21% 下げる（【ラット】3.4mV ÷【ヒト】16mV）／ 眠るのは 眠りの圧 ≥ A のとき\n#  42  驚きでその歩の学ぶ速さを上げる（青斑核、版38〜39 の形）を戻した。版40 で外したのは誤り\nN_MEL, N_SYL, LATENCY = 40, 8, 5\nDEV = \"cuda\"\n",
"#      表現の閾値を最大 21% 下げる（【ラット】3.4mV ÷【ヒト】16mV）／ 眠るのは 眠りの圧 ≥ A のとき\n#  42  驚きでその歩の学ぶ速さを上げる（青斑核、版38〜39 の形）を戻した。版40 で外したのは誤り\n#  43  a 前頭前野・背内側視床・嗅内・側坐核 ／ b 上丘・下丘・中脳の核（黒質の門）／\n#      d 口が運動の領野の発火を読んで、VOICEVOX（疑核と筋の代わり【み】）が声にする。\n#        写しは運動の領野の表現から。蹴りは語の指令の前で終える（config.CMD_LEAD）\nN_MEL, N_SYL, LATENCY = 40, 8, 5\nDEV = \"cuda\"\n"
],
[
"\n\ndef reply(b, rp, rng):\n    \"\"\"返し手が返すなら、それを聞く（写しなし。学ぶ。版40）\"\"\"\n",
"\n\ndef own_voice(b, vo, mo, rng, learn):\n    \"\"\"自分で言う（版43d）。返り値 (耳の発火, 運動の領野に入れた指令, 誤差の和, 選んだモーラ, 言えたモーラ)\n    合図 → 蹴り → 語の指令（補助輪：何を言うかは乱数が選ぶ）→ 口が運動の領野の発火を読む\n    → VOICEVOX が声にする → 自分の耳で聞く\n    口の読み方：指令の窓（CMD_LEAD 歩）のうち半分以上の歩で撃った体を「撃った」とみる\n      ⚠️【C】筋が運動ニューロンの発火をならして動く、の代わり。窓の長さでならすのは人の値ではない\"\"\"\n    morae, vol = vo.random_morae(rng)\n    if not morae:                                       # 黙る（合図も蹴りも無し）\n        ear = vo.encode(vo.say(morae, vol))\n        cmd = np.zeros((len(ear), mo.n_cmd), np.float32)\n        return ear, cmd, run(b, cmd, ear, True, learn), [], []\n    one_np = mo.encode(morae, vol)[0]\n    one = torch.tensor(one_np, device=DEV)\n    zero = torch.zeros_like(one)\n    quiet_np = vo.encode(np.full((1, N_MEL), vo.floor, np.float32))[0]\n    quiet = torch.tensor(quiet_np, device=DEV)\n    w0 = C.GO_LEAD - C.CMD_LEAD\n    b.new_utterance(True)\n    n, reps = 0.0, []\n    for t in range(C.GO_LEAD):                          # 準備（声はまだ出ない）\n        _, e, _, _ = b.step(quiet, one if t >= w0 else zero, learn=learn)\n        n += float(e.sum())\n        if t >= w0:\n            reps.append(b.motor_rep.clone())\n    said, v = mo.decode((torch.stack(reps).mean(0) >= 0.5).float().cpu().numpy()[None])\n    snd = vo.encode(vo.sound(said, v))                  # 言えたものが last_morae に入る（返し手はそれを返す）\n    for t in range(len(snd)):                           # 声のあいだも指令は運動の領野に入り続ける\n        _, e, _, _ = b.step(torch.tensor(snd[t], device=DEV), one, learn=learn)\n        n += float(e.sum())\n    ear = np.concatenate([np.repeat(quiet_np[None], C.GO_LEAD, 0), snd])\n    cmd = np.zeros((len(ear), mo.n_cmd), np.float32)\n    cmd[w0:] = one_np\n    return ear, cmd, n, [m for m, *_ in morae], [m for m, *_ in said]\n\n\ndef voices(b, fake):\n    \"\"\"声の用意。返り値 (乱数を進めるだけの関数, 言う関数 say(乱数, learn), 返し手)\n    say の返り値は own_voice と同じ形\"\"\"\n    if fake:\n        # ⚠️ 作り物の声には口が無い（指令がそのまま音になる）。版43d の「口が運動の領野を読む」は通らない\n        inv = make_inventory(np.random.default_rng(0))\n\n        def say(r, learn):\n            mel, cmd, syl = utter(inv, r, b.n[\"運動\"])\n            ear = encode(mel)\n            return ear, cmd, run(b, cmd, ear, True, learn), syl, syl\n        return (lambda r: utter(inv, r, b.n[\"運動\"])), say, None   # 返し手（版40）は作り物の声では返さない\n    from voice import Voice, Replier\n    from mouth import Mouth\n    vo, mo = Voice(), Mouth()\n    return vo.random_morae, (lambda r, learn: own_voice(b, vo, mo, r, learn)), Replier(vo)\n\n\ndef reply(b, rp, rng):\n    \"\"\"返し手が返すなら、それを聞く（写しなし。学ぶ。版40）\"\"\"\n"
],
[
"\n\ndef measure(b, speak, n_test=20):\n    \"\"\"自分で言う（写しあり）と 録音で聞く（写しなし）で、聴覚の誤差を比べる\"\"\"\n    own = rec = sp = 0.0\n    for i in range(n_test):\n        mel, cmd, enc = speak(np.random.default_rng(10000 + i))\n        ear = enc(mel)\n        if ear.sum() == 0:\n            continue\n        own += run(b, cmd, ear, True, False)\n        rec += run(b, cmd, ear, False, False)\n        sp += float(ear.sum())\n    return own / max(sp, 1), rec / max(sp, 1), rec / max(own, 1)\n\n\n",
"\n\ndef measure(b, say, n_test=20):\n    \"\"\"自分で言う（写しあり）と 録音で聞く（写しなし）で、聴覚の誤差を比べる\n    版43d：録音は、自分で言ったときに耳に届いたものをそのまま流す（口が読み違えた声も含む）\n    返り値に「選んだとおりに言えた回数 ／ 声を出そうとした回数」を足した\"\"\"\n    own = rec = sp = 0.0\n    ok = tried = 0\n    for i in range(n_test):\n        ear, cmd, n, want, got = say(np.random.default_rng(10000 + i), False)\n        if want:\n            tried += 1; ok += int(want == got)\n        if ear.sum() == 0:\n            continue\n        own += n\n        rec += run(b, cmd, ear, False, False)\n        sp += float(ear.sum())\n    return own / max(sp, 1), rec / max(sp, 1), rec / max(own, 1), ok, tried\n\n\n"
],
[
"    #   ⟲ 09-26 まで babble.py だけ止めていなかった（版30 までの babble の結果は、起きているあいだも下げていた）\n    rng = np.random.default_rng(0)\n    rp = None                                   # 返し手（版40。作り物の声では返さない）\n    if fake:\n        inv = make_inventory(np.random.default_rng(0))\n        speak = lambda r: utter(inv, r, b.n[\"運動\"])[:2] + (encode,)\n    else:\n        from voice import Voice, Replier\n        from mouth import Mouth\n        vo, mo = Voice(), Mouth()\n        rp = Replier(vo)\n\n        def speak(r):\n            morae, vol = vo.random_morae(r)\n            mel = vo.say(morae, vol)\n            cmd = vo.cmd_stream(morae, vol, len(mel), mo)   # 写しは「始まる歩」に届く\n            return mel, cmd, vo.encode\n    for _ in range(done):                       # 声の乱数を進めて、同じ列を繰り返さない\n        speak(rng)\n    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（喃語の列とは別）\n    t0 = time.time()\n    for i in range(n_add):\n        mel, cmd, enc = speak(rng)\n        run(b, cmd, enc(mel), True, True)\n        if rp is not None:\n            reply(b, rp, rng_r)\n",
"    #   ⟲ 09-26 まで babble.py だけ止めていなかった（版30 までの babble の結果は、起きているあいだも下げていた）\n    rng = np.random.default_rng(0)\n    draw, say, rp = voices(b, fake)\n    for _ in range(done):                       # 声の乱数を進めて、同じ列を繰り返さない\n        draw(rng)\n    rng_r = np.random.default_rng(1000 + done)   # 返し手の乱数（喃語の列とは別）\n    t0 = time.time()\n    for i in range(n_add):\n        say(rng, True)\n        if rp is not None:\n            reply(b, rp, rng_r)\n"
],
[
"            print(f\"  喃語 {done} 回  門 {b.bs.gate_mean():.3f}  覚醒 {float(b.bs.arousal):.2f}  経過 {el/60:.1f}分  残り見込み \"\n                  f\"{el/(i+1)*(n_add-i-1)/60:.1f}分\", flush=True)\n    o, r, s = measure(b, speak)\n    print(f\"喃語 {done} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}\", flush=True)\n    # VRAM は2つある。rocm-smi が見ているのは「確保」の方\n    #   使った山 ＝ こちらが実際に要る量\n",
"            print(f\"  喃語 {done} 回  門 {b.bs.gate_mean():.3f}  覚醒 {float(b.bs.arousal):.2f}  経過 {el/60:.1f}分  残り見込み \"\n                  f\"{el/(i+1)*(n_add-i-1)/60:.1f}分\", flush=True)\n    o, r, s, ok, tried = measure(b, say)\n    print(f\"喃語 {done} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  選んだとおりに言えた {ok}/{tried}\", flush=True)\n    # VRAM は2つある。rocm-smi が見ているのは「確保」の方\n    #   使った山 ＝ こちらが実際に要る量\n"
]
],
"day.py": [
[
"\n    rng = np.random.default_rng(0)\n    rp = None                                 # 返し手（版40。作り物の声では返さない）\n    if fake:\n        inv = B.make_inventory(np.random.default_rng(0))\n        speak = lambda r: B.utter(inv, r, b.n[\"運動\"])[:2] + (B.encode,)\n    else:\n        from voice import Voice, Replier\n        from mouth import Mouth\n        vo, mo = Voice(), Mouth()\n        rp = Replier(vo)\n\n        def speak(r):\n            morae, vol = vo.random_morae(r)\n            mel = vo.say(morae, vol)\n            return mel, vo.cmd_stream(morae, vol, len(mel), mo), vo.encode\n    for _ in range(done):                     # 声の乱数を進めて、同じ列を繰り返さない\n        speak(rng)\n\n    b.bs.set_state(BS.WAKE)                   # 起きている（版38：恒常性・上からの戻りは脳幹が決める）\n",
"\n    rng = np.random.default_rng(0)\n    draw, say, rp = B.voices(b, fake)        # 版43d：口が運動の領野を読む（babble.own_voice）\n    for _ in range(done):                     # 声の乱数を進めて、同じ列を繰り返さない\n        draw(rng)\n\n    b.bs.set_state(BS.WAKE)                   # 起きている（版38：恒常性・上からの戻りは脳幹が決める）\n"
],
[
"        i = 0\n        while (i < wake) if wake is not None else not b.bs.sleep_due:\n            mel, cmd, enc = speak(rng)\n            B.run(b, cmd, enc(mel), True, True)\n            if rp is not None:\n                B.reply(b, rp, rng_r)\n",
"        i = 0\n        while (i < wake) if wake is not None else not b.bs.sleep_due:\n            say(rng, True)\n            if rp is not None:\n                B.reply(b, rp, rng_r)\n"
],
[
"        n_wake = i\n        ns = n_sleep if n_sleep is not None else b.bs.sleep_steps()\n        o, r, s = B.measure(b, speak)\n        print(f\"  [{cyc+1}] 眠る前  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}\", flush=True)\n        B.save(b, done, log=log)\n\n",
"        n_wake = i\n        ns = n_sleep if n_sleep is not None else b.bs.sleep_steps()\n        o, r, s, ok, tr = B.measure(b, say)\n        print(f\"  [{cyc+1}] 眠る前  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  言えた {ok}/{tr}\", flush=True)\n        B.save(b, done, log=log)\n\n"
],
[
"              f\"眠りの圧 {b.bs.pressure:.2f}  疲れ {b.bs.fatigue:.2f}\", flush=True)\n        S.sleep(b, n_step=ns, report_every=max(1, ns // 4))\n        o2, r2, s2 = B.measure(b, speak)\n        print(f\"  [{cyc+1}] 起きたあと 自分 {o2:.1%} ／ 録音 {r2:.1%}  supp {s2:.2f}\"\n              f\"  （眠る前 {s:.2f} → {s2:.2f}）\", flush=True)\n        log.append(f\"{time.strftime('%m-%d %H:%M')} 喃語 {done}回（+{n_wake}）\"\n",
"              f\"眠りの圧 {b.bs.pressure:.2f}  疲れ {b.bs.fatigue:.2f}\", flush=True)\n        S.sleep(b, n_step=ns, report_every=max(1, ns // 4))\n        o2, r2, s2, ok2, tr2 = B.measure(b, say)\n        print(f\"  [{cyc+1}] 起きたあと 自分 {o2:.1%} ／ 録音 {r2:.1%}  supp {s2:.2f}  言えた {ok2}/{tr2}\"\n              f\"  （眠る前 {s:.2f} → {s2:.2f}）\", flush=True)\n        log.append(f\"{time.strftime('%m-%d %H:%M')} 喃語 {done}回（+{n_wake}）\"\n"
],
[
"\n    if n_cycle == 0:\n        o, r, s = B.measure(b, speak)\n        print(f\"喃語 {done} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}\", flush=True)\n\n\n",
"\n    if n_cycle == 0:\n        o, r, s, ok, tr = B.measure(b, say)\n        print(f\"喃語 {done} 回  自分 {o:.1%} ／ 録音 {r:.1%}  supp {s:.2f}  言えた {ok}/{tr}\", flush=True)\n\n\n"
]
]
}''')
NEW = json.loads(r'''{
"measure_v43d.py": "\"\"\"版43d の試し（arkhe）：口が運動の領野を読んで言う。学ぶにつれて、選んだとおりに言えるか\n  python3 measure_v43d.py [喃語の回数=100]\n覚えは残さない（state.pt は読まない・書かない）。返し手も返す（babble.py と同じ回し方）\n10回ごとに出すもの\n  言えた      選んだモーラ列どおりに言えた回数 ／ 声を出そうとした回数\n  指令の体    語の指令が入っている歩で、指令した体のうち運動の領野で撃った割合\n  指令外      同じ歩で、指令していない体が撃った数（1歩あたり）\n  抑制の重み  運動の領野の表現の体1つに入る抑制の重みの合計（学びで育つか）\n⟲ コンテナ（相手 200、CPU）では 8回で 抑制 0 → 36、指令の体 1.00 → 0.4 前後、学びを止めると 1.00 のまま（09-29）\"\"\"\nimport sys, time, torch, numpy as np\nimport babble as B, brainstem as BS, torch_area as T\nfrom torch_brain import Brain, NAMES\n\nN = int(sys.argv[1]) if len(sys.argv) > 1 else 100\ntorch.manual_seed(0)\nprior = T.Prior.load(\"prior_v1.npz\", dev=B.DEV)\nb = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)\nb.bs.set_state(BS.WAKE)\n_, say, rp = B.voices(b, False)\ng = b.g\nlo = int(b.off_ch[NAMES.index(\"運動\")]); nm = b.n[\"運動\"]\nis_m = torch.zeros(g.n_recv_all, dtype=torch.bool, device=B.DEV)\nis_m[torch.nonzero(g.is_rep).squeeze(1)[lo:lo + nm]] = True\nm_inh = is_m[g.e_recv.long()] & g.e_inh\n\n\ndef inh_w():\n    return float(g.w[m_inh].sum()) / nm      # 辺の並びは刈り込み（rewire）で変わらない前提。変わったら作り直す\n\n\nbox = {\"hit\": 0.0, \"want\": 0.0, \"extra\": 0.0, \"n\": 0}\nostep = b.step\n\n\ndef step(ear, cmd, *a, **k):\n    r = ostep(ear, cmd, *a, **k)\n    if float(cmd.sum()) > 0:\n        rep = b.motor_rep\n        box[\"hit\"] += float((rep * cmd).sum()); box[\"want\"] += float(cmd.sum())\n        box[\"extra\"] += float((rep * (1 - cmd)).sum()); box[\"n\"] += 1\n    return r\n\n\nb.step = step\nrng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)\nok = tried = 0; ex = \"\"; t0 = time.time()\nprint(\"回   言えた  指令の体  指令外/歩  抑制の重み/体  例（選んだ → 言えた）\")\nfor i in range(N):\n    ear, cmd, n, want, got = say(rng, True)\n    if want:\n        tried += 1; ok += int(want == got)\n        ex = f\"{want} → {got}\"\n    B.reply(b, rp, rng_r)\n    if (i + 1) % 10 == 0:\n        m_inh = is_m[g.e_recv.long()] & g.e_inh\n        print(f\"{i+1:4d}  {ok:2d}/{tried:2d}   {box['hit']/max(box['want'],1):6.2f}   {box['extra']/max(box['n'],1):7.1f}\"\n              f\"   {inh_w():10.2f}    {ex}  （{(time.time()-t0)/60:.1f}分）\", flush=True)\n        ok = tried = 0; box.update(hit=0.0, want=0.0, extra=0.0, n=0)\nprint(f\"VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB\")\n"
}''')
import os
for f in NEW:
    if os.path.exists(f):
        sys.exit(f"⚠️ {f} がもうある。何も書いていない（もう当ててあるか）")
out = {}
for f, hs in HUNKS.items():
    s = open(f, encoding="utf-8").read()
    for i, (old, new) in enumerate(hs):
        c = s.count(old)
        if c != 1:
            sys.exit(f"⚠️ {f} の {i+1} 番目の印が {c} 回。何も書いていない（もう当ててあるか、43b2 がまだ）\n--- 探した行 ---\n{old}")
        s = s.replace(old, new)
    ast.parse(s)
    out[f] = s
for f, s in out.items():
    shutil.copy(f, f + ".v43b")
    open(f, "w", encoding="utf-8").write(s)
for f, s in NEW.items():
    ast.parse(s)
    open(f, "w", encoding="utf-8").write(s)
print("版43d にした（元は *.v43b に残した。新しく measure_v43d.py）")
print("次：python3 measure_v43d.py 100")
