"""版43d → 版44：口の連続値の段数を聞き分けで決める ／ 喃語の高さを人の赤ちゃんの振れ方に
  arkhe の ~/anaplex で  python3 apply_v44.py

設計と出どころは claude/設計案_脳幹.md §12-63〜73
  mouth.py   高さ 120〜600Hz を対数で 30段【み】の耳 ／ 母音 10〜350ms 33段・子音 1〜150ms 15段（網の耳の1コマ）／
             大きさ 0.05〜2倍を対数で 16段【み】。段 k は k＋1 体で表す（撃った数が値のまま）
             運動の領野 1,812 → 840体（mouth.N_CMD）
  voice.py   喃語の高さ：発話ごとに中心 300Hz【み】・3.3半音【ヒト】1人、発話の中は 4.47半音の幅【ヒト】15人を一様に【C】
  config.py  BABBLE_PITCH_*（前の PITCH_MEAN・PITCH_STD は使わなくなった）
  torch_brain.py  n_cmd の既定を mouth.N_CMD に
  babble.py  VERSION 44（網の大きさが変わる。版43 の state.pt は読めない）
先に apply_v43d.py を当てておくこと。印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import json
import shutil
import sys

HUNKS = json.loads(r'''{
"mouth.py": [
[
"  位置ごとの表現は構造を共有する（別々の入口ではない）\n人の運動単位\n  筋あたり約100個。動員した数が出力の強さになる ⇒ 連続値はこの形\n\nこれまで（人と違った）\n",
"  位置ごとの表現は構造を共有する（別々の入口ではない）\n人の運動単位\n  筋あたり約100個。動員した数が出力の強さになる ⇒ 連続値はこの形（撃った体の数が値）\n  版44：体の数（＝段数）は筋の数ではなく、聞き分けで決める（設計案_脳幹.md §12-63〜73）\n    疑核と筋は VOICEVOX が代わる【み】09-29 ⇒ 100体の動員は筋の側の細工で、皮質に持たせる理由が無い\n    段の間が聞き分けられないなら、網にとっても人にとっても無いのと同じ\n\nこれまで（人と違った）\n"
],
[
"              （型はモーラごとに決まった疎な符号。位置をまたいで同じ符号を使う ＝ 構造を共有）\n  位置        4つのスロットは並列のまま（人の運動前野と同じ）。位置ごとに層を持つ\n  連続値      1つにつき n_pool 体。撃った数が値（人の運動単位の動員）\n  言うか黙るか  型がどのモーラにも近くなければ沈黙（0モーラ。人も常時発声しない）\n\"\"\"\n",
"              （型はモーラごとに決まった疎な符号。位置をまたいで同じ符号を使う ＝ 構造を共有）\n  位置        4つのスロットは並列のまま（人の運動前野と同じ）。位置ごとに層を持つ\n  連続値      値ごとに POOL 体。撃った数が値（人の運動単位の動員）。段は体の数\n  言うか黙るか  型がどのモーラにも近くなければ沈黙（0モーラ。人も常時発声しない）\n\"\"\"\n"
],
[
"N_CODE = 128               # モーラの層の体数（集団の型を作る）\nCODE_SPARSE = 0.10         # 1つのモーラの型で撃つ割合（人：音素を符号化するのは約1割）\nN_POOL = 100               # 連続値1つあたりの体数（人の運動単位は筋あたり約100個）\nMATCH_MIN = 0.5            # 型がどれだけ合えばそのモーラと見なすか（下回れば沈黙）\n\nRANGE = {\"consonant_length\": (0.001, 0.150),\n         \"vowel_length\": (0.010, 0.350),\n         \"pitch\": (0.001, 6.0)}\nVOLUME_RANGE = (0.05, 2.0)\nCONT_KEYS = (\"consonant_length\", \"vowel_length\", \"pitch\")\n\n\nclass Mouth:\n    def __init__(self, n_code=N_CODE, n_pool=N_POOL, seed=0):\n        self.n_code, self.n_pool = n_code, n_pool\n        rng = np.random.default_rng(seed)\n        k = max(1, int(round(n_code * CODE_SPARSE)))\n",
"N_CODE = 128               # モーラの層の体数（集団の型を作る）\nCODE_SPARSE = 0.10         # 1つのモーラの型で撃つ割合（人：音素を符号化するのは約1割）\nMATCH_MIN = 0.5            # 型がどれだけ合えばそのモーラと見なすか（下回れば沈黙）\n\n# 連続値の範囲と段数（版44）\n#   高さ      ln 120〜600Hz を対数で等間隔 30段【み】の耳（30段と140段が聞き分けられない §12-66）\n#             下 120：話者102 は 134Hz 未満が同じに聞こえる（§12-67）／ 上 600：08-02 に崩れないと聞いた端\n#   母音の長さ 10〜350ms を等間隔 33段 ／ 子音の長さ 1〜150ms を等間隔 15段\n#             網の耳は 1コマ（10.67ms）ごとにしか長さを聞き分けない【測】measure_ear_res（§12-65）\n#   大きさ    0.05〜2.0倍を対数で等間隔 16段（1段 約2.1dB）【み】09-30\n#   ⟲ 版43 までは どれも 100段（運動単位の動員の数）、範囲の中を等間隔。高さは ln 0.001〜6.0（1〜400Hz）\nRANGE = {\"consonant_length\": (0.001, 0.150),\n         \"vowel_length\": (0.010, 0.350),\n         \"pitch\": (float(np.log(120.0)), float(np.log(600.0)))}\nVOLUME_RANGE = (0.05, 2.0)\nCONT_KEYS = (\"consonant_length\", \"vowel_length\", \"pitch\")\nPOOL = {\"consonant_length\": 15, \"vowel_length\": 33, \"pitch\": 30}\nVOL_POOL = 16\nN_CMD = MAX_MORAE * N_CODE + MAX_MORAE * sum(POOL.values()) + VOL_POOL   # 840\n\n\ndef grid(key):\n    \"\"\"その値の段の並び（高さは ln Hz なので等間隔 ＝ 対数で等間隔。大きさは対数で等間隔）\"\"\"\n    if key == \"volume\":\n        lo, hi = VOLUME_RANGE\n        return np.exp(np.linspace(np.log(lo), np.log(hi), VOL_POOL))\n    lo, hi = RANGE[key]\n    return np.linspace(lo, hi, POOL[key])\n\n\ndef to_step(key, x):\n    \"\"\"値 → 段（0 から）。一番近い段に丸める\"\"\"\n    return int(np.argmin(np.abs(grid(key) - x)))\n\n\nclass Mouth:\n    def __init__(self, n_code=N_CODE, seed=0):\n        self.n_code = n_code\n        rng = np.random.default_rng(seed)\n        k = max(1, int(round(n_code * CODE_SPARSE)))\n"
],
[
"            self.code[m, rng.choice(n_code, k, replace=False)] = 1.0\n        self.k = k\n        self.n_cmd = MAX_MORAE * n_code + MAX_MORAE * 3 * n_pool + n_pool\n\n    def decode(self, spikes):\n",
"            self.code[m, rng.choice(n_code, k, replace=False)] = 1.0\n        self.k = k\n        self.n_cmd = MAX_MORAE * n_code + MAX_MORAE * sum(POOL.values()) + VOL_POOL\n\n    def decode(self, spikes):\n"
],
[
"            m = int(np.argmax(hit))\n            picks.append(m if hit[m] >= MATCH_MIN and extra <= 2.0 else None)\n        cont = np.zeros((MAX_MORAE, 3), np.float64)\n        for i in range(MAX_MORAE):\n            for j, kk in enumerate(CONT_KEYS):\n                n = int(s[off:off + self.n_pool].sum()); off += self.n_pool\n                lo, hi = RANGE[kk]\n                cont[i, j] = lo + (hi - lo) * n / self.n_pool\n        n = int(s[off:off + self.n_pool].sum())\n        vol = VOLUME_RANGE[0] + (VOLUME_RANGE[1] - VOLUME_RANGE[0]) * n / self.n_pool\n        return [(m, *cont[i]) for i, m in enumerate(picks) if m is not None], vol\n\n    def encode(self, morae, volume):\n        \"\"\"指令 → 随伴放電に渡す写し（運動の領野が出したものと同じ並び）\"\"\"\n        v = np.zeros(self.n_cmd, np.float32); off = 0\n        for i in range(MAX_MORAE):\n",
"            m = int(np.argmax(hit))\n            picks.append(m if hit[m] >= MATCH_MIN and extra <= 2.0 else None)\n        # 連続値：撃った体の数 n → 段 n − 1（段 k は k ＋ 1 体で表す。1体も撃たなければ一番下の段）\n        cont = np.zeros((MAX_MORAE, 3), np.float64)\n        for i in range(MAX_MORAE):\n            for j, kk in enumerate(CONT_KEYS):\n                P = POOL[kk]\n                n = int(s[off:off + P].sum()); off += P\n                cont[i, j] = grid(kk)[min(max(n - 1, 0), P - 1)]\n        n = int(s[off:off + VOL_POOL].sum())\n        vol = float(grid(\"volume\")[min(max(n - 1, 0), VOL_POOL - 1)])\n        return [(m, *cont[i]) for i, m in enumerate(picks) if m is not None], vol\n\n    def encode(self, morae, volume):\n        \"\"\"指令 → 随伴放電に渡す写し（運動の領野が出したものと同じ並び）。値は一番近い段に丸める\"\"\"\n        v = np.zeros(self.n_cmd, np.float32); off = 0\n        for i in range(MAX_MORAE):\n"
],
[
"            for j, kk in enumerate(CONT_KEYS):\n                if i < len(morae):\n                    lo, hi = RANGE[kk]\n                    n = int(round((morae[i][1 + j] - lo) / (hi - lo) * self.n_pool))\n                    v[off:off + max(0, min(n, self.n_pool))] = 1.0\n                off += self.n_pool\n        lo, hi = VOLUME_RANGE\n        n = int(round((volume - lo) / (hi - lo) * self.n_pool))\n        v[off:off + max(0, min(n, self.n_pool))] = 1.0\n        return v[None, :]\n\n",
"            for j, kk in enumerate(CONT_KEYS):\n                if i < len(morae):\n                    v[off:off + to_step(kk, morae[i][1 + j]) + 1] = 1.0\n                off += POOL[kk]\n        v[off:off + to_step(\"volume\", volume) + 1] = 1.0\n        return v[None, :]\n\n"
]
],
"voice.py": [
[
"        n = int(rng.integers(1, MAX_MORAE + 1))\n        out = []\n        for _ in range(n):\n            mid = int(rng.integers(0, self.n_mora))\n",
"        n = int(rng.integers(1, MAX_MORAE + 1))\n        out = []\n        st = np.log(2.0) / 12.0                                   # 1半音（ln）\n        center = rng.normal(C.BABBLE_PITCH_CENTER, C.BABBLE_PITCH_SD_ST * st)   # 版44：発話ごと\n        for _ in range(n):\n            mid = int(rng.integers(0, self.n_mora))\n"
],
[
"            v = float(np.clip(rng.normal(C.VOWEL_LENGTH_MEAN, C.VOWEL_LENGTH_STD),\n                              *RANGE[\"vowel_length\"]))\n            p = float(np.clip(rng.normal(C.PITCH_MEAN, C.PITCH_STD), *RANGE[\"pitch\"]))\n            out.append((mid, c, v, p))\n        vol = float(np.clip(rng.normal(C.VOLUME_SCALE_MEAN, C.VOLUME_SCALE_STD), *VOLUME_RANGE))\n",
"            v = float(np.clip(rng.normal(C.VOWEL_LENGTH_MEAN, C.VOWEL_LENGTH_STD),\n                              *RANGE[\"vowel_length\"]))\n            p = float(np.clip(center + rng.uniform(-0.5, 0.5) * C.BABBLE_PITCH_WIDTH_ST * st, *RANGE[\"pitch\"]))\n            out.append((mid, c, v, p))\n        vol = float(np.clip(rng.normal(C.VOLUME_SCALE_MEAN, C.VOLUME_SCALE_STD), *VOLUME_RANGE))\n"
]
],
"config.py": [
[
"VOWEL_LENGTH_MEAN = 0.170\nVOWEL_LENGTH_STD = 0.050\nPITCH_MEAN = 4.80      # 自然対数Hz。e^4.80 = 122 Hz\nPITCH_STD = 0.35\nVOLUME_SCALE_MEAN = 1.0\nVOLUME_SCALE_STD = 0.15\n",
"VOWEL_LENGTH_MEAN = 0.170\nVOWEL_LENGTH_STD = 0.050\n# 喃語の高さ（版44。設計案_脳幹.md §12-67〜70）\n#   発話ごとに中心を引き、その発話の中はモーラごとに幅の中で振る\n#   中心 300Hz【み】の耳（話者102 の自然な声 268〜330Hz）\n#   発話どうしのばらつき 3.3半音【ヒト】Gregory ほか 2010（1人、3〜26週、[a] の声 343〜382 ± 72〜79Hz）\n#   発話の中の幅 4.47半音【ヒト】F1000Research 2024（15人、4〜16か月、9,737発話の平均）\n#     ⚠️ 幅の中を一様に振るのは【C】\n#   打ち切りは口の範囲（120〜600Hz、mouth.RANGE）\n#   ⟲ 版43 までは モーラごとに独立に ln 4.80 ± 0.35（122Hz ± 約6半音）【前の作りの置き値】。約6割が 134Hz 未満（話者の下限で同じ声）\nBABBLE_PITCH_CENTER = 5.70378   # ln 300\nBABBLE_PITCH_SD_ST = 3.3        # 半音\nBABBLE_PITCH_WIDTH_ST = 4.47    # 半音\nVOLUME_SCALE_MEAN = 1.0\nVOLUME_SCALE_STD = 0.15\n"
]
],
"torch_brain.py": [
[
"\nclass Brain:\n    def __init__(self, prior, n_ear=40, n_cmd=1812, n_eye=40, n_assoc=240, n_high=263,\n                 hippo_scale=4.0, n_partner=9100, n_cell=128, eta=0.3,\n                 stp_path=\"synapse_model_human.json\", dev=\"cuda\", seed=0, eye_open=False):\n        h = {k: max(4, int(round(v * hippo_scale))) for k, v in HIPPO_RATIO.items()}\n        self.h = h\n",
"\nclass Brain:\n    def __init__(self, prior, n_ear=40, n_cmd=None, n_eye=40, n_assoc=240, n_high=263,\n                 hippo_scale=4.0, n_partner=9100, n_cell=128, eta=0.3,\n                 stp_path=\"synapse_model_human.json\", dev=\"cuda\", seed=0, eye_open=False):\n        if n_cmd is None:                    # 版44：運動の領野の体数は口の指令の数（mouth.N_CMD ＝ 840）\n            from mouth import N_CMD as n_cmd  # ⟲ 版43 までは 1,812（連続値1つに 100体）\n        h = {k: max(4, int(round(v * hippo_scale))) for k, v in HIPPO_RATIO.items()}\n        self.h = h\n"
]
],
"babble.py": [
[
"\nSTATE = \"state.pt\"\nVERSION = 43          # コードの版。網の作りを変えたら上げる\n#   1  最初の torch 版\n#   2  周回のあいだも発火が伝わるように直した\n",
"\nSTATE = \"state.pt\"\nVERSION = 44          # コードの版。網の作りを変えたら上げる\n#   1  最初の torch 版\n#   2  周回のあいだも発火が伝わるように直した\n"
],
[
"#      d 口が運動の領野の発火を読んで、VOICEVOX（疑核と筋の代わり【み】）が声にする。\n#        写しは運動の領野の表現から。蹴りは語の指令の前で終える（config.CMD_LEAD）\nN_MEL, N_SYL, LATENCY = 40, 8, 5\nDEV = \"cuda\"\n",
"#      d 口が運動の領野の発火を読んで、VOICEVOX（疑核と筋の代わり【み】）が声にする。\n#        写しは運動の領野の表現から。蹴りは語の指令の前で終える（config.CMD_LEAD）\n#  44  口の連続値の段数を聞き分けで決めた（高さ30・母音33・子音15・大きさ16。運動の領野 1,812 → 840体）／\n#      喃語の高さを中心 300Hz・発話ごと 3.3半音・発話の中 4.47半音に（設計案_脳幹.md §12-63〜73）\nN_MEL, N_SYL, LATENCY = 40, 8, 5\nDEV = \"cuda\"\n"
]
]
}''')
NEW = {}
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
            sys.exit(f"⚠️ {f} の {i+1} 番目の印が {c} 回。何も書いていない（もう当ててあるか、43d がまだ）\n--- 探した行 ---\n{old}")
        s = s.replace(old, new)
    ast.parse(s)
    out[f] = s
for f, s in out.items():
    shutil.copy(f, f + ".v43d")
    open(f, "w", encoding="utf-8").write(s)
for f, s in NEW.items():
    ast.parse(s)
    open(f, "w", encoding="utf-8").write(s)
print("版44 にした（元は *.v43d に残した）")
print("次：python3 measure_size.py → python3 measure_v43d.py 100")
