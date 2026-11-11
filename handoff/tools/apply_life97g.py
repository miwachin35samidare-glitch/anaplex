"""11-11：measure_life97 の「網だけの1歩」を、1歩ごとに GPU の終わりを待ち・5% 休み・見張りに伝える形に（網の値は変わらない）
  ⟲ 11-11 23:42：600歩を GPU へ積みっぱなしで GPU が休まず 98% のまま 4秒 ⇒ apply_guard109 の見張りが落とした（650歩のあと）
  python3 apply_life97g.py   （arkhe の ~/anaplex で。版109 の上に）
"""

FILES = {'measure_life97.py': ('a1465adec9bcfbdeec78233a080568467e4163772b7a6039705e26193996bcea', '"""版97 の速さ：続けて流す暮らし（Life.step）の1歩と、網だけ（b.step）の1歩を比べる（arkhe の ~/anaplex で。保存しない）\n  python3 measure_life97.py [歩=600]\n  （コンテナで道具を確かめるとき：python3 measure_life97.py 40 --test ⇒ CPU・相手 200。時間は読まない）\n声は作り物（FakeVoice・FakeReplier。VOICEVOX を待たない）⇒ 網の外の重さ（音の重ね・耳の符号・口の筋と読み）だけを見る\n"""\nimport sys, time, numpy as np, torch\nimport babble as B, brainstem as BS, torch_area as T, life as L\nfrom torch_brain import Brain\n\nif B.VERSION not in (97, 98, 99, 100, 101, 102, 103, 107, 109):\n    sys.exit(f"⚠️ babble.py が版97〜100 ではない（版 {B.VERSION}）")\nTEST = "--test" in sys.argv\nargs = [a for a in sys.argv[1:] if not a.startswith("--")]\nNS = int(args[0]) if args else 600\nif TEST:\n    B.DEV = "cpu"\ndev = B.DEV\nsync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)\ntorch.manual_seed(0)\nb = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=200 if TEST else 9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)\nb.bs.set_state(BS.WAKE)\nlife = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))\nfor _ in range(50):                      # 温める\n    life.step()\nsync(); t0 = time.time()\nfor _ in range(NS):\n    life.step()\nsync(); t_life = (time.time() - t0) / NS * 1000\near = torch.zeros(B.N_EAR, device=dev); cmd = torch.zeros(life.mo.n_cmd, device=dev)\n# 11-11：網だけの歩も1歩ごとに GPU の終わりを待ち、5% 休み、life.py の見張りに「進んだ」と伝える\n#   （⟲ 前は 600歩を GPU へ積みっぱなしで、GPU が休まず 98% のまま 4秒 ⇒ 見張りが落とした。画面を塞ぐ形そのもの）\nt_net = 0.0\nfor _ in range(NS):\n    t0 = time.time()\n    b.step(ear, cmd, learn=True)\n    sync()\n    el = time.time() - t0\n    t_net += el\n    if hasattr(L, "_G") and L._G.get("started"):\n        L._G["last"] = time.perf_counter()\n        L._G["n"] += 1\n    if getattr(L, "DUTY", 1.0) < 1.0:\n        time.sleep(el * (1.0 / L.DUTY - 1.0))\nt_net = t_net / NS * 1000\nst = life.stats\nprint(f"暮らしの1歩 {t_life:.1f}ms ／ 網だけ（無音）の1歩 {t_net:.1f}ms ／ 差 {t_life - t_net:+.1f}ms", flush=True)\nprint(f"（{life.t}歩・合図 {st[\'plans\']}・声 {st[\'voices\']}・自発 {st[\'spont\']}・返し手 {st[\'replies\']}）", flush=True)\nprint(f"人の速さ（1歩 {T.DT_MS}ms）に対して {t_life / T.DT_MS:.2f}倍の時間", flush=True)\n')}

import ast, hashlib, pathlib, sys

bad = []
for f, (h, text) in FILES.items():
    ast.parse(text)
    p = pathlib.Path(f)
    if p.exists() and p.read_text() == text:
        continue
    if h is None:                      # 新しく置くファイル（あれば止まる）
        if p.exists():
            bad.append(f)
        continue
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
        bad.append(f)
if bad:
    sys.exit(f"⚠️ {bad} が思っていた形ではない（手で直した所があるか、版が違う）。何も書き換えていない")
for f, (_, text) in FILES.items():
    pathlib.Path(f).write_text(text)
print("✅ measure_life97 の網だけの歩も休むようにした")
