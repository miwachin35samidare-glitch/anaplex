"""黒質へ入る手前の領野が、声の始まりにどれだけの時間撃つか（arkhe で。網は変えない・何も書かない）
  python3 probe_sn_input.py [発話の数=40]
【み】10-15「それ、跳ねるの？」⇒ bench_sn（コンテナ）：黒質の跳ねの幅は、入力が続く長さでほぼ決まる（NMDA 75.5ms の残りではほとんど延びない）
  ⇒ 人の跳ね（約3倍・幅 約150ms・山は返りの 約250ms 後）には、手前が 150ms 前後撃ち続けることが要る
見るもの（発話の始まりから 40歩 ＝ 427ms、発話で平均。⚠️ 数値を合わせ直す前の網）
  上丘の誤差（今 黒質へ入っている）／ 前頭前野の誤差・表現（これから入れる）／ 黒質（今の midbrain）
  ふだん（始まりの前 10歩）からの倍率・2倍以上の歩の長さ・山の時刻
"""
import sys, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T
import torch_brain as TB
from torch_brain import Brain, NAMES

NU = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 40
TEST = "--test" in sys.argv
if TEST:
    B.DEV = "cpu"
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
inv = B.make_inventory(np.random.default_rng(0)); rng = np.random.default_rng(0)
up = lambda x: TB.to_dev(x, dev) if hasattr(TB, "to_dev") else torch.tensor(x, device=dev)
PRE, POST = 10, 40
rows = {"上丘の誤差": [], "前頭前野の誤差": [], "前頭前野の表現": [], "黒質": []}


def snap():
    i = NAMES.index("前頭前野")
    lo, n = int(b.off_ch[i]), b.n["前頭前野"]
    return {"上丘の誤差": b.up["上丘"].mean(), "前頭前野の誤差": b.up["前頭前野"].mean(),
            "前頭前野の表現": b.g._any[0][lo:lo + n].mean(), "黒質": b.mb.spk["黒質"].mean()}


hist = []
for u in range(NU + 3):
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"]); ear = B.encode(mel)
    sil = np.repeat(ear[:1] * 0 + ear.min(), PRE, 0)          # 始まりの前は静か（声の一番小さい値）
    b.new_utterance(bool(np.any(cmd)))
    seq = list(sil) + list(ear[:POST])
    cseq = [np.zeros_like(cmd[0])] * PRE + list(cmd[:POST])
    rec = []
    for t in range(len(seq)):
        b.step(up(np.asarray(seq[t], np.float32)), up(np.asarray(cseq[t], np.float32)), learn=True)
        rec.append({k: v for k, v in snap().items()})
    if u >= 3 and len(rec) == PRE + POST:                      # 慣らしの3つは捨てる
        hist.append({k: torch.stack([r[k] for r in rec]).float().cpu().numpy() for k in rows})
print(f"発話 {len(hist)} ／ 1歩 {T.DT_MS}ms ／ 始まりの前 {PRE}歩・後 {POST}歩", flush=True)
for k in rows:
    m = np.mean([h[k] for h in hist], 0)
    base = m[:PRE].mean()
    x = m[PRE:] / max(base, 1e-9)
    o2 = (x >= 2).sum() * T.DT_MS
    print(f"  {k:<10} ふだん {base * 1000 / T.DT_MS:7.2f}Hz ／ 山 {m[PRE:].max() * 1000 / T.DT_MS:7.2f}Hz ×{x.max():5.2f}（{(x.argmax() + 0.5) * T.DT_MS:4.0f}ms）／ "
          f"2倍以上 {o2:4.0f}ms ／ 歩ごとの倍率 " + " ".join(f"{v:.1f}" for v in x[:20]), flush=True)
