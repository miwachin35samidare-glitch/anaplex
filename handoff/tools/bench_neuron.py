"""素子1つの試し台（段1、組み表_版116 §42・36_仕組みの棚卸し）

いまの錐体の素子（版116c の _drive_core・_out_core・_advance_tail と同じ式）1つに、
人の数・人の率の入力のくじを入れ、出る発火の並びを人の自然な並びの物差しで出す。
⚠️ 試し台の入力は測るためだけ（網には入れない）。電流は流さない（入力はシナプスのくじだけ）。

素子の式（コードのまま）
  1歩の頭に、前の歩に撃った送り手の分が届く：興奮 E ＝ Σ w（重さは 1.12mV 単位、接続1本ぶん）・抑制 I ＝ Σ 4 × W_INH
  周 c ＝ 0〜9：駆動 ＝ E ÷ (1 ＋ I) ＋ 持ち越し ＞ 14.3 なら撃つ（前に撃った周から 3周以上）
  歩の終わり：撃たなかったら 持ち越し ＝ 0.49 × 駆動、撃ったら 0
入力（人の値。どれも 36 の「置」の行）
  相手 9,100（興奮 84%・抑制 16%）・錐体 3.3Hz・抑制 4.7Hz・接続の重さは対数正規（中央 0.41・平均 1）を 0.1〜6mV に収めて引き直す
  倍率 s：本当に撃っている相手の割合 × 効き（§40 違い2 の「どれかが桁で違う」を掃く）
出す物差し：率・間隔が 2.2ms／3.3ms／10.67ms／21.3ms 未満の割合・間隔の CV・1歩に撃った周の数・網が見る率（1歩に1回まで）
比べる人の値（作るのに使っていない側）：Kyzar 2024 皮質 10.67ms 未満 2.1%（遅い体）・5.7%（速い体）、21.3ms 未満 5.2%・13.1%（§43）／ Ison 2011 10ms 未満 17〜25%（定義が違う、台帳_健診 §5）

使い方：python3 bench_neuron.py [秒 100] [種 0]
"""
import sys
import numpy as np

DT_MS = 10.67
CYC = 10
CYC_MS = 1.102
TH = 16.0 / 1.12
CARRY = 0.49
REFR = 3
W_INH = 0.02648 / 4.0
N_PART = 9100
N_E = int(round(N_PART * 0.84))
N_I = N_PART - N_E
R_E, R_I = 3.3, 4.7


def draw_w(n, rng):
    mu, sig = np.log(0.41), np.sqrt(2 * np.log(1 / 0.41))
    lo, hi = 0.1 / 1.12, 6.0 / 1.12
    w = rng.lognormal(mu, sig, n)
    bad = (w < lo) | (w > hi)
    while bad.any():
        w[bad] = rng.lognormal(mu, sig, bad.sum())
        bad = (w < lo) | (w > hi)
    return w


def run(sec, s, rng, w_e):
    n_step = int(sec * 1000 / DT_MS)
    p_e, p_i = R_E * DT_MS / 1000, R_I * DT_MS / 1000
    n_e_act = max(1, int(round(N_E * s)))
    w_act = w_e[:n_e_act]
    n_i_act = max(1, int(round(N_I * s)))
    carry, last = 0.0, -10 ** 9
    spikes = []
    per_step = np.zeros(n_step, dtype=np.int8)
    for t in range(n_step):
        fe = rng.random(n_e_act) < p_e
        E = w_act[fe].sum()
        I = 4.0 * W_INH * (rng.random(n_i_act) < p_i).sum()
        d = E / (1.0 + I) + carry
        fired = False
        for c in range(CYC):
            pc = t * CYC + c
            if d > TH and pc - last >= REFR:
                last = pc
                spikes.append(pc)
                per_step[t] += 1
                fired = True
        carry = 0.0 if fired else CARRY * d
    return np.array(spikes), per_step, n_step


def stats(spikes, per_step, n_step):
    sec = n_step * DT_MS / 1000
    out = {"率Hz": len(spikes) / sec, "網が見る率Hz": (per_step > 0).sum() / sec}
    if len(spikes) > 2:
        isi = np.diff(spikes) * CYC_MS
        for ms in (2.3, 3.4, DT_MS, 2 * DT_MS):
            out[f"間隔<{ms:.1f}ms"] = float((isi < ms).mean())
        out["CV"] = float(isi.std() / isi.mean())
    on = per_step[per_step > 0]
    out["撃った歩の周の数(平均)"] = float(on.mean()) if len(on) else 0.0
    return out


def selftest():
    rng = np.random.default_rng(1)
    w = draw_w(20000, rng)
    assert w.min() >= 0.1 / 1.12 and w.max() <= 6.0 / 1.12
    assert 0.35 < np.median(w) < 0.6, np.median(w)
    # 入力なしなら撃たない ・ 閾値の上で一定なら 3周ごと ⇒ 1歩に 3〜4発（平均 10/3）
    sp, ps, n = run(2, 1e-9, rng, np.zeros(N_E))
    assert len(sp) == 0
    sp, ps, n = run(1, 1.0, rng, np.full(N_E, 50.0))
    assert ps.min() == 3 and ps.max() == 4 and abs(ps.mean() - 10 / 3) < 0.02, (ps.min(), ps.max(), ps.mean())
    assert np.all(np.diff(sp) == REFR)
    print("selftest ✓")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        sys.exit()
    sec = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    w_e = draw_w(N_E, rng)
    E_mean = N_E * R_E * DT_MS / 1000 * w_e.mean()
    I_mean = 4 * W_INH * N_I * R_I * DT_MS / 1000
    print(f"人の数・率のとき 1歩の平均：興奮 {E_mean:.1f}（閾値の {E_mean / TH:.1f}倍）・1＋抑制 {1 + I_mean:.2f}・割ったあと 閾値の {E_mean / (1 + I_mean) / TH:.2f}倍【計算】")
    print("倍率 s | 率Hz | 網が見る率Hz | 間隔<2.3ms | <3.4ms | <10.7ms | <21.3ms | CV | 撃った歩の周の数")
    for s in (1.0, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01):
        st = stats(*run(sec, s, rng, w_e))
        row = [f"{s:g}", f"{st['率Hz']:.2f}", f"{st['網が見る率Hz']:.2f}"]
        for k in ("間隔<2.3ms", "間隔<3.4ms", "間隔<10.7ms", "間隔<21.3ms", "CV"):
            row.append(f"{st[k]:.3f}" if k in st else "—")
        row.append(f"{st['撃った歩の周の数(平均)']:.2f}")
        print(" | ".join(row))
    print("人（自然な並び・作るのに使っていない）：Kyzar 2024 皮質 <10.7ms 2.1%（遅い体）・5.7%（速い体）／ <21.3ms 5.2%・13.1% ／ 3ms 未満 0.3〜0.8%")
