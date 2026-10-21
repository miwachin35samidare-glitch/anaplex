"""核の素子に、人・霊長類の撃ち方の型を焼けるか試す（設計案_素子の焼き §20・引き継ぎ §28）
  python3 bake_nuc.py [秒=120] [体=48]        （網は使わない。何も書き換えない。CPU）
【み】10-21 18:06「進めて」
手順
  1 核ごとに、記録した値（率・CV・休み・はじけ）から ms の刻みの発火の列を作る【C】（人の列そのものは公開に無い ⇒ 作った列）
  2 1歩 10.67ms に区切る（その歩のどこかで撃ったか ＝ 網と同じ数え方）
  3 今の素子と同じ形（自分の発火の残り τ 1〜128歩の8本 → 32 → 1）で焼く（bake_prior と同じ）
  4 焼いた素子だけを、入力なしで「見込みに従って撃つ」で回し、歩の列の数（率・歩の間隔の CV・長い休み）が 2 の列と合うかを見る
作る列の形【C】：ガンマの間隔（形 k）＋ 休み（ポアソンで来て、決まった長さ黙る）＋ はじけ（ポアソンで来て、前に長い間隔・中は速く n 発）
  k は ms の列の CV が的になるよう二分法で（はじけ・休みを足したあと）。的の CV が無い核は ⚠️ 下の表のとおり借りる
"""
import sys, math, numpy as np, torch

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
NCELL = int(sys.argv[2]) if len(sys.argv) > 2 else 48
DT = 10.67
import os
EPOCH = int(os.environ.get("EPOCH", "20"))
ONLY = os.environ.get("ONLY")
TAUS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float32)
rng = np.random.default_rng(0); torch.manual_seed(0)

# 核ごとの値（設計案_凍結する部位の表）。rate_sd は体ごとの率のばらつき（無ければ 0）
NUC = {
    "外節": dict(rate=69.7, rate_sd=0.0, cv=1.15,
               pause=dict(frac_cells=0.56, per_min=13.0, dur=0.62),
               burst=dict(frac_spk=0.115, f_in=205.0, n=3.0, pre=1.7),
               src="率【マカク】Starr 2005・CV【マカク】2022 Nat Commun・休み【マカク】Elias 2007・はじけ【マカク】Wichmann & Soares 2006"),
    "内節": dict(rate=82.5, rate_sd=0.0, cv=0.89,
               pause=dict(frac_cells=0.06, per_min=13.0, dur=0.62),
               burst=dict(frac_spk=0.109, f_in=233.0, n=2.7, pre=2.0),
               src="率 Starr・CV 2022・休み 6%（回数・長さは外節の値を借りた ⚠️）・はじけ W&S"),
    "境": dict(rate=33.0, rate_sd=20.0, cv=0.89,
              pause=dict(frac_cells=0.06, per_min=13.0, dur=0.62),
              burst=dict(frac_spk=0.109, f_in=233.0, n=2.7, pre=2.0),
              src="率【マカク】Hong & Hikosaka 2008 ・⚠️ 撃ち方は内節の値を借りた（内節の一部）"),
    "網様部": dict(rate=60.5, rate_sd=18.7, cv=0.89,
                pause=dict(frac_cells=0.06, per_min=13.0, dur=0.62), burst=None,
                src="率【マカク】Elias 2008・休み 6%（内節と合わせた値）・⚠️ CV は的にしない決め【み】10-17 ⇒ 内節の 0.89 を仮に"),
    "視床下核": dict(rate=19.3, rate_sd=20.1, cv=1.29, pause=None,
                 burst=dict(frac_spk=0.329, f_in=102.0, n=3.7, pre=2.5),
                 src="率・CV【ヒト】Steigerwald 2008・はじけ【マカク】W&S"),
    "外側手綱核": dict(rate=30.0, rate_sd=0.0, cv=1.0, pause=None, burst=None,
                  src="率【マカク】・⚠️ 撃ち方の値なし ⇒ CV 1（ポアソン）を仮に"),
    "吻内側被蓋核": dict(rate=17.8, rate_sd=12.1, cv=1.08, pause=None, burst=None,
                   src="率【マカク】Hong 2011・⚠️ 不規則さの指数 1.08 を CV の代わりに（CV ではない）"),
    "黒質": dict(rate=4.56, rate_sd=0.0, cv=0.61, pause=None, burst=None,
               src="率【ヒト】Ramayya 2014・CV・ガンマ（形 約3）【マカク】Bayer 2007"),
}


def train_ms(rate, k, T, pause, burst, is_pauser):
    """1体の発火の時刻（ms）"""
    # はじけに入る発火 frac ⇒ はじけの来る率 ＝ rate·frac ÷ n。ふだんの発火は残り
    fb = burst["frac_spk"] if burst else 0.0
    r_base = rate * (1 - fb)
    out = []
    t = rng.gamma(k, 1000.0 / (r_base * k)) if r_base > 0 else T
    while t < T:
        out.append(t)
        t += rng.gamma(k, 1000.0 / (r_base * k))
    base = np.array(out)
    ev = []
    if burst:
        nb = rng.poisson(rate * fb / burst["n"] * T / 1000.0)
        for tb in rng.uniform(0, T, nb):
            n = max(2, rng.poisson(burst["n"]))
            gap = burst["pre"] * 1000.0 / rate
            base = base[(base < tb - gap) | (base > tb + n * 1000.0 / burst["f_in"])]   # 前に長い間隔
            ev += list(tb + np.arange(n) * 1000.0 / burst["f_in"])
    s = np.sort(np.concatenate([base, np.array(ev)])) if len(ev) else base
    if pause and is_pauser:
        npz = rng.poisson(pause["per_min"] * T / 60000.0)
        for tp in rng.uniform(0, T, npz):
            s = s[(s < tp) | (s > tp + pause["dur"] * 1000.0)]
    return s[s < T]


def cv_of(trains):
    isi = [np.diff(s) for s in trains if len(s) > 2]
    return float(np.mean([i.std() / i.mean() for i in isi])) if isi else float("nan")


def make(nm, T):
    c = NUC[nm]
    rates = (rng.gamma((c["rate"] / c["rate_sd"]) ** 2, c["rate_sd"] ** 2 / c["rate"], NCELL) if c["rate_sd"] > 0 and not os.environ.get("NOSD")
             else np.full(NCELL, c["rate"]))
    rates = np.clip(rates, 0.5, None)
    pausers = rng.random(NCELL) < (c["pause"]["frac_cells"] if c["pause"] else 0)
    lo, hi = 0.3, 30.0
    for _ in range(14):                                   # 形 k を CV の的へ（短い列で）
        k = math.sqrt(lo * hi)
        cv = cv_of([train_ms(r, k, 20000.0, c["pause"], c["burst"], p) for r, p in zip(rates[:16], pausers[:16])])
        lo, hi = (k, hi) if cv > c["cv"] else (lo, k)
    k = math.sqrt(lo * hi)
    # 版2：はじけの前の間隔・休みで消えた分だけ、ふだんの率を上げて的の率に（体ごとの率の比はそのまま）
    g = 1.0
    for _ in range(4):
        tr_ = [train_ms(r * g, k, 20000.0, c["pause"], c["burst"], p) for r, p in zip(rates[:16], pausers[:16])]
        got = np.mean([len(s) for s in tr_]) / 20.0
        g *= float(np.mean(rates[:16])) / max(got, 0.1)
    trains = [train_ms(r * g, k, T, c["pause"], c["burst"], p) for r, p in zip(rates, pausers)]
    return trains, k


def binarize(trains, T):
    n = int(T / DT)
    x = np.zeros((n, len(trains)), np.float32)
    for j, s in enumerate(trains):
        x[np.minimum((s / DT).astype(int), n - 1), j] = 1
    return x


def step_stats(x):
    """歩の列：率（Hz の目盛り ＝ 撃った歩の割合 ÷ 1歩）・歩の間隔の CV・40歩（427ms）以上黙る割合（1分あたりの回数）"""
    rate = x.mean() * 1000 / DT
    cvs, longp = [], 0
    for j in range(x.shape[1]):
        t = np.nonzero(x[:, j])[0]
        if len(t) > 2:
            d = np.diff(t); cvs.append(d.std() / d.mean()); longp += int((d >= 40).sum())
    mins = x.shape[0] * x.shape[1] * DT / 60000
    return rate, float(np.mean(cvs)) if cvs else float("nan"), longp / mins


def hist(x):
    dec = np.exp(-1.0 / TAUS)
    tb = np.zeros((x.shape[1], len(TAUS)), np.float32); out = np.empty((x.shape[0], x.shape[1], len(TAUS)), np.float32)
    for t in range(x.shape[0]):
        out[t] = tb
        tb = tb * dec + x[t][:, None] * (1 - dec)
    return out


def bake(x):
    X = torch.tensor(hist(x)).reshape(-1, len(TAUS)); Y = torch.tensor(x).reshape(-1)
    rate = float(Y.mean())
    W1 = (torch.randn(len(TAUS), 32) / len(TAUS) ** 0.5).requires_grad_(); b1 = torch.zeros(32, requires_grad=True)
    W2 = (torch.randn(32, 1) / 32 ** 0.5).requires_grad_(); b2 = torch.tensor([math.log(rate / (1 - rate))], requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.01); lf = torch.nn.BCEWithLogitsLoss()
    f = lambda z: (torch.tanh(z @ W1 + b1) @ W2 + b2).squeeze(-1)
    n = len(Y); idx = torch.randperm(n); cut = int(n * 0.8); tr, te = idx[:cut], idx[cut:]
    for ep in range(EPOCH):
        perm = tr[torch.randperm(len(tr))]
        for s in range(0, len(perm), 65536):
            i = perm[s:s + 65536]; opt.zero_grad(); loss = lf(f(X[i]), Y[i]); loss.backward(); opt.step()
    with torch.no_grad():
        l = float(lf(f(X[te]), Y[te])); l0 = float(lf(torch.full((len(te),), math.log(rate / (1 - rate))), Y[te]))
    return (W1.detach(), b1.detach(), W2.detach(), b2.detach()), l, l0


@torch.no_grad()
def free_run(P, nstep, ncell):
    W1, b1, W2, b2 = P
    dec = torch.exp(-1.0 / torch.tensor(TAUS))
    tb = torch.zeros(ncell, len(TAUS)); out = torch.zeros(nstep, ncell)
    for t in range(nstep):
        p = torch.sigmoid((torch.tanh(tb @ W1 + b1) @ W2 + b2).squeeze(-1))
        s = (torch.rand(ncell) < p).float(); out[t] = s
        tb = tb * dec + s.unsqueeze(1) * (1 - dec)
    return out.numpy()


def main():
    T = SEC * 1000.0
    print(f"核の素子に撃ち方を焼く試し（{SEC:.0f}秒 × {NCELL}体、1歩 {DT}ms）\n", flush=True)
    print("核          | ms の列：率・CV（的）     | 歩の列：率・CV・長い休み/分 | 焼いた素子だけで回す：率・CV・長い休み/分 | 当たり（率だけ）", flush=True)
    res = {}
    for nm in NUC:
        if ONLY and nm not in ONLY.split(","):
            continue
        trains, k = make(nm, T)
        ms_rate = np.mean([len(s) for s in trains]) / SEC
        ms_cv = cv_of(trains)
        x = binarize(trains, T)
        sr, scv, slp = step_stats(x)
        P, l, l0 = bake(x)
        y = free_run(P, int(60000 / DT), NCELL)
        fr, fcv, flp = step_stats(y)
        cond = []
        for m in (1, 2, 4, 8, 16, 32):
            # 列：直前 m 歩が黙っていたときに撃つ割合 ／ 素子：同じ形の履歴での見込み（最後に撃ったのが m 歩前）
            run = np.zeros(x.shape[1]); hit = 0; tot = 0
            for t in range(x.shape[0]):
                sel = run == m; tot += int(sel.sum()); hit += int(x[t][sel].sum())
                run = np.where(x[t] > 0, 0, run + 1)
            W1, b1, W2, b2 = P
            dec = np.exp(-1.0 / TAUS); tb = (1 - dec) * dec ** m
            pe = float(torch.sigmoid((torch.tanh(torch.tensor(tb) @ W1 + b1) @ W2 + b2)))
            cond.append(f"{m}歩 {hit / max(tot, 1):.2f}/{pe:.2f}")
        res[nm] = dict(k=k, ms=(ms_rate, ms_cv), step=(sr, scv, slp), free=(fr, fcv, flp), loss=(l, l0))
        print(f"{nm:<7} | {ms_rate:5.1f}Hz・{ms_cv:.2f}（{NUC[nm]['cv']}）k {k:.2f} | {sr:5.1f}Hz・{scv:.2f}・{slp:5.1f}"
              f"      | {fr:5.1f}Hz・{fcv:.2f}・{flp:5.1f}                      | {l:.4f}（{l0:.4f}）", flush=True)
        print("          黙ったあとに撃つ割合（列/素子。素子は1回撃って m 歩黙った履歴）：" + "・".join(cond), flush=True)
    print("\n出どころ", flush=True)
    for nm, c in NUC.items():
        print(f"  {nm}：{c['src']}", flush=True)


if __name__ == "__main__":
    main()
