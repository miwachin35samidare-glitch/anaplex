"""核の素子の「自分の発火の残り」を 1.1ms（周）ごとにする必要があるかを、サルの本物の列で測る
  python3 bake_nuc2.py 核 [1体の秒=120] [周=40]      核：外節・内節・視床下核
【み】10-21 21:44「必要があるなら」
列：【マカク】健康・課題なし（Zenodo 6581800、Galvan & Wichmann。handoff/data）
比べる2つ（どちらも撃つかは周 1.067ms ごと・不応期 3周 ＝ 網と同じ）
  A 今の形：残り（τ 1〜128歩の8本）と見込みは 1歩（10周）に1回。焼く的 ＝ その歩のどこかで撃ったか
       回すとき：歩の見込み p を周ごとの撃つ割合 1−(1−p)^(1/10) に割って、周ごとに撃つ（不応期つき）
  B 周の形：残り（τ 1〜1280周の11本）と見込みを周ごとに。焼く的 ＝ その周に撃ったか
焼いたあと、入力なしで回して、ms の間隔の 率・CV・IR・5ms 未満・300ms 超の休み を列と比べる
"""
import sys, os, math, numpy as np, torch

NUC = sys.argv[1] if len(sys.argv) > 1 else "外節"
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
EPOCH = int(sys.argv[3]) if len(sys.argv) > 3 else 40
DATA = os.environ.get("ISI_DIR", "/home/claude/dl/isi")
DP = 1.067; R = 10; REF = 3
FOLD = {"外節": "GPe normal", "内節": "GPi normal", "視床下核": "STN normal"}
TA = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.float32)                       # 歩
TB = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1280], np.float32)       # 周
torch.manual_seed(0); rng = np.random.default_rng(0)


def load():
    d = os.path.join(DATA, FOLD[NUC]); out = []
    for f in sorted(x for x in os.listdir(d) if x.endswith(".bis")):
        isi = np.loadtxt(os.path.join(d, f)).ravel(); isi = isi[isi > 0]
        out.append(np.cumsum(isi))
    if os.environ.get("WARP"):    # 版2：体ごとの率のばらつきを除く（間隔を 体の率 ÷ 核の平均の率 倍に。CV・IR は変わらない）
        m = np.mean([len(s) / s[-1] for s in out])
        out = [s * (len(s) / s[-1]) / m for s in out]
    return out


def stats_ms(trains):
    R_, C, I, B, P = [], [], [], [], []
    for s in trains:
        x = np.diff(s)
        if len(x) < 10:
            continue
        R_.append(1000 / x.mean()); C.append(x.std() / x.mean()); I.append(np.mean(np.abs(np.log(x[1:] / x[:-1]))))
        B.append((x < 5).mean()); P.append((x > 300).sum() / (s[-1] - s[0]) * 60000)
    return np.mean(R_), np.mean(C), np.mean(I), np.mean(B) * 100, np.mean(P)


def to_pass(trains, sec):
    n = int(sec * 1000 / DP)
    x = np.zeros((n, len(trains)), np.float32)
    for j, s in enumerate(trains):
        s = s[s < sec * 1000]
        x[np.minimum((s / DP).astype(int), n - 1), j] = 1
    return x


def hist(x, taus):
    dec = np.exp(-1.0 / taus)
    tb = np.zeros((x.shape[1], len(taus)), np.float32); out = np.empty((x.shape[0], x.shape[1], len(taus)), np.float32)
    for t in range(x.shape[0]):
        out[t] = tb
        tb = tb * dec + x[t][:, None] * (1 - dec)
    return out


def bake(X, Y, nin):
    X = torch.tensor(X).reshape(-1, nin); Y = torch.tensor(Y).reshape(-1)
    rate = float(Y.mean())
    H = int(os.environ.get('HID', '32'))
    W1 = (torch.randn(nin, H) / nin ** 0.5).requires_grad_(); b1 = torch.zeros(H, requires_grad=True)
    W2 = (torch.randn(H, 1) / H ** 0.5).requires_grad_(); b2 = torch.tensor([math.log(rate / (1 - rate))], requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.01); lf = torch.nn.BCEWithLogitsLoss()
    f = lambda z: (torch.tanh(z @ W1 + b1) @ W2 + b2).squeeze(-1)
    n = len(Y); idx = torch.randperm(n); cut = int(n * 0.8); tr, te = idx[:cut], idx[cut:]
    for ep in range(EPOCH):
        perm = tr[torch.randperm(len(tr))]
        for s in range(0, len(perm), 131072):
            i = perm[s:s + 131072]; opt.zero_grad(); loss = lf(f(X[i]), Y[i]); loss.backward(); opt.step()
    with torch.no_grad():
        l = float(lf(f(X[te]), Y[te])); l0 = float(lf(torch.full((len(te),), math.log(rate / (1 - rate))), Y[te]))
    return (W1.detach(), b1.detach(), W2.detach(), b2.detach()), l, l0


def net(P, z):
    W1, b1, W2, b2 = P
    return torch.sigmoid((torch.tanh(z @ W1 + b1) @ W2 + b2).squeeze(-1))


@torch.no_grad()
def run_A(P, sec, ncell):
    dec = torch.exp(-1.0 / torch.tensor(TA)); tb = torch.zeros(ncell, len(TA))
    last = torch.full((ncell,), -99); sp = [[] for _ in range(ncell)]
    t = 0
    for st in range(int(sec * 1000 / (DP * R))):
        h = 1 - (1 - net(P, tb)).clamp(max=1 - 1e-6) ** (1.0 / R)
        anyf = torch.zeros(ncell)
        for _ in range(R):
            f = (torch.rand(ncell) < h) & (t - last >= REF)
            last = torch.where(f, torch.full_like(last, t), last); anyf = torch.maximum(anyf, f.float())
            for j in torch.nonzero(f).squeeze(1).tolist():
                sp[j].append(t * DP)
            t += 1
        tb = tb * dec + anyf.unsqueeze(1) * (1 - dec)
    return [np.array(s) for s in sp]


@torch.no_grad()
def run_B(P, sec, ncell):
    dec = torch.exp(-1.0 / torch.tensor(TB)); tb = torch.zeros(ncell, len(TB))
    last = torch.full((ncell,), -99); sp = [[] for _ in range(ncell)]
    for t in range(int(sec * 1000 / DP)):
        f = (torch.rand(ncell) < net(P, tb)) & (t - last >= REF)
        last = torch.where(f, torch.full_like(last, t), last)
        for j in torch.nonzero(f).squeeze(1).tolist():
            sp[j].append(t * DP)
        tb = tb * dec + f.float().unsqueeze(1) * (1 - dec)
    return [np.array(s) for s in sp]


def main():
    trains = load()
    fmt = lambda s: f"率 {s[0]:5.1f}Hz・CV {s[1]:.2f}・IR {s[2]:.2f}・5ms 未満 {s[3]:4.1f}%・300ms 超 {s[4]:4.1f}回/分"
    print(f"{NUC}：{len(trains)}体（1体 {SEC:.0f}秒を焼く）", flush=True)
    print("  列（全部の長さ）    ：" + fmt(stats_ms(trains)), flush=True)
    xp = to_pass(trains, SEC)                                  # [周, 体]
    n_st = xp.shape[0] // R
    xs = xp[: n_st * R].reshape(n_st, R, -1).max(1)            # [歩, 体]
    PA, la, la0 = bake(hist(xs, TA), xs, len(TA))
    PB, lb, lb0 = bake(hist(xp, TB), xp, len(TB)) if not os.environ.get('ONLYA') else (None, 0, 0)
    print(f"  当たり A {la:.4f}（率だけ {la0:.4f}）／ B {lb:.4f}（率だけ {lb0:.4f}）", flush=True)
    sa = run_A(PA, 60.0, len(trains)); print("  A 今の形（歩ごと）：" + fmt(stats_ms(sa)), flush=True)
    if PB is not None:
        sb = run_B(PB, 60.0, len(trains)); print("  B 周の形（周ごと）：" + fmt(stats_ms(sb)), flush=True)


if __name__ == "__main__":
    main()
