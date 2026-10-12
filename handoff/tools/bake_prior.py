"""素子の焼いた網（見込み）を焼き直す（コンテナでも arkhe でも。torch の CPU で回る）
  python3 bake_prior.py streams_v53.npz [組 …]      ⇒ prior_v2_<組>.npz
  組の書き方：名前=領野,領野,…  例  皮質=聴覚,連合,高次,前頭前野,運動  視床=視床  海馬=歯状回,CA3,CA1,嗅内浅,嗅内深
  （組を書かなければ、全領野で1つ（今と同じ共有））
【み】10-10「Aやね」「ニューロンとしての質が上がるなら複数回焼いたっていい」「焼きはニューロンにする手段であって目的じゃない」
役目（決まったこと 09-11 のイ・役割3）：自分の細胞体の少し先を当てる
  入力  tb ＝ その口の表現の発火の減衰の和（τ 1〜128歩、torch_group2 と同じ式）。t−1 まで
  当てる その歩の的（下から越えた）
  形    8 → 32 → 1（prior_v1 と同じ。tanh）。交差エントロピーを最小に
⚠️ 直前だけで次が決まる列で焼くと、記憶なしでも当たる（前の作りの教訓。決まったこと 09-11）⇒ 列は網の実際の列
出すもの（組ごと）：的の率 ／ 履歴が空の見込み（床）／ 焼いた網の当たり（交差エントロピー、率だけで当てた場合と比べる）／
  1回撃ってから k 歩あとの見込み ／ 見込みの平均と的の率（ずれていないか）
"""
import sys, numpy as np, torch

path = sys.argv[1]
d = np.load(path)
n_ch = int(d["n_ch"]); names = list(d["names"]); off = list(d["off_ch"])
rep = np.unpackbits(d["rep"], axis=1)[:, :n_ch].astype(np.float32)
tgt = np.unpackbits(d["tgt"], axis=1)[:, :n_ch].astype(np.float32)
taus = torch.tensor(d["taus"], dtype=torch.float32)
dec = torch.exp(-1.0 / taus)
groups = {}
for a in sys.argv[2:]:
    k, v = a.split("="); groups[k] = v.split(",")
if not groups:
    groups = {"全部": names}
torch.manual_seed(0)


def hist(r):
    """r [歩, 口] → tb [歩, 口, 8]（t の行は t−1 までの減衰の和）"""
    T_, C = r.shape
    tb = torch.zeros(C, len(taus)); out = torch.empty(T_, C, len(taus))
    rr = torch.tensor(r)
    for t in range(T_):
        out[t] = tb
        tb = tb * dec + rr[t].unsqueeze(1) * (1 - dec)
    return out


for gname, members in groups.items():
    cols = []
    for nm in members:
        j = names.index(nm); cols += list(range(off[j], off[j + 1]))
    cols = np.array(cols)
    r, y = rep[:, cols], tgt[:, cols]
    rate = float(y.mean())
    if rate == 0:
        print(f"[{gname}] 的が1つも無い。焼かない", flush=True); continue
    X = hist(r).reshape(-1, len(taus)); Y = torch.tensor(y).reshape(-1)
    W1 = (torch.randn(len(taus), 32) / len(taus) ** 0.5).requires_grad_()
    b1 = torch.zeros(32, requires_grad=True)
    W2 = (torch.randn(32, 1) / 32 ** 0.5).requires_grad_()
    b2 = torch.tensor([float(np.log(rate / (1 - rate)))], requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(Y); idx_all = torch.randperm(n); cut = int(n * 0.8)
    tr, te = idx_all[:cut], idx_all[cut:]
    f = lambda x: (torch.tanh(x @ W1 + b1) @ W2 + b2).squeeze(-1)
    for ep in range(30):
        perm = tr[torch.randperm(len(tr))]
        for k in range(0, len(perm), 65536):
            i = perm[k:k + 65536]
            opt.zero_grad(); loss = lossf(f(X[i]), Y[i]); loss.backward(); opt.step()
    with torch.no_grad():
        l_te = float(lossf(f(X[te]), Y[te]))
        l_base = float(lossf(torch.full((len(te),), float(np.log(rate / (1 - rate)))), Y[te]))
        p_te = torch.sigmoid(f(X[te]))
        floor = float(torch.sigmoid(f(torch.zeros(1, len(taus)))))
        tb = (1 - dec).unsqueeze(0); ks = []
        for k in range(21):
            if k in (0, 1, 2, 5, 10, 20):
                ks.append(f"{k}歩 {float(torch.sigmoid(f(tb))):.3f}")
            tb = tb * dec
    np.savez(f"prior_v2_{gname}.npz", W1=W1.detach().numpy(), b1=b1.detach().numpy(),
             W2=W2.detach().numpy(), b2=b2.detach().numpy())
    print(f"[{gname}] 口 {len(cols)}・歩 {r.shape[0]} ｜ 的の率 {rate:.2%} ｜ 床（履歴が空）{floor:.4f} ｜ 見込みの平均 {float(p_te.mean()):.4f}"
          f" ｜ 当たり 交差エントロピー {l_te:.4f}（率だけ {l_base:.4f}）｜ 1回撃ってから " + "・".join(ks), flush=True)
    print(f"   ⇒ prior_v2_{gname}.npz", flush=True)
