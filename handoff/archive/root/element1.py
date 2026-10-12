"""デフォルメ版の素子 v1 —— 役割 1〜6

v0 との違い
  樹状突起の予測を2つの和にする（ロジットで足す）
    ① 焼いた網：自分の細胞体に来た入力の残り（ゆっくりした状態）から出す「いつもの見込み」
    ② シナプス：樹状突起に着いた送り手の発火 × 重み の和（どれが当てになるかはここが学ぶ）
  役割 6：素子は自分の中の誤差 e = b − p を出し、樹状突起のシナプスはそれで変わる
    Δw_j = η · e · x_j     （送り手 j が撃っていなければ変わらない。送り手と自分だけで閉じる）
    ② がロジットで足されているので、この形は予測の当たりを良くする向きにそのままなる

焼いた網（①）は同じ種類で共有・凍結。変わるのはシナプス（1本ずつ別々）だけ。
"""
import numpy as np

TAUS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float32)
DECAY = np.exp(-1.0 / TAUS).astype(np.float32)
THETA = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Prior:
    """①：自分の入力の残り → いつもの見込み（ロジット）。小さな2層の網（焼いて凍結）"""

    def __init__(self, hidden=32, seed=0):
        r = np.random.default_rng(seed)
        n = len(TAUS)
        self.W1 = (r.standard_normal((n, hidden)) / np.sqrt(n)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (r.standard_normal((hidden, 1)) / np.sqrt(hidden)).astype(np.float32)
        self.b2 = np.zeros(1, np.float32)

    def n_params(self):
        return sum(p.size for p in (self.W1, self.b1, self.W2, self.b2))

    def logit(self, f):
        h = np.tanh(f @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[..., 0], h

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(hidden=d["W1"].shape[1])
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return m


def run(prior, b, x, eta=0.1, learn=True):
    """素子を列の上で動かす（本数ぶん並べて、1本ずつ別々のシナプスを持つ）
    b: 細胞体に来る入力 [本数, 歩]      x: 樹状突起の送り手の発火 [本数, 歩, 送り手]
    返り値: 予測 p、誤差 e、発火 z [本数, 歩]、最後の重み w [本数, 送り手]"""
    n, T = b.shape
    w = np.zeros((n, x.shape[2]), np.float32)
    tb = np.zeros((n, len(TAUS)), np.float32)
    p = np.zeros((n, T), np.float32)
    for t in range(T):
        prior_logit, _ = prior.logit(tb)                 # ① いつもの見込み（t−1 までの残り）
        s = np.einsum("nj,nj->n", w, x[:, t])            # ② 樹状突起のシナプスの和
        p[:, t] = sigmoid(prior_logit + s)
        e_t = b[:, t] - p[:, t]
        if learn:
            w += eta * e_t[:, None] * x[:, t]            # 役割 6：自分の誤差で自分のシナプスを変える
        tb = tb * DECAY + b[:, t:t + 1] * (1 - DECAY)
    e = b - p
    z = (e > THETA).astype(np.float32)
    return p, e, z, w
