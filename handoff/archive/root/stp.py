"""短期可塑性 —— デフォルメなし。ヒト側頭葉の当てはめを1行そのまま引く

出所  Campagnola ら 2022（Allen、synphys_r2.1_small.sqlite の synapse_model 表）
      ヒト・TCx（素子と同じ側頭葉）・ex/in の組。172行が synapse_model_human.json に入っている
式    経過   dep *= exp(-dt/τd)   fac *= exp(-dt/τf)
      放出   P = Pr0 + (1-Pr0)·fac        伝わる量 ∝ N · avail · P
      発火後 avail *= (1-k)   k = 抑圧量（-1 のときは k = P ＝ 小胞枯渇）
             fac += (1-fac)·増強量
放出  二項分布 n = Binom(avail, P)。期待値で置き換えない
状態  接続ごと（束で1つ）。1人の相手に来る接触は小胞の在庫を共有する

⚠️ 持ち方（2026-09-11）：値は172行しか無いので、共有の表に置き、1本は「どの行か」だけ持つ。
   抑圧（dep）と枯渇（avail）は行がどちらを使うか決めるので排他 ⇒ 1つの状態に畳む。
   単価 53B → 9B（索引1B ＋ 状態2つ）。引く値も式も変えていない
"""
import json
import numpy as np

DT_S = 0.01067          # 1歩 ＝ 10.67ms
KEYS = ("ex2ex", "ex2in", "in2ex", "in2in")


class STP:
    def __init__(self, shape, sender_inh, recv_inh, rng, path="synapse_model_human.json"):
        d = json.load(open(path, encoding="utf-8"))
        tbl, off = [], {}
        for k in KEYS:                                    # 4群を1枚の共有の表に積む
            off[k] = len(tbl)
            tbl += [list(map(float, r)) for r in d["行"][k]]
        self.tbl = np.asarray(tbl, np.float32)            # [行, 8]（共有。1本ずつは持たない）
        self.tbl_dec_d = np.exp(-DT_S / np.maximum(self.tbl[:, 2], 1e-4)).astype(np.float32)
        self.tbl_dec_f = np.exp(-DT_S / np.maximum(self.tbl[:, 4], 1e-4)).astype(np.float32)
        self.tbl_N = np.maximum(np.round(self.tbl[:, 5]), 1).astype(np.float32)
        grp = np.where(sender_inh, np.where(recv_inh, "in2in", "in2ex"),
                       np.where(recv_inh, "ex2in", "ex2ex"))
        row = np.zeros(shape, np.uint16)                  # ← 1本が持つのはこれだけ
        for k in KEYS:
            m = grp == k
            if m.any():
                row[m] = off[k] + rng.integers(0, len(d["行"][k]), int(m.sum()))
        self.row = row
        self.state = np.where(self.tbl[row, 1] < 0, 1.0, 0.0).astype(np.float32)  # 枯渇なら在庫1／抑圧なら0
        self.fac = np.zeros(shape, np.float32)
        self.rng = rng

    @property
    def Pr0(self):
        return self.tbl[self.row, 0]

    @property
    def N(self):
        return self.tbl_N[self.row]

    def reorder(self, o):
        """辺の並べ替えに合わせて状態も並べ替える"""
        self.row, self.state, self.fac = self.row[o], self.state[o], self.fac[o]
        if hasattr(self, "last"):
            self.last = self.last[o]

    def release_at(self, e, t):
        """触った辺 e だけ、最後に触った歩からいまの歩 t までまとめて減らして放出する
        （毎歩ぜんぶ減衰させると密になるので、触ったときにまとめる。減衰は指数なので1回で足りる）"""
        if not hasattr(self, "last"):
            self.last = np.zeros(self.row.shape, np.int32)
        r = self.row[e]
        n = (t - self.last[e]).astype(np.float32)
        self.state[e] *= np.where(self.tbl[r, 1] < 0, 1.0, self.tbl_dec_d[r] ** n)
        self.fac[e] *= self.tbl_dec_f[r] ** n
        self.last[e] = t
        Pr0 = self.tbl[r, 0]
        P = np.clip(Pr0 + (1 - Pr0) * self.fac[e], 0, 1)
        deplete = self.tbl[r, 1] < 0
        N = self.tbl_N[r]
        avail = np.where(deplete, self.state[e], 1 - self.state[e])
        cnt = self.rng.binomial(np.maximum(np.round(N * avail), 0).astype(np.int64), P)
        out = cnt / np.maximum(N * Pr0, 1e-6)
        k = np.where(deplete, P, self.tbl[r, 1])
        self.state[e] = np.where(deplete, self.state[e] * (1 - k),
                                 self.state[e] + (1 - self.state[e]) * k)
        self.fac[e] = self.fac[e] + (1 - self.fac[e]) * self.tbl[r, 3]
        return out.astype(np.float32)

    def release(self, fired):
        """fired: この歩に撃った接続 → 伝わる量の倍率（静止を1とする）"""
        r = self.row
        k_col, Pr0 = self.tbl[r, 1], self.tbl[r, 0]
        deplete = k_col < 0
        self.state *= np.where(deplete, 1.0, self.tbl_dec_d[r])   # 抑圧は減衰、在庫はそのまま
        self.fac *= self.tbl_dec_f[r]
        P = np.clip(Pr0 + (1 - Pr0) * self.fac, 0, 1)
        f = fired > 0
        if not f.any():
            return np.zeros_like(P)
        N = self.tbl_N[r]
        avail = np.where(deplete, self.state, 1 - self.state)
        n = self.rng.binomial(np.maximum(np.round(N * avail), 0).astype(np.int64), P)
        out = np.where(f, n / np.maximum(N * Pr0, 1e-6), 0.0)
        kk = np.where(deplete, P, k_col)
        self.state = np.where(f, np.where(deplete, self.state * (1 - kk),
                                          self.state + (1 - self.state) * kk), self.state)
        self.fac = np.where(f, self.fac + (1 - self.fac) * self.tbl[r, 3], self.fac)
        return out.astype(np.float32)
