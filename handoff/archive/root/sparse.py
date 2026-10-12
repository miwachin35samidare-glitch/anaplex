"""疎を計算に効かせる持ち方 —— 送り手ごとに連ねる（CSR）

これまで  受け手ごとに「自分の相手2,000人」を並べ、毎歩ぜんぶ読んでいた
          ⇒ 撃った数が0でも2,000でも同じ時間。発火の疎が計算に効かない
これから  送り手ごとに「自分の届け先」を連ねて持つ
          撃った送り手（毎歩 0.3%）だけを取り、その届け先に配る
          ⇒ 触るバイトが発火率に比例する。しかも続きの領域なので運搬に無駄が出ない

短期可塑性の状態も、毎歩ぜんぶ減衰させると密になる。
「最後に触った歩」を持ち、次に触ったときに何歩ぶんかまとめて減らす（減衰は指数なので1回で足りる）
"""
import numpy as np


class SparseFanout:
    """送り手 → 受け手 の疎な繋がりを、送り手ごとに連ねて持つ"""

    def __init__(self, n_send, n_recv, idx, live):
        """idx / live: [受け手, 枠] の形（これまでの持ち方）から作り直す"""
        r, s = np.nonzero(live)
        tgt = idx[r, s]                                  # 送り手の番号
        order = np.argsort(tgt, kind="stable")
        self.recv = r[order].astype(np.int32)            # 届け先（送り手ごとに連なる）
        self.slot = s[order].astype(np.int32)            # 受け手の中の枠（重みの置き場）
        src = tgt[order]
        self.start = np.zeros(n_send + 1, np.int64)      # 送り手 i の分は start[i]:start[i+1]
        np.add.at(self.start, src + 1, 1)
        self.start = np.cumsum(self.start)
        self.n_send, self.n_recv = n_send, n_recv
        self.n_edge = len(self.recv)

    def fired_slice(self, fired_ids):
        """撃った送り手の番号 → 触る辺の位置（続きの領域の連結）"""
        if len(fired_ids) == 0:
            return np.empty(0, np.int64)
        return np.concatenate([np.arange(self.start[i], self.start[i + 1]) for i in fired_ids])

    def accumulate(self, fired_ids, w, out):
        """撃った送り手の分だけを、受け手ごとに足し込む"""
        e = self.fired_slice(fired_ids)
        if len(e) == 0:
            return out
        np.add.at(out, self.recv[e], w[self.recv[e], self.slot[e]])
        return out


class LazyDecay:
    """最後に触った歩を覚えておき、次に触ったときにまとめて減らす（毎歩ぜんぶ触らない）"""

    def __init__(self, shape, dtype=np.float32):
        self.val = np.zeros(shape, dtype)
        self.last = np.zeros(shape, np.int32)

    def read(self, where, t, decay):
        """where の位置を、いまの歩 t まで減らしてから返す"""
        n = (t - self.last[where]).astype(np.float32)
        self.val[where] *= decay[where] ** n if np.ndim(decay) else decay ** n
        self.last[where] = t
        return self.val[where]
