"""版104：強まった後ふさがる仕組み（皮質の学ぶ興奮のシナプス、繋がり1本ずつ）— 組み表_版104.md
  【み選】11-03 12:17「ア」（繋がりごと）・【み】12:44・21:42（時計は実時間）・21:48（皮質だけ）・21:53「人にあってモデルにないものなら入れて」
  ・22:01（嗅内は外す）

人で揃う条件（【ヒト】一次運動野・網全体の MEP。台帳_人の値「可塑性」）
  ① 強まりが起きたときだけふさがる（撃つだけではふさがらない）
  ② 直後は次の強まりがふさがるか、弱まりへ逆転する
  ③ ふさがりは数分で育つ（2分ではまだ足される・3〜5分でふさがる）・前に強めた量が多いほど強い
  ④ 1〜3時間は、強さが戻っていても効かない（上限・飽和ではない）
  ⑤ 6時間までに戻り、強さは保つ
  ⑥ 強めた所を弱めると、ふさがりも解ける

形（繋がりごとに q：育ちかけ・m：効いている痕跡。単位は重さ ＝ 1.12mV）
  1 前に見たときからの重さの変わり d（学びの式は今のまま。変わりを後から見る）
  2 強まり（d ＞ 0）だけ g 倍：g ＝ max(1 − m／M, −r)。弱まりはそのまま【C】
  3 q ← q ＋ 実際の変わり（強まりで増え、弱まりで減る ⇒ ⑥）
  4 q の分が τ育 で m へ移る（⇒ ③）・m は τ残 で減る（⇒ ⑤）。m は 0 未満にしない
  ⚠️ 変わりは OCC_EVERY 歩ごとにまとめて見る（τ育・τ残 に比べて十分短い。速さのため）【C】
  ⚠️ 領野の中の辺は「たまった分を後で足す」形なので、変わりが見えるのは足したとき（送り手が撃ったとき）【C】
  ⚠️ q・m は保存しない（読み込むと 0 から）
ANAPLEX_OCC=0 で版103g と同じ（比べる用）
"""
import math
import os

import torch

from torch_area import DT_MS

OCC_ON = os.environ.get("ANAPLEX_OCC", "1") != "0"
# 当てる領野（新皮質。嗅内・海馬・扁桃体・線条体・核・小脳・視床は当てない）【み】11-03 21:48・22:01
OCC_AREAS = ("聴覚", "運動", "視覚", "連合", "高次", "前頭前野", "眼窩前頭", "前帯状", "補足", "前補足",
             "44", "45", "STS1", "STS2", "中側頭後", "TPJ")
TAU_GROW = 2.0 * 60.0 * 1000.0 / DT_MS      # 2分（ノブ。【ヒト】窓 2〜5分：Gamboa 2011・Fricke 2011・Tse 2018）
TAU_KEEP = 1.3 * 3600.0 * 1000.0 / DT_MS    # 1.3時間（ノブ）【計算】m の始め 10M で 3時間 m ＝ M・6時間 m ＝ 0.1M
OCC_M = 1.0                                 # ふさがりきる痕跡（ノブ）【C】
OCC_R = 0.5                                 # 逆へ転ぶ深さ（ノブ）【C】
OCC_EVERY = 16                              # 変わりを見る間（歩）【C】速さのため
OCC_DECAY_EVERY = 1024                      # 育つ・減るを進める間（歩）【C】半精度で減りが丸められないように
CHUNK = 1 << 22                             # 1回に見る本数（VRAM の山を増やさない）


def gain(m):
    """痕跡 m のときの強まりの効き g"""
    return (1.0 - m / OCC_M).clamp(min=-OCC_R)


class OccSet:
    """1つのシナプスの組（重さの配列の一部 idx、または全部）"""

    def __init__(self, get_w, idx, half=False):
        self.get_w = get_w
        self.idx = idx
        cur = self._cur()
        self.prev = cur.detach().clone().float()
        # q は小さな変わりを足し込むので単精度。m は OCC_DECAY_EVERY 歩ごとにしか動かないので、領野の中（本数が多い）は半精度
        self.q = torch.zeros(len(cur), dtype=torch.float32, device=cur.device)
        self.m = torch.zeros(len(cur), dtype=torch.float16 if half else torch.float32, device=cur.device)

    def _cur(self):
        w = self.get_w()
        return w if self.idx is None else torch.index_select(w, 0, self.idx)

    @torch.no_grad()
    def apply(self):
        """前に見たときからの変わりを読み、強まりに g を掛けて書き戻す。実際の変わりを q へ（CHUNK 本ずつ。VRAM の山を増やさない）"""
        w = self.get_w()
        n = len(self.q)
        for a in range(0, n, CHUNK):
            sl = slice(a, min(n, a + CHUNK))
            ix = None if self.idx is None else self.idx[sl].long()
            cur = (w[sl] if ix is None else w[ix]).float()
            pv = self.prev[sl]
            d = cur - pv
            g = gain(self.m[sl].float())
            ch = (d > 0) & (g < 1.0)                  # g ＝ 1 の所は触らない（足し直しの丸めで値を変えない）
            new = torch.where(ch, (pv + d * g).clamp(min=0.0), cur)
            if bool(ch.any()):
                if ix is None:
                    w[sl] = new.to(w.dtype)
                else:
                    w[ix] = new.to(w.dtype)
            self.q[sl] += new - pv
            pv.copy_(new)

    @torch.no_grad()
    def advance(self, n):
        """n 歩ぶん：q の分が m へ移り、m が減る（CHUNK 本ずつ）"""
        a = (1.0 - 1.0 / TAU_GROW) ** n
        b = (1.0 - 1.0 / TAU_KEEP) ** n
        N = len(self.q)
        for i in range(0, N, CHUNK):
            sl = slice(i, min(N, i + CHUNK))
            q = self.q[sl]
            tr = q * (1.0 - a)
            q.sub_(tr)
            self.m[sl] = ((self.m[sl].float() + tr) * b).clamp(min=0.0).to(self.m.dtype)


class Occ:
    """網全体のふさがり：領野の中の辺（Group2）・下からの線（_BUTable）・CL → 皮質"""

    def __init__(self, brain):
        from torch_brain import NAMES
        b = brain
        self.b = b
        dev = b.g.w.device
        self.t = 0
        cx = [NAMES.index(nm) for nm in OCC_AREAS]
        # 領野の中：受け手が皮質の表現・誤差・来ない体で、興奮の辺
        g = b.g
        rc = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev)
        for i in cx:
            rc[int(g.off_recv[i]):int(g.off_recv[i + 1])] = True
        self._rc = rc & ~g.is_inh_r.bool()
        self.grp = OccSet(lambda: b.g.w, self._grp_idx(), half=True)
        # 下から：受け手が皮質の線（道の ch0 で見る）
        sets = {}
        ch0 = {int(b.off_ch[i]) for i in cx}
        but = getattr(b, "but", None)
        if but is not None:
            sel = torch.zeros(len(but.w), dtype=torch.bool, device=dev)
            for i, k in enumerate(but.keys):
                if int(b.bu[k].ch0) in ch0:
                    sel[but.eo[i]:but.eo[i + 1]] = True
            if bool(sel.any()):
                sets["下から"] = OccSet(lambda: b.but.w, torch.nonzero(sel).squeeze(1).int())
        if getattr(b, "cl_on", False):
            sets["CL"] = OccSet(lambda: b.cl_w, None)
        self.sets = sets
        self.keys_bu = [k for i, k in enumerate(but.keys) if int(b.bu[k].ch0) in ch0] if but is not None else []

    def _grp_idx(self):
        """当てる辺の番号（昇順・32bit）。番号の列は CPU で作る（GPU で nonzero すると 64bit の大きな列ができる）"""
        g = self.b.g
        rc = self._rc.cpu()
        sel = rc[g.e_recv.cpu().long()] & ~g.e_inh.cpu().bool()
        return torch.nonzero(sel).squeeze(1).int().to(g.w.device)

    def all_sets(self):
        return [self.grp] + list(self.sets.values())

    @torch.no_grad()
    def tick(self):
        """毎歩（学ぶ歩も学ばない歩も）呼ぶ"""
        self.t += 1
        if self.t % OCC_EVERY == 0:
            for s in self.all_sets():
                s.apply()
        if self.t % OCC_DECAY_EVERY == 0:
            for s in self.all_sets():
                s.advance(OCC_DECAY_EVERY)

    # ── 刈り込み（Group2.rewire）の前後 ──
    @torch.no_grad()
    def rewire_pre(self):
        """刈り込みで切る前に、学びの変わり（flush_all の分も）を読んでおく"""
        self.grp.apply()

    def _grp_idx_gpu(self):
        """版110b：当てる辺の番号（昇順・32bit）を GPU の上で塊ごとに（_grp_idx と同じ番号。塊ごとなので 64bit の大きな列はできない）"""
        g = self.b.g
        out = []
        for a in range(0, len(g.e_recv), CHUNK):
            sel = self._rc[g.e_recv[a:a + CHUNK].long()] & ~g.e_inh[a:a + CHUNK].bool()
            out.append((torch.nonzero(sel).squeeze(1) + a).int())
        return torch.cat(out)

    @staticmethod
    def _in_sorted(x, s):
        """x の各値が、昇順の s に入っているか"""
        if len(s) == 0:
            return torch.zeros(len(x), dtype=torch.bool, device=x.device)
        j = torch.searchsorted(s, x).clamp(max=len(s) - 1)
        return s[j] == x

    @torch.no_grad()
    def rewire_map(self, pos, cuts):
        """並べ直しの直後（新しい位置 ＝ pos[古い位置]、切って生やした辺 ＝ 新しい位置 cuts）。
        版110b：残る辺は並びの順が変わらない（_order_after_cut）ので、痕跡は「切った辺の分を抜く」だけ（GPU の上で塊ごと）。
          ⟲ 版104〜110：全部の辺の長さの表を CPU に作って置き直し（0.49秒）。入る値は同じ"""
        cs = torch.sort(cuts.to(pos.dtype)).values
        idx = self.grp.idx
        keep = torch.empty(len(idx), dtype=torch.bool, device=idx.device)       # 切られずに残る辺か
        last = -1
        for a in range(0, len(idx), CHUNK):
            npos = pos[idx[a:a + CHUNK].long()]
            k = ~self._in_sorted(npos, cs)
            keep[a:a + CHUNK] = k
            nk = npos[k]                                                         # 残る辺の新しい位置：昇順のままのはず
            if len(nk):
                if int(nk[0]) <= last or (len(nk) > 1 and bool((nk[1:] <= nk[:-1]).any())):
                    raise RuntimeError("ふさがり：残る辺の順が変わった（_order_after_cut の前提が崩れた）")
                last = int(nk[-1])
        self._moved = (keep, cs)

    @torch.no_grad()
    def rewire_done(self):
        """刈り込みの終わり（切った辺の重さを置き直したあと）：当てる辺を選び直し、痕跡を戻し、prev を今の重さに"""
        g = self.b.g
        keep, cs = self._moved
        self._moved = None
        idx = self._grp_idx_gpu()
        fresh = self._in_sorted(idx, cs)                                         # 切って生やした辺（痕跡は 0 から）
        if int((~fresh).sum()) != int(keep.sum()):
            raise RuntimeError("ふさがり：残る辺の数が合わない")
        for nm in ("q", "m"):
            old = getattr(self.grp, nm)
            kept = torch.cat([old[a:a + CHUNK][keep[a:a + CHUNK]] for a in range(0, len(old), CHUNK)])
            del old
            setattr(self.grp, nm, None)
            new = torch.zeros(len(idx), dtype=kept.dtype, device=kept.device)
            new.masked_scatter_(~fresh, kept)
            setattr(self.grp, nm, new)
            del kept, new
        self.grp.idx = idx
        self.grp.prev = torch.index_select(g.w, 0, idx).float()

    def counts(self):
        return {"領野の中": len(self.grp.q), **{k: len(s.q) for k, s in self.sets.items()}}
