"""版54b（コンテナの写し用。編集は claude/道具_0926/apply_v54b.py と同じ）"""
import ast, pathlib, sys


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:70]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


G2 = [
("""        self.off = torch.cumsum(torch.tensor([0] + self.n_src, device=dev), 0)
""", """        self.off = torch.cumsum(torch.tensor([0] + self.n_src, device=dev), 0)
        self.off_cpu = [sum(self.n_src[:k]) for k in range(len(self.n_src) + 1)]   # 版54b：境目は CPU の数で
"""),
("""            lo, hi = int(self.off[i]), int(self.off[i + 1])
""", """            lo, hi = self.off_cpu[i], self.off_cpu[i + 1]
"""),
("""        idx = torch.nonzero(fired_all > 0).squeeze(1)
        if len(idx):
            oldest = self.age.argmax(1)
            self.age[idx, oldest[idx]] = 0
""", """        # 版54b：撃った送り手だけ一番古い枠を 0 に（撃っていない所は同じ値を書き戻す。nonzero で待たない）
        oldest = self.age.argmax(1)
        rows = torch.arange(len(fired_all), device=self.dev)
        self.age[rows, oldest] = torch.where(fired_all > 0, torch.zeros_like(oldest), self.age[rows, oldest])
"""),
("""        self.is_inh_r = self.kind == 2
""", """        self.is_inh_r = self.kind == 2
        # 版54b：決まった印は番号の列でも持つ（印で選ぶと、数を読むために GPU を待つ）
        self.err_i, self.rep_i, self.inh_i = [torch.nonzero(m).squeeze(1) for m in (self.is_err, self.is_rep, self.is_inh_r)]
"""),
("""        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        fired, lens = fired[keep], lens[keep]
        if len(fired) == 0:
            return empty
        base = self.start[fired].repeat_interleave(lens)
        off = torch.arange(int(lens.sum()), device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens)
        return base + off
""", """        lens = self.start[fired + 1] - self.start[fired]
        # 版54b：総数を1回だけ読む（長さ 0 の送り手は repeat_interleave で何も出さないので、外さなくても同じ）
        tot = int(lens.sum())
        if tot == 0:
            return empty
        base = self.start[fired].repeat_interleave(lens, output_size=tot)
        off = torch.arange(tot, device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens, output_size=tot)
        return base + off
"""),
("""        self.k_start[idx] = self.kcur + 1
""", """        self.k_start.index_fill_(0, idx, self.kcur + 1)     # 版54b
"""),
("""                    self._amp_buf[self._amp_prev] = 1.0
""", """                    self._amp_buf.index_fill_(0, self._amp_prev, 1.0)     # 版54b
"""),
("""        mod = mod.index_add(0, re_[mm], amp[mm])
""", """        mod = mod.index_add(0, re_, amp * mm)     # 版54b：選ばない辺は 0 を足す（選んで足すのと同じ値）
"""),
("""        x_t = torch.where(self.tgt_by_fire, (x_all * self.x_gain > self.fire_th).float(), x_all)
        self.last_xt = x_t                         # 測る道具用
        tgt = x_t[self.x_index]
        tgt = torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], tgt)
""", """        if first:      # 版54b：入力 x は1歩のあいだ同じ ⇒ 的は1周目に出して使い回す
            x_t = torch.where(self.tgt_by_fire, (x_all * self.x_gain > self.fire_th).float(), x_all)
            self.last_xt = x_t                         # 測る道具用
            tgt = x_t[self.x_index]
            self._tgt = torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], tgt)
        tgt = self._tgt
"""),
("""        pr = p[self.is_rep]
""", """        pr = p[self.rep_i]
"""),
("""        drive = ((exc - exc_mod)[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])
""", """        drive = ((exc - exc_mod)[self.rep_i] + x_all * self.x_gain) / (1.0 + ihn[self.rep_i])
"""),
("""            if bool(ltd.any()):
                self.w[e[ltd]] *= INH_LTD_F
""", """            self.w[e] = torch.where(ltd, self.w[e] * INH_LTD_F, self.w[e])     # 版54b：判定で待たない
"""),
("""        err = (err_sig[self.is_err] > THETA).float()
""", """        err = (err_sig[self.err_i] > THETA).float()
"""),
("""        d_inh = (exc - exc_mod)[self.is_inh_r] / (1.0 + ihn[self.is_inh_r])
""", """        d_inh = (exc - exc_mod)[self.inh_i] / (1.0 + ihn[self.inh_i])
"""),
("""            fired[self.is_err], fired[self.is_rep], fired[self.is_inh_r] = err, rep, inh_spk
""", """            fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = err, rep, inh_spk
"""),
("""            now[self.is_err] = rep[self.x_index[self.is_err]]
            now[self.is_rep] = rep[self.x_index[self.is_rep]]
            now[self.is_inh_r] = inh_spk
""", """            now[self.err_i] = rep[self.x_index[self.err_i]]
            now[self.rep_i] = rep[self.x_index[self.rep_i]]
            now[self.inh_i] = inh_spk
"""),
("""        return rep * self.gate, err * self.gate, p[self.is_err]
""", """        return rep * self.gate, err * self.gate, p[self.err_i]
"""),
]
AR = [("""        self.last[e] = t
""", """        self.last.index_fill_(0, e, t)     # 版54b
""")]
BR = [("""        dg[top] = 1.0
""", """        dg.index_fill_(0, top, 1.0)     # 版54b
""")]
BB = [("#  54  意味を変えずに1歩を縮める：周回の使い回し ／ 学びを触ったときにまとめる ／ 焼いた網を1歩1回（【み】10-10）\n",
       "#  54  意味を変えずに1歩を縮める：周回の使い回し ／ 学びを触ったときにまとめる ／ 焼いた網を1歩1回（【み】10-10）\n"
       "#  54b 意味を変えずに CPU の待ちを減らす（印で選ぶ → 番号で選ぶ ほか。measure_sync 10-10）\n")]

if __name__ == "__main__":
    out = [edit("torch_group2.py", G2), edit("torch_area.py", AR), edit("torch_brain.py", BR), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版54b を当てた")
