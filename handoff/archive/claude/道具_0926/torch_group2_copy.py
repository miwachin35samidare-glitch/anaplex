"""領野をまとめて1回で進める v2（arkhe 用）—— 領野ごとのループを全部消す

⚠️ 切り分け用の写し：プロジェクトの torch_group2.py（版35）をそのまま写した
"""
import torch
from torch_area import TAUS, THETA


class TimeCellsGroup:
    """8領野ぶんの時間細胞を、ひとつの行列で持つ（領野をまたいで混ざらない）"""

    def __init__(self, areas):
        dev = areas[0].dev
        self.dev = dev
        self.areas = areas
        self.M = areas[0].ch.M
        self.n_src = [a.ch.n_src for a in areas]
        self.off = torch.cumsum(torch.tensor([0] + self.n_src, device=dev), 0)
        self.age = torch.cat([a.ch.age for a in areas])          # [送り元の合計, M]
        self.n_cell = [a.ch.n_cell for a in areas]
        self.n_ramp = [a.ch.n_ramp for a in areas]
        self.pref = [a.ch.pref for a in areas]
        self.half = [a.ch.half for a in areas]
        self.ramp_len = [a.ch.ramp_len for a in areas]
        self.gen = areas[0].ch.gen
        self._cache = None

    @torch.no_grad()
    def spikes(self):
        if self._cache is not None:
            return self._cache
        out = []
        for i, a in enumerate(self.areas):
            lo, hi = int(self.off[i]), int(self.off[i + 1])
            age = self.age[lo:hi]
            parts = []
            if self.n_cell[i]:
                f = age.unsqueeze(-1).float()
                hit = ((f - self.pref[i]).abs() <= self.half[i]) & (f <= self.pref[i].max())
                parts.append(hit.any(1).float())
            y = (age.min(1).values.float() / self.ramp_len[i]).clamp(0, 1).unsqueeze(-1)
            h = self.n_ramp[i] // 2
            up = torch.rand(hi - lo, h, device=self.dev, generator=self.gen) < y * 0.3
            dn = torch.rand(hi - lo, self.n_ramp[i] - h, device=self.dev,
                            generator=self.gen) < (1 - y) * 0.3
            parts.append(torch.cat([up, dn], 1).float())
            out.append(torch.cat(parts, 1).reshape(-1))
        self._cache = torch.cat(out)
        return self._cache

    @torch.no_grad()
    def step(self, fired_all):
        self._cache = None
        self.age = torch.where(self.age < 10 ** 8, self.age + 1, self.age)
        idx = torch.nonzero(fired_all > 0).squeeze(1)
        if len(idx):
            oldest = self.age.argmax(1)
            self.age[idx, oldest[idx]] = 0


class Group2:
    def __init__(self, areas):
        self.areas = list(areas)
        dev = self.areas[0].dev
        self.dev, self.recur = dev, self.areas[0].recur
        self.prior = self.areas[0].prior
        self.eta = self.areas[0].eta
        self.homeo_tau = self.areas[0].homeo_tau
        self.scaling_target = self.areas[0].scaling_target

        cs = lambda xs: torch.cumsum(torch.tensor([0] + xs, device=dev), 0)
        self.off_recv = cs([a.n_recv for a in self.areas])
        self.off_ch = cs([a.n_ch for a in self.areas])
        self.off_src = cs([a.n_src for a in self.areas])
        self.n_recv_all = int(self.off_recv[-1])
        self.n_ch_all = int(self.off_ch[-1])
        self.n_src_all = int(self.off_src[-1])

        ch_off = torch.cumsum(torch.tensor([0] + [a.ch.n_cells for a in self.areas], device=dev), 0)
        buf_base = int(ch_off[-1])
        buf_off = torch.cumsum(torch.tensor([0] + [a.n_src * a.D for a in self.areas], device=dev), 0)
        remap = []
        for i, a in enumerate(self.areas):
            m = torch.empty(a.n_send, dtype=torch.long, device=dev)
            m[:a.ch.n_cells] = torch.arange(a.ch.n_cells, device=dev) + ch_off[i]
            m[a.ch.n_cells:] = torch.arange(a.n_src * a.D, device=dev) + buf_base + buf_off[i]
            remap.append(m)
        self.off_send = torch.cat([ch_off[:-1], torch.tensor([buf_base], device=dev)])
        src = torch.cat([remap[i][a.e_src_sorted] for i, a in enumerate(self.areas)])
        recv = torch.cat([a.e_recv.long() + self.off_recv[i] for i, a in enumerate(self.areas)])
        w = torch.cat([a.w for a in self.areas])
        inh = torch.cat([a.e_inh for a in self.areas])
        self.n_send_all = buf_base + int(buf_off[-1])
        self.is_inh_all = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)
        for i, a in enumerate(self.areas):
            self.is_inh_all[remap[i]] = a.is_inh_send
        self.pool_send = remap
        o = torch.argsort(src)
        self.e_src, self.e_recv, self.w, self.e_inh = src[o], recv[o], w[o], inh[o]
        self.start = torch.zeros(self.n_send_all + 1, dtype=torch.long, device=dev)
        self.start.scatter_add_(0, self.e_src + 1, torch.ones(len(o), dtype=torch.long, device=dev))
        self.start = self.start.cumsum(0)

        self.stp = None
        if self.areas[0].stp is not None:
            import copy
            g = copy.copy(self.areas[0].stp)
            for k in ("row", "state", "fac", "last", "deplete"):
                setattr(g, k, torch.cat([getattr(a.stp, k) for a in self.areas])[o])
            self.stp = g

        self.scale = torch.cat([a.scale for a in self.areas])
        self.drive = torch.cat([a.drive for a in self.areas])
        self.recv_ema = torch.cat([a.recv_ema for a in self.areas])
        self.rate_ema = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)
        self.decay = torch.cat([a.decay.unsqueeze(0).expand(a.n_recv, -1) for a in self.areas])
        self.tb = torch.cat([a.tb for a in self.areas])
        self.bufs = [a.buf for a in self.areas]
        self.Ds = [a.D for a in self.areas]
        self.ch = TimeCellsGroup(self.areas)

        kind, area_of, ch_of = [], [], []
        for i, a in enumerate(self.areas):
            kind += [0] * a.n_ch + [1] * a.n_ch + [2] * a.n_inh
            area_of += [i] * a.n_recv
            ch_of += list(range(a.n_ch)) + list(range(a.n_ch)) + [-1] * a.n_inh
        self.kind = torch.tensor(kind, device=dev)
        self.area_of = torch.tensor(area_of, device=dev)
        self.ch_of = torch.tensor(ch_of, device=dev)
        self.is_err = self.kind == 0
        self.is_rep = self.kind == 1
        self.is_inh_r = self.kind == 2
        self.x_index = torch.zeros(self.n_recv_all, dtype=torch.long, device=dev)
        for i, a in enumerate(self.areas):
            lo = int(self.off_recv[i]); c = int(self.off_ch[i])
            self.x_index[lo:lo + a.n_ch] = torch.arange(a.n_ch, device=dev) + c
            self.x_index[lo + a.n_ch:lo + 2 * a.n_ch] = torch.arange(a.n_ch, device=dev) + c
        self.inh_area = self.area_of[self.is_inh_r]
        self.mean_map = torch.zeros(self.n_ch_all, len(self.areas), device=dev)
        for i, a in enumerate(self.areas):
            c = int(self.off_ch[i])
            self.mean_map[c:c + a.n_ch, i] = 1.0 / a.n_ch
        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])
        self.complete = torch.cat([torch.full((a.n_ch,), float(a.complete), device=dev)
                                   for a in self.areas])
        self.pred_boost = torch.cat([torch.full((a.n_ch,), a.pred_boost, device=dev)
                                     for a in self.areas])
        self.gate = torch.cat([torch.full((a.n_ch,), 1.0 if a.gate else 0.0, device=dev)
                               for a in self.areas])
        self.gen = self.areas[0].gen
        self.fire_th = torch.cat([torch.full((a.n_ch,), a.fire_th, device=dev) for a in self.areas])
        self.x_gain = torch.cat([torch.full((a.n_ch,), a.x_gain, device=dev) for a in self.areas])
        self.c_exc = torch.cat([torch.full((a.n_recv,), float(getattr(a, "contacts_exc", 3.3)), device=dev)
                                for a in self.areas])
        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
        self.t = 0
        self._e_cache = None
        self._within = None
        self._base = None
        self.scaling_on = True
        self.exc_gain = None
        self.sleep_mode = False
        self.is_hippo = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)
        self.rewire_every = self.areas[0].rewire_every
        self.n_partner = torch.cat([torch.full((a.n_recv,), float(a.n_partner), device=dev)
                                    for a in self.areas])
        self.budget = torch.cat([a.budget for a in self.areas])
        self.within_index = torch.cat(
            [torch.arange(a.n_ch, device=dev) * a.D + buf_base + int(buf_off[i])
             for i, a in enumerate(self.areas)])
        self._release_areas()

    def _release_areas(self):
        for a in self.areas:
            for k in ("w", "e_src_sorted", "e_recv", "e_inh", "start", "is_inh_send"):
                if getattr(a, k, None) is not None:
                    setattr(a, k, None)
            if getattr(a, "stp", None) is not None:
                for k in ("row", "state", "fac", "last", "deplete"):
                    setattr(a.stp, k, None)
            if getattr(a.ch, "age", None) is not None:
                a.ch.age = None
        if str(self.dev) != "cpu":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _fired_edges(self):
        if self._e_cache is not None:
            return self._e_cache
        if self._base is None:
            self._base = torch.cat([self.ch.spikes()] + [b.reshape(-1) for b in self.bufs])
        s = self._base
        if self._within is not None:
            s = torch.maximum(s, self._within)
        fired = torch.nonzero(s).squeeze(1)
        empty = torch.empty(0, dtype=torch.long, device=self.dev)
        if len(fired) == 0:
            self._e_cache = empty
            return empty
        lens = self.start[fired + 1] - self.start[fired]
        keep = lens > 0
        fired, lens = fired[keep], lens[keep]
        if len(fired) == 0:
            self._e_cache = empty
            return empty
        base = self.start[fired].repeat_interleave(lens)
        off = torch.arange(int(lens.sum()), device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens)
        self._e_cache = base + off
        return self._e_cache

    @torch.no_grad()
    def rewire(self):
        target = self.n_partner * 0.5
        self.budget = torch.minimum(
            (self.budget + 0.5 * torch.sign(target - self.recv_ema)).clamp(min=1.0),
            self.n_partner * 3)
        self.n_rewire = getattr(self, "n_rewire", 0) + 1
        big = self.w.abs().max() + 1.0
        key = self.e_recv.float() * big + self.w.abs()
        order = torch.argsort(key)
        r_sorted = self.e_recv[order]
        first = torch.ones_like(r_sorted, dtype=torch.bool)
        first[1:] = r_sorted[1:] != r_sorted[:-1]
        cuts = order[first]
        if len(cuts) == 0:
            return
        self.n_cut = getattr(self, "n_cut", 0) + len(cuts)
        ar = self.area_of[self.e_recv[cuts]]
        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)
        for i in range(len(self.areas)):
            m = ar == i
            k = int(m.sum())
            if k:
                pool = self.pool_send[i]
                new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev,
                                            generator=self.gen)]
        self.e_src[cuts] = new
        o = torch.argsort(self.e_src)
        self.e_src, self.e_recv, self.w = self.e_src[o], self.e_recv[o], self.w[o]
        self.e_inh = self.is_inh_all[self.e_src]
        self.w[cuts] = 0.0
        if self.stp is not None:
            self.stp.reorder(o)
        self.start = torch.zeros(self.n_send_all + 1, dtype=torch.long, device=self.dev)
        self.start.scatter_add_(0, self.e_src + 1,
                                torch.ones(len(self.w), dtype=torch.long, device=self.dev))
        self.start = self.start.cumsum(0)

    @torch.no_grad()
    def step(self, x_all, ext_all, learn=True):
        self._within = None
        for i in range(self.recur):
            out = self._pass(x_all, ext_all, learn and i == 0, i == self.recur - 1, i == 0)
        if learn and self.rewire_every and self.t % self.rewire_every == 0:
            self.rewire()
        return out

    @torch.no_grad()
    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
        exc = torch.zeros(self.n_recv_all, device=self.dev)
        ihn = torch.zeros(self.n_recv_all, device=self.dev)
        if len(e):
            amp = torch.where(self.e_inh[e], 4.0, self.c_exc[self.e_recv[e]]) * self.w[e]
            if self.stp is not None:
                if first:
                    self._amp = torch.ones(len(self.w), device=self.dev)
                    self._amp[e] = self.stp.release_at(e, self.t)
                amp = amp * self._amp[e]
            r = self.e_recv[e]
            both = torch.zeros(self.n_recv_all * 2, device=self.dev)
            both.scatter_add_(0, r + self.n_recv_all * self.e_inh[e].long(), amp)
            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]
        exc = exc * self.scale; ihn = ihn * self.scale
        if getattr(self, "exc_gain", None) is not None:
            exc = exc * self.exc_gain
        p = torch.sigmoid(self.prior.logit(self.tb) + exc / (1.0 + ihn))

        x_t = torch.where(self.tgt_by_fire, (x_all * self.x_gain > self.fire_th).float(), x_all)
        self.last_xt = x_t
        tgt = x_t[self.x_index]
        tgt = torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], tgt)
        err_sig = tgt - p

        if learn and len(e):
            r = self.e_recv[e]
            gg = self.eta * err_sig[r]
            shunt = (exc / (1.0 + ihn) ** 2)[r]
            self.w[e] += torch.where(self.e_inh[e], -gg * shunt, gg)
            self.w[e] = torch.where(self.e_inh[e], self.w[e].clamp(min=0.0), self.w[e])
            if getattr(self, "exc_nonneg", False):      # 切り分け用：興奮の重みも 0 以上（人の形）
                self.w[e] = self.w[e].clamp(min=0.0)
            a_ = 1.0 / self.homeo_tau
            cnt = torch.zeros(self.n_recv_all, device=self.dev)
            cnt.scatter_add_(0, r, torch.ones(len(e), device=self.dev))
            self.recv_ema += a_ * (cnt - self.recv_ema)
            self.drive += a_ * (exc.abs() - self.drive)
            if getattr(self, "scaling_on", True):
                f = (1 - 0.01 * (self.rate_ema / self.scaling_target - 1)).clamp(0.5, 1.5) ** a_
                if getattr(self, "sleep_mode", False):
                    f = torch.where(self.is_hippo, torch.ones_like(f), f)
                self.scale = self.scale * f

        err = (err_sig[self.is_err] > THETA).float()
        pr = p[self.is_rep]
        if getattr(self, "rep_below_only", False):   # 切り分け用：表現の細胞体は下からだけ（案1）
            drive = x_all * self.x_gain
        else:
            drive = (exc[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])
        rep = (drive > self.fire_th).float()
        if self.complete.any():
            fill = (self.complete > 0) & ((drive + self.pred_boost * pr) > self.fire_th) & (x_all > 0)
            rep = torch.where(fill, torch.ones_like(rep), rep)
        rep = torch.maximum(rep, (torch.rand(self.n_ch_all, device=self.dev,
                                             generator=self.gen) < self.spont_p).float())
        inh_spk = (p[self.is_inh_r] > 0.5).float()
        if learn:
            fired = torch.zeros(self.n_recv_all, device=self.dev)
            fired[self.is_err], fired[self.is_rep], fired[self.is_inh_r] = err, rep, inh_spk
            self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)
        self.pred_rep = pr
        for i, a in enumerate(self.areas):
            a.pred_rep = pr[int(self.off_ch[i]):int(self.off_ch[i]) + a.n_ch]

        w_new = torch.zeros(self.n_send_all, device=self.dev)
        w_new[self.within_index] = rep
        self._within = w_new if self._within is None else torch.maximum(self._within, w_new)
        self._e_cache = None

        # ── 切り分け用の記録（中身は変えない。見るだけ）──
        self.last_exc, self.last_ihn = exc, ihn
        self.last_drive, self.last_rep_below = drive, (drive > self.fire_th).float()

        if advance:
            self.tb = self.tb * self.decay + torch.cat(
                [rep[self.x_index[self.is_err]], rep[self.x_index[self.is_rep]], inh_spk]
            ).unsqueeze(1) * (1 - self.decay)
            src = torch.empty(self.n_src_all, device=self.dev)
            eo = io = 0
            for i, a in enumerate(self.areas):
                lo = int(self.off_src[i]); c = int(self.off_ch[i])
                src[lo:lo + a.n_ch] = rep[c:c + a.n_ch]
                if a.n_ext:
                    src[lo + a.n_ch:lo + a.n_ch + a.n_ext] = ext_all[eo:eo + a.n_ext]
                    eo += a.n_ext
                src[lo + a.n_ch + a.n_ext:int(self.off_src[i + 1])] = inh_spk[io:io + a.n_inh]
                io += a.n_inh
            self.ch.step(src)
            for i, a in enumerate(self.areas):
                lo, hi = int(self.off_src[i]), int(self.off_src[i + 1])
                self.bufs[i] = torch.cat([src[lo:hi].unsqueeze(1), self.bufs[i][:, :-1]], 1)
            self.t += 1
            self._e_cache = None
            self._base = None
            self._within = None
        return rep * self.gate, err * self.gate, p[self.is_err]
