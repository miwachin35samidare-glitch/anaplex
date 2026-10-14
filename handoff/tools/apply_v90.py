'''版90：1周目から最後の周までを「1つの」グラフに（1歩の網の中で CPU が命令を1つずつ渡すのは、くじ・放出・歩の終わりだけ）
（~/anaplex で。版89 の上に当てる。印が1つでも合わなければ何も書かずに止まる。速さだけ。式・くじの順は同じ）
  python3 apply_v90.py
【み】10-15「稼げるならやっておいて」
【測】arkhe 版88：1歩 9.8ms（実時間の 0.92倍）／ 1周目 7.28ms（GPU の中身は約3.5ms）／ GPU 7.8ms
直すもの（torch_group2）
  1周目の「学びの分を足す＋足し込み」「学ぶ＋2周目の頭」を Triton の核に（辺の番号の列と本数は GPU の箱に置き、本数まで回る）
  ⇒ 1周目の残り（駆動・的・恒常性・誤差・不応期）と 2周目から最後の周までを1つのグラフで（1歩に1回の再生）
  グラフの外（今まで通り、1つずつ）：時間細胞のくじ・撃った辺を集める（辺の数を1回読む）・発火の履歴・放出（くじ binomial）・
    焼いた網の見込み・自発のくじ（1周目の分のあとに 2周目からの分。今と同じ順）・学びの刻み・歩の終わり
  学びの区切り（LAZY_EPOCH）の歩・撃った辺が無い歩・測る道具が中を差し替えているときは版89 と同じ形
  ANAPLEX_V90=0 で版89 と同じ
【確】コンテナ（CPU。Triton の核の代わりに同じ式の torch で）版78 と 200歩 ／ Triton の核は解釈実行で式どおりの値と照合
'''
import ast, pathlib, re, sys

KERNELS = r'''
# 版90：1周目から最後の周までを1つのグラフに。ANAPLEX_V90=0 で版89 と同じ
V90 = _os.environ.get("ANAPLEX_V90", "1") != "0"
if _TRITON:
    @triton.jit
    def _p1_flush_k(E_p, ne_p, k_start_p, elig_p, A_p, anow_p, kc1_p, w_p, e_recv_p, e_src_p, e_inh_p, c_exc_p,
                    amp_p, send_gain_p, mod_send_p, mdi_p, iir_p, both_p, mod_p, n_recv,
                    HAS_AMP: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        """版90：_flush_sum_core（1周目の 学びの分を足す＋足し込み）と同じ式。辺は E の先頭 ne 本（重なりなし）"""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ne = tl.load(ne_p).to(tl.int32)
        a_now = tl.load(anow_p)
        kc1 = tl.load(kc1_p)
        rep = pid % K
        for base in range(pid * BLOCK, ne, nprog * BLOCK):
            j = base + tl.arange(0, BLOCK)
            m = j < ne
            e = tl.load(E_p + j, mask=m, other=0)
            ks = tl.load(k_start_p + e, mask=m, other=1).to(tl.int64)
            ak = tl.load(A_p + ks - 1, mask=m, other=0.0)
            w1 = tl.maximum(tl.load(w_p + e, mask=m, other=0.0) + tl.load(elig_p + e, mask=m, other=0.0) * (a_now - ak).to(tl.float32), 0.0)
            tl.store(w_p + e, w1, mask=m)
            tl.store(k_start_p + e, (tl.zeros([BLOCK], dtype=tl.int32) + kc1.to(tl.int32)).to(k_start_p.dtype.element_ty), mask=m)
            r = tl.load(e_recv_p + e, mask=m, other=0)
            se = tl.load(e_src_p + e, mask=m, other=0)
            ie = tl.load(e_inh_p + e, mask=m, other=0).to(tl.int32)
            amp = tl.where(ie != 0, 4.0, tl.load(c_exc_p + r, mask=m, other=0.0)) * w1
            if HAS_AMP:
                amp = amp * tl.load(amp_p + e, mask=m, other=1.0)
            amp = amp * tl.load(send_gain_p + se, mask=m, other=0.0)
            ms = tl.load(mod_send_p + se, mask=m, other=0).to(tl.int32)
            mdi = tl.load(mdi_p + se, mask=m, other=0).to(tl.int32)
            iir = tl.load(iir_p + r, mask=m, other=0).to(tl.int32)
            mm = (ms != 0) & (ie == 0) & ((mdi == 0) | (iir == 0))
            tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie, amp, mask=m, sem="relaxed")
            tl.atomic_add(mod_p + rep * n_recv + r, amp, mask=m & mm, sem="relaxed")

    @triton.jit
    def _p1_learn_k(E_p, ne_p, e_recv_p, e_src_p, e_inh_p, iir_p, err_p, g_p, pw_p, elig_p, hist_n_p, w_p, c_exc_p,
                    k_start_p, A_p, ak_p, kck_p, amp_p, send_gain_p, mod_send_p, mdi_p, both_p, mod_p, cnt_p, n_recv,
                    LTD_N, LTD_TH, LTD_F, HAS_AMP: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        """版90：_learn_sum_core（学ぶ ＋ 2周目の頭の 学びの分を足す・足し込み）と同じ式。辺の数も数える"""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ne = tl.load(ne_p).to(tl.int32)
        g = tl.load(g_p)
        pw = tl.load(pw_p)
        a_k = tl.load(ak_p)
        kck = tl.load(kck_p)
        rep = pid % K
        for base in range(pid * BLOCK, ne, nprog * BLOCK):
            j = base + tl.arange(0, BLOCK)
            m = j < ne
            e = tl.load(E_p + j, mask=m, other=0)
            r = tl.load(e_recv_p + e, mask=m, other=0)
            gg = g * tl.load(err_p + r, mask=m, other=0.0)
            to_inh = tl.load(iir_p + r, mask=m, other=0).to(tl.int32) != 0
            ie = tl.load(e_inh_p + e, mask=m, other=0).to(tl.int32) != 0
            el = tl.load(elig_p + e, mask=m, other=0.0) + tl.where(ie | to_inh, 0.0, gg) * pw
            tl.store(elig_p + e, el, mask=m)
            se = tl.load(e_src_p + e, mask=m, other=0)
            w0 = tl.load(w_p + e, mask=m, other=0.0)
            cx = tl.load(c_exc_p + r, mask=m, other=0.0)
            hn = tl.load(hist_n_p + se, mask=m, other=0).to(tl.int32)
            ltd = to_inh & (~ie) & (hn >= LTD_N) & (w0 * cx > LTD_TH)
            wl = tl.where(ltd, w0 * LTD_F, w0)
            ks = tl.load(k_start_p + e, mask=m, other=1).to(tl.int64)
            ak = tl.load(A_p + ks - 1, mask=m, other=0.0)
            w2 = tl.maximum(wl + el * (a_k - ak).to(tl.float32), 0.0)
            tl.store(w_p + e, w2, mask=m)
            tl.store(k_start_p + e, (tl.zeros([BLOCK], dtype=tl.int32) + kck.to(tl.int32)).to(k_start_p.dtype.element_ty), mask=m)
            amp = tl.where(ie, 4.0, cx) * w2
            if HAS_AMP:
                amp = amp * tl.load(amp_p + e, mask=m, other=1.0)
            amp = amp * tl.load(send_gain_p + se, mask=m, other=0.0)
            ms = tl.load(mod_send_p + se, mask=m, other=0).to(tl.int32) != 0
            mdi = tl.load(mdi_p + se, mask=m, other=0).to(tl.int32) != 0
            mm = ms & (~ie) & ~(mdi & to_inh)
            tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie.to(tl.int32), amp, mask=m, sem="relaxed")
            tl.atomic_add(mod_p + rep * n_recv + r, amp, mask=m & mm, sem="relaxed")
            tl.atomic_add(cnt_p + rep * n_recv + r, tl.full([BLOCK], 1, tl.int32), mask=m, sem="relaxed")


'''

METHODS = r'''
    def _fast_ok(self, learn):
        """版90：1周目から最後の周までを1つのグラフの形で回すか"""
        if not (V90 and V88 and self._gp_ok() and self._one_ok() and self._fuse_ok() and self.stp is not None):
            return False
        if type(self)._pass is not _G2_ORIG["_pass"] or type(self.areas[0].prior).logit is not _PRIOR_LOGIT0:
            return False
        if learn and self.kcur >= LAZY_EPOCH:          # 学びの区切りの歩は今まで通り
            return False
        return (_TRITON and str(self.dev).startswith("cuda")) or not str(self.dev).startswith("cuda")

    def _own(self, name):
        """版90：グラフが中身を書き換える値は、網が持つ1つの置き場所に（読み込みなどで置き換わったら、その置き場所を持ち直す）"""
        v = getattr(self, name)
        if getattr(self, "_own_" + name, None) is not v:
            setattr(self, "_own_" + name, v)
        return v

    @torch.no_grad()
    def _prior_now(self):
        """版90：焼いた網の見込み（1歩に1回。_pass と同じ）"""
        if self._prior_l is None:
            if len(self._prior_segs) == 1:
                self._prior_l = self.prior.logit(self.tb)
            else:
                if getattr(self, "_pri_stack", None) is None:
                    ps = [pr for _, _, pr in self._prior_segs]
                    self._pri_stack = tuple(torch.stack([getattr(p, k) for p in ps]) for k in ("W1", "b1", "W2", "b2"))
                    self._pri_segs = tuple((lo, hi) for lo, hi, _ in self._prior_segs)
                self._prior_l = _run(_prior_core, self.tb, *self._pri_stack, self._pri_segs)
        return self._prior_l

    @torch.no_grad()
    def _fast_step(self, x_all, ext_all, learn):
        """版90：1歩の網（版89 の _pass の1周目 ＋ _gp_passes と同じ式・同じくじの順）。撃った辺が無い歩は None（今まで通りで回す）"""
        e = self._fired_edges()
        if len(e) == 0:
            return None
        ne = len(e)
        # 発火の履歴（_pass と同じ）
        now = (self._base > 0).to(torch.uint8)
        self.hist_n += now.to(torch.int16) - self.hist[self.hist_i].to(torch.int16)
        self.hist[self.hist_i] = now
        self.hist_i = (self.hist_i + 1) % INH_LTD_WIN
        # 放出（_flush_sum の前半と同じ。くじ binomial はここ）
        if self._amp_buf is None or len(self._amp_buf) != len(self.w):
            self._amp_buf = torch.ones(len(self.w), device=self.dev)
        elif self._amp_prev is not None:
            self._amp_buf.index_fill_(0, self._amp_prev, 1.0)
        self._amp_buf[e] = self.stp.release_at(e, self.t)
        self._amp_prev = e
        self._amp = self._amp_buf
        self._both_base = None
        prior_l = self._prior_now()
        S = self._gp_state()
        F = getattr(self, "_fS", None)
        if F is None or F["E"].shape[0] < ne:
            cap = max(1 << 22, int(ne * 1.25))
            z = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device=self.dev)
            F = self._fS = dict(E=z(cap, torch.long), ne=z((), torch.long), spont1=z(self.n_ch_all),
                                a_now1=z((), self.A.dtype), kc1=z((), self.k_start.dtype), xt=z(self.n_ch_all))
            self._graphs.clear()
        sp = S.get("spont_all")
        if sp is None or sp.shape[0] != self.recur - 1:
            sp = S["spont_all"] = torch.zeros(self.recur - 1, self.n_ch_all, device=self.dev)
            self._graphs.clear()
        # 自発のくじ：1周目の分、続けて 2周目からの分（今と同じ順。その間に他のくじは引かない）
        torch.rand(self.n_ch_all, device=self.dev, generator=self.gen, out=F["spont1"])
        for i in range(self.recur - 1):
            torch.rand(self.n_ch_all, device=self.dev, generator=self.gen, out=sp[i])
        # 学びの刻み（_pass と同じ。区切りの歩はここに来ない）
        F["a_now1"].copy_(self.A[self.kcur])
        F["kc1"].fill_(self.kcur + 1)
        if learn:
            k = self.kcur + 1
            if getattr(self, "_g_buf", None) is None:
                self._g_buf = torch.zeros((), dtype=self.w.dtype, device=self.dev)
                self._pw_buf = torch.zeros((), dtype=self.w.dtype, device=self.dev)
            eg = self.eta * self.eta_gain
            self._g_buf.copy_(eg) if torch.is_tensor(eg) else self._g_buf.fill_(eg)
            self._pw_buf.fill_(self.elig_decay ** (-k))
            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(()) * (self.elig_decay ** k)
            self.kcur = k
        S["a_now"].copy_(self.A[self.kcur])           # 2周目から（と学ぶときの 2周目の頭）の刻み
        S["kcur1"].fill_(self.kcur + 1)
        F["E"][:ne].copy_(e)
        F["ne"].fill_(ne)
        # 周をまたぐ値の初め（_pass の1周目と同じ）
        if getattr(self, "v_carry", None) is not None and self.v_carry_inh is None:
            self.v_carry_inh = torch.zeros(len(self.inh_i), device=self.dev)
        self._learn_step = learn
        if getattr(self, "_last_fire", None) is None:
            self._pc = 0
            self._pc_t = torch.zeros((), dtype=torch.long, device=self.dev)
            self._last_fire = [torch.full((n,), -10 ** 6, dtype=torch.long, device=self.dev)
                               for n in (len(self.rep_i), len(self.err_i), len(self.inh_i))]
        if getattr(self, "_pc_t", None) is None:
            self._pc_t = torch.full((), self._pc, dtype=torch.long, device=self.dev)
        for i in range(3):
            if self._last_fire[i] is not S[f"lf{i}"]:
                S[f"lf{i}"].copy_(self._last_fire[i])
        S["pc"].copy_(self._pc_t)
        # グラフに渡す値（版86 と同じ：置き場所の変わらない値は直に読む・変わる値は写す）
        vc = getattr(self, "v_carry", None)
        vin = dict(base_on=self._base_on, prior_l=prior_l, x_all=x_all, scale=self._own("scale"),
                   scale_up=self._own("scale_up"), exc_gain=getattr(self, "exc_gain", None),
                   cl_drive=getattr(self, "cl_drive", None), v_carry=vc,
                   v_carry_inh=(self.v_carry_inh if vc is not None else None), th_gain=self.th_gain,
                   x_gain=self.x_gain, fire_th=self.fire_th, complete=(self.complete if self._complete_any else None),
                   pred_boost=self.pred_boost, spont_p_pass=self.spont_p_pass, send_gain=self.send_gain)
        X, key = self._gX, []
        AL = self.__dict__.setdefault("_galias", {})
        CP = self.__dict__.setdefault("_gcopy", set())
        for k_, v in vin.items():
            if v is None:
                key.append((k_, False))
                continue
            if k_ in ("scale", "scale_up"):            # グラフが中身を書き換える ⇒ 網の置き場所を直に
                if X.get(k_) is not v:
                    X[k_] = v
                    self._graphs.clear()
                key.append((k_, True))
                continue
            if k_ in AL:
                if v is AL[k_]:
                    key.append((k_, True))
                    continue
                del AL[k_]
                X.pop(k_, None)
                CP.add(k_)
                self._graphs.clear()
            elif torch.is_tensor(v) and k_ not in CP and k_ not in X:
                AL[k_] = X[k_] = v
                self._graphs.clear()
                key.append((k_, True))
                continue
            shp, dt = (tuple(v.shape), v.dtype) if torch.is_tensor(v) else ((), torch.float32)
            b = X.get(k_)
            if b is None or tuple(b.shape) != shp or b.dtype != dt or b is v:
                b = X[k_] = torch.empty(shp, dtype=dt, device=self.dev)
                self._graphs.clear()
            if torch.is_tensor(v):
                b.copy_(v)
            else:
                b.fill_(float(v))
            key.append((k_, True))
        for k_, n in (("both_base", self.n_recv_all * 2), ("mod_base", self.n_recv_all), ("tgt", self.n_recv_all)):
            if X.get(k_) is None or k_ in AL or X[k_].shape[0] != n:   # グラフの中で作る値は自分の置き場所に
                AL.pop(k_, None)
                CP.add(k_)
                X[k_] = torch.zeros(n, device=self.dev)
                self._graphs.clear()
        key += [("learn", bool(learn)), ("scaling", bool(getattr(self, "scaling_on", True))),
                ("sleep", bool(getattr(self, "sleep_mode", False)))]
        key = tuple(key)
        pins = (self.w, self.e_recv, self.e_src, self.e_inh, self.c_exc, self.k_start, self.elig, self.A, self.start,
                self.mod_send, self.mod_drive_inh, self.is_inh_r, self.rep_i, self.err_i, self.inh_i,
                self.within_index, self.within_inh_index, self._amp_buf, self.recv_ema, self.drive, self.hist_n,
                self.x_index, self.mean_map, self.tgt_by_fire, self.is_inh_r, self.area_of, self.target_recv,
                self.rate_ema, self.rate_long, self.scale_skip, self.is_hippo, F["E"])
        pk = tuple((t.data_ptr(), t.numel()) for t in pins)
        if pk != self._gpins:
            self._graphs.clear()
            self._gpins = pk
        self._fast_run(key, e)
        self.fast_steps = getattr(self, "fast_steps", 0) + 1
        # 歩の終わり（_gp_passes の終わりと同じ）
        self._pc += self.recur
        self._last_fire = [S["lf0"], S["lf1"], S["lf2"]]
        self._any = [S["any0"], S["any1"], S["any2"]]
        self._pc_t.copy_(S["pc"])
        self._drive_last, self._dinh_last = S["drive"], S["dinh"]
        self._within = S["within"]
        self._tgt = X["tgt"]
        self.last_xt = F["xt"]
        self._both_base, self._mod_base = X["both_base"], X["mod_base"]
        if learn:
            self._rate_obs()
        pr = S["pr"].clone()
        self.pred_rep = pr
        for i, a in enumerate(self.areas):
            a.pred_rep = pr[int(self.off_ch[i]):int(self.off_ch[i]) + a.n_ch]
        rep, err = self._advance_tail(ext_all)
        return rep * self.gate, err * self.gate, S["perr"].clone()

    def _fast_run(self, key, e):
        """版90：1歩の網を1回の再生で（1回だけ撮る）。撮る前の慣らしで変わる値は、慣らしの前に戻す"""
        if GRAPH == "eager" or not str(self.dev).startswith("cuda"):
            return self._fast_body(key)
        g = self._graphs.get(("fast", key))
        if g is None:
            S, X, F = self._gS, self._gX, self._fS
            keep = [S[k] for k in S] + [X["both_base"], X["mod_base"], X["tgt"], F["xt"], self.recv_ema, self.drive,
                                        self.scale, self.scale_up]
            snap = [t.clone() for t in keep]
            we, ee, ke = self.w[e].clone(), self.elig[e].clone(), self.k_start[e].clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._fast_body(key)                   # 慣らし（compile・Triton を済ませる）
            torch.cuda.current_stream().wait_stream(s)
            for t, v in zip(keep, snap):
                t.copy_(v)
            self.w.index_put_((e,), we)
            self.elig.index_put_((e,), ee)
            self.k_start.index_put_((e,), ke)
            if getattr(self, "_gpool", None) is None:
                self._gpool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._gpool):
                self._fast_body(key)
            self._graphs[("fast", key)] = g
            self.n_capture = getattr(self, "n_capture", 0) + 1
        g.replay()

    def _fast_body(self, key):
        """版90：1周目（学びの分を足す・足し込み・駆動・的・学び・恒常性・誤差・不応期）→ 2周目から最後の周まで"""
        S, X, F = self._gS, self._gX, self._fS
        on = dict(key)
        opt = lambda k: X[k] if on.get(k) else None
        # 1周目：学びの分を足す ＋ 足し込み（版80 の _flush_sum と同じ式）
        if str(self.dev).startswith("cuda"):
            K, BL = _SC_K, 256
            b0 = torch.zeros(K, self.n_recv_all * 2, device=self.dev)
            m0 = torch.zeros(K, self.n_recv_all, device=self.dev)
            _p1_flush_k[(1024,)](F["E"], F["ne"], self.k_start, self.elig, self.A, F["a_now1"], F["kc1"], self.w,
                                 self.e_recv, self.e_src, self.e_inh, self.c_exc, self._amp_buf, X["send_gain"],
                                 self.mod_send, self.mod_drive_inh, self.is_inh_r, b0, m0, self.n_recv_all,
                                 HAS_AMP=True, K=K, BLOCK=BL)
            both0, mod0 = b0.sum(0), m0.sum(0)
        else:
            e = F["E"][:int(F["ne"])]
            both0, mod0 = _run(_flush_sum_core, e, self.k_start, self.elig, self.A, F["a_now1"], self.w, F["kc1"],
                               self.e_recv, self.e_src, self.e_inh, self.c_exc, self._amp_buf, X["send_gain"],
                               self.mod_send, self.mod_drive_inh, self.is_inh_r, self.n_recv_all)
        exc, ihn, exc_mod, p, pr, drive, rep = _run(
            _drive_core, both0, mod0, X["scale"], X["scale_up"], opt("exc_gain"), X["prior_l"], self.rep_i,
            X["x_all"], X["x_gain"], opt("cl_drive"), opt("v_carry"), X["fire_th"], X["th_gain"], opt("complete"),
            X["pred_boost"], F["spont1"], X["spont_p_pass"], self.n_recv_all)
        # 的（_pass の1周目と同じ）
        x_t = torch.where(self.tgt_by_fire, (X["x_all"] * X["x_gain"] > X["fire_th"]).float(), X["x_all"])
        F["xt"].copy_(x_t)
        X["tgt"].copy_(torch.where(self.is_inh_r, (x_t @ self.mean_map)[self.area_of], x_t[self.x_index]))
        err_sig = X["tgt"] - p
        if on["learn"]:      # 学ぶ ＋ 2周目の頭（版80 の _learn_sum_core と同じ式）＋ 恒常性（版88）
            if str(self.dev).startswith("cuda"):
                K, BL = _SC_K, 256
                b2 = torch.zeros(K, self.n_recv_all * 2, device=self.dev)
                m2 = torch.zeros(K, self.n_recv_all, device=self.dev)
                c2 = torch.zeros(K, self.n_recv_all, device=self.dev, dtype=torch.int32)
                _p1_learn_k[(1024,)](F["E"], F["ne"], self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,
                                     self._g_buf, self._pw_buf, self.elig, self.hist_n, self.w, self.c_exc, self.k_start,
                                     self.A, S["a_now"], S["kcur1"], self._amp_buf, X["send_gain"], self.mod_send,
                                     self.mod_drive_inh, b2, m2, c2, self.n_recv_all, INH_LTD_N, INH_LTD_TH, INH_LTD_F,
                                     HAS_AMP=True, K=K, BLOCK=BL)
                cnt, bb, mb = c2.sum(0).to(torch.float32), b2.sum(0), m2.sum(0)
            else:
                e = F["E"][:int(F["ne"])]
                cnt, bb, mb = _run(_learn_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,
                                   self._g_buf, self._pw_buf, self.elig, self.hist_n, self.w, self.c_exc, self.k_start,
                                   self.A, S["a_now"], S["kcur1"], self._amp_buf, X["send_gain"], self.mod_send,
                                   self.mod_drive_inh, self.n_recv_all)
            X["both_base"].copy_(bb)
            X["mod_base"].copy_(mb)
            a_ = 1.0 / self.homeo_tau
            r_, d_ = _run(_homeo_core, cnt, exc, self.recv_ema, self.drive, a_)
            self.recv_ema.copy_(r_)
            self.drive.copy_(d_)
            if on["scaling"]:
                X["scale"].copy_(_run(_scale_down_core, self.rate_ema, self.target_recv, self.is_hippo, self.scale_skip,
                                      X["scale"], a_, on["sleep"]))
            else:
                X["scale_up"].copy_(_run(_scale_up_core, self.rate_long, self.target_recv, self.scale_skip, X["scale_up"]))
        else:                # 学ばない歩：2周目の頭は1周目と同じ辺・同じ重さ（学びの分は足してある）
            X["both_base"].copy_(both0)
            X["mod_base"].copy_(mod0)
        # 誤差・抑制の体・不応期（_pass の1周目と同じ）
        S["pc"].add_(1)
        (d_inh, rep, err, inh_spk, lf0, lf1, lf2, a0, a1, a2) = _run(
            _out_core, err_sig, exc, exc_mod, ihn, self.err_i, self.inh_i, opt("v_carry_inh"), rep,
            S["lf0"], S["lf1"], S["lf2"], S["pc"], None, None, None)
        for k_, v in (("lf0", lf0), ("lf1", lf1), ("lf2", lf2), ("any0", a0), ("any1", a1), ("any2", a2),
                      ("drive", drive), ("dinh", d_inh), ("pr", pr)):
            S[k_].copy_(v)
        S["perr"].copy_(p[self.err_i])
        wi = S["within"]
        wi.zero_()
        wi[self.within_index] = rep
        if WITHIN_INH:
            wi[self.within_inh_index] = inh_spk
        lens, csum = _run(_new_core, wi, X["base_on"], self._gW, self.start)
        S["lens"].copy_(lens)
        S["csum"].copy_(csum)
        # 2周目から最後の周まで（版83 と同じ）
        self._gp_body1(key)

'''

G2 = [
    ("class Group2:\n", KERNELS + "class Group2:\n"),
    ("        self._pre2 = None                    # 版80：1周目の学びと一緒に作った2周目の頭の足し込み\n"
     "        gp = self._gp_ok()                   # 版79：2周目からをグラフの形で\n"
     "        for i in range(1 if gp else self.recur):\n"
     "            out = self._pass(x_all, ext_all, learn and i == 0, i == self.recur - 1, i == 0)\n"
     "        if gp:\n"
     "            out = self._gp_passes(x_all, ext_all)\n",
     "        self._pre2 = None                    # 版80：1周目の学びと一緒に作った2周目の頭の足し込み\n"
     "        out = self._fast_step(x_all, ext_all, learn) if self._fast_ok(learn) else None   # 版90：1つのグラフで\n"
     "        if out is None:\n"
     "            gp = self._gp_ok()               # 版79：2周目からをグラフの形で\n"
     "            for i in range(1 if gp else self.recur):\n"
     "                out = self._pass(x_all, ext_all, learn and i == 0, i == self.recur - 1, i == 0)\n"
     "            if gp:\n"
     "                out = self._gp_passes(x_all, ext_all)\n"),
    ("    def _fuse_ok(self):\n", METHODS.lstrip("\n") + "    def _fuse_ok(self):\n"),
]
BB = [("VERSION = 89          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88)",
       "VERSION = 90          # コードの版。網の作りを変えたら上げる\n"
       "SAME_NET = (66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89)"),
      ("#  89  恒常性の観測で誤差と表現の行が入れ違っていた（版59 から）のを直す【み】10-15「バグは直して」⚠️ 値が変わる\n",
       "#  89  恒常性の観測で誤差と表現の行が入れ違っていた（版59 から）のを直す【み】10-15「バグは直して」⚠️ 値が変わる\n"
       "#  90  1周目から最後の周までを1つのグラフに（1周目の辺は Triton の核。くじ・放出・歩の終わりは外）【み】10-15「稼げるならやっておいて」\n")]
TOOLS = ("measure_check.py", "measure_check2.py", "measure_v59a.py", "measure_v59b.py", "measure_v59c.py", "measure_v59d.py",
         "measure_v62a.py", "measure_v62b.py", "measure_v65a.py", "measure_v66a.py", "measure_v67a.py",
         "measure_v67b.py", "measure_v67d.py", "measure_speed75.py")


def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f"⚠️ {path} に印が {s.count(a)} 個（1 個のはず）：{a[:60]!r}。何も書いていない")
        s = s.replace(a, b)
    ast.parse(s)
    return p, s


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 89:
        sys.exit(f"⚠️ babble.py が版89 ではない（版 {babble.VERSION}）")
    out = [edit("torch_group2.py", G2), edit("babble.py", BB)]
    for p, s in out:
        p.write_text(s)
    print("✅ 版90 を当てた（torch_group2・babble）。1周目から最後の周までを1つのグラフに", flush=True)
    for f in TOOLS:
        q = pathlib.Path(f)
        if not q.exists():
            continue
        s = q.read_text()
        m = re.search(r"if B\.VERSION not in \(([^)]*)\):", s)
        if m and "90" not in m.group(1):
            nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
            s = s[:m.start()] + f"if B.VERSION not in ({', '.join(nums + ['90'])}):" + s[m.end():]
            q.write_text(s); print(f"   {f}：版90 でも回るようにした", flush=True)
