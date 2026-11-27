"""切片の試し台（段2、36_仕組みの棚卸し）【み】11-28 05:55「組織になると撃ち方が変わるんじゃないの？」・07:07「まぁやってみて」

皮質の領野1つ（既定 前頭前野：表現 318・ext 661・抑制の体 61）を、網と同じ作りで1つだけ回す。
  繋がり：torch_area.Area をそのまま組む（相手 9,100・抑制の相手は切片の見方 23%・表現 → 表現 rr 15.8%・抑制の体の興奮の相手 180）
  重さ：網と同じ（rr ＝ 人の1本の幅 ÷ 接触・時間細胞の遅れの体 → 表現 ＝ 人の幅 × 0.35 ÷ 接触・錐体 → 抑制の体 ＝ Wilbers の幅 ÷ 接触・抑制 4 × W_INH）
  版116：抑制の代わり 1,456本（抑制の行の遅れの体 → 表現の分母）・版116c：抑制の体へ切片の外の錐体の代わり 290本（× 0.35）
  1歩の中：10周。周で撃った表現・抑制の体は次の周から届く（版63）。持ち越し 15ms・6.8ms・不応期 3周。短期の増減（人の表）あり
入れないもの（網にはある）：学び・恒常性の倍率（1 のまま）・CL・下からの入口・誤差の体・ほかの領野
  ext の行（ほかの領野の表現）：くじで ext_hz（既定 3.3Hz【ヒト】Verma 2024）
型（--mode）
  net      網と同じ：自分の行の遅れの体は自分の発火の写し（写しの輪あり）
  poisson  切片の中は網と同じ・切片の外の遅れの体を、ばらばらのくじ（自分の行も ext_hz）に差し替え
  alone    切片の中の繋がり（rr・抑制の体 ⇄ 表現）を外し、外はくじ ＝ 組織なしの素子の集まり（段1 と同じ段）
⚠️ 測るためだけ。くじの入力は網に入れない（【み】10-16）
使い方：python3 bench_slice.py [歩 2000] [--mode net|poisson|alone] [--ext-hz 3.3] [--no-stp] [--area 前頭前野 318 661] [--seed 0]
         python3 bench_slice.py selftest
"""
import sys, os, math, time
import numpy as np
import torch

TREE = os.environ.get("ANAPLEX_TREE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import torch_area as T   # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = "cpu"
DT = T.DT_MS
RECUR = T.RECUR
REFR = 3
TH_REP = 16.0 / 1.12
INH_TH = T.INH_TH
W_INH = T.W_INH
W_INH_BU = W_INH * 4.0
TC_W_DIST = 14.3 / (134.0 / 3.3)
N_TC_INH = int(round(9100 * 0.16))
N_TC_EI = 470 - T.INH_N_EXC
VD_REP = math.exp(-DT / 15.0)
VD_INH = math.exp(-DT / 6.8)
W_MED, W_MEAN = 0.46 / 1.12, 1.0
W_LO, W_HI = 0.1 / 1.12, 6.0 / 1.12
C_EXC = 3.3


def ln_w(n, gen):
    sig = math.sqrt(2.0 * math.log(W_MEAN / W_MED))
    w = torch.exp(math.log(W_MED) + sig * torch.randn(n, generator=gen))
    for _ in range(64):
        bad = (w < W_LO) | (w > W_HI)
        nb = int(bad.sum())
        if nb == 0:
            break
        w[bad] = torch.exp(math.log(W_MED) + sig * torch.randn(nb, generator=gen))
    return w.clamp_(W_LO, W_HI)


def inh_w(n, gen):
    z = torch.randn(n, generator=gen)
    return torch.exp(T.INH_W_MU + T.INH_W_SIG * z) * T.INH_W_MEAN


class Slice:
    def __init__(self, n_ch, n_ext, mode="net", ext_hz=3.3, stp=True, seed=0, n_partner=9100, ei_slice=0.0):
        self.mode, self.ext_p = mode, ext_hz * DT / 1000.0
        g = torch.Generator(); g.manual_seed(seed + 7)
        self.gen = g
        rr_k = int(round(0.158 * (n_ch - 1)))
        stp_path = os.path.join(TREE, "synapse_model_human.json") if stp else None
        a = T.Area(n_ch, prior=None, n_ext=n_ext, n_partner=n_partner, inh_frac=0.16, stp_path=stp_path,
                   spont_hz=0.0, dev=DEV, seed=seed, fire_th=TH_REP, contacts_exc=C_EXC,
                   inh_n_exc=T.INH_N_EXC, inh_slice=0.23, rr_k=rr_k, rr_seed=seed + 113000,
                   rr_w=lambda n_, g_: ln_w(n_, g_))
        self.a = a
        self.n_ch, self.n_ext, self.n_inh = n_ch, n_ext, a.n_inh
        self.n_src = a.n_src
        self.per = a.ch.n_cell + a.ch.n_ramp
        self.nc = a.ch.n_cell
        self.n_cells = a.ch.n_cells
        n_exc_rows = n_ch + n_ext
        src = a.e_src_sorted.long()
        recv = a.e_recv.long()
        e_inh = a.e_inh.clone()
        w = a.w.clone()
        is_rep_r = (recv >= n_ch) & (recv < 2 * n_ch)
        is_inh_r = (recv >= 2 * n_ch) & (recv < 2 * n_ch + a.n_inh)
        in_tc = src < self.n_cells
        row = torch.where(in_tc, src // self.per, src - self.n_cells)          # D ＝ 1
        cell = torch.where(in_tc, src % self.per, torch.full_like(src, -1))
        is_delay = in_tc & (cell < self.nc)
        is_ramp = in_tc & (cell >= self.nc)
        exc_row = row < n_exc_rows
        own_col = (~in_tc) & (row < n_ch)
        inh_col = (~in_tc) & (row >= n_exc_rows)
        # 重さ（網と同じ）
        m_ei = is_inh_r & ~e_inh                                    # 錐体 → 抑制の体：全部の興奮の相手（Group2 と同じ）
        w[m_ei] = inh_w(int(m_ei.sum()), g) / C_EXC
        m_tc = is_rep_r & is_delay & exc_row & ~e_inh               # 版116 B：遅れの体 → 表現
        w[m_tc] = ln_w(int(m_tc.sum()), g) * TC_W_DIST / C_EXC
        # 調整する側（傾きの体の興奮の行）は撃たせない ⇒ 落とす。誤差の行も落とす
        keep = (is_rep_r | is_inh_r) & ~(is_ramp & exc_row)
        if mode == "alone":                                          # 切片の中の繋がりを外す（外の代わりだけ残す）
            keep &= ~(own_col | inh_col)
        # --ei-slice p：抑制の体の興奮の相手のうち round(p × 表現の数) 本を、自分の領野の表現の列（1歩前＋周の中）に差し替える
        #   網（torch_area）は抑制の体の興奮の相手を興奮の池（ほぼ時間細胞）から引く ⇒ 自分の切片の錐体は 平均 0.5本
        #   【ヒト】Allen synphys の局所 錐体 → 抑制 20〜23%（組み表_版112 §17）。試しのためだけ（網は変えない）
        self.ei_slice = ei_slice
        if ei_slice > 0 and mode != "alone":
            k_ei = int(round(ei_slice * n_ch))
            cand = is_inh_r & ~e_inh & is_delay & exc_row
            for r in range(2 * n_ch, 2 * n_ch + a.n_inh):
                idx = torch.nonzero(cand & (recv == r)).squeeze(1)
                assert len(idx) >= k_ei, (len(idx), k_ei)
                pick = idx[torch.randperm(len(idx), generator=g)[:k_ei]]
                src[pick] = a.ch.n_cells + torch.randperm(n_ch, generator=g)[:k_ei]
        amp = torch.where(e_inh, torch.full_like(w, 4.0), torch.full_like(w, C_EXC)) * w
        keep &= amp != 0
        self.k_idx = torch.nonzero(keep).squeeze(1)                  # 元の辺の番号（短期の増減の状態を引く）
        self.src, self.amp, self.e_inh = src[self.k_idx], amp[self.k_idx], e_inh[self.k_idx]
        r_ = recv[self.k_idx]
        self.rr = torch.where(r_ < 2 * n_ch, r_ - n_ch, r_ - 2 * n_ch + n_ch)   # 0..n_ch−1 表現・n_ch.. 抑制の体
        self.n_r = n_ch + a.n_inh
        self.stp = a.stp
        # 送り手ごとの辺の範囲（周の中で新しく撃った送り手の分を足す）
        o = torch.argsort(self.src, stable=True)
        self.src, self.amp, self.e_inh, self.rr, self.k_idx = self.src[o], self.amp[o], self.e_inh[o], self.rr[o], self.k_idx[o]
        n_send = self.n_cells + self.n_src
        self.start = torch.zeros(n_send + 1, dtype=torch.long)
        self.start[1:] = torch.bincount(self.src, minlength=n_send).cumsum(0)
        # 版116 C：表現 ← 抑制の行の遅れの体 1,456本 ／ 版116c：抑制の体 ← 興奮の行の遅れの体 290本
        k_ = torch.arange(self.nc)
        ih_rows = torch.arange(n_exc_rows, self.n_src)
        ih_cells = (ih_rows.unsqueeze(1) * self.per + k_).reshape(-1)
        ex_cells = (torch.arange(n_exc_rows).unsqueeze(1) * self.per + k_).reshape(-1)
        self.tci = ih_cells[torch.randint(0, len(ih_cells), (n_ch, N_TC_INH), generator=g)]
        self.eii = ex_cells[torch.randint(0, len(ex_cells), (a.n_inh, N_TC_EI), generator=g)]
        self.eiw = inh_w(a.n_inh * N_TC_EI, g).view(a.n_inh, N_TC_EI) * TC_W_DIST
        self.delay_mask = torch.zeros(self.n_cells, dtype=torch.bool)
        self.delay_mask[(torch.arange(self.n_src).unsqueeze(1) * self.per + k_).reshape(-1)] = True
        # 状態
        self.buf = torch.zeros(self.n_src)
        self.carry = torch.zeros(n_ch)
        self.carry_i = torch.zeros(a.n_inh)
        self.lf = torch.full((self.n_r,), -10 ** 6, dtype=torch.long)
        self.pc = 0
        self.t = 0

    def tc_spikes(self):
        s = self.a.ch.spikes().clone()
        if self.mode in ("poisson", "alone"):                       # 切片の外の遅れの体を、ばらばらのくじに
            p = torch.rand(self.n_cells, generator=self.gen) < self.ext_p
            s = torch.where(self.delay_mask, p.float(), s)
        return s

    def step(self):
        n_ch, ni = self.n_ch, self.n_inh
        spk = self.tc_spikes()
        base = torch.cat([spk, self.buf])
        on = base > 0
        e = torch.nonzero(on[self.src]).squeeze(1)
        amp = self.amp[e]
        if self.stp is not None:
            amp = amp * self.stp.release_at(self.k_idx[e], self.t)
        ie = self.e_inh[e]
        exc = torch.zeros(self.n_r).index_add_(0, self.rr[e], torch.where(ie, torch.zeros_like(amp), amp))
        ihn = torch.zeros(self.n_r).index_add_(0, self.rr[e], torch.where(ie, amp, torch.zeros_like(amp)))
        ihn[:n_ch] += spk[self.tci].sum(1) * W_INH_BU                       # 版116 C
        exc[n_ch:] += (spk[self.eii] * self.eiw).sum(1)                       # 版116c
        sent = on.clone()
        cyc = torch.zeros((RECUR, self.n_r), dtype=torch.bool)
        any_ = torch.zeros(self.n_r, dtype=torch.bool)
        col_off = self.n_cells
        for c in range(RECUR):
            self.pc += 1
            d_rep = exc[:n_ch] / (1.0 + ihn[:n_ch]) + self.carry
            d_inh = exc[n_ch:] / (1.0 + ihn[n_ch:]) + self.carry_i
            f = torch.cat([d_rep > TH_REP, d_inh > INH_TH]) & ((self.pc - self.lf) >= REFR)
            self.lf = torch.where(f, torch.full_like(self.lf, self.pc), self.lf)
            cyc[c] = f
            any_ |= f
            if self.mode != "alone":                                       # 周で撃った表現・抑制の体は次の周から届く（1歩に1回）
                sid = torch.cat([col_off + torch.nonzero(f[:n_ch]).squeeze(1),
                                 col_off + self.n_ch + self.n_ext + torch.nonzero(f[n_ch:]).squeeze(1)])
                sid = sid[~sent[sid]]
                if len(sid):
                    sent[sid] = True
                    lo, hi = self.start[sid], self.start[sid + 1]
                    idx = torch.cat([torch.arange(int(l), int(h)) for l, h in zip(lo, hi)]) if len(sid) else None
                    if idx is not None and len(idx):
                        am, ii = self.amp[idx], self.e_inh[idx]
                        exc.index_add_(0, self.rr[idx], torch.where(ii, torch.zeros_like(am), am))
                        ihn.index_add_(0, self.rr[idx], torch.where(ii, am, torch.zeros_like(am)))
        rep_any, inh_any = any_[:n_ch], any_[n_ch:]
        self.carry = VD_REP * d_rep * (~rep_any).float()
        self.carry_i = VD_INH * d_inh * (~inh_any).float()
        ext = (torch.rand(self.n_ext, generator=self.gen) < self.ext_p).float()
        src_now = torch.cat([rep_any.float(), ext, inh_any.float()])
        self.a.ch.step(src_now)
        self.buf = src_now
        self.t += 1
        return cyc, d_rep


def run(n_step, **kw):
    s = Slice(**kw)
    n_ch = s.n_ch
    cyc_all = np.zeros((n_step, RECUR, s.n_r), dtype=bool)
    dr = np.zeros(n_step)
    t0 = time.time()
    for t in range(n_step):
        cyc, d_rep = s.step()
        cyc_all[t] = cyc.numpy()
        dr[t] = float(d_rep.mean()) / TH_REP
        if (t + 1) % 200 == 0:
            print(f"  {t + 1}歩 {time.time() - t0:.0f}秒", flush=True)
    return s, cyc_all, dr


def stats(cyc_all, n_ch, skip):
    c = cyc_all[skip:]
    n_step = c.shape[0]
    sec = n_step * DT / 1000.0
    out = {}
    for nm, sl in (("表現", slice(0, n_ch)), ("抑制の体", slice(n_ch, None))):
        x = c[:, :, sl]                                   # 歩 × 周 × 体
        n_b = x.shape[2]
        spikes = x.sum()
        per_step = x.sum(1)                               # 歩 × 体
        on = per_step > 0
        r = {"率Hz": spikes / sec / n_b, "網が見る率Hz": on.sum() / sec / n_b}
        r["撃った歩のうち2発以上"] = float((per_step[on] >= 2).mean()) if on.any() else float("nan")
        isis = []
        flat = x.transpose(2, 0, 1).reshape(n_b, -1)
        for b in range(n_b):
            pos = np.nonzero(flat[b])[0]
            if len(pos) > 1:
                isis.append(np.diff(pos))
        if isis:
            isi = np.concatenate(isis) * T.SYN_DELAY_MS
            for ms in (3.4, DT, 2 * DT):
                r[f"間隔<{ms:.1f}ms"] = float((isi < ms).mean())
            r["CV"] = float(isi.std() / isi.mean())
        frac = on.mean(1)                                 # 歩ごとの撃った体の割合
        r["揃う歩(20%以上)の発火の割合"] = float(on[frac >= 0.2].sum() / max(1, on.sum()))
        r["黙った歩"] = float((frac == 0).mean())
        out[nm] = r
    return out


def selftest():
    # 小さい切片で：作りが組めて回る・alone は周の中で届かない・持ち越し0 の体は撃ったら次の歩は持ち越しなし
    s = Slice(40, 60, mode="net", seed=1, n_partner=600)
    for _ in range(30):
        cyc, d = s.step()
    assert s.n_inh == 8 and s.n_r == 48
    assert (s.amp > 0).all()
    # 興奮の辺の重さの上限（6mV ÷ 1.12 × 接触の人の幅の上・抑制の体の幅の上）を越えない
    assert float(s.amp[~s.e_inh].max()) < 10.0
    sa = Slice(40, 60, mode="alone", seed=1, n_partner=600)
    col_row = sa.src[sa.src >= sa.n_cells] - sa.n_cells
    assert not bool(((col_row < sa.n_ch) | (col_row >= sa.n_ch + sa.n_ext)).any()), "alone に切片の中の列の辺が残っている"
    s_net_cols = s.src[s.src >= s.n_cells] - s.n_cells
    assert bool((s_net_cols < s.n_ch).any()) and bool((s_net_cols >= s.n_ch + s.n_ext).any()), "net に切片の中の列の辺が無い"
    for _ in range(10):
        sa.step()
    se = Slice(40, 60, mode="net", seed=1, n_partner=600, ei_slice=0.23)
    colr = se.src - se.n_cells
    m = (se.src >= se.n_cells) & (colr < se.n_ch) & (se.rr >= se.n_ch) & ~se.e_inh
    cnt = torch.bincount(se.rr[m] - se.n_ch, minlength=se.n_inh)
    assert int(cnt.min()) >= int(round(0.23 * 40)), cnt
    for _ in range(5):
        se.step()
    print("selftest ✓")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest(); sys.exit()
    args = sys.argv[1:]
    n_step = int(args[0]) if args and args[0].isdigit() else 2000
    mode = args[args.index("--mode") + 1] if "--mode" in args else "net"
    ext_hz = float(args[args.index("--ext-hz") + 1]) if "--ext-hz" in args else 3.3
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
    if "--area" in args:
        i = args.index("--area"); name, n_ch, n_ext = args[i + 1], int(args[i + 2]), int(args[i + 3])
    else:
        name, n_ch, n_ext = "前頭前野", 318, 661
    stp = "--no-stp" not in args
    ei_slice = float(args[args.index("--ei-slice") + 1]) if "--ei-slice" in args else 0.0
    print(f"切片 {name}（表現 {n_ch}・ext {n_ext}）型 {mode}・ext {ext_hz}Hz・短期の増減 {'あり' if stp else 'なし'}・抑制の体 ← 切片の錐体 {ei_slice}・{n_step}歩")
    s, cyc_all, dr = run(n_step, n_ch=n_ch, n_ext=n_ext, mode=mode, ext_hz=ext_hz, stp=stp, seed=seed, ei_slice=ei_slice)
    skip = min(500, n_step // 4)
    st = stats(cyc_all, n_ch, skip)
    print(f"（始めの {skip}歩を除く）表現の駆動の平均 ÷ 閾値（最後の周）：中央 {np.median(dr[skip:]):.2f}")
    for nm, r in st.items():
        print(nm + "：" + "・".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v:.2f}" for k, v in r.items()))
    rr_ = st["抑制の体"]["率Hz"] / max(1e-9, st["表現"]["率Hz"])
    print(f"抑制の体 ÷ 表現 の率：{rr_:.2f}")
    print("人（組織の中・作るのに使っていない）：皮質 3ms 未満 0.3〜0.8%・10.7ms 未満 2.1〜5.7%・21.3ms 未満 5.2〜13.1%（Kyzar 2024）"
          "／ 10ms 未満 17〜25%（Ison 2011、定義が違う）／ 揃い RS 9%・FS 13%【ヒト】／ 抑制 ÷ 錐体 1.4〜2.5倍")
    np.savez_compressed(f"bench_slice_{mode}_ei{ei_slice:g}_hz{ext_hz:g}.npz", cyc=np.packbits(cyc_all, axis=-1), n_ch=n_ch, dr=dr)
