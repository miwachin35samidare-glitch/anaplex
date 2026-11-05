"""領野をまとめて1回で進める v2（arkhe 用）—— 領野ごとのループを全部消す

v1 との違いは、書く場所だけ。仕組みは1つも増えていない。
  v1  周回の中に「的を作るループ」と「出力を作るループ」、歩を進めるところに時間細胞のループ
      ⇒ 8領野 × 周回10 で、小さな呼び出しが何度も走る
  v2  入口・出力・時間細胞を、それぞれ1本の配列／行列にまとめて一度に扱う
      領野の境目は番号で残る（辺と同じやり方）

⚠️ 一致することを確かめてから使う（check(areas) が v1 と同じ結果になるか見る）

arkhe の実測（2026-09-11）
  別々 41.95ms → v1 23.90ms（周回1回 1.94ms ＋ 歩の固定費 5.23ms）
"""
import math
import torch
from torch_area import TAUS, THETA, DT_MS, W_INH
from torch_area import INH_W_SIG, INH_W_MU, INH_W_MEAN, INH_LTD_TH, INH_LTD_F, INH_LTD_WIN, INH_LTD_N   # 版47
from torch_area import INH_TH   # 版48

# 傾きの細胞の発火（版39）：【ヒト】皮質の単一細胞の発火率 0.3Hz を1歩に（0.32%）
#   ⟲ 版38 までは 30%/歩（chains.py の既定値。出どころの記録なし【C】。人の 30〜90倍）
RAMP_P = 0.3 * DT_MS / 1000.0
# 学んだ跡の長さ（版40）【ラット】大人の聴覚野・生体：修飾の信号が 1秒前・同時なら効き、
#   3秒前後では効かない（Brain Stimulation 2025）⇒ 時定数 1秒。【ヒト】【霊長類】は見つからない【み】09-27
ELIG_TAU = 1000.0 / DT_MS
# 版53：辺1本の寿命（【ヒト】の神経の棘 約2週。マウスに移植した人の神経、移植後 3〜6か月。Linaro 2019）
LIFE_STEPS = 14 * 24 * 3600 * 1000.0 / DT_MS
LAZY_EPOCH = 1024                   # 版54：学びをまとめる区切り（刻み）。減り^(−1024) ≈ 5.6万で float32 に収まる
REWIRE_STEPS = 1000                 # 刈り込みを見る間隔（約10.7秒）。寿命は確率で合わせる【C の置き方】
# 版53：静か・撃ちすぎを見る発火の長い平均（1時間【C】）。起きているあいだだけ進める
LONG_TAU = 3600 * 1000.0 / DT_MS
# 版59：不応期（周の数）。【ヒト】起きている人の内側側頭葉の単一細胞で 3ms 以内の間隔は 0.10%（Ison 2011）⇒ 3ms ÷ 1.102ms ≒ 3周
REFRACT_PASSES = 3
# 版63：周回の中で撃った抑制の体の発火も次の周で届ける【ヒト】Szegedi 2017：籠細胞の抑制は錐体の発火の 4.1ms 後【み】10-11
#   ANAPLEX_WITHIN_INH=0 で版62 と同じ（比べる用）
import os as _os

# 版76：辺ごとの計算を torch.compile でまとめる（小さな命令を1つにして、呼ぶ回数と GPU の時間を減らす）
#   【測】probe_gpu2（10-14、torch 2.14.1・ROCm 7.14）：gfx1201 で動く・小さい命令の列で 9.6倍・同じ値（丸めの範囲）
#   撃った辺の数は毎歩変わる ⇒ dynamic（数が変わっても作り直さない）
#   動かないとき（作るときの誤り）は1回だけ知らせて、まとめない形で回す（値は同じ）
_CC = {}


def _cc(fn):
    c = _CC.get(fn)
    if c is None:
        c = _CC[fn] = torch.compile(fn, dynamic=True)
    return c


# 版77：核の中の「番号が外に出ていないか」の assert を外す（RX 9070 XT で、assert の入った核が起動できない）
#   【測】probe_compile2・3（10-14）：assert ありは番号で引く計算が全部 ✗（driver error 401）、外すと全部 ○。値には関わらない
import torch._inductor.config as _ind_cfg
_ind_cfg.assert_indirect_indexing = False


def _run(fn, *a):
    """版77：torch.compile（dynamic）で回す"""
    return _cc(fn)(*a)


def _sum_core(e, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r, n_recv: int):
    """版76：Group2._sum の辺の計算（式は版75 と同じ）"""
    re_ = e_recv[e].long()
    se_ = e_src[e].long()
    ie = e_inh[e]
    amp = torch.where(ie, 4.0, c_exc[re_]) * w[e]
    if amp_s is not None:
        amp = amp * amp_s[e]
    amp = amp * send_gain[se_]
    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)
    both.scatter_add_(0, re_ + n_recv * ie.long(), amp)
    mm = mod_send[se_] & ~ie
    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])
    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, re_, amp * mm)
    return both, mod


def _flush_core(idx, k_start, elig, A, a_now, w, e_recv, recv_grp):
    """版76：Group2._flush の辺の計算（a_now ＝ A[kcur]）／ 版107d：門は受け手の組ごと"""
    ks = k_start[idx].long()
    gi = recv_grp[e_recv[idx].long()].long()
    pend = elig[idx] * (a_now[gi] - A[ks - 1, gi]).float()
    w[idx] = (w[idx] + pend).clamp(min=0.0)


def _learn_core(e, e_recv, e_src, e_inh, is_inh_r, err_sig, g, pw, elig, hist_n, w, c_exc, n_recv: int):
    """版76：Group2._pass の学びの辺の計算（式は版75 と同じ。g ＝ eta × eta_gain、pw ＝ 減り^(−k)）"""
    r = e_recv[e].long()
    gg = g * err_sig[r]
    to_inh = is_inh_r[r]
    elig[e] += torch.where(e_inh[e] | to_inh, torch.zeros_like(gg), gg) * pw
    se = e_src[e].long()
    ltd = to_inh & ~e_inh[e] & (hist_n[se] >= INH_LTD_N) & (w[e] * c_exc[r] > INH_LTD_TH)
    w[e] = torch.where(ltd, w[e] * INH_LTD_F, w[e])
    cnt = torch.zeros(n_recv, device=w.device, dtype=w.dtype)
    cnt.scatter_add_(0, r, torch.ones(e.shape[0], device=w.device, dtype=w.dtype))
    return cnt


def _drive_core(both, exc_mod, scale, scale_up, exc_gain, prior_l, rep_i, x_all, x_gain, cl_drive, v_carry,
                fire_th, th_gain, complete, pred_boost, spont_u, spont_p_pass, ext_ihn, nuc_x, nuc_ihn, n_recv: int):
    """版78：周回の中の『届いた入力 → 見込み → 表現が撃つか』（式は版77 の _pass と同じ。くじ spont_u は外で引く）"""
    exc, ihn = both[:n_recv], both[n_recv:]
    exc = exc * (scale * scale_up); ihn = ihn * scale
    if ext_ihn is not None:                  # 版94：外からの抑制の線（中脳の核へ）。学ばない
        ihn = ihn + ext_ihn
    if nuc_ihn is not None:                  # 版95：核どうしの抑制（周ごと。NucNet）
        ihn = ihn + nuc_ihn
    exc_mod = exc_mod * (scale * scale_up)
    if exc_gain is not None:
        exc = exc * exc_gain
        exc_mod = exc_mod * exc_gain
    p = torch.sigmoid(prior_l + exc / (1.0 + ihn))
    pr = p[rep_i]
    xx = x_all if nuc_x is None else x_all + nuc_x   # 版95：核どうしの興奮（周ごと。x と同じ入り方。的には入れない）
    drive = ((exc - exc_mod)[rep_i] + xx * x_gain) / (1.0 + ihn[rep_i])
    if cl_drive is not None:
        drive = drive + cl_drive / (1.0 + ihn[rep_i])
    if v_carry is not None:
        drive = drive + v_carry
    rep = (drive > fire_th * th_gain).float()
    if complete is not None:
        fill = (complete > 0) & ((drive + pred_boost * pr) > fire_th * th_gain) & (x_all > 0)
        rep = torch.where(fill, torch.ones_like(rep), rep)
    rep = torch.maximum(rep, (spont_u < spont_p_pass).float())
    return exc, ihn, exc_mod, p, pr, drive, rep


def _out_core(err_sig, exc, exc_mod, ihn, err_i, inh_i, v_carry_inh, rep, lf_r, lf_e, lf_i, pc, any_r, any_e, any_i):
    """版78：周回の中の『誤差 → 抑制の体 → 不応期 → この歩のどこかで撃ったか』（式は版77 と同じ。pc は周の番号の箱）"""
    err = (err_sig[err_i] > THETA).float()
    d_inh = (exc - exc_mod)[inh_i] / (1.0 + ihn[inh_i])
    if v_carry_inh is not None:
        d_inh = d_inh + v_carry_inh
    inh_spk = (d_inh > INH_TH).float()
    outs, lfs = [], []
    for v, lf in ((rep, lf_r), (err, lf_e), (inh_spk, lf_i)):
        v = v * ((pc - lf) >= REFRACT_PASSES).float()
        outs.append(v)
        lfs.append(torch.where(v > 0, pc, lf))
    if any_r is None:
        anys = list(outs)
    else:
        anys = [torch.maximum(a, v) for a, v in zip((any_r, any_e, any_i), outs)]
    return d_inh, outs[0], outs[1], outs[2], lfs[0], lfs[1], lfs[2], anys[0], anys[1], anys[2]
WITHIN_INH = _os.environ.get("ANAPLEX_WITHIN_INH", "1") != "0"
# ⟲ 版60 の f-I は外した（10-11【み】「電極刺して生体にありえない電流を流して測った数字ってことでしょ？」）
# 版53：起きているあいだ上げる恒常性の速さ（【ラット】Hengen 2016：率が約60%落ちたのが 3.5日ほどで戻る）
#   足りない割合 0.6 が 3.5日続くと 2.5倍 ⇒ 1歩あたり ln(2.5) ÷ (0.6 × 3.5日の歩数)【C の当てはめ】
K_WAKE_UP = math.log(2.5) / (0.6 * 3.5 * 24 * 3600 * 1000.0 / DT_MS)


def _tc_core(age, prefs, halfs, pmax, ups, dns, offs, n_cell, n_ramp, ramp_len):
    """版78：時間細胞の撃つか（領野ごとの塊を順に）。くじ ups・dns は外で今の順に引く。式は版77 と同じ"""
    out = []
    for i in range(len(n_ramp)):
        a = age[offs[i]:offs[i + 1]]
        parts = []
        if n_cell[i]:
            f = a.unsqueeze(-1).float()
            hit = ((f - prefs[i]).abs() <= halfs[i]) & (f <= pmax[i])
            parts.append(hit.any(1).float())
        y = (a.min(1).values.float() / ramp_len[i]).clamp(0, 1).unsqueeze(-1)
        parts.append(torch.cat([ups[i] < y * RAMP_P, dns[i] < (1 - y) * RAMP_P], 1).float())
        out.append(torch.cat(parts, 1).reshape(-1))
    return torch.cat(out)


class TimeCellsGroup:
    """8領野ぶんの時間細胞を、ひとつの行列で持つ（領野をまたいで混ざらない）"""

    def __init__(self, areas):
        dev = areas[0].dev
        self.dev = dev
        self.areas = areas
        self.M = areas[0].ch.M
        self.n_src = [a.ch.n_src for a in areas]
        self.off = torch.cumsum(torch.tensor([0] + self.n_src, device=dev), 0)
        self.off_cpu = [sum(self.n_src[:k]) for k in range(len(self.n_src) + 1)]   # 版54b：境目は CPU の数で
        self.age = torch.cat([a.ch.age for a in areas])          # [送り元の合計, M]
        # 領野ごとに時間細胞の数が違う（運動は 0）ので、出す幅も領野ごとに持つ
        self.n_cell = [a.ch.n_cell for a in areas]
        self.n_ramp = [a.ch.n_ramp for a in areas]
        self.pref = [a.ch.pref for a in areas]
        self.half = [a.ch.half for a in areas]
        self.ramp_len = [a.ch.ramp_len for a in areas]
        self.gen = areas[0].ch.gen
        self._cache = None

    @torch.no_grad()
    def spikes(self):
        """[送り手の口の合計] を1本で返す（領野ごとの塊が順に並ぶ）"""
        if self._cache is not None:
            return self._cache
        # 版78：くじは領野の順・上り下りの順に外で引き、撃つかは _tc_core（torch.compile）で1回に
        ups, dns = [], []
        for i in range(len(self.areas)):
            lo, hi = self.off_cpu[i], self.off_cpu[i + 1]
            h = self.n_ramp[i] // 2
            ups.append(torch.rand(hi - lo, h, device=self.dev, generator=self.gen))
            dns.append(torch.rand(hi - lo, self.n_ramp[i] - h, device=self.dev, generator=self.gen))
        if getattr(self, "_pmax", None) is None:
            self._pmax = [(p.max() if p is not None else None) for p in self.pref]
        self._cache = _run(_tc_core, self.age, self.pref, self.half, self._pmax, ups, dns, tuple(self.off_cpu),
                           tuple(self.n_cell), tuple(self.n_ramp), tuple(float(x) for x in self.ramp_len))
        return self._cache

    @torch.no_grad()
    def step(self, fired_all):
        """fired_all: [送り元の合計]（領野ごとの塊が順に並ぶ）。1回で全部進める"""
        self._cache = None
        self.age = torch.where(self.age < 10 ** 8, self.age + 1, self.age)
        # 版54b：撃った送り手だけ一番古い枠を 0 に（撃っていない所は同じ値を書き戻す。nonzero で待たない）
        oldest = self.age.argmax(1)
        rows = torch.arange(len(fired_all), device=self.dev)
        self.age[rows, oldest] = torch.where(fired_all > 0, torch.zeros_like(oldest), self.age[rows, oldest])


# 版79：2周目から最後の周までを HIP グラフで（形を毎周同じにして、1周を1回の呼び出しに）
#   【測】版78 1歩 23.2ms ／ 2周目から 1.19ms/周（GPU の中身より、呼び出しと待ちが大きい）
#   ANAPLEX_GRAPH=1（既定）：GPU でグラフ ／ 0：版78 と同じ周ごと ／ eager：グラフの形のまま撮らずに回す（確かめ用）
GRAPH = _os.environ.get("ANAPLEX_GRAPH", "1")
_GP_MIN, _GP_RATIO = 1024, 1.25


def _gp_cap(n):
    """版79：辺の箱の大きさ（1024 から 1.25倍ずつの段で、n 以上の一番小さい段。0 なら箱なし）"""
    if n <= 0:
        return 0
    c = _GP_MIN
    while c < n:
        c = int(math.ceil(c * _GP_RATIO / 256.0)) * 256
    return c


def _new_core(within, base_on, W, start):
    """版79：周回の中で新しく撃った送り手（W ＝ 周回の中で撃ちうる行）の辺の数と、その累積（版78 の nonzero・_expand と同じ並び）"""
    newW = (within[W] > 0) & ~base_on[W]
    lens = torch.where(newW, start[W + 1] - start[W], torch.zeros_like(W))
    return lens, lens.cumsum(0)


def _edges_core(lens, csum, W, start, C: int):
    """版79：辺の番号を大きさ C の箱に詰める。余りは本当の辺の写し（足すときは valid で 0 にする）
    版79b：余りの j 番目は j ÷ 本当の数 の余りの辺（⟲ 版79 は全部先頭の辺。1か所にぶつかって遅かった）"""
    j = torch.arange(C, device=csum.device)
    valid = j < csum[-1]
    jj = torch.where(valid, j, j % csum[-1])
    s = torch.searchsorted(csum, jj, right=True)
    e = start[W[s]] + jj - (csum[s] - lens[s])
    return e, valid


def _flush_vals(e, k_start, elig, A, a_now, w, e_recv, recv_grp):
    """版79：_flush_core の値だけ（書くのは外で。箱の余りは写した辺と同じ値を同じ所へ書く）／ 版107d：門は受け手の組ごと"""
    ks = k_start[e].long()
    gi = recv_grp[e_recv[e].long()].long()
    return (w[e] + elig[e] * (a_now[gi] - A[ks - 1, gi]).float()).clamp(min=0.0)


def _sum_core_v(e, valid, e_recv, e_src, e_inh, w, c_exc, amp_s, send_gain, mod_send, mod_drive_inh, is_inh_r,
                n_recv: int):
    """版79：_sum_core と同じ式。箱の余り（valid でない所）は 0 を足す"""
    re_ = e_recv[e].long()
    se_ = e_src[e].long()
    ie = e_inh[e]
    amp = torch.where(ie, 4.0, c_exc[re_]) * w[e]
    if amp_s is not None:
        amp = amp * amp_s[e]
    amp = amp * send_gain[se_]
    amp = amp * valid.to(amp.dtype)
    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)
    both.scatter_add_(0, re_ + n_recv * ie.long(), amp)
    mm = mod_send[se_] & ~ie
    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])
    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, re_, amp * mm)
    return both, mod


def _flush_sum_core(e, k_start, elig, A, a_now, w, kc1, e_recv, e_src, e_inh, c_exc, amp_s, send_gain, mod_send,
                    mod_drive_inh, is_inh_r, n_recv: int, recv_grp):
    """版80：_flush_core（学びの分を足す）と _sum_core（足し込み）を1つに。式は同じ（足した重さをそのまま足し込みに使う）"""
    ks = k_start[e].long()
    gi = recv_grp[e_recv[e].long()].long()   # 版107d
    w1 = (w[e] + elig[e] * (a_now[gi] - A[ks - 1, gi]).float()).clamp(min=0.0)
    w[e] = w1
    k_start[e] = kc1
    re_ = e_recv[e].long()
    se_ = e_src[e].long()
    ie = e_inh[e]
    amp = torch.where(ie, 4.0, c_exc[re_]) * w1
    if amp_s is not None:
        amp = amp * amp_s[e]
    amp = amp * send_gain[se_]
    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)
    both.scatter_add_(0, re_ + n_recv * ie.long(), amp)
    mm = mod_send[se_] & ~ie
    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])
    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, re_, amp * mm)
    return both, mod


def _learn_sum_core(e, e_recv, e_src, e_inh, is_inh_r, err_sig, g, pw, elig, hist_n, w, c_exc, k_start, A, a_k, kc1,
                    amp_s, send_gain, mod_send, mod_drive_inh, n_recv: int, recv_grp):
    """版80：_learn_core（学ぶ）と、2周目の頭の _flush_core・_sum_core を1つに。式は同じ（a_k ＝ 進めたあとの A[k]）"""
    r = e_recv[e].long()
    gg = g * err_sig[r]
    to_inh = is_inh_r[r]
    ie = e_inh[e]
    el = elig[e] + torch.where(ie | to_inh, torch.zeros_like(gg), gg) * pw
    elig[e] = el
    se = e_src[e].long()
    w0 = w[e]
    ltd = to_inh & ~ie & (hist_n[se] >= INH_LTD_N) & (w0 * c_exc[r] > INH_LTD_TH)
    wl = torch.where(ltd, w0 * INH_LTD_F, w0)
    ks = k_start[e].long()
    gi = recv_grp[r].long()   # 版107d
    w2 = (wl + el * (a_k[gi] - A[ks - 1, gi]).float()).clamp(min=0.0)
    w[e] = w2
    k_start[e] = kc1
    cnt = torch.zeros(n_recv, device=w.device, dtype=w.dtype)
    cnt.scatter_add_(0, r, torch.ones(e.shape[0], device=w.device, dtype=w.dtype))
    amp = torch.where(ie, 4.0, c_exc[r]) * w2
    if amp_s is not None:
        amp = amp * amp_s[e]
    amp = amp * send_gain[se]
    both = torch.zeros(n_recv * 2, device=w.device, dtype=amp.dtype)
    both.scatter_add_(0, r + n_recv * ie.long(), amp)
    mm = mod_send[se] & ~ie
    mm = mm & ~(mod_drive_inh[se] & to_inh)
    mod = torch.zeros(n_recv, device=w.device, dtype=amp.dtype).index_add(0, r, amp * mm)
    return cnt, both, mod


# 版81：1周目の足し込みを Triton の核で（印のある所だけ足す・辺の数は整数で数える）。GPU のときだけ
#   【測】probe_atomic（10-14）：230万本 小数の足し込み 0.67ms ／ 整数 0.158ms
#   ANAPLEX_SCATTER=0 で版80 と同じ（比べる用）
SCATTER = _os.environ.get("ANAPLEX_SCATTER", "1") != "0"
_SC_K, _SC_BLOCK = int(_os.environ.get("ANAPLEX_SCATTER_K", "16")), 256   # 版103c：64 → 16（【測】measure_life97 11-01：64 68.3 ／ 32 65.8 ／ 16 63.9 ／ 8 63.7ms。丸めの順だけ）    # 版82：下書きの枚数・束の大きさ（K=1 で版81 と同じ足し方）
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _scatter3_k(r_p, ie_p, amp_p, mm_p, both_p, mod_p, cnt_p, n, n_recv, DO_CNT: tl.constexpr, BLOCK: tl.constexpr,
                    K: tl.constexpr):
        # 版82：束ごとに K 枚の下書きのどれか（束の番号 ÷ K の余り）へ足す。1か所に同時に来る数が 1/K に
        pid = tl.program_id(0)
        o = pid * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        rep = pid % K
        r = tl.load(r_p + o, mask=m, other=0)
        ie = tl.load(ie_p + o, mask=m, other=0).to(tl.int32)
        a = tl.load(amp_p + o, mask=m, other=0.0)
        mm = tl.load(mm_p + o, mask=m, other=0).to(tl.int32)
        tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie, a, mask=m, sem="relaxed")
        tl.atomic_add(mod_p + rep * n_recv + r, a, mask=m & (mm != 0), sem="relaxed")
        if DO_CNT:
            tl.atomic_add(cnt_p + rep * n_recv + r, tl.full([BLOCK], 1, tl.int32), mask=m, sem="relaxed")
    _TRITON = True
except Exception:          # Triton が無い（CPU のコンテナなど）⇒ 版80 と同じ
    _TRITON = False


def _scatter3(r, ie, amp, mm, n_recv, do_cnt):
    """版81：both（興奮｜抑制）・mod（印 mm の辺だけ）・cnt（辺の数。整数で数えて小数に）"""
    # 版82：下書きを K 枚に分けて足し、受け手が読む前に足し合わせる（届く周・相手・量は同じ。小数の丸めの順だけ変わる）
    #   【測】probe_scatter_real2：写し1つ 2.37ms ／ 64枚・束 256 1.10ms
    n = r.shape[0]
    K, BL = _SC_K, _SC_BLOCK
    both = torch.zeros(K, n_recv * 2, device=amp.device, dtype=amp.dtype)
    mod = torch.zeros(K, n_recv, device=amp.device, dtype=amp.dtype)
    cnt = torch.zeros(K, n_recv if do_cnt else 1, device=amp.device, dtype=torch.int32)
    if n:
        _scatter3_k[(triton.cdiv(n, BL),)](r, ie, amp, mm, both, mod, cnt, n, n_recv, DO_CNT=do_cnt, BLOCK=BL, K=K)
    return both.sum(0), mod.sum(0), (cnt.sum(0).to(amp.dtype) if do_cnt else None)


def _expand_mask_core(lens, csum, start, C: int):
    """版84：累積和から辺の番号を C 本（C ＝ 辺の数ちょうど）。_expand と同じ並び（送り手の順・送り手の中の順）"""
    j = torch.arange(C, device=csum.device)
    s = torch.searchsorted(csum, j, right=True)
    return start[s] + j - (csum[s] - lens[s])


def _prior_core(tb, W1, b1, W2, b2, segs):
    """版86：型ごとの塊の見込みを1回に（式は Prior.logit と同じ。W1・b1・W2・b2 は塊ごとに積んだもの）"""
    out = []
    for i, (lo, hi) in enumerate(segs):
        out.append((torch.tanh(tb[lo:hi] @ W1[i] + b1[i]) @ W2[i] + b2[i]).squeeze(-1))
    return torch.cat(out)


# 版103c：焼いた素子の見込みを1つの核で（⟲ 版86〜103b は型の塊ごとに行列の積 2回 ⇒ 1歩 45回・約10ms【測】measure_prof103）
#   式は _prior_core と同じ（tanh(x W1 + b1) W2 + b2）。丸めの順だけ変わる。ANAPLEX_PRIOR_K=0 で前の形
PRIOR_K = _os.environ.get("ANAPLEX_PRIOR_K", "1") != "0"
if _TRITON:
    @triton.jit
    def _prior_k(tb_p, seg_p, W1_p, b1_p, W2_p, b2_p, out_p, n, F: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
                 ):
        pid = tl.program_id(0)
        o = pid * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        s = tl.load(seg_p + o, mask=m, other=0)
        fi = tl.arange(0, F)
        x = tl.load(tb_p + o[:, None] * F + fi[None, :], mask=m[:, None], other=0.0)
        acc = tl.load(b2_p + s, mask=m, other=0.0)
        for j in tl.static_range(H):
            w = tl.load(W1_p + s[:, None] * (F * H) + fi[None, :] * H + j, mask=m[:, None], other=0.0)
            h = tl.sum(x * w, axis=1) + tl.load(b1_p + s * H + j, mask=m, other=0.0)
            h = 2.0 * tl.sigmoid(2.0 * h) - 1.0          # tanh(h)（同じ式。丸めだけ違う）
            acc += h * tl.load(W2_p + s * H + j, mask=m, other=0.0)
        tl.store(out_p + o, acc, mask=m)


def _prior_fast(tb, seg_id, W1, b1, W2, b2):
    """版103c：_prior_core と同じ式を核1つで。W1 [型, F, H]・b1 [型, H]・W2 [型, H, 1]・b2 [型, 1]"""
    n, F = tb.shape
    H = W1.shape[2]
    out = torch.empty(n, device=tb.device, dtype=tb.dtype)
    BL = 128
    _prior_k[(triton.cdiv(n, BL),)](tb.contiguous(), seg_id, W1.contiguous(), b1.contiguous(), W2.reshape(-1, H).contiguous(),
                                    b2.reshape(-1).contiguous(), out, n, F=F, H=H, BLOCK=BL)
    return out


def _homeo_core(cnt, exc, recv_ema, drive, a_: float):
    """版88：受け手の平均・駆動の平均（式は版87 の _pass と同じ）"""
    return recv_ema + a_ * (cnt - recv_ema), drive + a_ * (exc.abs() - drive)


def _scale_down_core(rate_ema, target, is_hippo, scale_skip, scale, a_: float, sleep: bool):
    """版88：眠りは下げるだけ（式は版87 と同じ）"""
    f = (1 - 0.01 * (rate_ema / target - 1)).clamp(0.5, 1.0) ** a_
    if sleep:
        f = torch.where(is_hippo, torch.ones_like(f), f)
    f = torch.where(scale_skip, torch.ones_like(f), f)
    return scale * f


def _scale_up_core(rate_long, target, scale_skip, scale_up):
    """版88：起きているあいだは上げるだけ（式は版87 と同じ）"""
    up = torch.exp(K_WAKE_UP * (1.0 - rate_long / target).clamp(min=0.0))
    return scale_up * torch.where(scale_skip, torch.ones_like(up), up)


def _rate_core(a0, a1, a2, err_i, rep_i, inh_i, rate_ema, rate_long, n_recv: int, h: float, lt: float, awake: bool):
    """版88：恒常性の観測。a0・a1・a2 ＝ _any ＝ [表現, 誤差, 抑制]
    版89：誤差の行に誤差、表現の行に表現を入れる（⟲ 版59〜88 は入れ違い【み】10-15「バグは直して」）"""
    fired = torch.zeros(n_recv, device=a0.device)
    fired[err_i] = a1
    fired[rep_i] = a0
    fired[inh_i] = a2
    re_ = rate_ema + h * (fired - rate_ema)
    rl = rate_long + lt * (fired - rate_long) if awake else rate_long
    return re_, rl


def _flush_amp_core(e, k_start, elig, A, a_now, w, kc1, e_recv, e_src, e_inh, c_exc, amp_s, send_gain, mod_send,
                    mod_drive_inh, is_inh_r, recv_grp):
    """版81：_flush_sum_core の足し込みの前まで（式は同じ）。足し込みは _scatter3"""
    ks = k_start[e].long()
    gi = recv_grp[e_recv[e].long()].long()   # 版107d
    w1 = (w[e] + elig[e] * (a_now[gi] - A[ks - 1, gi]).float()).clamp(min=0.0)
    w[e] = w1
    k_start[e] = kc1
    re_ = e_recv[e].long()
    se_ = e_src[e].long()
    ie = e_inh[e]
    amp = torch.where(ie, 4.0, c_exc[re_]) * w1
    if amp_s is not None:
        amp = amp * amp_s[e]
    amp = amp * send_gain[se_]
    mm = mod_send[se_] & ~ie
    mm = mm & ~(mod_drive_inh[se_] & is_inh_r[re_])
    return re_.int(), ie, amp, mm


def _learn_amp_core(e, e_recv, e_src, e_inh, is_inh_r, err_sig, g, pw, elig, hist_n, w, c_exc, k_start, A, a_k, kc1,
                    amp_s, send_gain, mod_send, mod_drive_inh, recv_grp):
    """版81：_learn_sum_core の足し込みの前まで（式は同じ）。足し込み・辺の数は _scatter3"""
    r = e_recv[e].long()
    gg = g * err_sig[r]
    to_inh = is_inh_r[r]
    ie = e_inh[e]
    el = elig[e] + torch.where(ie | to_inh, torch.zeros_like(gg), gg) * pw
    elig[e] = el
    se = e_src[e].long()
    w0 = w[e]
    ltd = to_inh & ~ie & (hist_n[se] >= INH_LTD_N) & (w0 * c_exc[r] > INH_LTD_TH)
    wl = torch.where(ltd, w0 * INH_LTD_F, w0)
    ks = k_start[e].long()
    gi = recv_grp[r].long()   # 版107d
    w2 = (wl + el * (a_k[gi] - A[ks - 1, gi]).float()).clamp(min=0.0)
    w[e] = w2
    k_start[e] = kc1
    amp = torch.where(ie, 4.0, c_exc[r]) * w2
    if amp_s is not None:
        amp = amp * amp_s[e]
    amp = amp * send_gain[se]
    mm = mod_send[se] & ~ie
    mm = mm & ~(mod_drive_inh[se] & to_inh)
    return r.int(), ie, amp, mm


# 版83：2周目から最後の周までを1つのグラフに（辺の数は GPU の中で読む）。ANAPLEX_ONEGRAPH=0 で版82 と同じ
ONEGRAPH = _os.environ.get("ANAPLEX_ONEGRAPH", "1") != "0"
if _TRITON:
    @triton.jit
    def _pass_edges_k(csum_p, lens_p, W_p, start_p, k_start_p, elig_p, A_p, anow_p, rg_p, kc1_p, w_p,
                      e_recv_p, e_src_p, e_inh_p, c_exc_p, amp_p, send_gain_p, mod_send_p, mod_drive_inh_p, is_inh_r_p,
                      both_p, mod_p, nW, n_recv, HAS_AMP: tl.constexpr, LOG: tl.constexpr, K: tl.constexpr,
                      BLOCK: tl.constexpr):
        """版83：周回の中で新しく撃った送り手の辺（_edges_core）→ 学びの分を足す（_flush_vals）→ 足し込み（_sum_core_v）を1つに。
        辺の数（csum の最後）は GPU の中で読み、決まった数の束で辺の数まで回る（箱の余りは無い ⇒ 1本の辺は1か所だけが書く）"""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        tot = tl.load(csum_p + nW - 1).to(tl.int32)
        kc1 = tl.load(kc1_p)
        rep = pid % K
        for base in range(pid * BLOCK, tot, nprog * BLOCK):
            j = base + tl.arange(0, BLOCK)
            m = j < tot
            lo = tl.zeros([BLOCK], dtype=tl.int32)
            hi = tl.zeros([BLOCK], dtype=tl.int32) + nW
            for _ in tl.static_range(LOG):          # csum[s] > j の最初の s（searchsorted right と同じ）
                act = lo < hi
                mid = (lo + hi) // 2
                cv = tl.load(csum_p + mid, mask=m & act, other=0)
                go_r = act & (cv <= j)
                go_l = act & (cv > j)
                lo = tl.where(go_r, mid + 1, lo)
                hi = tl.where(go_l, mid, hi)
            s = lo
            cs = tl.load(csum_p + s, mask=m, other=0)
            ln = tl.load(lens_p + s, mask=m, other=0)
            snd = tl.load(W_p + s, mask=m, other=0)
            e = tl.load(start_p + snd, mask=m, other=0) + j - (cs - ln)
            ks = tl.load(k_start_p + e, mask=m, other=1).to(tl.int64)
            r = tl.load(e_recv_p + e, mask=m, other=0)
            gi = tl.load(rg_p + r, mask=m, other=0).to(tl.int64)       # 版107d：受け手の組の門
            a_now = tl.load(anow_p + gi, mask=m, other=0.0)
            ak = tl.load(A_p + (ks - 1) * 2 + gi, mask=m, other=0.0)
            pend = tl.load(elig_p + e, mask=m, other=0.0) * (a_now - ak).to(tl.float32)
            w1 = tl.maximum(tl.load(w_p + e, mask=m, other=0.0) + pend, 0.0)
            tl.store(w_p + e, w1, mask=m)
            tl.store(k_start_p + e, (tl.zeros([BLOCK], dtype=tl.int32) + kc1.to(tl.int32)).to(k_start_p.dtype.element_ty), mask=m)
            se = tl.load(e_src_p + e, mask=m, other=0)
            ie = tl.load(e_inh_p + e, mask=m, other=0).to(tl.int32)
            cx = tl.load(c_exc_p + r, mask=m, other=0.0)
            amp = tl.where(ie != 0, 4.0, cx) * w1
            if HAS_AMP:
                amp = amp * tl.load(amp_p + e, mask=m, other=1.0)
            amp = amp * tl.load(send_gain_p + se, mask=m, other=0.0)
            ms = tl.load(mod_send_p + se, mask=m, other=0).to(tl.int32)
            mdi = tl.load(mod_drive_inh_p + se, mask=m, other=0).to(tl.int32)
            iir = tl.load(is_inh_r_p + r, mask=m, other=0).to(tl.int32)
            mm = (ms != 0) & (ie == 0) & ((mdi == 0) | (iir == 0))
            tl.atomic_add(both_p + rep * 2 * n_recv + r + n_recv * ie, amp, mask=m, sem="relaxed")
            tl.atomic_add(mod_p + rep * n_recv + r, amp, mask=m & mm, sem="relaxed")


# 版88：学び・恒常性の小さな命令を compile で。ANAPLEX_V88=0 で版87 と同じ形
V88 = _os.environ.get("ANAPLEX_V88", "1") != "0"



# 版90：1周目から最後の周までを1つのグラフに。ANAPLEX_V90=0 で版89 と同じ
V90 = _os.environ.get("ANAPLEX_V90", "1") != "0"
if _TRITON:
    @triton.jit
    def _p1_flush_k(E_p, ne_p, k_start_p, elig_p, A_p, anow_p, rg_p, kc1_p, w_p, e_recv_p, e_src_p, e_inh_p, c_exc_p,
                    amp_p, send_gain_p, mod_send_p, mdi_p, iir_p, both_p, mod_p, n_recv,
                    HAS_AMP: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        """版90：_flush_sum_core（1周目の 学びの分を足す＋足し込み）と同じ式。辺は E の先頭 ne 本（重なりなし）"""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ne = tl.load(ne_p).to(tl.int32)
        kc1 = tl.load(kc1_p)
        rep = pid % K
        for base in range(pid * BLOCK, ne, nprog * BLOCK):
            j = base + tl.arange(0, BLOCK)
            m = j < ne
            e = tl.load(E_p + j, mask=m, other=0)
            ks = tl.load(k_start_p + e, mask=m, other=1).to(tl.int64)
            r = tl.load(e_recv_p + e, mask=m, other=0)
            gi = tl.load(rg_p + r, mask=m, other=0).to(tl.int64)       # 版107d：受け手の組の門
            a_now = tl.load(anow_p + gi, mask=m, other=0.0)
            ak = tl.load(A_p + (ks - 1) * 2 + gi, mask=m, other=0.0)
            w1 = tl.maximum(tl.load(w_p + e, mask=m, other=0.0) + tl.load(elig_p + e, mask=m, other=0.0) * (a_now - ak).to(tl.float32), 0.0)
            tl.store(w_p + e, w1, mask=m)
            tl.store(k_start_p + e, (tl.zeros([BLOCK], dtype=tl.int32) + kc1.to(tl.int32)).to(k_start_p.dtype.element_ty), mask=m)
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
                    k_start_p, A_p, ak_p, rg_p, kck_p, amp_p, send_gain_p, mod_send_p, mdi_p, both_p, mod_p, cnt_p, n_recv,
                    LTD_N, LTD_TH, LTD_F, HAS_AMP: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        """版90：_learn_sum_core（学ぶ ＋ 2周目の頭の 学びの分を足す・足し込み）と同じ式。辺の数も数える"""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ne = tl.load(ne_p).to(tl.int32)
        g = tl.load(g_p)
        pw = tl.load(pw_p)
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
            gi = tl.load(rg_p + r, mask=m, other=0).to(tl.int64)       # 版107d：受け手の組の門
            a_k = tl.load(ak_p + gi, mask=m, other=0.0)
            ak = tl.load(A_p + (ks - 1) * 2 + gi, mask=m, other=0.0)
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

        # 送り手の口の並び：まとめると「時間細胞ぜんぶ → 各領野の buf」になる。
        # Area 側は「自分の時間細胞 → 自分の buf」なので、番号を写し直す
        ch_off = torch.cumsum(torch.tensor([0] + [a.ch.n_cells for a in self.areas], device=dev), 0)
        buf_base = int(ch_off[-1])
        buf_off = torch.cumsum(torch.tensor([0] + [a.n_src * a.D for a in self.areas], device=dev), 0)
        remap = []
        for i, a in enumerate(self.areas):
            m = torch.empty(a.n_send, dtype=torch.long, device=dev)
            m[:a.ch.n_cells] = torch.arange(a.ch.n_cells, device=dev) + ch_off[i]
            m[a.ch.n_cells:] = torch.arange(a.n_src * a.D, device=dev) + buf_base + buf_off[i]
            remap.append(m)
        self.off_send = torch.cat([ch_off[:-1], torch.tensor([buf_base], device=dev)])  # 参考用
        # 版92：番号は最初から 32bit（値は同じ。組む途中の GPU の山を下げる）
        src = torch.cat([remap[i][a.e_src_sorted].int() for i, a in enumerate(self.areas)])
        recv = torch.cat([(a.e_recv.long() + self.off_recv[i]).int() for i, a in enumerate(self.areas)])
        w = torch.cat([a.w for a in self.areas])
        inh = torch.cat([a.e_inh for a in self.areas])
        self.n_send_all = buf_base + int(buf_off[-1])
        # 送り手が抑制性か（刈り込みで辺を並べ直したときに作り直すのに要る）
        self.is_inh_all = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)
        for i, a in enumerate(self.areas):
            self.is_inh_all[remap[i]] = a.is_inh_send
        # 刈り込みで生やす先の候補（領野ごと。番号は全体の通し）
        #   Area.rewire は自分の領野の送り元から引いていた。まとめたときに全領野に広がっていた
        self.pool_send = remap
        o = torch.argsort(src)
        # 辺の番号は 32bit で持つ（版43a-m【み】09-29 許可。番号は1つもずれない。64bit の半分）
        # 版92：写しを1本ずつ作り、元をすぐ手放す（⟲ 版91 まで4本の写しを全部作ってから）
        self.e_src = src[o]; del src
        self.e_recv = recv[o]; del recv
        self.w = w[o]; del w
        self.e_inh = inh[o]; del inh
        self.start = self._starts()

        self.stp = None
        if self.areas[0].stp is not None:
            import copy
            g = copy.copy(self.areas[0].stp)
            for k in ("row", "state", "fac", "last", "deplete"):
                setattr(g, k, torch.cat([getattr(a.stp, k) for a in self.areas])[o])
            self.stp = g
        del o                              # 版92

        self.scale = torch.cat([a.scale for a in self.areas])
        self.drive = torch.cat([a.drive for a in self.areas])
        self.recv_ema = torch.cat([a.recv_ema for a in self.areas])
        # 細胞自身の発火率（恒常性が感じるもの）。目標から始める
        self.rate_ema = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)
        # 恒常性の目標を受け手ごとに（版39b。torch_brain が種類ごとの【ヒト】の値を入れる）
        self.target_recv = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)
        self.scale_skip = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)   # 人の値が無い受け手は下げない
        self.rate_long = self.target_recv.clone()   # 版53：発火の長い平均（1時間。静か・撃ちすぎの判定）
        self.scale_up = torch.ones(self.n_recv_all, device=dev)   # 版53：起きているあいだ上げる恒常性（興奮だけに掛ける）
        # 学びの門（版40）：学んだ跡を辺ごとに持ち、脳幹が配る第三の信号 third の分だけ重みを変える
        self.elig = torch.zeros_like(self.w)
        self.elig_decay = math.exp(-1.0 / ELIG_TAU)
        # 版54：学びを触ったときにまとめる。elig は「跡 × 減り^(−k)」（区切りの頭からの刻み k）、k_start は辺がまだ足していない最初の刻み
        self.k_start = torch.ones(len(self.w), dtype=torch.int16, device=dev)
        self.A = torch.zeros(LAZY_EPOCH + 1, 2, dtype=torch.float64, device=dev)   # 版107d：[刻み, 組（黒質・VTA）]
        self.recv_grp = torch.zeros(self.n_recv_all, dtype=torch.int32, device=dev)  # 版107d：受け手の行の門の組（脳が決める）
        self.kcur = 0
        self._prior_l = None        # 版54：焼いた網の見込み（1歩に1回）
        self._amp_buf = None        # 版54：短期可塑性の値を持ち続ける配列
        self._amp_prev = None
        self._new_e = None
        self._both_base = None
        self.third = torch.ones(2, device=dev)       # 版107d：(黒質の門, VTA の門)
        self.th_gain = torch.ones((), device=dev)   # 覚醒で表現の閾値に掛ける倍率（版41。脳幹が決める）
        self.decay = torch.cat([a.decay.unsqueeze(0).expand(a.n_recv, -1) for a in self.areas])
        self.tb = torch.cat([a.tb for a in self.areas])
        # 短い遅れ D は領野ごとに違う（運動は自分の発火が戻る道を広く持つので 8、他は 1）
        # 揃えると中身が変わるので、buf は領野ごとに別々の幅のまま持ち、口の番号だけ繋ぐ
        self.bufs = [a.buf for a in self.areas]
        self.Ds = [a.D for a in self.areas]
        self.ch = TimeCellsGroup(self.areas)

        # 受け手が「どの領野の、どの役か」を印で持つ（ループの代わり）
        kind, area_of, ch_of, om_of = [], [], [], []
        for i, a in enumerate(self.areas):
            n_om = getattr(a, "n_om", 0)          # 版101：来ない体の行は誤差の種類（0）で、撃つ向きだけ逆
            kind += [0] * a.n_ch + [1] * a.n_ch + [2] * a.n_inh + [0] * n_om
            area_of += [i] * a.n_recv
            ch_of += list(range(a.n_ch)) + list(range(a.n_ch)) + [-1] * a.n_inh + list(range(n_om))
            om_of += [False] * (2 * a.n_ch + a.n_inh) + [True] * n_om
        self.kind = torch.tensor(kind, device=dev)
        self.area_of = torch.tensor(area_of, device=dev)
        self.ch_of = torch.tensor(ch_of, device=dev)
        self.is_err = self.kind == 0
        self.is_rep = self.kind == 1
        self.is_inh_r = self.kind == 2
        # 版54b：決まった印は番号の列でも持つ（印で選ぶと、数を読むために GPU を待つ）
        self.err_i, self.rep_i, self.inh_i = [torch.nonzero(m).squeeze(1) for m in (self.is_err, self.is_rep, self.is_inh_r)]
        # 版101：来ない体（誤差の種類の行のうち、撃つ ＝ (p − 的) ＞ THETA）。誤差の行の並び（err_i）には領野ごとに 誤差 → 来ない の順で入る
        self.is_om = torch.tensor(om_of, dtype=torch.bool, device=dev)
        self.fire_sgn = torch.where(self.is_om, -1.0, 1.0).to(dev)        # 撃つ向き（学びは 的 − p のまま）
        self.err_ch_i = torch.nonzero(self.is_err & ~self.is_om).squeeze(1)   # 誤差の体の行（帯の順。版100 の err_i と同じ）
        om_e = self.is_om[self.err_i]
        self.err_pos_ch = torch.nonzero(~om_e).squeeze(1)                  # 誤差の発火の並び（err_i の順）のうち、誤差の体の所
        self.om_pos = torch.nonzero(om_e).squeeze(1)                       # 同じく来ない体の所
        # 表現・誤差の受け手が、まとめた入口のどこに当たるか
        self.x_index = torch.zeros(self.n_recv_all, dtype=torch.long, device=dev)
        for i, a in enumerate(self.areas):
            lo = int(self.off_recv[i]); c = int(self.off_ch[i])
            self.x_index[lo:lo + a.n_ch] = torch.arange(a.n_ch, device=dev) + c
            self.x_index[lo + a.n_ch:lo + 2 * a.n_ch] = torch.arange(a.n_ch, device=dev) + c
            n_om = getattr(a, "n_om", 0)          # 版101：来ない体も同じ帯の入口（的）
            if n_om:
                o_om = lo + 2 * a.n_ch + a.n_inh
                self.x_index[o_om:o_om + n_om] = torch.arange(n_om, device=dev) + c
        # 抑制の受け手には、その領野の入口の平均を渡す
        self.inh_area = self.area_of[self.is_inh_r]
        # 版103f：領野の入口の平均は累積和の差で（⟲ 版54b〜103e は 帯 × 領野 の密な行列 mean_map との積。1歩 5.36ms・VRAM も 帯 × 58 × 4バイト）
        self._a_lo = torch.tensor([int(self.off_ch[i]) for i in range(len(self.areas))], dtype=torch.long, device=dev)
        self._a_hi = self._a_lo + torch.tensor([a.n_ch for a in self.areas], dtype=torch.long, device=dev)
        self._a_n = torch.tensor([float(a.n_ch) for a in self.areas], dtype=torch.float64, device=dev)
        self.spont_p = torch.cat([torch.full((a.n_ch,), a.spont_p, device=dev) for a in self.areas])
        # 版58：自発のくじは周ごと（1周 ＝ 【ヒト】シナプス1本の遅れ 1.1ms）。1回あたりは1周ぶんの確率にする
        #   ⟲ 版57 まで1歩ぶんの確率で周ごと10回引き、相手に届く自発が置いた率の約10倍だった【測】measure_v57b
        self.spont_p_pass = 1.0 - (1.0 - self.spont_p) ** (1.0 / self.recur)
        self.complete = torch.cat([torch.full((a.n_ch,), float(a.complete), device=dev)
                                   for a in self.areas])
        self.pred_boost = torch.cat([torch.full((a.n_ch,), a.pred_boost, device=dev)
                                     for a in self.areas])
        self.gate = torch.cat([torch.full((a.n_ch,), 1.0 if a.gate else 0.0, device=dev)
                               for a in self.areas])
        self.gate_err = self.gate[self.x_index[self.err_i]]   # 版101：誤差の発火の並び（来ない体を含む）の門
        self.gen = self.areas[0].gen
        # 閾値：シナプスの合計が「同時に 134本」に当たるところ（人の値）
        #   x_gain は、入口から直に来る駆動を「何本ぶん」と数えるか
        self.fire_th = torch.cat([torch.full((a.n_ch,), a.fire_th, device=dev) for a in self.areas])
        self.x_gain = torch.cat([torch.full((a.n_ch,), a.x_gain, device=dev) for a in self.areas])
        # 興奮性の接触の数（受け手ごと）。既定は【ヒト】皮質 3.3
        self.c_exc = torch.cat([torch.full((a.n_recv,), float(getattr(a, "contacts_exc", 3.3)), device=dev)
                                for a in self.areas])
        self.tgt_by_fire = torch.cat([torch.full((a.n_ch,), bool(getattr(a, "tgt_by_fire", False)),
                                                 dtype=torch.bool, device=dev) for a in self.areas])
        # 版47：錐体 → 抑制の体 の重みは人の分布で始める（予測誤差では学ばない。形A の直し）
        ei = self.is_inh_r[self.e_recv.long()] & ~self.e_inh
        self.w[ei] = self._inh_w(self.e_recv[ei].long())
        # 送り手ごとの直近 INH_LTD_WIN 歩の発火（弱まりの条件を見るため）
        self.hist = torch.zeros((INH_LTD_WIN, self.n_send_all), dtype=torch.uint8, device=dev)
        self.hist_n = torch.zeros(self.n_send_all, dtype=torch.int16, device=dev)
        self.hist_i = 0
        self.t = 0
        self._e_cache = None
        self._within = None
        self._base = None
        self.scaling_on = True          # 脳幹（brainstem.py）が状態で切り替える（版38）
        self.mod_send = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 調整する側の送り手（版39。torch_brain が印を付ける）
        self.mod_drive_inh = torch.zeros(self.n_send_all, dtype=torch.bool, device=dev)   # 版66：調整する側のうち抑制の体は撃たせる送り手
        self.send_gain = torch.ones(self.n_send_all, device=dev)   # 送り手ごとの重さ。眠りで上からの戻りを落とす（版38）
        self.eta_gain = 1.0             # 学ぶ速さの掛け算。起きているあいだの驚き（版38）
        self.exc_gain = None            # リップルのときだけ、興奮の入力を底上げする
        self.sleep_mode = False
        self.is_hippo = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)
        self.rewire_every = REWIRE_STEPS      # 版53（⟲ 版52 まで 100歩ごとに受け手ごと1本）
        # 相手の数は受け手ごと（【ヒト】1体 30,000シナプス・相手 9,100人）。
        #   ⚠️ 先頭の領野の値を全体に当てていた。視床は時間細胞を持たないので口が小さく、
        #      その値が全部の領野に効いていた（視床の相手が少ないこと自体は人の形）
        self.n_partner = torch.cat([a.n_partner_vec for a in self.areas])   # 版47：抑制の体は相手が少ない
        self.budget = torch.cat([a.budget for a in self.areas])
        # 周回の中で撃った発火を、短い遅れの「1歩前」の枠に重ねるための番号
        self.within_index = torch.cat(
            [torch.arange(a.n_ch, device=dev) * a.D + buf_base + int(buf_off[i])
             for i, a in enumerate(self.areas)])
        # 版63：抑制の体の行（送り元の並び 表現・ext・抑制 の最後）。並びは inh_spk と同じ（領野ごとに n_inh）
        self.within_inh_index = torch.cat(
            [(a.n_ch + a.n_ext + torch.arange(a.n_inh, device=dev)) * a.D + buf_base + int(buf_off[i])
             for i, a in enumerate(self.areas)])
        # ここまでで、領野ごとの辺・短期可塑性・時間細胞の中身はすべて group 側へ写してある。
        # 元を抱えたままだと、同じものを2つ持つことになる
        self._release_areas()
        # 版52 A・B：領野の境目と complete の有無は変わらないので CPU の数で持つ
        #   ⟲ 版51 まで GPU の配列。周回ごとに int() で読むたび GPU を待っていた（1歩 約360回）
        self.off_recv, self.off_ch, self.off_src = (self.off_recv.cpu().numpy(), self.off_ch.cpu().numpy(),
                                                    self.off_src.cpu().numpy())
        self._complete_any = bool(self.complete.any())
        # 版55：焼いた網は型ごと。受け手の並びで同じ網が続く塊にまとめる（全部同じなら1つ）
        segs = []
        for i, a in enumerate(self.areas):
            lo, hi = int(self.off_recv[i]), int(self.off_recv[i + 1])
            if segs and segs[-1][2] is a.prior and segs[-1][1] == lo:
                segs[-1][1] = hi
            else:
                segs.append([lo, hi, a.prior])
        self._prior_segs = segs
        self._base_of = None

    def _inh_w(self, r):
        """版47：錐体 → 抑制の体 の1本（繋がり1つ。平均 1.65mV を版45 の単位にした INH_W_MEAN）を接触の数で割って w に"""
        z = torch.randn(len(r), device=self.dev, generator=self.gen)
        return torch.exp(INH_W_MU + INH_W_SIG * z) * INH_W_MEAN / self.c_exc[r]

    def _area_mean(self, x):
        """版103f：領野ごとの x の平均（＝ x @ mean_map。float64 の累積和の差）"""
        cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=x.device), x.double().cumsum(0)])
        return ((cs[self._a_hi] - cs[self._a_lo]) / self._a_n).to(x.dtype)

    def _starts(self):
        """送り手ごとの辺の始まり（e_src は並んでいる）。数えるだけなので 64bit の写しを作らない（版43a-m）"""
        cnt = torch.bincount(self.e_src, minlength=self.n_send_all)
        return torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev), cnt.cumsum(0)])

    def _release_areas(self):
        """まとめ終わった領野ごとの配列を手放す（中身は group に移っている）

        ⚠️ 手放したあと Area 単体では回せない。Group2 に入れたら group で回す
        ⚠️ buf は残す（self.bufs が同じものを指している）
        """
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

    @property
    def n_cells(self):
        return sum(a.n_cells for a in self.areas)

    @torch.no_grad()
    def _expand(self, fired):
        """撃った送り手 → その辺の番号（送り手ごとに連ねて持っているので、start から切り出す）"""
        empty = torch.empty(0, dtype=torch.long, device=self.dev)
        if len(fired) == 0:
            return empty
        lens = self.start[fired + 1] - self.start[fired]
        # 版54b：総数を1回だけ読む（長さ 0 の送り手は repeat_interleave で何も出さないので、外さなくても同じ）
        tot = int(lens.sum())
        if tot == 0:
            return empty
        base = self.start[fired].repeat_interleave(lens, output_size=tot)
        off = torch.arange(tot, device=self.dev) - torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev),
             lens.cumsum(0)[:-1]]).repeat_interleave(lens, output_size=tot)
        return base + off

    @torch.no_grad()
    def _expand_mask(self, mask):
        """版84：撃った送り手（mask）→ その辺の番号。_expand(nonzero(mask)) と同じ並び。CPU が待つのは辺の数の1回"""
        lens = torch.where(mask, self.start[1:] - self.start[:-1], torch.zeros_like(self.start[1:]))
        csum = lens.cumsum(0)
        tot = int(csum[-1])
        if tot == 0:
            return torch.empty(0, dtype=torch.long, device=self.dev)
        return _run(_expand_mask_core, lens, csum, self.start, tot)

    def _buf_flat_get(self):
        """版84：buf を1本の配列で（領野ごとの [送り元, D] を順に並べたもの。読み込みなどで置き換わったら作り直す）"""
        refs = getattr(self, "_buf_refs", None)
        if refs is None or len(refs) != len(self.bufs) or any(a is not b for a, b in zip(refs, self.bufs)):
            self._buf_flat = torch.cat([b.reshape(-1) for b in self.bufs])
            self._buf_refs = tuple(self.bufs)
        return self._buf_flat

    def _build_tail_idx(self, n_rep, n_ext, n_inh):
        """版84：歩の終わりの番号の表（入口の並べ直し・buf のずらし）。領野の形だけで決まるので1回だけ作る"""
        dev = self.dev
        si = torch.empty(self.n_src_all, dtype=torch.long, device=dev)
        eo = io = 0
        for i, a in enumerate(self.areas):        # cat([rep, ext, inh]) の中の番号（版79 の切り貼りと同じ並び）
            lo = int(self.off_src[i]); c = int(self.off_ch[i])
            si[lo:lo + a.n_ch] = torch.arange(c, c + a.n_ch, device=dev)
            if a.n_ext:
                si[lo + a.n_ch:lo + a.n_ch + a.n_ext] = n_rep + torch.arange(eo, eo + a.n_ext, device=dev)
                eo += a.n_ext
            si[lo + a.n_ch + a.n_ext:int(self.off_src[i + 1])] = n_rep + n_ext + torch.arange(io, io + a.n_inh, device=dev)
            io += a.n_inh
        bi, shapes, bo = [], [], 0
        for i, b in enumerate(self.bufs):        # cat([src, 古い buf]) の中の番号：列 0 は新しい入口、列 d は古い列 d−1
            n, d = b.shape
            r = torch.arange(n, device=dev).unsqueeze(1)
            col = torch.arange(d, device=dev).unsqueeze(0)
            old = self.n_src_all + bo + r * d + (col - 1)
            bi.append(torch.where(col == 0, int(self.off_src[i]) + r + 0 * col, old).reshape(-1))
            shapes.append((bo, n, d))
            bo += n * d
        self._src_idx, self._buf_idx, self._buf_shapes = si, torch.cat(bi), shapes
        self._tail_n = (n_rep, n_ext, n_inh)

    def _fired_edges(self):
        # 版52 C：時間細胞と buf の辺は1歩に1回だけ作り、周回の中で新しく撃った送り手の辺だけ足す
        #   ⟲ 版51 まで周回ごとに全部作り直していた。足す辺の組は同じ（並びだけ違う）【測】measure_speed 10-07
        if self._e_cache is not None:
            return self._e_cache
        if self._base is None or self._base_of is not self._base:
            if self._base is None:                # 時間細胞と buf は1歩に1回でいい
                self._base = torch.cat([self.ch.spikes(), self._buf_flat_get()])   # 版84：buf は1本の配列で（並びは同じ）
            self._base_of = self._base
            self._base_on = self._base > 0
            if type(self)._expand is _G2_ORIG["_expand"]:   # 版84：数えずに累積和から（CPU が待つのは辺の数の1回）
                self._base_e = self._expand_mask(self._base_on)
            else:
                self._base_e = self._expand(torch.nonzero(self._base_on).squeeze(1))
        if self._within is None:
            self._e_cache = self._base_e
            self._new_e = None
        else:                                     # 周回の中で撃った分も送り手になる
            new = torch.nonzero((self._within > 0) & ~self._base_on).squeeze(1)
            self._new_e = self._expand(new) if len(new) else None     # 版54：周回の中の新しい分（別に足す）
            self._e_cache = torch.cat([self._base_e, self._new_e]) if self._new_e is not None else self._base_e
        return self._e_cache

    @torch.no_grad()
    def _flush(self, idx):
        """版54：idx の辺に、まだ足していない学びの分（k_start から今の刻みまで）を足して 0 で止める"""
        if idx is None or len(idx) == 0:
            return
        _run(_flush_core, idx, self.k_start, self.elig, self.A, self.A[self.kcur], self.w, self.e_recv, self.recv_grp)   # 版76 ／ 版107d
        self.k_start.index_fill_(0, idx, self.kcur + 1)     # 版54b

    @torch.no_grad()
    def flush_all(self):
        """版54：全部の辺へまとめて足す（刈り込み・保存・区切りの前）。メモリを抑えるため塊ごとに"""
        n = len(self.w); step = 1 << 24
        for a in range(0, n, step):
            s = slice(a, min(n, a + step))
            gi = self.recv_grp[self.e_recv[s].long()].long()            # 版107d：受け手の組の門
            pend = self.elig[s] * (self.A[self.kcur][gi] - self.A[self.k_start[s].long() - 1, gi]).float()
            self.w[s] = (self.w[s] + pend).clamp(min=0.0)
        self.k_start.fill_(self.kcur + 1)

    @torch.no_grad()
    def _new_epoch(self):
        """版54：区切り。全部の辺へ足し、跡を新しい区切りの頭（今の刻み）に合わせ直す"""
        self.flush_all()
        self.elig.mul_(self.elig_decay ** self.kcur)
        self.k_start.fill_(1)
        self.A.zero_()
        self.kcur = 0

    def _order_after_cut(self, cuts, new):
        """版107h：送り元の順に並んだ辺のうち cuts の送り元を new に変えたあとの並び（32bit の番号）。
        残る辺は今の順のまま、変えた辺だけを送り元の値の所へ差し込む（同じ送り元の中では残る辺のあと）。
        argsort と同じく送り元の順に並ぶ（同じ送り元の中の順だけが違いうる。辺の中身・送り元ごとの組は同じ）"""
        E, dev = len(self.e_src), self.dev
        nv = new.to(self.e_src.dtype)
        ns, nperm = torch.sort(nv)
        cut_sorted = cuts[nperm]
        old_cut = torch.sort(self.e_src[cuts]).values                       # 抜く辺の今の送り元
        kept_le = (torch.searchsorted(self.e_src, ns, right=True)
                   - torch.searchsorted(old_cut, ns, right=True))            # 値以下で残る辺の数
        final_new = kept_le + torch.arange(len(ns), device=dev)
        o = torch.empty(E, dtype=torch.int32, device=dev)
        o[final_new] = cut_sorted.to(torch.int32)
        slot_kept = torch.ones(E, dtype=torch.bool, device=dev)
        slot_kept[final_new] = False
        keep = torch.ones(E, dtype=torch.bool, device=dev)
        keep[cuts] = False
        kept = torch.arange(E, dtype=torch.int32, device=dev)[keep]
        del keep
        o.masked_scatter_(slot_kept, kept)
        return o

    def _first_per_recv(self, key):
        """受け手ごとに key が一番小さい辺（key が inf の辺しか無い受け手は −1）。同じ値なら番号が一番小さい辺
        版91：並べ替えなしで塊ごとに（⟲ 版90 まで argsort 2回 ⇒ 刈り込みで GPU の山 +8.4GB【測】watch_vram2）"""
        n, m, step = self.n_recv_all, len(key), 1 << 24
        mn = torch.full((n,), float("inf"), dtype=key.dtype, device=self.dev)
        for a in range(0, m, step):
            b = min(m, a + step)
            mn.scatter_reduce_(0, self.e_recv[a:b].long(), key[a:b], reduce="amin")
        big = torch.iinfo(torch.long).max
        pick = torch.full((n,), big, dtype=torch.long, device=self.dev)
        for a in range(0, m, step):
            b = min(m, a + step)
            r = self.e_recv[a:b].long()
            k = key[a:b]
            hit = (k == mn[r]) & torch.isfinite(k)
            idx = torch.arange(a, b, device=self.dev)
            pick.scatter_reduce_(0, r[hit], idx[hit], reduce="amin")
        pick[pick == big] = -1
        return pick

    @torch.no_grad()
    def rewire(self):
        """版53：細胞の撃ち具合で刈り方を決める（受け手ごと。寿命 LIFE_STEPS の確率で）
        ふだん          一番弱い興奮の辺を切り、興奮で生やす   【ヒト】刈り込みで消えるのは小さい棘
        静か            抑制の辺を切り、興奮で生やす           【ネズミ】入力が減ると抑制が外れる ／【ヒト iPS】止めると興奮性が増える
        撃ちすぎ        一番弱い興奮の辺を切り、抑制で生やす   【ネズミ】学ぶと抑制が増える
        ⇒ 人の率ちょうどで釣り合い、興奮と抑制の割合が保たれる（【ヒト】割合は領野・個人でほぼ一定）
        ⟲ 版52 まで：種類を見ずに一番弱い1本を切り、送り元の割合で生やした。抑制は固定の強さ・興奮は 0 で生えて 0 へ痩せるので
           切られるのはいつも興奮 ⇒ 興奮 → 抑制 の片道（【測】measure_v52a 視床の興奮 64 → 11本）"""
        if str(self.dev).startswith("cuda"):     # 版99b：大きな作業の前に、使っていない取り置きを返す（値は変わらない）
            torch.cuda.empty_cache()             #   【測】measure_mem99：使用 4.86GB・取り置き 8.86GB の上に刈り込みで +4.04 ⇒ 山 13.11GB（12GB【み】10-15 を越えた）
        self.flush_all()          # 版54：一番弱い辺を選ぶ前に、学びの分を全部足しておく
        if getattr(self, "_occ", None) is not None:     # 版104：ふさがり。切る前に学びの変わりを読む
            self._occ.rewire_pre()
        self.n_rewire = getattr(self, "n_rewire", 0) + 1        # 何回まわしたか
        n = self.n_recv_all
        r_all = self.e_recv.long()
        inh = self.e_inh
        n_e = torch.bincount(r_all[~inh], minlength=n).float()
        n_i = torch.bincount(r_all[inh], minlength=n).float()
        R = float(self.rewire_every)
        do_e = (torch.rand(n, device=self.dev, generator=self.gen) < R * n_e / LIFE_STEPS) & (n_e > 0)
        do_i = (torch.rand(n, device=self.dev, generator=self.gen) < R * n_i / LIFE_STEPS) & (n_i > 0)
        judge = ~self.scale_skip
        quiet = judge & (self.rate_long < self.target_recv)
        over = judge & (self.rate_long > self.target_recv)
        # 版107h：鍵は 1つの置き場所を使い回す（⟲ inf と where を 2つずつ ＝ 辺の数 × 4バイト × 4 を一度に持っていた）
        key = self.w.clone()
        key.masked_fill_(inh, float("inf"))
        weak_e = self._first_per_recv(key)                                                 # 一番弱い興奮
        key.uniform_(generator=self.gen)
        key.masked_fill_(~inh, float("inf"))
        rand_i = self._first_per_recv(key)                                                 # でたらめな抑制1本
        del key                            # 版92：選び終えたら手放す
        cut_e = (do_e | (over & do_i)) & (weak_e >= 0)
        cut_i = do_i & ~over & (rand_i >= 0)
        cuts = torch.cat([weak_e[cut_e], rand_i[cut_i]])
        if len(cuts) == 0:
            return
        to_inh = torch.cat([(over & do_i)[cut_e], ~quiet[cut_i]])   # 生やす種類（True ＝ 抑制）
        # 版61：中継（no_lat_exc の領野）の表現・誤差の行には興奮を生やさない（【マカク】中継どうしの興奮はほぼ無い）
        if getattr(self, "_no_lat_exc_r", None) is None:
            fl = torch.tensor([bool(getattr(a, "no_lat_exc", False)) for a in self.areas], device=self.dev)
            self._no_lat_exc_r = fl[self.area_of] & ~self.is_inh_r
        to_inh = to_inh | self._no_lat_exc_r[self.e_recv[cuts].long()]
        self.n_cut = getattr(self, "n_cut", 0) + len(cuts)      # 何本 切ったか
        self.n_e2i = getattr(self, "n_e2i", 0) + int((over & do_i & cut_e).sum())   # 興奮 → 抑制（撃ちすぎ）
        self.n_i2e = getattr(self, "n_i2e", 0) + int((quiet & cut_i).sum())          # 抑制 → 興奮（静か）
        # 生やす先：受け手と同じ領野の、決めた種類の送り元から一様に
        if getattr(self, "_pool_ei", None) is None:
            self._pool_ei = [(p[~self.is_inh_all[p]], p[self.is_inh_all[p]]) for p in self.pool_send]
            # 版57：海馬の抑制の体へ生やす興奮は、遅れの列の興奮の行（時間細胞ではない）から
            nb = self.n_send_all - sum(a.n_src * a.D for a in self.areas)      # 遅れの列の始まり（通し番号）
            self._pool_e_rep = [pe[pe >= nb] for pe, _ in self._pool_ei]
            self._inh_rep_area = [bool(getattr(a, "inh_exc_from_rep", False)) for a in self.areas]
            # 版64：inh_slice の領野の表現・誤差の行へ生やす抑制は、抑制の体の1歩前の列から
            self._inh_slice_area = [bool(getattr(a, "inh_slice", False)) for a in self.areas]
            oi = 0
            self._pool_i_col0 = []
            for a in self.areas:
                self._pool_i_col0.append(self.within_inh_index[oi:oi + a.n_inh])
                oi += a.n_inh
        ar = self.area_of[self.e_recv[cuts].long()]
        new = torch.empty(len(cuts), dtype=torch.long, device=self.dev)
        for i in range(len(self.areas)):
            for kind in (0, 1):
                m = (ar == i) & (to_inh == bool(kind))
                k = int(m.sum())
                if k:
                    pool = self._pool_ei[i][kind]
                    new[m] = pool[torch.randint(0, len(pool), (k,), device=self.dev, generator=self.gen)]
                    if kind == 1 and self._inh_slice_area[i]:   # 版64：表現・誤差の受け手は抑制の体の1歩前の列から
                        mr = m & ~self.is_inh_r[self.e_recv[cuts].long()]
                        kr = int(mr.sum())
                        if kr:
                            pc = self._pool_i_col0[i]
                            new[mr] = pc[torch.randint(0, len(pc), (kr,), device=self.dev, generator=self.gen)]
                    if kind == 0 and self._inh_rep_area[i]:     # 版57：抑制の体の受け手だけ引き直す
                        mi = m & self.is_inh_r[self.e_recv[cuts].long()]
                        ki = int(mi.sum())
                        if ki:
                            pr_ = self._pool_e_rep[i]
                            new[mi] = pr_[torch.randint(0, len(pr_), (ki,), device=self.dev, generator=self.gen)]
        o = self._order_after_cut(cuts, new)    # 版107h：並べ替えなしで差し込む（⟲ argsort：64bit の番号と並べ替えの作業で 約3GB）
        self.e_src[cuts] = new.to(self.e_src.dtype)
        # 版79：並べ直しは同じ置き場所へ写す（グラフが配列の置き場所を覚えているため。値は同じ）
        for nm in ("e_src", "e_recv", "w", "elig", "k_start"):     # k_start は版54（flush_all のあとなので全部同じ値）
            t_ = getattr(self, nm)
            t_.copy_(t_[o])
        if self._amp_buf is not None and len(self._amp_buf) == len(self.w):
            self._amp_buf.fill_(1.0)       # 版54：辺の並びが変わったので作り直す（版79：同じ置き場所で 1 に）
        else:
            self._amp_buf = None
        self._amp_prev = None
        torch.index_select(self.is_inh_all, 0, self.e_src, out=self.e_inh)
        # ⟲ 版40 で直した：並べ直したあとに、並べ直す前の位置 cuts で 0 にしていた。
        #    開通した辺は切った辺の重みを持ち越し、代わりに無関係の辺の重みが 100歩ごとに受け手の数だけ消えていた
        pos = torch.empty_like(o)
        pos[o] = torch.arange(len(o), device=self.dev, dtype=o.dtype)    # 版92：32bit
        cuts = pos[cuts].long()
        if getattr(self, "_occ", None) is not None:     # 版104：ふさがりの痕跡を新しい並びへ（CPU で。VRAM を増やさない）
            self._occ.rewire_map(pos, cuts)
        del pos
        self.w[cuts] = 0.0             # 開通の値（決まったこと 09-11：開通は w0、学びの跡は 0）
        self.w[self.e_inh] = W_INH     # 版45：開通した抑制も人の強さ（抑制は学ばないので全部同じ値）
        ei = cuts[self.is_inh_r[self.e_recv[cuts].long()] & ~self.e_inh[cuts]]
        if len(ei):                    # 版47：開通した 錐体 → 抑制の体 も人の分布から
            self.w[ei] = self._inh_w(self.e_recv[ei].long())
        self.elig[cuts] = 0.0
        if getattr(self, "_occ", None) is not None:     # 版104：当てる辺を選び直し、生やした辺の痕跡は 0 から
            self._occ.rewire_done()
        if self.stp is not None:
            self.stp.reorder(o)
            # 版53：生やした辺の短期可塑性は新しい種類の組から（⟲ 版52 まで切った辺の行を持ち越していた）
            self.stp.redraw(cuts, self.e_inh[cuts], self.is_inh_r[self.e_recv[cuts].long()])
        self.start.copy_(self._starts())     # 版79：同じ置き場所へ
        if str(self.dev).startswith("cuda"):     # 版99b：刈り込みの作業の分を持ち越さない（値は変わらない）
            torch.cuda.empty_cache()

    @torch.no_grad()
    def step(self, x_all, ext_all, learn=True):
        """x_all: まとめた入口 [n_ch_all]  ext_all: まとめた ext [送り元の合計 − n_ch_all − 抑制]"""
        self._within = None                  # 周回の中の発火（1歩の中だけ。歩はまたがない）
        self._pre2 = None                    # 版80：1周目の学びと一緒に作った2周目の頭の足し込み
        out = self._fast_step(x_all, ext_all, learn) if self._fast_ok(learn) else None   # 版90：1つのグラフで
        if out is None:
            gp = self._gp_ok()               # 版79：2周目からをグラフの形で
            for i in range(1 if gp else self.recur):
                out = self._pass(x_all, ext_all, learn and i == 0, i == self.recur - 1, i == 0)
            if gp:
                out = self._gp_passes(x_all, ext_all)
        # 刈り込みは、周をまわし終えてから
        #   ⚠️ 学習するのは1周目、歩を進めるのは最後の周。両方を条件にすると永久に呼ばれない
        if learn and self.rewire_every and self.t % self.rewire_every == 0:
            self.rewire()
        return out

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
    def _prior_any(self):
        """版103c：型の塊ごとの見込み。GPU で Triton があれば核1つ（_prior_fast）、無ければ _prior_core（式は同じ）"""
        if PRIOR_K and _TRITON and str(self.dev).startswith("cuda"):
            if getattr(self, "_pri_seg_id", None) is None:
                sid = torch.empty(len(self.tb), dtype=torch.int32, device=self.dev)
                for i, (lo, hi) in enumerate(self._pri_segs):
                    sid[lo:hi] = i
                assert self._pri_segs[0][0] == 0 and self._pri_segs[-1][1] == len(self.tb) and all(
                    a[1] == b[0] for a, b in zip(self._pri_segs, self._pri_segs[1:])), "塊が受け手を隙間なく覆っていない"
                self._pri_seg_id = sid
            return _prior_fast(self.tb, self._pri_seg_id, *self._pri_stack)
        return _run(_prior_core, self.tb, *self._pri_stack, self._pri_segs)

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
                self._prior_l = self._prior_any()   # 版103c
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
                                a_now1=z(2, self.A.dtype), kc1=z((), self.k_start.dtype), xt=z(self.n_ch_all))
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
            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(-1).expand(2) * (self.elig_decay ** k)
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
                   cl_drive=getattr(self, "cl_drive", None), v_carry=vc, ext_ihn=getattr(self, "ext_ihn", None),
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
                self.x_index, self._a_lo, self.tgt_by_fire, self.is_inh_r, self.area_of, self.target_recv,
                self.rate_ema, self.rate_long, self.scale_skip, self.is_hippo, F["E"]) + self._nuc_pins()
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
        return rep * self.gate, err * self.gate_err, S["perr"].clone()

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
            _p1_flush_k[(1024,)](F["E"], F["ne"], self.k_start, self.elig, self.A, F["a_now1"], self.recv_grp, F["kc1"], self.w,
                                 self.e_recv, self.e_src, self.e_inh, self.c_exc, self._amp_buf, X["send_gain"],
                                 self.mod_send, self.mod_drive_inh, self.is_inh_r, b0, m0, self.n_recv_all,
                                 HAS_AMP=True, K=K, BLOCK=BL)
            both0, mod0 = b0.sum(0), m0.sum(0)
        else:
            e = F["E"][:int(F["ne"])]
            both0, mod0 = _run(_flush_sum_core, e, self.k_start, self.elig, self.A, F["a_now1"], self.w, F["kc1"],
                               self.e_recv, self.e_src, self.e_inh, self.c_exc, self._amp_buf, X["send_gain"],
                               self.mod_send, self.mod_drive_inh, self.is_inh_r, self.n_recv_all, self.recv_grp)
        exc, ihn, exc_mod, p, pr, drive, rep = _run(
            _drive_core, both0, mod0, X["scale"], X["scale_up"], opt("exc_gain"), X["prior_l"], self.rep_i,
            X["x_all"], X["x_gain"], opt("cl_drive"), opt("v_carry"), X["fire_th"], X["th_gain"], opt("complete"),
            X["pred_boost"], F["spont1"], X["spont_p_pass"], opt("ext_ihn"), *self._nuc_in(), self.n_recv_all)
        # 的（_pass の1周目と同じ）
        x_t = torch.where(self.tgt_by_fire, (X["x_all"] * X["x_gain"] > X["fire_th"]).float(), X["x_all"])
        F["xt"].copy_(x_t)
        X["tgt"].copy_(torch.where(self.is_inh_r, self._area_mean(x_t)[self.area_of], x_t[self.x_index]))
        err_sig = X["tgt"] - p
        if on["learn"]:      # 学ぶ ＋ 2周目の頭（版80 の _learn_sum_core と同じ式）＋ 恒常性（版88）
            if str(self.dev).startswith("cuda"):
                K, BL = _SC_K, 256
                b2 = torch.zeros(K, self.n_recv_all * 2, device=self.dev)
                m2 = torch.zeros(K, self.n_recv_all, device=self.dev)
                c2 = torch.zeros(K, self.n_recv_all, device=self.dev, dtype=torch.int32)
                _p1_learn_k[(1024,)](F["E"], F["ne"], self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,
                                     self._g_buf, self._pw_buf, self.elig, self.hist_n, self.w, self.c_exc, self.k_start,
                                     self.A, S["a_now"], self.recv_grp, S["kcur1"], self._amp_buf, X["send_gain"], self.mod_send,
                                     self.mod_drive_inh, b2, m2, c2, self.n_recv_all, INH_LTD_N, INH_LTD_TH, INH_LTD_F,
                                     HAS_AMP=True, K=K, BLOCK=BL)
                cnt, bb, mb = c2.sum(0).to(torch.float32), b2.sum(0), m2.sum(0)
            else:
                e = F["E"][:int(F["ne"])]
                cnt, bb, mb = _run(_learn_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,
                                   self._g_buf, self._pw_buf, self.elig, self.hist_n, self.w, self.c_exc, self.k_start,
                                   self.A, S["a_now"], S["kcur1"], self._amp_buf, X["send_gain"], self.mod_send,
                                   self.mod_drive_inh, self.n_recv_all, self.recv_grp)
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
            _out_core, err_sig * self.fire_sgn, exc, exc_mod, ihn, self.err_i, self.inh_i, opt("v_carry_inh"), rep,
            S["lf0"], S["lf1"], S["lf2"], S["pc"], None, None, None)
        if getattr(self, "nucnet", None) is not None:      # 版95：核どうしの線（次の周・その次の周へ）
            self.nucnet.after(S, rep, err)
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

    def _nuc_in(self):
        """版95：この周に届く核どうしの興奮（受け手の体の駆動へ）・抑制（表現・誤差の行へ）。NucNet が無ければ None"""
        if getattr(self, "nucnet", None) is None:
            return None, None
        S = self._gp_state()
        return S["nuc_x"], S["nuc_ihn"]

    def _nuc_pins(self):
        nn_ = getattr(self, "nucnet", None)
        return () if nn_ is None else (nn_.w, nn_.src) + tuple(g_[5] for g_ in nn_.groups)   # 版95d

    def _fuse_ok(self):
        """版80：辺を読む回数を減らした形で回すか（測る道具が _flush・_sum を差し替えているときは今まで通り）"""
        return type(self)._flush is _G2_ORIG["_flush"] and type(self)._sum is _G2_ORIG["_sum"]

    def _kc(self, v):
        """版80：刻みの数を GPU の箱で渡す（CPU が待たない）"""
        if getattr(self, "_kc_buf", None) is None:
            self._kc_buf = torch.zeros((), dtype=self.k_start.dtype, device=self.dev)
        return self._kc_buf.fill_(v)

    @torch.no_grad()
    def _flush_sum(self, e):
        """版80：1周目の _flush(e) → _sum(e, True) と同じ（放出を先に。放出は重さを読まない）"""
        if self.stp is not None:       # 放出は _sum の1周目と同じ
            if self._amp_buf is None or len(self._amp_buf) != len(self.w):
                self._amp_buf = torch.ones(len(self.w), device=self.dev)
            elif self._amp_prev is not None:
                self._amp_buf.index_fill_(0, self._amp_prev, 1.0)     # 版54b
            self._amp_buf[e] = self.stp.release_at(e, self.t)
            self._amp_prev = e
            self._amp = self._amp_buf
        if self._tri_ok():               # 版81：足し込みは Triton の核で（印のある所だけ）
            r_, ie_, amp_, mm_ = _run(_flush_amp_core, e, self.k_start, self.elig, self.A, self.A[self.kcur], self.w,
                                      self._kc(self.kcur + 1), self.e_recv, self.e_src, self.e_inh, self.c_exc,
                                      self._amp if self.stp is not None else None, self.send_gain, self.mod_send,
                                      self.mod_drive_inh, self.is_inh_r, self.recv_grp)
            both, mod, _ = _scatter3(r_, ie_, amp_, mm_, self.n_recv_all, False)
            return both, mod
        return _run(_flush_sum_core, e, self.k_start, self.elig, self.A, self.A[self.kcur], self.w,
                    self._kc(self.kcur + 1), self.e_recv, self.e_src, self.e_inh, self.c_exc,
                    self._amp if self.stp is not None else None, self.send_gain, self.mod_send,
                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all, self.recv_grp)

    def _tri_ok(self):
        """版81：足し込みを Triton の核で回すか（GPU・Triton あり・ANAPLEX_SCATTER が 0 でない）"""
        return SCATTER and _TRITON and str(self.dev).startswith("cuda")

    def _gp_ok(self):
        """版79：2周目からをグラフの形で回すか（測る道具が中を差し替えているときは版78 と同じ周ごと）"""
        if self.recur < 2 or GRAPH not in ("1", "eager"):
            return False
        if GRAPH == "1" and not str(self.dev).startswith("cuda"):
            return False
        return all(getattr(type(self), k) is f for k, f in _G2_ORIG.items())

    def _gp_state(self):
        """版79：周をまたぐ値の決まった置き場所（1回だけ作る）"""
        S = getattr(self, "_gS", None)
        if S is not None:
            return S
        dev = self.dev
        z = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device=dev)
        nr, ne, ni = len(self.rep_i), len(self.err_i), len(self.inh_i)
        self._gW = torch.sort(torch.cat([self.within_index, self.within_inh_index])).values   # 周回の中で撃ちうる行
        S = dict(within=z(self.n_send_all), lf0=z(nr, torch.long), lf1=z(ne, torch.long), lf2=z(ni, torch.long),
                 any0=z(nr), any1=z(ne), any2=z(ni), pc=z((), torch.long), a_now=z(2, self.A.dtype),
                 kcur1=z((), self.k_start.dtype), spont=z(self.n_ch_all), lens=z(len(self._gW), torch.long),
                 csum=z(len(self._gW), torch.long), drive=z(self.n_ch_all), dinh=z(ni), pr=z(self.n_ch_all), perr=z(ne))
        if getattr(self, "nucnet", None) is not None:      # 版95：核どうしの線の周ごとの置き場所
            S.update(self.nucnet.state(self.n_ch_all, self.n_recv_all))
        self._gS, self._gX, self._graphs, self._gpins = S, {}, {}, None
        self.gp_hist = {}           # 箱の大きさごとの周の数（測る道具用）
        return S

    @torch.no_grad()
    def _gp_passes(self, x_all, ext_all):
        """版79：2周目から最後の周まで（版78 の _pass の2周目からと同じ式・同じくじの順）"""
        # 2周目の頭：1歩ぶんの辺（時間細胞と buf）を、1周目に学んだあとの重みで（版78 と同じ）
        if self._pre2 is not None:            # 版80：1周目の学びと一緒に作ってある（学びの分も足してある）
            both_base, mod_base = self._pre2
        else:
            self._flush(self._base_e)
            both_base, mod_base = self._sum(self._base_e, False)
        self._both_base, self._mod_base = both_base, mod_base
        S = self._gp_state()
        S["within"].copy_(self._within)
        for i in range(3):
            S[f"lf{i}"].copy_(self._last_fire[i])
            S[f"any{i}"].copy_(self._any[i])
        S["pc"].copy_(self._pc_t)
        S["a_now"].copy_(self.A[self.kcur])
        S["kcur1"].fill_(self.kcur + 1)
        vc = getattr(self, "v_carry", None)
        vin = dict(both_base=both_base, mod_base=mod_base, base_on=self._base_on, prior_l=self._prior_l, x_all=x_all,
                   scale=self.scale, scale_up=self.scale_up, exc_gain=getattr(self, "exc_gain", None),
                   cl_drive=getattr(self, "cl_drive", None), v_carry=vc, ext_ihn=getattr(self, "ext_ihn", None),
                   v_carry_inh=(self.v_carry_inh if vc is not None else None), th_gain=self.th_gain, tgt=self._tgt,
                   x_gain=self.x_gain, fire_th=self.fire_th, complete=(self.complete if self._complete_any else None),
                   pred_boost=self.pred_boost, spont_p_pass=self.spont_p_pass, send_gain=self.send_gain)
        X, key = self._gX, []
        AL = self.__dict__.setdefault("_galias", {})      # 版86：置き場所の変わらない値（グラフが直に読む）
        CP = self.__dict__.setdefault("_gcopy", set())    # 版86：置き場所が変わったことのある値（写す）
        for k, v in vin.items():          # 歩ごとに変わる値は決まった置き場所へ写す
            if v is None:
                key.append((k, False))
                continue
            if k in AL:
                if v is AL[k]:                # 同じ置き場所 ⇒ 写さない（グラフは再生のときの中身を読む）
                    key.append((k, True))
                    continue
                del AL[k]                     # 置き場所が変わった ⇒ これからは写す
                X.pop(k, None)
                CP.add(k)
                self._graphs.clear()
            elif torch.is_tensor(v) and k not in CP and k not in X:
                AL[k] = X[k] = v              # 初めて見た値は、まず直に読む形で
                self._graphs.clear()
                key.append((k, True))
                continue
            shp, dt = (tuple(v.shape), v.dtype) if torch.is_tensor(v) else ((), torch.float32)
            b = X.get(k)
            if b is None or tuple(b.shape) != shp or b.dtype != dt:
                b = X[k] = torch.empty(shp, dtype=dt, device=self.dev)
                self._graphs.clear()
            if torch.is_tensor(v):
                b.copy_(v)
            else:
                b.fill_(float(v))
            key.append((k, True))
        key = tuple(key)
        pins = (self.w, self.e_recv, self.e_src, self.e_inh, self.c_exc, self.k_start, self.elig, self.A, self.start,
                self.mod_send, self.mod_drive_inh, self.is_inh_r, self.rep_i, self.err_i, self.inh_i,
                self.within_index, self.within_inh_index, self._amp_buf if self.stp is not None else None) + self._nuc_pins()
        pk = tuple((t.data_ptr(), t.numel()) if t is not None else None for t in pins)
        if pk != self._gpins:              # 配列が置き換わった（読み込みなど）⇒ 撮り直す
            self._graphs.clear()
            self._gpins = pk
        lens, csum = _run(_new_core, S["within"], X["base_on"], self._gW, self.start)
        S["lens"].copy_(lens)
        S["csum"].copy_(csum)
        if self._one_ok():                 # 版83：9周を1つのグラフに（自発のくじは今と同じ順に先に引く）
            sp = S.get("spont_all")
            if sp is None or sp.shape[0] != self.recur - 1:
                sp = S["spont_all"] = torch.zeros(self.recur - 1, self.n_ch_all, device=self.dev)
                self._graphs.clear()
            for i in range(self.recur - 1):
                torch.rand(self.n_ch_all, device=self.dev, generator=self.gen, out=sp[i])
            self._gp_run1(key)
            self._pc += self.recur - 1
        for i in (range(1, self.recur) if not self._one_ok() else ()):
            tot = int(S["csum"][-1])     # 周回の中で新しく撃った送り手の辺の数（1周に1回だけ待つ）
            C = _gp_cap(tot)
            self.gp_hist[C] = self.gp_hist.get(C, 0) + 1
            torch.rand(self.n_ch_all, device=self.dev, generator=self.gen, out=S["spont"])   # 自発のくじ（版78 と同じ所）
            self._gp_run(C, key)
            self._pc += 1
        self._last_fire = [S["lf0"], S["lf1"], S["lf2"]]
        self._any = [S["any0"], S["any1"], S["any2"]]
        self._pc_t.copy_(S["pc"])
        self._drive_last, self._dinh_last = S["drive"], S["dinh"]
        self._within = S["within"]
        if getattr(self, "_learn_step", False):
            self._rate_obs()
        pr = S["pr"].clone()
        self.pred_rep = pr
        for i, a in enumerate(self.areas):
            a.pred_rep = pr[int(self.off_ch[i]):int(self.off_ch[i]) + a.n_ch]
        rep, err = self._advance_tail(ext_all)
        return rep * self.gate, err * self.gate_err, S["perr"].clone()

    def _gp_run(self, C, key):
        """版79：1周を回す。GPU では箱の大きさごとに1回だけ撮って、あとは再生"""
        if GRAPH == "eager" or not str(self.dev).startswith("cuda"):
            return self._gp_body(C, key)
        g = self._graphs.get((C, key))
        if g is None:
            S = self._gS
            snap = {k: v.clone() for k, v in S.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._gp_body(C, key)         # 慣らし（compile を済ませる）。学びの分を足すのは同じ刻みでは2回目が 0
            torch.cuda.current_stream().wait_stream(s)
            for k, v in snap.items():         # 周をまたぐ値は慣らしの前に戻す
                S[k].copy_(v)
            if getattr(self, "_gpool", None) is None:
                self._gpool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._gpool):
                self._gp_body(C, key)
            self._graphs[(C, key)] = g
            self.n_capture = getattr(self, "n_capture", 0) + 1
        g.replay()

    def _one_ok(self):
        """版83：9周を1つのグラフに（GPU は Triton の核で。CPU（eager の確かめ）は同じ式の torch で）"""
        if not ONEGRAPH:
            return False
        return _TRITON or not str(self.dev).startswith("cuda")

    def _gp_run1(self, key):
        """版83：2周目から最後の周までを1回の再生で（1回だけ撮る）"""
        if GRAPH == "eager" or not str(self.dev).startswith("cuda"):
            return self._gp_body1(key)
        g = self._graphs.get(("one", key))
        if g is None:
            S = self._gS
            snap = {k: v.clone() for k, v in S.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._gp_body1(key)            # 慣らし。学びの分を足すのは同じ刻みでは2回目が 0
            torch.cuda.current_stream().wait_stream(s)
            for k, v in snap.items():
                S[k].copy_(v)
            if getattr(self, "_gpool", None) is None:
                self._gpool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._gpool):
                self._gp_body1(key)
            self._graphs[("one", key)] = g
            self.n_capture = getattr(self, "n_capture", 0) + 1
        g.replay()

    def _gp_body1(self, key):
        """版83：2周目から最後の周まで（周ごとの中身は _gp_body と同じ式）"""
        S = self._gS
        for i in range(self.recur - 1):
            S["spont"].copy_(S["spont_all"][i])
            self._gp_body(None, key)

    def _pass_edges(self):
        """版83：周回の中で新しく撃った送り手の辺の分（学びの分を足してから足し込む）。返り値は足し込みの増え分"""
        S, X = self._gS, self._gX
        nW = len(self._gW)
        if str(self.dev).startswith("cuda"):
            K, BL = _SC_K, 256
            b2 = torch.zeros(K, self.n_recv_all * 2, device=self.dev)
            m2 = torch.zeros(K, self.n_recv_all, device=self.dev)
            amp = self._amp_buf if self.stp is not None else self.w
            _pass_edges_k[(512,)](S["csum"], S["lens"], self._gW, self.start, self.k_start, self.elig, self.A,
                                  S["a_now"], self.recv_grp, S["kcur1"], self.w, self.e_recv, self.e_src, self.e_inh, self.c_exc,
                                  amp, X["send_gain"], self.mod_send, self.mod_drive_inh, self.is_inh_r,
                                  b2, m2, nW, self.n_recv_all, HAS_AMP=self.stp is not None,
                                  LOG=max(1, int(math.ceil(math.log2(nW + 1)))) + 1, K=K, BLOCK=BL)
            return b2.sum(0), m2.sum(0)
        # CPU（確かめ用）：版82 の _gp_body と同じ式。辺の数ちょうどの箱で
        tot = int(S["csum"][-1])
        if tot == 0:
            return None
        e, valid = _run(_edges_core, S["lens"], S["csum"], self._gW, self.start, tot)
        vals = _run(_flush_vals, e, self.k_start, self.elig, self.A, S["a_now"], self.w, self.e_recv, self.recv_grp)
        self.w.index_put_((e,), vals)
        self.k_start.index_put_((e,), S["kcur1"])
        return _run(_sum_core_v, e, valid, self.e_recv, self.e_src, self.e_inh, self.w, self.c_exc,
                    self._amp_buf if self.stp is not None else None, X["send_gain"], self.mod_send,
                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)

    def _gp_body(self, C, key):
        """版79：2周目からの1周（版78 の _pass の first=False・learn=False と同じ式）。読み書きは決まった置き場所だけ"""
        S, X = self._gS, self._gX
        on = dict(key)
        opt = lambda k: X[k] if on[k] else None
        if C is None:    # 版83：辺の数は GPU の中で読む（1つのグラフの形）
            inc = self._pass_edges()
            if inc is None:
                both, exc_mod = X["both_base"], X["mod_base"]
            else:
                both, exc_mod = X["both_base"] + inc[0], X["mod_base"] + inc[1]
        elif C:          # 周回の中で新しく撃った送り手の辺：学びの分を足してから足し込む
            e, valid = _run(_edges_core, S["lens"], S["csum"], self._gW, self.start, C)
            vals = _run(_flush_vals, e, self.k_start, self.elig, self.A, S["a_now"], self.w, self.e_recv, self.recv_grp)
            self.w.index_put_((e,), vals)
            self.k_start.index_put_((e,), S["kcur1"])
            b2, m2 = _run(_sum_core_v, e, valid, self.e_recv, self.e_src, self.e_inh, self.w, self.c_exc,
                          self._amp_buf if self.stp is not None else None, X["send_gain"], self.mod_send,
                          self.mod_drive_inh, self.is_inh_r, self.n_recv_all)
            both, exc_mod = X["both_base"] + b2, X["mod_base"] + m2
        else:
            both, exc_mod = X["both_base"], X["mod_base"]
        exc, ihn, exc_mod, p, pr, drive, rep = _run(
            _drive_core, both, exc_mod, X["scale"], X["scale_up"], opt("exc_gain"), X["prior_l"], self.rep_i,
            X["x_all"], X["x_gain"], opt("cl_drive"), opt("v_carry"), X["fire_th"], X["th_gain"], opt("complete"),
            X["pred_boost"], S["spont"], X["spont_p_pass"], opt("ext_ihn"), *self._nuc_in(), self.n_recv_all)
        err_sig = X["tgt"] - p
        S["pc"].add_(1)
        (d_inh, rep, err, inh_spk, lf0, lf1, lf2, a0, a1, a2) = _run(
            _out_core, err_sig * self.fire_sgn, exc, exc_mod, ihn, self.err_i, self.inh_i, opt("v_carry_inh"), rep,
            S["lf0"], S["lf1"], S["lf2"], S["pc"], S["any0"], S["any1"], S["any2"])
        if getattr(self, "nucnet", None) is not None:      # 版95
            self.nucnet.after(S, rep, err)
        for k, v in (("lf0", lf0), ("lf1", lf1), ("lf2", lf2), ("any0", a0), ("any1", a1), ("any2", a2),
                     ("drive", drive), ("dinh", d_inh), ("pr", pr)):
            S[k].copy_(v)
        S["perr"].copy_(p[self.err_i])
        wi = S["within"]                 # 周回の中の発火は次の周で届く（版78 の w_new と maximum と同じ）
        wi.index_put_((self.within_index,), torch.maximum(wi[self.within_index], rep))
        if WITHIN_INH:
            wi.index_put_((self.within_inh_index,), torch.maximum(wi[self.within_inh_index], inh_spk))
        lens, csum = _run(_new_core, wi, X["base_on"], self._gW, self.start)   # 次の周の辺の数
        S["lens"].copy_(lens)
        S["csum"].copy_(csum)

    def _rate_obs(self):
        """細胞自身がこの歩に撃ったか（恒常性の観測。版59：歩のどこか）。版79 で _pass から出した（式は同じ）"""
        if V88:                                    # 版88：compile で（式・入れる行は下と同じ）
            re_, rl = _run(_rate_core, *self._any, self.err_i, self.rep_i, self.inh_i, self.rate_ema, self.rate_long,
                           self.n_recv_all, 1.0 / self.homeo_tau, 1.0 / LONG_TAU, not getattr(self, "scaling_on", True))
            self.rate_ema.copy_(re_)
            if not getattr(self, "scaling_on", True):
                self.rate_long.copy_(rl)
            return
        fired = torch.zeros(self.n_recv_all, device=self.dev)
        # 版89：_any ＝ [表現, 誤差, 抑制]。誤差の行に誤差、表現の行に表現（⟲ 版59〜88 は入れ違い）
        fired[self.err_i], fired[self.rep_i], fired[self.inh_i] = self._any[1], self._any[0], self._any[2]
        self.rate_ema += (1.0 / self.homeo_tau) * (fired - self.rate_ema)
        if not getattr(self, "scaling_on", True):   # 版53：長い平均は起きているあいだだけ
            self.rate_long += (1.0 / LONG_TAU) * (fired - self.rate_long)

    def _advance_tail(self, ext_all):
        """歩を進める（最後の周のあと）。版79 で _pass から出した（式は同じ）"""
        rep, err, inh_spk = self._any     # 版59：最後の周ではなく、この歩のどこかで撃ったか
        if getattr(self, "v_carry", None) is not None:   # 版72：撃たなかった体だけ残りを次の歩へ（撃ったら 0）
            self.v_carry = self.v_decay * self._drive_last * (1.0 - rep)
            self.v_carry_inh = self.v_decay_inh * self._dinh_last * (1.0 - inh_spk)
        # 版50：行（領野ごとに 誤差・表現・抑制）と同じ並びで足す
        #   ⟲ 版49 まで cat([全領野の rep, 全領野の rep, 全領野の inh])（種類ごとの並び）。
        #      先頭の領野の誤差の行のほかは、別の体の発火を履歴にしていた【測】chk_tb（10-04）
        now = torch.zeros(self.n_recv_all, device=self.dev)
        now[self.err_i] = rep[self.x_index[self.err_i]]
        now[self.rep_i] = rep[self.x_index[self.rep_i]]
        now[self.inh_i] = inh_spk
        self.tb = self.tb * self.decay + now.unsqueeze(1) * (1 - self.decay)
        self._prior_l = None      # 版54
        # 版84：入口の並べ直しを番号の表1つで（領野ごとの切り貼りと同じ並び・同じ値）
        nn = (len(rep), len(ext_all), len(inh_spk))
        if getattr(self, "_src_idx", None) is None or self._tail_n != nn \
                or [tuple(b.shape) for b in self.bufs] != [(n, d) for _, n, d in self._buf_shapes]:
            self._build_tail_idx(*nn)
        src = torch.cat([rep.float(), ext_all.float(), inh_spk.float()])[self._src_idx]
        self.ch.step(src)
        # 版84：buf は1本の配列で1回に進める（領野ごとに 列 0 ＝ 新しい入口、列 d ＝ 古い列 d−1。値を写すだけ）
        new = torch.cat([src, self._buf_flat_get()])[self._buf_idx]
        self.bufs = [new[o:o + n * d].view(n, d) for o, n, d in self._buf_shapes]
        self._buf_flat, self._buf_refs = new, tuple(self.bufs)
        self.t += 1
        self._e_cache = None
        self._base = None        # 時間細胞と buf が進んだので作り直す
        self._within = None      # 歩をまたいだら、周回の中の発火は消える
        return rep, err

    @torch.no_grad()
    def _sum(self, e, first):
        """版54：辺 e の分を受け手へ足す（版53 の _pass の中身と同じ式）。返り値は [興奮 ｜ 抑制]・調整する側の興奮"""
        if e is None or len(e) == 0:
            return (torch.zeros(self.n_recv_all * 2, device=self.dev), torch.zeros(self.n_recv_all, device=self.dev))
        # 1つの繋がりの接触の数（強さに畳んである）。興奮は受け手の領野ごと、抑制は【ヒト】皮質の 4.0
        # 版76：放出（くじ）は今までどおり先に。辺の計算は _sum_core（torch.compile）で1回に
        if self.stp is not None:
            if first:      # 放出は1周目だけ。値の配列は持ち続け、前の歩に入れた所だけ 1 に戻す（版53 は毎歩作り直し）
                if self._amp_buf is None or len(self._amp_buf) != len(self.w):
                    self._amp_buf = torch.ones(len(self.w), device=self.dev)
                elif self._amp_prev is not None:
                    self._amp_buf.index_fill_(0, self._amp_prev, 1.0)     # 版54b
                self._amp_buf[e] = self.stp.release_at(e, self.t)
                self._amp_prev = e
                self._amp = self._amp_buf
        # 脳幹：眠りで上からの戻りを落とす（版38）／ 足し込みは1回（抑制を負にして繋ぎ、あとで分ける）
        # 版66：印の送り手 → 抑制の体 は撃たせる側 ／ 版54b：選ばない辺は 0 を足す（選んで足すのと同じ値）
        return _run(_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.w, self.c_exc,
                    self._amp if self.stp is not None else None, self.send_gain, self.mod_send,
                    self.mod_drive_inh, self.is_inh_r, self.n_recv_all)

    def _pass(self, x_all, ext_all, learn, advance, first):
        e = self._fired_edges()
        if first and self._base is not None:   # 版47：送り手ごとの直近 INH_LTD_WIN 歩の発火を数える
            now = (self._base > 0).to(torch.uint8)
            self.hist_n += now.to(torch.int16) - self.hist[self.hist_i].to(torch.int16)
            self.hist[self.hist_i] = now
            self.hist_i = (self.hist_i + 1) % INH_LTD_WIN
        # 版54：1周目は撃った辺（時間細胞と buf）をそのまま。2周目は学んだあとの重みで同じ辺の分を作って残し、
        #   3〜10周目はそれを使い回す。どの周も、周回の中で新しく撃った送り手の分だけ足す（足す辺の組は版53 と同じ）
        if first:
            if len(e) and self._fuse_ok():      # 版80：学びの分を足すのと足し込みを1つに
                both, exc_mod = self._flush_sum(e)
            else:
                self._flush(e)
                both, exc_mod = self._sum(e, True)
            self._both_base = None
        else:
            if self._both_base is None:
                if getattr(self, "_pre2", None) is not None:   # 版80：1周目の学びと一緒に作ってある
                    self._both_base, self._mod_base = self._pre2
                else:
                    self._flush(self._base_e)
                    self._both_base, self._mod_base = self._sum(self._base_e, False)
            if self._new_e is not None:
                self._flush(self._new_e)
                b2, m2 = self._sum(self._new_e, False)
                both, exc_mod = self._both_base + b2, self._mod_base + m2
            else:
                both, exc_mod = self._both_base, self._mod_base
        # 版53：起きて上げた分は興奮だけに（版78：_drive_core の中）
        # リップルのとき、興奮の入力だけ底上げする（【ヒト】錐体細胞 5.2倍）
        #   抑制は外さない。抑制性の体は、増えた発火を受けて自分で強まる（人では 3.8倍）
        #   （版78：_drive_core の中）
        if self._prior_l is None:        # 版54：tb は歩の終わりにしか変わらない ⇒ 1歩に1回
            if len(self._prior_segs) == 1:      # 版55
                self._prior_l = self.prior.logit(self.tb)
            elif type(self.areas[0].prior).logit is not _PRIOR_LOGIT0:   # 測る道具が差し替えているときは今まで通り
                self._prior_l = torch.cat([pr.logit(self.tb[lo:hi]) for lo, hi, pr in self._prior_segs])
            else:                               # 版86：8つの塊を1回に
                if getattr(self, "_pri_stack", None) is None:
                    ps = [pr for _, _, pr in self._prior_segs]
                    self._pri_stack = tuple(torch.stack([getattr(p, k) for p in ps]) for k in ("W1", "b1", "W2", "b2"))
                    self._pri_segs = tuple((lo, hi) for lo, hi, _ in self._prior_segs)
                self._prior_l = self._prior_any()   # 版103c
        # 版78：自発のくじは今と同じ順（見込みのあと・学びの前）に引いて、_drive_core に渡す
        spont_u = torch.rand(self.n_ch_all, device=self.dev, generator=self.gen)
        exc, ihn, exc_mod, p, pr, drive, rep = _run(
            _drive_core, both, exc_mod, self.scale, self.scale_up, getattr(self, "exc_gain", None), self._prior_l,
            self.rep_i, x_all, self.x_gain, getattr(self, "cl_drive", None), getattr(self, "v_carry", None),
            self.fire_th, self.th_gain, self.complete if self._complete_any else None, self.pred_boost,
            spont_u, self.spont_p_pass, getattr(self, "ext_ihn", None), *self._nuc_in(), self.n_recv_all)

        # 表現の発火は学びの前に出す（版36。的に使うため）。学びはこの周の exc・ihn を変えないので値は同じ
        # 表現のニューロンも、シナプスの合計が閾値を越えたら撃つ
        #   【ヒト】L2/3 錐体細胞は、同時に活動した興奮性シナプス 134±28本 で 50% の確率で撃つ
        #   （人が毎歩受けている入力 135本＝30,000シナプス×0.45%/歩 と、ほぼ同じ）
        #   ⇒ 平均的な入力がちょうど閾値。だから疎で、入力の揺らぎに敏感になる
        #   抑制（短絡）はこの合計に掛かるので、ここで初めて表現の発火に効く
        # 調整する側（上からの戻り・指令の写し）は予測 p には入るが、表現を撃たせない（版39）
        # 版67：CL からの駆動（誤差の的には入れない）／ 版72：持ち越した膜の残り ／ 版41：覚醒で閾値が下がる
        # 版52 B：入力が無いところは予測が足しになる ／ 版58：自発（以上 版78 で _drive_core の中）
        self._drive_last = drive

        # 的：誤差と表現は自分の入口、抑制はその領野の入口の平均（ループなしで一度に）
        # 誤差の的：下からの入力でこの細胞が撃つか（0か1）。予測はシグモイドなので的も 0か1 の目盛り
        # 的は「下からの入力（本数 × 重さ）がこの細胞を撃たせるか」。学んでいる重み（抑制）は入れない
        #   ⟲ 版31 で表現と同じ抑制を入れたら、学びが抑制を育てて的そのものを消し、皮質が全部止まった
        #      （的が 0 ⇒ e ＝ −p ⇒ 抑制が強まる ⇒ 的がさらに立たない、の輪）。的は学ぶ量に依存させない
        if first:      # 版54b：入力 x は1歩のあいだ同じ ⇒ 的は1周目に出して使い回す
            x_t = torch.where(self.tgt_by_fire, (x_all * self.x_gain > self.fire_th).float(), x_all)
            self.last_xt = x_t                         # 測る道具用
            tgt = x_t[self.x_index]
            self._tgt = torch.where(self.is_inh_r, self._area_mean(x_t)[self.area_of], tgt)
        tgt = self._tgt
        # 表現の的は下からの入力（版39。版36 の「自分の発火」を外した）
        #   ⟲ 自分の発火を的にすると、学んだシナプスがそのまま駆動に入るので「いつも撃つ」が誤差0の解になり、
        #      全領野が撃ちっぱなしになった（版36〜38、表現 50〜64%）
        #   ⚠️ 何を的にするかは人からは決まらない【C】。区画（予測する網と駆動する網）を作る段で見直す
        err_sig = tgt - p

        if learn and len(e):
            # eta_gain：驚きで上がる（青斑核。版42 で戻した）
            # 学びの門（版40）：その歩の学びは跡にためる。重みを変えるのは脳幹の第三の信号の分だけ
            #   【ヒト】皮質の古典的な対ではドーパミンを入れて初めて強まる（前の作り 生-217）
            # 版54：全部の辺の跡を減らす代わりに、刻みを1つ進める（跡は 減り^(−k) を掛けて持つ）
            if self.kcur >= LAZY_EPOCH:
                self._new_epoch()
            k = self.kcur + 1
            # 版45：抑制は学ばない（人の強さ W_INH のまま。【み】10-01）
            # 版47：錐体 → 抑制の体 も予測誤差では学ばない（下の弱まりだけ。⟲ 版45 は的 ＝ 入口の平均で学び上下した：形A）
            # 版47：大きい繋がりだけ、送り手が 10歩に5回以上撃った歩に弱まる（【ヒト】Molnár 2016）
            # 版76：辺の計算は _learn_core（torch.compile）で1回に（式は同じ。減り^(−k) と eta は値で渡す）
            if getattr(self, "_g_buf", None) is None:     # 版77：値は GPU の箱へ fill_ で入れる（CPU が待たない。値は同じ）
                self._g_buf = torch.zeros((), dtype=self.w.dtype, device=self.dev)
                self._pw_buf = torch.zeros((), dtype=self.w.dtype, device=self.dev)
            eg = self.eta * self.eta_gain
            g_ = self._g_buf.copy_(eg) if torch.is_tensor(eg) else self._g_buf.fill_(eg)
            pw_ = self._pw_buf.fill_(self.elig_decay ** (-k))
            # 版80：2周目の頭（学びの分を足す・1歩ぶんの辺の足し込み）も一緒に（A を先に進める。A は跡を読まない）
            fuse = self.recur > 1 and e is getattr(self, "_base_e", None) and self._fuse_ok()
            if not fuse:
                cnt = _run(_learn_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig, g_, pw_,
                           self.elig, self.hist_n, self.w, self.c_exc, self.n_recv_all)
            #   third ＝ 1 が続けば、跡を使わない今までの学びとほぼ同じ量（1 − 減り を掛ける）
            # 版54：全部の辺へ足す代わりに A を1つ進める（辺は読む前にまとめて足す）
            self.A[k] = self.A[k - 1] + (self.third * (1.0 - self.elig_decay)).double().reshape(-1).expand(2) * (self.elig_decay ** k)
            if fuse and self._tri_ok():   # 版81：足し込み・辺の数は Triton の核で
                r_, ie_, amp_, mm_ = _run(_learn_amp_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r,
                                          err_sig, g_, pw_, self.elig, self.hist_n, self.w, self.c_exc, self.k_start,
                                          self.A, self.A[k], self._kc(k + 1),
                                          self._amp if self.stp is not None else None, self.send_gain,
                                          self.mod_send, self.mod_drive_inh, self.recv_grp)
                b2, m2, cnt = _scatter3(r_, ie_, amp_, mm_, self.n_recv_all, True)
                self._pre2 = (b2, m2)
            elif fuse:
                cnt, b2, m2 = _run(_learn_sum_core, e, self.e_recv, self.e_src, self.e_inh, self.is_inh_r, err_sig,
                                   g_, pw_, self.elig, self.hist_n, self.w, self.c_exc, self.k_start, self.A,
                                   self.A[k], self._kc(k + 1), self._amp if self.stp is not None else None,
                                   self.send_gain, self.mod_send, self.mod_drive_inh, self.n_recv_all, self.recv_grp)
                self._pre2 = (b2, m2)
            self.kcur = k
            # 興奮も抑制も 0 以上（版36。【ヒト】繋がりの符号は送り手の型で決まる。前は興奮が負になれた）
            a_ = 1.0 / self.homeo_tau
            if V88:                                # 版88：compile で（式は同じ。中身を書き換える）
                r_, d_ = _run(_homeo_core, cnt, exc, self.recv_ema, self.drive, a_)
                self.recv_ema.copy_(r_)
                self.drive.copy_(d_)
            else:
                self.recv_ema += a_ * (cnt - self.recv_ema)
                self.drive += a_ * (exc.abs() - self.drive)
            # シナプススケーリング：受け手ごとに入辺を一括で乗算して、自分の発火率を目標に保つ
            #   目標は【ヒト】0.45%/歩。撃ちすぎなら下げ、足りなければ上げる
            #   【ヒト】iPS 由来の皮質細胞：活動を止めると興奮性シナプス全体が強まって活動が戻る
            #   ⚠️ 起きているあいだは止める。【ヒト】起きているあいだは強くなる方へ、
            #      眠っているあいだに戻す（TMS の応答が起きている時間で増え、眠ると元に戻る）
            #   ⚠️ 下げるのは皮質だけ。海馬は眠っているあいだ強くなるという報告がある【げっ歯類】
            if V88 and getattr(self, "scaling_on", True):     # 版88：compile で（式は下と同じ）
                self.scale = _run(_scale_down_core, self.rate_ema, self.target_recv, self.is_hippo, self.scale_skip,
                                  self.scale, a_, bool(getattr(self, "sleep_mode", False)))
            elif V88:
                self.scale_up = _run(_scale_up_core, self.rate_long, self.target_recv, self.scale_skip, self.scale_up)
            elif getattr(self, "scaling_on", True):
                # 版53：眠りは下げるだけ（⟲ 版52 まで両向き。上げる向きは人の根拠なしで【み】保留 10-02）
                f = (1 - 0.01 * (self.rate_ema / self.target_recv - 1)).clamp(0.5, 1.0) ** a_
                if getattr(self, "sleep_mode", False):
                    f = torch.where(self.is_hippo, torch.ones_like(f), f)
                # ⟲ 下限 0.05・上限 20 は torch 版だけにあった（numpy 原本は1歩の変化を 0.5〜1.5 に収めるだけ）
                #    版23 で7領野が下限に当たって止まり、発火率が人の 10〜50倍のままだった
                f = torch.where(self.scale_skip, torch.ones_like(f), f)   # 版39b：人の値が無い受け手
                self.scale = self.scale * f
            else:
                # 版53：起きているあいだは上げるだけ（【ラット】Hengen 2016。【み】10-08）。静かな細胞の興奮を強める
                up = torch.exp(K_WAKE_UP * (1.0 - self.rate_long / self.target_recv).clamp(min=0.0))
                self.scale_up = self.scale_up * torch.where(self.scale_skip, torch.ones_like(up), up)

        # 出力もループなしで（誤差・表現・抑制を印で切り出す）
        # 版48：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ。焼いた見込み・p は使わない
        #   ⟲ 版47 まで p ＞ 0.5【C】。撃つと見込みが上がって撃ち続けた（形B）
        #   【ヒト】速い抑制細胞は 1歩の刻みでは履歴にほぼ左右されない（Lee・Dalley 2023）
        if getattr(self, "v_carry", None) is not None and self.v_carry_inh is None:   # 版72：抑制の体の足し合わせ
            self.v_carry_inh = torch.zeros(len(self.inh_i), device=self.dev)
        # 版59：不応期。撃った周から REFRACT_PASSES 周たつまで撃たない（歩をまたいでも数える）
        if first:
            self._learn_step = learn
            if getattr(self, "_last_fire", None) is None:
                self._pc = 0
                self._pc_t = torch.zeros((), dtype=torch.long, device=self.dev)   # 版78：周の番号の箱（GPU）
                self._last_fire = [torch.full((n,), -10 ** 6, dtype=torch.long, device=self.dev)
                                   for n in (len(rep), len(self.err_i), len(self.inh_i))]
        self._pc += 1
        if getattr(self, "_pc_t", None) is None:          # 版77 の state から続けるとき
            self._pc_t = torch.full((), self._pc, dtype=torch.long, device=self.dev)
        else:
            self._pc_t.add_(1)
        # 版48：抑制の体は届いた興奮（上からの戻りを除く）を閾値と比べて撃つ ／ 版59：不応期・この歩のどこかで撃ったか
        # 版78：以上を _out_core（torch.compile）で1つに
        an = (None, None, None) if first else tuple(self._any)
        (d_inh, rep, err, inh_spk, lf0, lf1, lf2, a0, a1, a2) = _run(
            _out_core, err_sig * self.fire_sgn, exc, exc_mod, ihn, self.err_i, self.inh_i,
            self.v_carry_inh if getattr(self, "v_carry", None) is not None else None,
            rep, *self._last_fire, self._pc_t, *an)
        self._dinh_last = d_inh
        if getattr(self, "nucnet", None) is not None:      # 版95
            self.nucnet.after(self._gp_state(), rep, err)
        self._last_fire = [lf0, lf1, lf2]
        self._any = [a0, a1, a2]
        if advance and getattr(self, "_learn_step", False):   # 細胞自身がこの歩に撃ったか（恒常性の観測。版59：歩のどこか）
            self._rate_obs()                                  # 版79：中身は _rate_obs（2周目からをグラフで回すときも同じ）
        self.pred_rep = pr
        for i, a in enumerate(self.areas):        # 領野ごとに切り出して持たせる
            a.pred_rep = pr[int(self.off_ch[i]):int(self.off_ch[i]) + a.n_ch]

        # 周回の中の発火は、次の周で他の体に届く（1周 ＝ シナプス1本ぶんの遅れ 1.1ms）
        w_new = torch.zeros(self.n_send_all, device=self.dev)
        w_new[self.within_index] = rep
        if WITHIN_INH:                            # 版63：抑制の体の発火も次の周で届く
            w_new[self.within_inh_index] = inh_spk
        self._within = w_new if self._within is None else torch.maximum(self._within, w_new)
        self._e_cache = None                      # 送り手が変わったので作り直す

        if advance:
            rep, err = self._advance_tail(ext_all)     # 版79：中身は _advance_tail（式は同じ）
        return rep * self.gate, err * self.gate_err, p[self.err_i]


# 版86：焼いた網の元の関数（測る道具が差し替えているときは塊ごとに呼ぶ）
from torch_area import Prior as _Prior
_PRIOR_LOGIT0 = _Prior.logit
# 版79：差し替えられていない元の関数（測る道具が差し替えているときは、版78 と同じ周ごとで回す）
_G2_ORIG = {k: getattr(Group2, k) for k in ("_pass", "_sum", "_flush", "_fired_edges", "_expand")}


class NucNet:
    """版95：核どうしの線（周 1.1ms ごとに届く。核どうし 1周・線条体から 2周）【み】10-16「核は周回ごと。核どうしの遅れは周の数」
    送り手 ＝ 網の表現・誤差の体（[表現 n_ch | 誤差 n_ch] の並び）。興奮は受け手の表現の駆動へ（x と同じ入り方・誤差の的には入れない）、
    抑制は受け手の表現・誤差の行の抑制へ（÷(1＋抑制)）。興奮には周ごとの NMDA の尾（版93 の式）
    学び（決め A【み】10-15 23:51）：送り手の強い入力（直近 NUC_A_WIN 歩のうち NUC_A_N 歩以上撃った）のあとに強まる × 門。
      受け手の発火は条件にしない。⚠️ 量 NUC_A_ETA・強い入力の数・上限 NUC_A_WMAX はノブ【C】（弱まる側は人の値なし）
    版95d：線を（符号・遅れ・受け手1体の本数）の組に並べ、受け手1体ぶんずつ並べて合計（index_add の渋滞も累積和もなし）"""
    NUC_A_WIN, NUC_A_N, NUC_A_ETA, NUC_A_WMAX = 10, 8, 1e-3, 4.0

    def __init__(self, lines, n_ch, n_recv, rep_i, err_i, tail, dev):
        self.n_ch, self.n_recv, self.dev, self.tail = n_ch, n_recv, dev, tail
        grp = {}
        for L in lines:
            grp.setdefault((L["sign"], L["delay"], L["k"]), []).append(L)
        S_, W_, self.groups, self._tch = [], [], [], []
        o = 0
        for (sg, dl, k), Ls in sorted(grp.items()):
            src = torch.cat([L["src"] for L in Ls]).long()
            w = torch.cat([L["w"] for L in Ls]).float()
            tch = torch.cat([L["tgt_ch"] for L in Ls]).long()        # 受け手1体ぶん（k 本）ごとに1つ
            if sg > 0:
                tgt = tch
            else:                                                    # 抑制は表現の行と誤差の行の両方へ
                tgt = torch.cat([rep_i[tch], err_i[tch]]).long()
            self.groups.append((sg, dl, k, o, o + len(src), tgt))
            self._tch.append(tch)
            S_.append(src); W_.append(w); o += len(src)
        self.src = torch.cat(S_)
        self.w = torch.cat(W_)
        self.w0 = self.w.clone()
        self.hist = torch.zeros(self.NUC_A_WIN, n_ch * 2, dtype=torch.uint8, device=dev)
        self.cnt = torch.zeros(n_ch * 2, dtype=torch.int16, device=dev)
        self.hi = 0

    def flat(self):
        """測る道具用：線ごとの (送り手, 受け手の体（n_ch の並び）, 符号, 遅れ)"""
        D, Sg, Dl = [], [], []
        for (sg, dl, k, a, b, tgt), tch in zip(self.groups, self._tch):
            D.append(tch.repeat_interleave(k))
            Sg.append(torch.full((b - a,), sg, dtype=torch.long, device=self.dev))
            Dl.append(torch.full((b - a,), dl, dtype=torch.long, device=self.dev))
        return self.src, torch.cat(D), torch.cat(Sg), torch.cat(Dl)

    def state(self, n_ch, n_recv):
        z = lambda n: torch.zeros(n, device=self.dev)
        return {"nuc_x": z(n_ch), "nuc_x2": z(n_ch), "nuc_ihn": z(n_recv), "nuc_ihn2": z(n_recv),
                "nuc_s1": z(n_ch), "nuc_s2": z(n_ch)}

    def after(self, S, rep, err):
        """この周の発火 → 次の周（遅れ1）・その次の周（遅れ2）に届く分。置き場所は S（グラフの中で書き換える）"""
        v = torch.cat([rep, err])
        xs = {1: torch.zeros(self.n_ch, device=self.dev), 2: torch.zeros(self.n_ch, device=self.dev)}
        hs = {1: torch.zeros(self.n_recv, device=self.dev), 2: torch.zeros(self.n_recv, device=self.dev)}
        for sg, dl, k, a, b, tgt in self.groups:
            s = (self.w[a:b] * v[self.src[a:b]]).view(-1, k).sum(1)    # 受け手1体ぶんの合計
            if sg > 0:
                xs[dl].index_add_(0, tgt, s)
            else:
                hs[dl].index_add_(0, tgt, torch.cat([s, s]))
        raw = S["nuc_x2"] + xs[1]                                # 次の周に届く興奮
        T = self.tail
        s1 = S["nuc_s1"] * T["a1"] + raw * (1 - T["a1"])
        s2 = S["nuc_s2"] * T["a2"] + raw * (1 - T["a2"])
        S["nuc_s1"].copy_(s1); S["nuc_s2"].copy_(s2)
        S["nuc_x"].copy_((1 - T["f"]) * raw + T["f"] * (T["m1"] * s1 + T["m2"] * s2))
        S["nuc_x2"].copy_(xs[2])
        S["nuc_ihn"].copy_(S["nuc_ihn2"] + hs[1])
        S["nuc_ihn2"].copy_(hs[2])

    @torch.no_grad()
    def learn(self, any_src, d):
        """1歩に1回。any_src：送り手の体がこの歩のどこかで撃ったか（[表現 | 誤差]）／ d：学びの門（0〜1）"""
        now = (any_src > 0).to(torch.uint8)
        self.cnt += now.to(torch.int16) - self.hist[self.hi].to(torch.int16)
        self.hist[self.hi] = now
        self.hi = (self.hi + 1) % self.NUC_A_WIN
        g_on = (2.0 * torch.as_tensor(d, dtype=self.w.dtype, device=self.dev) - 1.0).clamp(0.0, 1.0)
        strong = (self.cnt[self.src] >= self.NUC_A_N).float()
        self.w.mul_(torch.exp(self.NUC_A_ETA * g_on * strong))
        torch.minimum(self.w, self.w0 * self.NUC_A_WMAX, out=self.w)

    def state_dict(self):
        return {"w": self.w, "cnt": self.cnt, "hist": self.hist, "hi": self.hi}

    def load_state_dict(self, d):
        if d.get("w") is None or d["w"].shape != self.w.shape:
            print("⚠️ 核どうしの線の重さは形が違うので読まない（初めの重さから）", flush=True)
            return
        self.w.copy_(d["w"]); self.cnt.copy_(d["cnt"]); self.hist.copy_(d["hist"]); self.hi = int(d["hi"])


def _segsum_nuc(v, ends):
    """版95b：受け手の順に並んだ値を受け手ごとに合計（累積和の差。float64 で足す）"""
    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=v.device), v.double().cumsum(0)])
    starts = torch.cat([torch.zeros(1, dtype=ends.dtype, device=ends.device), ends[:-1]])
    return (cs[ends] - cs[starts]).to(v.dtype)
