"""版45 の切り分け 2（arkhe）：言えなかった発話で、運動の領野の抑制の体はなぜ撃ったか
  python3 measure_v45b.py [喃語の回数=100]

measure_v45 100（10-01 夜）で分かったこと
  閾値は動いていない（14.31〜14.75）
  言えなかった歩は 抑制の本数が 83〜596本（言えた10回の平均は 15〜74本）、駆動の上が 29〜47（≒ 指令 28.6 だけ。横の興奮がほぼ 0）
  落ちた回は続けて起きる（30〜33・38〜39・75〜77・93〜94）。measure_v43d とは落ちた回の場所が違う
⇒ この道具で分けること
  抑制の体が撃つのは、焼いた見込み（自分の撃ち方の履歴）からか、シナプス（どこから来た興奮か）からか
  いつから多いか（前の返事のあいだ ／ 蹴り ／ 間 ／ 指令）

測るもの（運動の領野。最初の周の値。覚えは残さない。measure_v45 と同じ乱数）
  抑制の体   撃った割合 ／ 見込みの項 logit(履歴) ／ シナプスの項 exc÷(1＋ihn)
             exc の出どころ：運動の表現（横）／ 高次から ／ 前頭前野から ／ 傾きの細胞
             撃つのは 見込みの項 ＋ シナプスの項 ＞ 0（p ＞ 0.5 と同じ）
  表現の体   撃った割合（最初の周）
  指令の体（指令の歩だけ）指令 ／ 横（運動の表現）／ 傾き ／ ihn ／ 撃った割合（歩の終わり）
区間  返事（その発話の前の返事のあいだ）／ 蹴り ／ 間（蹴りのあと、指令の前）／ 指令 ／ 後（指令のあと）
言えた発話と言えなかった発話（指令の体が半分も撃たなかった）を分けて平均を出し、言えなかった発話は1回ずつ出す
⚠️ GPU の足し込みの順で、落ちる回の場所は回すたびに変わる。比べるのは同じ回し方の中の 言えた／言えなかった
"""
import sys
import time
import torch

PH = ("返事", "蹴り", "間", "指令", "後")
SRC = ("表現", "高次", "前頭前野", "傾き")          # 興奮の出どころ（0〜3）。4 抑制 ／ 5 抑制の傾き ／ 6 ほか
# 区間ごとに足すもの
K_INH_F, K_PRI, K_SYN, K_E0, K_IHN, K_REP, K_N = 0, 1, 2, 3, 7, 8, 9
NK = 10
# 指令の歩だけ足すもの
C_CMD, C_LAT, C_RAMP, C_TOP, C_IHN, C_CNT, C_HIT, C_N = range(8)
NC = 8


class Probe:
    """Group2 の _pass に掛けて、運動の領野の最初の周を分ける

    ext_split：運動の領野の ext の並び [(数, 出どころ番号)]（torch_brain では 高次 → 前頭前野）"""

    def __init__(self, g, areas, i_motor, ext_split):
        self.g = g
        dev = g.dev
        self.dev = dev
        a = areas[i_motor]
        self.n_ch, self.n_inh = a.n_ch, a.n_inh
        self.lo_r = int(g.off_recv[i_motor]); self.hi_r = int(g.off_recv[i_motor + 1])
        self.lo_c = int(g.off_ch[i_motor])
        n_rm = self.hi_r - self.lo_r
        self.n_rm = n_rm
        # 送り手の出どころ（全体の番号 → 0〜6）
        rowcat = torch.full((a.n_src,), 6, dtype=torch.long, device=dev)
        rowcat[:a.n_ch] = 0
        k = a.n_ch
        for n, c in ext_split:
            rowcat[k:k + n] = c
            k += n
        assert k == a.n_ch + a.n_ext, "ext の並びが領野の ext の数と合わない"
        rowcat[k:] = 4
        cat = torch.full((g.n_send_all,), 6, dtype=torch.long, device=dev)
        per = a.ch.n_cell + a.ch.n_ramp
        ch_lo = int(g.off_send[i_motor])
        rows = torch.arange(a.n_src, device=dev)
        for kk in range(per):
            c = rowcat.clone()
            if kk >= a.ch.n_cell:                    # 傾きの細胞
                c = torch.where(rowcat == 0, torch.full_like(c, 3), c)
                c = torch.where(rowcat == 4, torch.full_like(c, 5), c)
            cat[ch_lo + rows * per + kk] = c
        buf_base = int(g.off_send[-1])
        buf_off = sum(x.n_src * x.D for x in areas[:i_motor])
        for d in range(a.D):
            cat[buf_base + buf_off + rows * a.D + d] = rowcat
        self.cat = cat
        self.cmd = None
        self.acc = torch.zeros(NK, device=dev)
        self.cacc = torch.zeros(NC, device=dev)
        self.last = None            # 最後に測った値（試しで確かめる用）
        self.orig = type(g)._pass

    def install(self):
        probe = self
        orig = self.orig

        def hooked(gs, x_all, ext_all, learn, advance, first):
            if not first or gs is not probe.g:
                return orig(gs, x_all, ext_all, learn, advance, first)
            e = gs._fired_edges()
            r = gs.e_recv[e].long()
            e = e[(r >= probe.lo_r) & (r < probe.hi_r)]   # 運動の領野に届く辺だけ
            w_e = gs.w[e].clone()                    # 学びの前の重み（_pass の中で学ぶので先に取る）
            sc = gs.scale[probe.lo_r:probe.hi_r].clone()
            out = orig(gs, x_all, ext_all, learn, advance, first)
            probe._measure(gs, e, w_e, sc, x_all, out)
            return out

        type(self.g)._pass = hooked

    @torch.no_grad()
    def _measure(self, gs, e, w_e, sc, x_all, out):
        n_rm, dev = self.n_rm, self.dev
        sums = torch.zeros(n_rm * 7, device=dev)
        if len(e):
            e2, w2 = e, w_e                          # 運動の領野に届く辺だけ（hooked で絞ってある）
            r2 = gs.e_recv[e2].long() - self.lo_r
            se = gs.e_src[e2].long()
            inh = gs.e_inh[e2]
            amp = torch.where(inh, 4.0, gs.c_exc[r2 + self.lo_r]) * w2
            if gs.stp is not None:
                amp = amp * gs._amp[e2]
            amp = amp * gs.send_gain[se]
            sums.index_add_(0, r2 * 7 + self.cat[se], amp)
        sums = sums.view(n_rm, 7) * sc.unsqueeze(1)
        eg = getattr(gs, "exc_gain", None)              # リップルのときだけ（起きているあいだは None）
        if eg is not None:
            eg = eg[self.lo_r:self.hi_r].unsqueeze(1) if torch.is_tensor(eg) and eg.dim() else eg
            sums[:, :4] = sums[:, :4] * eg
            sums[:, 6] = sums[:, 6] * (eg.squeeze(1) if torch.is_tensor(eg) and eg.dim() else eg)
        exc = sums[:, :4].sum(1) + sums[:, 6]
        ihn = sums[:, 4] + sums[:, 5]
        nc = self.n_ch
        ii = slice(2 * nc, n_rm)
        pri = gs.prior.logit(gs.tb[self.lo_r + 2 * nc:self.hi_r])
        syn = exc[ii] / (1.0 + ihn[ii])
        fire = (pri + syn > 0).float()
        rep1 = out[0][self.lo_c:self.lo_c + nc]
        v = torch.zeros(NK, device=dev)
        v[K_INH_F] = fire.mean(); v[K_PRI] = pri.mean(); v[K_SYN] = syn.mean()
        v[K_E0:K_E0 + 4] = sums[ii, :4].mean(0)
        v[K_IHN] = ihn[ii].mean(); v[K_REP] = rep1.mean(); v[K_N] = 1.0
        self.acc += v
        c = torch.zeros(NC, device=dev)
        if self.cmd is not None:
            on = torch.nonzero(self.cmd > 0).squeeze(1)
            rr = nc + on
            c[C_CMD] = (x_all[self.lo_c + on] * gs.x_gain[self.lo_c + on]).mean()
            c[C_LAT] = sums[rr, 0].mean(); c[C_RAMP] = sums[rr, 3].mean()
            c[C_TOP] = (sums[rr, 1] + sums[rr, 2]).mean(); c[C_IHN] = ihn[rr].mean()
            cnt = torch.zeros(n_rm, device=dev)
            if len(e):
                ihr = r2[inh]
                cnt.index_add_(0, ihr, torch.ones(len(ihr), device=dev))
            c[C_CNT] = cnt[rr].mean(); c[C_N] = 1.0
            self.cacc += c
        self.last = dict(exc=exc, ihn=ihn, pri=pri, syn=syn, fire=fire, sums=sums)


def fmt_ph(s):
    """区間ごとの値 [NK] → 文字列（抑制の体の率と、見込み・シナプスの項）"""
    n = max(float(s[K_N]), 1.0)
    return f"{float(s[K_INH_F])/n:4.2f}（{float(s[K_PRI])/n:+5.2f} {float(s[K_SYN])/n:+5.2f}）"


def fmt_src(s):
    n = max(float(s[K_N]), 1.0)
    return " ".join(f"{SRC[k]} {float(s[K_E0+k])/n:5.2f}" for k in range(4)) + f"  ihn {float(s[K_IHN])/n:5.2f}"


def fmt_cmd(c):
    n = max(float(c[C_N]), 1.0)
    return (f"指令 {float(c[C_CMD])/n:5.1f} 横 {float(c[C_LAT])/n:6.1f} 傾き {float(c[C_RAMP])/n:4.1f} "
            f"上から（撃たせない側）{float(c[C_TOP])/n:5.1f} ／ 抑制 {float(c[C_CNT])/n:5.1f}本 ihn {float(c[C_IHN])/n:5.2f} "
            f"／ 撃った {float(c[C_HIT])/n:4.2f}")


def main():
    import numpy as np
    import babble as B, brainstem as BS, torch_area as T
    from torch_brain import Brain, NAMES
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    torch.manual_seed(0)
    prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
    b = Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
    b.bs.set_state(BS.WAKE)
    _, say, rp = B.voices(b, False)
    im = NAMES.index("運動")
    pr = Probe(b.g, [b.areas[nm] for nm in NAMES], im,
               [(b.n["高次"], 1), (b.n["前頭前野"], 2)])        # 運動の ext ＝ 高次 → 前頭前野（torch_brain.n_ext）
    pr.install()
    dev = pr.dev
    st = {"seg": "say", "seen": False}
    ph_acc = {p: torch.zeros(NK, device=dev) for p in PH}
    ostep = b.step

    def step(ear, cmd, *a, **k):
        if st["seg"] == "reply":
            ph = "返事"
        elif float(cmd.sum()) > 0:
            ph = "指令"; st["seen"] = True
        elif b.kick_left > 0:
            ph = "蹴り"
        else:
            ph = "後" if st["seen"] else "間"
        pr.cmd = cmd if ph == "指令" else None
        pr.acc.zero_()
        r = ostep(ear, cmd, *a, **k)
        ph_acc[ph] += pr.acc
        if pr.cmd is not None:
            pr.cacc[C_HIT] += (b.motor_rep * cmd).sum() / cmd.sum()
        pr.cmd = None
        return r

    b.step = step
    KEYS = ("言えた", "言えなかった")
    groups = {k: {p: torch.zeros(NK, device=dev) for p in PH} for k in KEYS}   # 発話ごとの区間の平均を足す
    gn = {k: {p: 0 for p in PH} for k in KEYS}                                 # その区間があった発話の数
    gcmd = {k: torch.zeros(NC, device=dev) for k in KEYS}
    gu = {k: 0 for k in KEYS}
    rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000)
    ok = tried = 0; t0 = time.time()
    prev_reply = torch.zeros(NK, device=dev)
    print("回   言えた")
    for i in range(N):
        for p in PH:
            ph_acc[p].zero_()
        ph_acc["返事"] += prev_reply
        pr.cacc.zero_()
        st["seg"], st["seen"] = "say", False
        ear, cmd, n, want, got = say(rng, True)
        aro = float(b.bs.arousal)
        if want:
            tried += 1; ok += int(want == got)
            nC = max(float(pr.cacc[C_N]), 1.0)
            bad = float(pr.cacc[C_HIT]) / nC < 0.5
            key = "言えなかった" if bad else "言えた"
            for p in PH:
                k_ = float(ph_acc[p][K_N])
                if k_:
                    v = ph_acc[p] / k_
                    v[K_N] = 1.0
                    groups[key][p] += v; gn[key][p] += 1
            c_ = pr.cacc / nC
            c_[C_N] = 1.0
            gcmd[key] += c_; gu[key] += 1
            if bad:
                print(f"   ✗ {i+1:3d}回目 {want} → {got}  覚醒 {aro:.2f}", flush=True)
                print("       抑制の体が撃った割合（見込みの項 シナプスの項）  " +
                      "  ".join(f"{p} {fmt_ph(ph_acc[p])}" for p in PH), flush=True)
                print(f"       指令の歩：抑制の体に来た興奮  {fmt_src(ph_acc['指令'])}", flush=True)
                print(f"       指令の歩：指令した体  {fmt_cmd(pr.cacc)}", flush=True)
        prev_reply = torch.zeros(NK, device=dev)
        st["seg"] = "reply"
        ph_acc["返事"].zero_()
        B.reply(b, rp, rng_r)
        prev_reply = ph_acc["返事"].clone()
        if (i + 1) % 10 == 0:
            print(f"{i+1:4d}  {ok:2d}/{tried:2d}   （{(time.time()-t0)/60:.1f}分）", flush=True)
            ok = tried = 0
    print("\n平均（1発話ずつ平均してから、発話の数で割る）")
    for key in KEYS:
        if not gu[key]:
            print(f"  {key}  0回"); continue
        print(f"  {key}  {gu[key]}回")
        print("    抑制の体が撃った割合（見込みの項 シナプスの項）  " +
              "  ".join(f"{p} {fmt_ph(groups[key][p])}" for p in PH if gn[key][p]))
        print(f"    指令の歩：抑制の体に来た興奮  {fmt_src(groups[key]['指令'])}")
        print(f"    指令の歩：指令した体  {fmt_cmd(gcmd[key])}")
    print(f"VRAM 予約の山 {torch.cuda.max_memory_reserved()/1e9:.2f}GB" if torch.cuda.is_available() else "")


if __name__ == "__main__":
    main()
