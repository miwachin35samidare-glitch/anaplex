"""版66：聴覚の抑制の体と表現の率を、喃語の段ごとに分け、届いた指令の写しを「運動の発火そのもの・写しの時間細胞・写しの傾きの細胞」に分ける
（arkhe の ~/anaplex で。版66。0から・学びながら・返し手あり・保存しない・網は書き換えない。暮らしと試しは measure_check2 と同じ）
  python3 measure_v66a.py [喃語の回数=50]
  （コンテナで道具を確かめるとき：python3 measure_v66a.py 3 --test）
【み】10-12 7時46分「いいよ」
【測】measure_v65a（版66）：聴覚の抑制の体 音あり 10.9% ／ 無音 12.2%。「無音」に 蹴り・声の前・返し手や録音の合間がまざり、分けられていない
【コード】写しの送り手 ＝ 運動の発火そのもの（遅れの列）＋ 写しの時間細胞（運動が撃ったあと 341ms まで）＋ 写しの傾きの細胞（入力によらず乱数で撃つ）
【マーモセット】Eliades & Wang 2003：自分の発声の数百 ms 前から聴覚が抑えられる（自分の声のときだけ）⇒ 段ごとに見る
段（自分の喃語は口の指令と蹴りで分ける）
  自分・蹴り（蹴りが入っている歩）／ 自分・声の前（指令あり・耳の和 0）／ 自分・声（指令あり・耳に音）／ 自分・ほか（指令も蹴りも無い：黙る回・声のあと）／ 返し手 ／ 録音
並べるもの（段ごと）
  歩の割合 ／ 聴覚の抑制の体の率 ／ 聴覚の表現の率 ／ 聴覚の抑制の体1体に届いた写し：そのもの・時間細胞・傾き（本数）／ 運動の表現の率（写しの元）
⚠️ 届いた本数は最後の周に届いた分（measure_v65a と同じ）
"""
import sys, time, torch, numpy as np
import babble as B, brainstem as BS, torch_area as T
import torch_group2 as G
from torch_brain import Brain, NAMES

TEST = "--test" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if B.VERSION not in (66,):
    sys.exit(f"⚠️ babble.py が版66 ではない（版 {B.VERSION}）")
N = int(args[0]) if args else 50
if TEST:
    B.DEV = "cpu"
DELAY = 2
N_PROBE = 3
torch.manual_seed(0)
prior = T.Prior.load("prior_v1.npz", dev=B.DEV)
b = Brain(prior, n_partner=(200 if TEST else 9100), hippo_scale=20.0, n_ear=B.N_EAR, dev=B.DEV)
b.bs.set_state(BS.WAKE)
_, say, rp = B.voices(b, TEST)
g = b.g
dev = g.dev
ia, im = NAMES.index("聴覚"), NAMES.index("運動")
aA = g.areas[ia]
nA, nM = b.n["聴覚"], b.n["運動"]

# ── 写しの送り手を3つに分ける（版39・66 と同じ組：聴覚の送り元の行 nA〜nA+nM）──
CK = ("そのもの", "時間細胞", "傾き")
ckind = torch.full((g.n_send_all,), -1, dtype=torch.long, device=dev)
buf_base = g.n_send_all - sum(a.n_src * a.D for a in g.areas)
buf_off = sum(a.n_src * a.D for a in g.areas[:ia])
per = aA.ch.n_cell + aA.ch.n_ramp
rows = torch.arange(nA, nA + nM, device=dev)
tc = int(g.off_send[ia]) + rows.unsqueeze(1) * per + torch.arange(per, device=dev)
ckind[tc[:, :aA.ch.n_cell].reshape(-1)] = 1
ckind[tc[:, aA.ch.n_cell:].reshape(-1)] = 2
bf = buf_base + buf_off + rows.unsqueeze(1) * aA.D + torch.arange(aA.D, device=dev)
ckind[bf.reshape(-1)] = 0
assert bool((g.mod_drive_inh == (ckind >= 0)).all()), "写しの組が版66 の印と合わない"
lo_inh = int(g.off_recv[ia]) + 2 * nA
is_ainh = torch.zeros(g.n_recv_all, dtype=torch.bool, device=dev)
is_ainh[lo_inh:lo_inh + aA.n_inh] = True
inh_rows = slice(nA + aA.n_ext, aA.n_src)
repA = slice(int(g.off_ch[ia]), int(g.off_ch[ia]) + nA)
repM = slice(int(g.off_ch[im]), int(g.off_ch[im]) + nM)

st = {"tag": "他", "ear": 0.0, "cmd": False, "kick": False}
_step = b.step


def step_wrap(ear, cmd, *a, **k):
    st["ear"] = float(ear.sum())
    st["cmd"] = cmd is not None and bool((cmd > 0).any())
    st["kick"] = b.kick_left > 0
    return _step(ear, cmd, *a, **k)


b.step = step_wrap
PH = ("自分・蹴り", "自分・声の前", "自分・声", "自分・ほか", "返し手", "録音")
rows_out = []     # (段, 抑制の体の率, 表現の率, 運動の率, 写し3つの本数)
orig = G.Group2._pass


def hooked(self, x_all, ext_all, learn, advance, first):
    e = self._fired_edges() if (advance and st["tag"] != "他") else None
    out = orig(self, x_all, ext_all, learn, advance, first)
    if e is None:
        return out
    if st["tag"] == "自分":
        ph = "自分・蹴り" if st["kick"] else ("自分・声" if st["ear"] > 0 else "自分・声の前") if st["cmd"] else "自分・ほか"
    else:
        ph = st["tag"]
    cn = np.zeros(3)
    if len(e):
        r = self.e_recv[e].long(); ck = ckind[self.e_src[e].long()]
        m = is_ainh[r] & (ck >= 0) & ~self.e_inh[e]
        if bool(m.any()):
            cn = torch.bincount(ck[m], minlength=3).float().cpu().numpy() / aA.n_inh
    rows_out.append((ph, float(self.bufs[ia][inh_rows, 0].mean()), float((out[0][repA] > 0).float().mean()),
                     float((out[0][repM] > 0).float().mean()), cn))
    return out


G.Group2._pass = hooked


def show(rs, head):
    if not rs:
        return
    print(f"── {head}  歩 {len(rs)}", flush=True)
    print("   段            歩（割合）     聴覚の抑制の体  聴覚の表現   運動の表現   届いた写し（本／抑制の体1体）そのもの・時間細胞・傾き", flush=True)
    for ph in PH:
        sel = [r for r in rs if r[0] == ph]
        if not sel:
            continue
        k = len(sel)
        a = np.array([[r[1], r[2], r[3]] for r in sel]).mean(0)
        c = np.stack([r[4] for r in sel]).mean(0)
        print(f"   {ph:8s}  {k:6d}（{k / len(rs):5.1%}）   {a[0]:7.2%}        {a[1]:7.2%}     {a[2]:7.2%}      "
              f"{c[0]:6.2f} ・ {c[1]:6.2f} ・ {c[2]:6.2f}", flush=True)


rng = np.random.default_rng(0); rng_r = np.random.default_rng(1000); rng_p = np.random.default_rng(2000)
past = []
blk = 0
t0 = time.time()
for i in range(N):
    st["tag"] = "自分"
    out = say(rng, True)
    ear, cmd = np.asarray(out[0]), out[1]
    if ear.sum() > 0:
        past.append((cmd, ear))
    if rp is not None:
        st["tag"] = "返し手"; B.reply(b, rp, rng_r)
    st["tag"] = "他"
    if (i + 1) % 10 == 0 or (TEST and i + 1 == N):
        for _ in range(N_PROBE):
            if len(past) > DELAY:
                c0, e0 = past[int(rng_p.integers(0, len(past) - DELAY))]
                st["tag"] = "録音"; B.run(b, c0, e0, False, True)
        st["tag"] = "他"
        show(rows_out[blk:], f"{i+1}回目まで（{(time.time()-t0)/60:.1f}分）この10回")
        blk = len(rows_out)
print("", flush=True)
show(rows_out, "全部")
print("読み：【マーモセット】の形なら、抑制の体は 自分・蹴り／声の前／声 で高く、返し手・録音 で低い", flush=True)
print("     写しの傾きの細胞は入力によらず乱数で撃つ（話しているかに関係ない分）。録音は口の指令を入れない（写しは運動の自発・横だけ）", flush=True)
