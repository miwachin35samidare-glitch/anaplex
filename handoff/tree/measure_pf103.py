"""版103d：平行線維（足す）が 1歩 7ms かかる訳を分ける（arkhe の ~/anaplex で。保存しない・網は変えない）
  python3 measure_pf103.py [歩=50]
  _BUSyn.drive（平行線維 → プルキンエ、44.4万本）の中を、同じ式の順に区切って GPU を待って測る
    1 送り手を引く（v[src] > 0）2 撃った線を数える（nonzero）3 放出（STP.release_at）4 受け手へ足す（index_add）
  撃った線の数・release_at の中（前半・くじ・後半）も出す
"""
import sys, time, collections, numpy as np, torch
import babble as B, brainstem as BS, torch_area as T, torch_brain as TB
from torch_brain import Brain
from torch_group2 import _run

if B.VERSION not in (103,):
    sys.exit(f"⚠️ 版 {B.VERSION}：この道具は版103 の網のため")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NS = int(args[0]) if args else 50
dev = B.DEV
torch.manual_seed(0)
b = Brain(T.Prior.load("prior_v1.npz", dev=dev), n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=dev)
b.bs.set_state(BS.WAKE)
life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
for _ in range(60):
    life.step()
acc, n_fired = collections.defaultdict(float), []
s = b.bu[TB.PF_KEY]
st = s.stp


def tick(name, t0):
    torch.cuda.synchronize()
    acc[name] += time.time() - t0
    return time.time()


def rel(e, t):
    t0 = time.time()
    r, dep, st_, fc, Pr0, P, N, cn = _run(T._rel_pre, e, st.row, st.last, st.deplete, st.state, st.fac, st.dec_d, st.dec_f, st.tbl, st.N, t)
    t0 = tick("  3a 放出の前半（状態を進める）", t0)
    st.last.index_fill_(0, e, t)
    t0 = tick("  3b 最後に触った歩", t0)
    cnt = T.binom(cn, P, st.gen)
    t0 = tick("  3c くじ", t0)
    out, st2, fc2 = _run(T._rel_post, cnt, r, dep, st_, fc, Pr0, P, N, st.tbl)
    st.state[e] = st2
    st.fac[e] = fc2
    tick("  3d 放出の後半・書き戻し", t0)
    return out


def drive(v):
    torch.cuda.synchronize(); t0 = time.time(); ta = t0
    s.pre = v
    out = torch.zeros(s.n_tgt, device=v.device)
    m = v[s.src] > 0
    t0 = tick("1 送り手を引く", t0)
    e = torch.nonzero(m).squeeze(1)
    t0 = tick("2 撃った線を数える（nonzero）", t0)
    n_fired.append(len(e))
    if len(e):
        a = rel(e, s.t)
        t0 = time.time()
        out.index_add_(0, s.tgt[e], s.w[e] * a)
        t0 = tick("4 受け手へ足す", t0)
    s.t += 1
    acc["ぜんぶ"] += time.time() - ta
    return out


s.drive = drive
for _ in range(NS):
    life.step()
print(f"平行線維 → プルキンエ：線 {len(s.w):,}本 ／ 撃った線 1歩 平均 {np.mean(n_fired):,.0f}本（最大 {max(n_fired):,}）", flush=True)
for k, v in acc.items():
    print(f"  {k:<28s} {v / NS * 1000:7.2f}ms", flush=True)
print(f"（参考：顆粒の体 {b.n['顆粒']:,}・プルキンエ {b.n['プルキンエ']}）", flush=True)
