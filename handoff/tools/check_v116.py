"""版116 の確かめ（容器・CPU）：python3 check_v116.py [歩=40]
  1 ANAPLEX_V116=0 で版115b と 毎歩の発火・終わりの重さが同じ（版115b の木 /home/claude/w/v115b で同じ道具を回して比べる）
  2 オン：遅れの体（皮質20）は撃たせる側・傾きの体は調整の側のまま
  3 オン：遅れの体 → 表現 の辺の初めの重さ ＝ 人の幅 × 0.35 ÷ 接触（0.1〜6mV × 0.35 の中）・誤差の跡の印（rr_send）
  4 オン：抑制の相手 1,456本・送り手は同じ領野の抑制の行の遅れの体
  5 オン：回って有限・遅れの体 → 表現 の辺の重さが学びで動く（初めと終わりが違う）
"""
import os, sys, subprocess, json, numpy as np
NS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 40
if "--run" in sys.argv:
    import torch, babble as B, torch_area as T, torch_brain as TB
    from torch_brain import Brain, NAMES
    B.DEV = "cpu"; torch.manual_seed(0)
    out = sys.argv[sys.argv.index("--run") + 1]
    b = Brain(T.Prior.load("prior_v1.npz", dev="cpu"), n_partner=200, hippo_scale=20.0, n_ear=B.N_EAR, dev="cpu")
    import life as L
    b.bs.set_state(__import__("brainstem").WAKE)
    life = B.life_of(b, True, np.random.default_rng(0), np.random.default_rng(1))
    g = b.g
    info = {}
    if getattr(TB, "V116", False):
        idx = b._tc_idx.long()
        w0 = g.w[idx].clone()
        ce = g.c_exc[g.e_recv[idx].long()]
        info["n_tc"] = int(idx.numel())
        info["w_conn_min"] = float((w0 * ce).min()); info["w_conn_max"] = float((w0 * ce).max())
        info["w_conn_mean"] = float((w0 * ce).mean())
        info["rr_send_all"] = bool(g.rr_send[g.e_src[idx].long()].all())
        info["drive_delay"] = bool((~g.mod_send[g.e_src[idx].long()]).all())
        # 傾きの体は調整の側のまま
        ok_r = True
        for nm in TB.CX_TC:
            a, i = b.areas[nm], NAMES.index(nm); per = a.ch.n_cell + a.ch.n_ramp; lo = int(g.off_send[i])
            r_ = (lo + torch.arange(a.n_ch + a.n_ext).unsqueeze(1) * per + a.ch.n_cell + torch.arange(a.ch.n_ramp)).reshape(-1)
            ok_r &= bool(g.mod_send[r_].all())
            ih_lo = lo + (a.n_ch + a.n_ext) * per
            o_ = int(b.off_ch[i]); pos = b._tc_rowmap[g.rep_i[o_:o_ + a.n_ch].long()].long()
            s_ = b._tc_inh_idx[pos]
            ok_r &= bool(((s_ >= ih_lo) & (s_ < lo + a.n_src * per)).all()) and bool((((s_ - lo) % per) < a.ch.n_cell).all())
        info["ramp_mod_and_inh_src"] = ok_r
        info["n_inh_per_row"] = int(b._tc_inh_idx.shape[1])
    spk = []
    for t in range(NS):
        life.step(learn=True)
        spk.append(g._any[0].clone().numpy())
    if getattr(TB, "V116", False):
        idx = b._tc_idx.long()
        info["w_moved"] = int((g.w[idx] != w0).sum())
        info["finite"] = bool(torch.isfinite(g.w).all())
    np.savez(out, spk=np.array(spk), w=g.w.numpy())
    json.dump(info, open(out + ".json", "w"))
    sys.exit(0)

def run(tree, env, out):
    e = dict(os.environ); e.update(env)
    import shutil
    me = os.path.join(tree, "_check_v116_run.py")      # 木の中に置いて回す（スクリプトの置き場所の .py が先に読まれるため）
    shutil.copy(os.path.abspath(__file__), me)
    r = subprocess.run([sys.executable, me, str(NS), "--run", out], cwd=tree, env=e, capture_output=True, text=True)
    os.remove(me)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:]); sys.exit("✗ 回らなかった")
    return np.load(out + ".npz"), json.load(open(out + ".json"))

ok = True
def ck(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("✓ " if cond else "✗ ") + name + extra, flush=True)

a, _ = run("/home/claude/w/v115b", {}, "/tmp/c116_v115b")
b, _ = run(os.getcwd(), {"ANAPLEX_V116": "0"}, "/tmp/c116_off")
ck("1 ANAPLEX_V116=0 は版115b と発火が毎歩同じ", np.array_equal(a["spk"], b["spk"]))
ck("1 ANAPLEX_V116=0 は版115b と終わりの重さが同じ", np.array_equal(a["w"], b["w"]))
c, i = run(os.getcwd(), {}, "/tmp/c116_on")
print("  （参考）", i, flush=True)
ck("2 遅れの体 → 表現 は撃たせる側", i["drive_delay"])
ck("2・4 傾きの体は調整の側・抑制の相手は同じ領野の抑制の行の遅れの体", i["ramp_mod_and_inh_src"])
lo, hi = 0.1 / 1.12 * 14.3 / (134 / 3.3), 6.0 / 1.12 * 14.3 / (134 / 3.3)
ck("3 初めの重さ（繋がり1本）が 人の幅 × 0.35 の中", lo * 0.999 <= i["w_conn_min"] and i["w_conn_max"] <= hi * 1.001, f"：{i['w_conn_min']:.4f}〜{i['w_conn_max']:.4f}（{lo:.4f}〜{hi:.4f}）・平均 {i['w_conn_mean']:.3f}")
ck("3 誤差の跡の印（rr_send）が全部に付く", i["rr_send_all"])
ck("4 抑制の相手 1,456本", i["n_inh_per_row"] == 1456)
ck("5 回って有限", i["finite"])
ck("5 遅れの体 → 表現 の重さが学びで動く", i["w_moved"] > 0, f"：動いた辺 {i['w_moved']:,} ／ {i['n_tc']:,}")
print("全部 ✓" if ok else "✗ あり", flush=True)
