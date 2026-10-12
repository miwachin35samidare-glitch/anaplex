"""どの版から撃ちっぱなしになったかを切り分ける（arkhe の ~/anaplex で回す。中身は何も書き換えない）

apply_* が残した控え（*.v35 ／ *.v36 ／ *.v37）をそのまま読み込んで、0からの網で作り物の声の喃語を3回だけ回し、
表現の発火の割合と1歩の時間を出す。state.pt は読まない・書かない

  python3 measure_versions.py 35 on     版35 の網（torch_group2.py.v35 ＋ torch_brain.py.v36）。起きていても下方調整あり（09-26 夜まで）
  python3 measure_versions.py 35 off    同じ網で、起きているあいだ下方調整を止める（apply_babble_awake の分）
  python3 measure_versions.py 36 off    版36（興奮 0以上 ＋ 表現の的＝自分の発火）。torch_group2.py.v37 ＋ torch_brain.py.v36
  python3 measure_versions.py 37 off    版37（上から送るのを発火に）。torch_group2.py.v37 ＋ torch_brain.py.v37
  python3 measure_versions.py 38 off    版38（いまのファイル。驚きは切る）
  版36 の2つを分ける（torch_group2.py.v37 の文字を、読み込むときだけ書き換える。ファイルは触らない）
  python3 measure_versions.py 36w off   興奮 0以上 だけ（表現の的は版35 と同じ「下からの入力」）
  python3 measure_versions.py 36t off   表現の的＝自分の発火 だけ（興奮の重みは版35 と同じく負になれる）
  python3 measure_versions.py 38m off   版38（驚きは切る）＋ 調整する側の経路は表現を撃たせない（a、09-27【み選】「一旦aで」）
        調整する側にするのは4本だけ：連合→聴覚 ／ 高次→運動 ／ 高次→連合（上からの戻り）と 指令の写し（運動→聴覚）
        この4本は予測 p には今までどおり入る。表現の駆動（合計 ＞ 閾値）からだけ外す。ほかの経路は今のまま
  軸1（今すでに人とずれているもの）を切り分ける（どれも 38m の上に重ねる）
  python3 measure_versions.py 38r off   傾きの細胞の発火を 30%/歩【C】→【ヒト】皮質の発火率 0.3Hz（0.32%/歩、timing.py の CORTEX_HZ）
  python3 measure_versions.py 38s off   自分自身の送り手（自分の少し前の発火・自分の時間細胞・傾きの細胞）から自分への辺を効かせない
  python3 measure_versions.py 38rs off  両方
  まとめて： for a in 38r 38s 38rs; do python3 measure_versions.py $a off; done
  python3 measure_versions.py 38w off   38r ＋ 表現の的を下からの入力に戻す（B を外す。36w の形。興奮 0以上 は残す）

まとめて回すなら（約20分）
  for a in "35 on" "35 off" "36 off" "37 off" "38 off"; do python3 measure_versions.py $a; done
"""
import sys
import time
import numpy as np
import torch

ver, scale = sys.argv[1], sys.argv[2]
FILES = {"35": ("torch_group2.py.v35", "torch_brain.py.v36"),
         "36": ("torch_group2.py.v37", "torch_brain.py.v36"),
         "37": ("torch_group2.py.v37", "torch_brain.py.v37"),
         "38": ("torch_group2.py", "torch_brain.py"),
         "36w": ("torch_group2.py.v37", "torch_brain.py.v36"),
         "36t": ("torch_group2.py.v37", "torch_brain.py.v36"),
         "38m": ("torch_group2.py", "torch_brain.py"),
         "38r": ("torch_group2.py", "torch_brain.py"),
         "38s": ("torch_group2.py", "torch_brain.py"),
         "38rs": ("torch_group2.py", "torch_brain.py"),
         "38w": ("torch_group2.py", "torch_brain.py")}
OWN = "        tgt = tgt.clone()\n        tgt[self.is_rep] = rep\n"
CLAMP = "            self.w[e] = self.w[e].clamp(min=0.0)\n"
CLAMP_OLD = "            self.w[e] = torch.where(self.e_inh[e], self.w[e].clamp(min=0.0), self.w[e])\n"


def load(name, path, edit=None):
    src = open(path, encoding="utf-8").read()
    for old, new in (edit or []):
        if src.count(old) != 1:
            sys.exit(f"⚠️ {path} に書き換えの印が {src.count(old)} 回。止める：{old[:40]!r}")
        src = src.replace(old, new)
    m = type(sys)(name)
    m.__file__ = path
    sys.modules[name] = m
    exec(compile(src, path, "exec"), m.__dict__)
    return m


fg, fb = FILES[ver]
# 38m：調整する側の経路（mod_send の送り手）は、表現の駆動に足さない。予測 p には今までどおり入る
MOD = [("        ihn = torch.zeros(self.n_recv_all, device=self.dev)\n",
        "        ihn = torch.zeros(self.n_recv_all, device=self.dev)\n"
        "        exc_mod = torch.zeros(self.n_recv_all, device=self.dev)\n"),
       ("            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n",
        "            exc, ihn = both[:self.n_recv_all], both[self.n_recv_all:]\n"
        "            mm = self.mod_send[self.e_src[e]] & ~self.e_inh[e]\n"
        "            exc_mod = exc_mod.index_add(0, r[mm], amp[mm])\n"),
       ("        exc = exc * self.scale; ihn = ihn * self.scale\n",
        "        exc = exc * self.scale; ihn = ihn * self.scale\n"
        "        exc_mod = exc_mod * self.scale\n"),
       ("            exc = exc * self.exc_gain\n",
        "            exc = exc * self.exc_gain\n"
        "            exc_mod = exc_mod * self.exc_gain\n"),
       ("        drive = (exc[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])\n",
        "        drive = ((exc - exc_mod)[self.is_rep] + x_all * self.x_gain) / (1.0 + ihn[self.is_rep])\n")]
RAMP = [("< y * 0.3\n", "< y * RAMP_P\n"),
        ("< (1 - y) * 0.3\n", "< (1 - y) * RAMP_P\n"),
        ("class TimeCellsGroup:", "RAMP_P = 0.3 * 10.67 / 1000.0   # 【ヒト】皮質の発火率 0.3Hz を1歩に\n\n\nclass TimeCellsGroup:")]
SELF = [("            amp = amp * self.send_gain[self.e_src[e]]",
         "            if getattr(self, 'no_self', False):\n"
         "                amp = amp * (self.send_row[self.e_src[e]] != self.recv_row[self.e_recv[e]]).float()\n"
         "            amp = amp * self.send_gain[self.e_src[e]]")]
EDIT = {"36w": [(OWN, "")], "36t": [(CLAMP, CLAMP_OLD)], "38m": MOD,
        "38r": MOD + RAMP, "38s": MOD + SELF, "38rs": MOD + RAMP + SELF,
        "38w": MOD + RAMP + [(OWN, "")]}.get(ver)
load("torch_group2", fg, EDIT)
TB = load("torch_brain", fb)
import torch_area as T
import babble as B                     # 作り物の声と run だけ使う（babble の Brain は上で読んだもの）

DEV = "cuda"
prior = T.Prior.load("prior_v1.npz", dev=DEV)
b = TB.Brain(prior, n_partner=9100, hippo_scale=20.0, n_ear=B.N_EAR, dev=DEV)
g = b.g
if hasattr(b, "bs"):
    import brainstem as BS
    BS.SURPRISE_K = 0.0
    b.bs.set_state(BS.WAKE)
g.scaling_on = (scale == "on")


def send_rows(area, first, n):
    """その領野の送り元の行 first〜first+n が持つ送り手の番号（時間細胞の分と、短い遅れの分）"""
    names = list(b.areas.keys())
    i = names.index(area)
    buf_off = sum(b.areas[nm].n_src * b.areas[nm].D for nm in names[:i])
    a = b.areas[area]
    per = a.ch.n_cell + a.ch.n_ramp
    rows = torch.arange(first, first + n, device=DEV)
    tc = int(g.off_send[i]) + rows.unsqueeze(1) * per + torch.arange(per, device=DEV)
    bf = int(g.off_send[-1]) + buf_off + rows.unsqueeze(1) * a.D + torch.arange(a.D, device=DEV)
    return torch.cat([tc.reshape(-1), bf.reshape(-1)])


if ver.startswith("38") and ver != "38":
    n = b.n
    g.mod_send = torch.zeros(g.n_send_all, dtype=torch.bool, device=DEV)
    for area, first, cnt, what in (
            ("聴覚", n["聴覚"], n["運動"], "指令の写し"),
            ("聴覚", n["聴覚"] + n["運動"], n["連合"], "連合から"),
            ("運動", n["運動"], n["高次"], "高次から"),
            ("連合", n["連合"], n["高次"], "高次から")):
        idx = send_rows(area, first, cnt)
        assert not bool(g.is_inh_all[idx].any()), "調整する側に抑制の送り手が混ざった"
        g.mod_send[idx] = True
        print(f"  調整する側：{area} の {what}（送り元 {cnt} 行・送り手 {len(idx):,}）", flush=True)
    e_mod = g.mod_send[g.e_src]
    print(f"  調整する側の辺 {int(e_mod.sum()):,} 本（全体の {float(e_mod.float().mean()):.1%}）", flush=True)
if "s" in ver[2:]:
    # 送り手 → その送り手の元の送り元の行（全体の通し番号）／ 受け手 → 自分の送り元の行
    names = list(b.areas.keys())
    send_row = torch.empty(g.n_send_all, dtype=torch.long, device=DEV)
    recv_row = torch.empty(g.n_recv_all, dtype=torch.long, device=DEV)
    buf_off = 0
    for i, nm in enumerate(names):
        a = b.areas[nm]
        per = a.ch.n_cell + a.ch.n_ramp
        rows = int(g.off_src[i]) + torch.arange(a.n_src, device=DEV)
        lo = int(g.off_send[i])
        send_row[lo:lo + a.n_src * per] = rows.repeat_interleave(per)
        lo = int(g.off_send[-1]) + buf_off
        send_row[lo:lo + a.n_src * a.D] = rows.repeat_interleave(a.D)
        buf_off += a.n_src * a.D
        r0 = int(g.off_recv[i])
        ch = int(g.off_src[i]) + torch.arange(a.n_ch, device=DEV)
        recv_row[r0:r0 + a.n_ch] = ch
        recv_row[r0 + a.n_ch:r0 + 2 * a.n_ch] = ch
        recv_row[r0 + 2 * a.n_ch:int(g.off_recv[i + 1])] = int(g.off_src[i]) + a.n_ch + a.n_ext + torch.arange(a.n_inh, device=DEV)
    g.send_row, g.recv_row, g.no_self = send_row, recv_row, True
    self_e = send_row[g.e_src] == recv_row[g.e_recv]
    print(f"  自分への辺 {int(self_e.sum()):,} 本（全体の {float(self_e.float().mean()):.2%}）。領野ごと（表現の受け手の1体あたり）：", flush=True)
    for i, nm in enumerate(names):
        a = b.areas[nm]
        r0 = int(g.off_recv[i]) + a.n_ch
        m = (g.e_recv >= r0) & (g.e_recv < r0 + a.n_ch)
        print(f"    {nm} {float(self_e[m].sum()) / a.n_ch:.1f}本", flush=True)
print(f"版{ver}（{fg} ＋ {fb}）  起きているあいだの下方調整 {scale}  体 {b.n_cells:,}  辺 {len(g.w):,}", flush=True)

NAMES = TB.NAMES
ms, rates, per_area, orig = [], [], [], b.step


def step(*a, **k):
    torch.cuda.synchronize(); t0 = time.time()
    out = orig(*a, **k)
    torch.cuda.synchronize(); ms.append((time.time() - t0) * 1000)
    return out


b.step = step
orig_pass = g._pass


def pas(*a, **k):
    out = orig_pass(*a, **k)
    if a[3]:                                   # 歩を進める周（最後の周）の発火
        r = out[0]
        rates.append(float(r.mean()))
        per_area.append(torch.stack([r[int(g.off_ch[i]):int(g.off_ch[i + 1])].mean() for i in range(len(NAMES))]))
    return out


g._pass = pas
inv = B.make_inventory(np.random.default_rng(0))
rng = np.random.default_rng(0)
for u in range(3):
    n0 = len(ms)
    mel, cmd, _ = B.utter(inv, rng, b.n["運動"])
    B.run(b, cmd, B.encode(mel), True, True)
    a = torch.stack(per_area[n0:]).mean(0)
    print(f"  喃語{u+1}  {len(ms)-n0:4d}歩  1歩 {np.mean(ms[n0:]):6.1f} ms  表現の発火 {np.mean(rates[n0:]):6.1%}   "
          + "  ".join(f"{nm} {float(v):.1%}" for nm, v in zip(NAMES, a)), flush=True)
print(f"  最初の20歩の表現の発火 {[round(x * 100, 1) for x in rates[:20]]} %", flush=True)
print(f"  scale の平均 {float(g.scale.mean()):.3f}", flush=True)
