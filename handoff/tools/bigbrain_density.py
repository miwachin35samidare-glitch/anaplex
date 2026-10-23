"""人（BigBrain、65歳男性1体、細胞染色）の HCP-MMP1 区画ごとの細胞の詰まり方から、区画の組の上下を読む（arkhe で。網は使わない）
  1 落とす（1回だけ。大きい）：
      curl -L https://fz-juelich.sciebo.de/s/HoKucnh8zFpSbKF/download -o BBW_BigData.zip && unzip -q BBW_BigData.zip
      curl -L -O https://raw.githubusercontent.com/DiedrichsenLab/fs_LR_32/master/Glasser_2016.32k.L.label.gii
      curl -L -O https://raw.githubusercontent.com/DiedrichsenLab/fs_LR_32/master/Glasser_2016.32k.R.label.gii
      pip install --user nibabel   （無ければ）
  2 python3 bigbrain_density.py BBW_BigData          （展開した場所。Glasser の2つは今いる所）
【み】10-23 22:00「作ってるのはマーモセットじゃないから人からとれるところまで横着しないでね」⇒ 7（上下）を人の組織で
読み方（Beul・Hilgetag 2017〜：細胞の密度が高い区画 → 低い区画 の線は上り。マカク・ネコ・マウスで確かめられた形）
  ⚠️ BigBrain の値は染色の濃さ（細胞体の体積の割合の目安）で、細胞の数の密度そのものではない ⇒ 係数は移さず、向き（差の符号）だけを見る
  先に確かめ：上下が分かっている組（V1→V2→V4、A1→ベルト）で「上りの送り手の方が濃い」になるか。ならなければ上下の根拠にしない
  ⚠️ 一次運動（4）は髄鞘・層の厚さで外れることが人で知られている（Burt 2018・Wagstyl 2020）⇒ 4 との組は特に慎重に
出すもの：区画ごとの 全体・浅い層（層1〜3）・深い層（層5〜6）の濃さ ／ 組ごとの差 ／ Hist-G1（人の細胞構築の主軸）
区画（HCP-MMP1）：補足 6mp ／ 前補足 6ma ／ SCEF ／ 一次運動 4 ／ 運動前 6d・6a・6v・6r ／ 前頭前野 8BL・8Ad・9-46d・46・p9-46v ／ 44 ／ 確かめ V1・V2・V4・A1・MBelt・LBelt・PBelt
"""
import sys, glob, os, re
import numpy as np, warnings
warnings.filterwarnings("ignore")
import nibabel as nib

root = sys.argv[1] if len(sys.argv) > 1 else "BBW_BigData"
files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True) if os.path.isfile(f)]


def pick(*pats, must=("fs_LR",)):
    c = [f for f in files if all(re.search(p, os.path.basename(f), re.I) for p in pats)
         and all(m.lower() in f.lower() for m in must)]
    c32 = [f for f in c if "32k" in f]
    return sorted(c32 or c)


print(f"展開した所：{root}（ファイル {len(files)}）")
need = {}
for h in ("L", "R"):
    need[("prof", h)] = pick("profiles", f"hemi-{h}") or pick("profiles")       # 版2：左右で1つのファイル（L の後に R）
    for k in range(1, 7):
        need[(f"l{k}", h)] = pick(f"layer{k}", "thick", f"hemi-{h}")
    need[("g1", h)] = pick("Hist-G1", f"hemi-{h}") or pick("Hist_G1", f"hemi-{h}") or pick("G1", "hist", f"hemi-{h}")
miss = [k for k, v in need.items() if not v and k[0] == "prof"]   # 版2：層の厚さは無くてもよい（全体の濃さだけ使う）
HAS_TH = all(need[(f"l{k}", h)] for k in range(1, 7) for h in ("L", "R"))
for k, v in need.items():
    print(f"  {k}: {os.path.basename(v[0]) if v else '― 見つからない'}{'' if len(v) < 2 else f'（ほか {len(v) - 1}）'}")
if miss:
    print("⚠️ 要るファイルが見つからない。fs_LR の名前の一覧を出して止める：")
    for f in files:
        if "fs_lr" in f.lower():
            print("   " + f)
    sys.exit(1)


def load_vec(path):
    if path.endswith(".gii"):
        g = nib.load(path)
        return np.concatenate([d.data.ravel() for d in g.darrays]).astype(float)
    if path.endswith((".txt", ".csv")):
        return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None).ravel().astype(float)
    if path.endswith(".npy"):
        return np.load(path).ravel().astype(float)
    try:
        return np.asarray(nib.load(path).get_fdata()).ravel().astype(float)
    except Exception:
        return np.loadtxt(path).ravel().astype(float)


def labels(h):
    g = nib.load(f"Glasser_2016.32k.{h}.label.gii")
    lab = g.darrays[0].data.astype(int)
    names = {k: v for k, v in g.labeltable.get_labels_as_dict().items()}
    return lab, names


def area_key(name):
    s = re.sub(r"^[LR]_", "", name)
    return re.sub(r"_ROI$", "", s)


res = {}
for h in ("L", "R"):
    lab, names = labels(h)
    nv = len(lab)
    if "_prof" not in globals():
        _prof = load_vec(need[("prof", h)][0])
    prof = _prof
    nvL = len(labels("L")[0]); nvR = len(labels("R")[0])
    if not need[("prof", h)][0].count(f"hemi-{h}"):                  # 左右で1つ ⇒ 50 段 × (L ＋ R)
        if prof.size % (nvL + nvR):
            sys.exit(f"⚠️ profiles の長さ {prof.size} が頂点 {nvL}＋{nvR} で割れない")
        nd = prof.size // (nvL + nvR)
        PP = prof.reshape(nd, nvL + nvR)
        P = PP[:, :nvL] if h == "L" else PP[:, nvL:]
        print(f"  profiles は左右で1つ：{nd} 段 × {nvL}＋{nvR}（⚠️ 並びは L の後に R と見た）")
    else:
        if prof.size % nv:
            sys.exit(f"⚠️ profiles の長さ {prof.size} が頂点 {nv} で割れない")
        nd = prof.size // nv
        P = prof.reshape(nd, nv)
    if nd != 50:
        print(f"  ⚠️ 深さの段が {nd}（50 のはず）")
    if HAS_TH:
        th = np.stack([load_vec(need[(f"l{k}", h)][0]) for k in range(1, 7)])        # [6, nv]
        tot = th.sum(0)
        tot[tot <= 0] = np.nan
        b_sup = (th[0] + th[1] + th[2]) / tot                                         # 層3の下 ＝ 浅い層の終わり（深さの割合）
        b_inf = (th[0] + th[1] + th[2] + th[3]) / tot                                 # 層4の下 ＝ 深い層の始まり
    else:                                                                             # 版2：層の厚さが無い ⇒ 浅・深は出さない
        b_sup = np.full(nv, -1.0); b_inf = np.full(nv, 2.0)
    # 深さの向き：表の1段目が軟膜か白質か ⇒ 層1（細胞が少ない）は薄い ⇒ 両端の濃さの低い方を軟膜と見る【C】
    top_first = np.nanmean(P[: nd // 10]) < np.nanmean(P[-nd // 10:])
    if not top_first:
        P = P[::-1]
    d = (np.arange(nd) + 0.5) / nd
    sup = np.where(d[:, None] < b_sup[None, :], P, np.nan)
    inf = np.where(d[:, None] > b_inf[None, :], P, np.nan)
    g1 = load_vec(need[("g1", h)][0]) if need[("g1", h)] else None
    if g1 is not None and g1.size != nv:
        g1 = None
    print(f"\n{h}：頂点 {nv}・深さ {nd} 段・1段目は {'軟膜の側' if top_first else '白質の側（逆にした）'}")
    for k, nm in names.items():
        m = lab == k
        if m.sum() < 5:
            continue
        a = area_key(nm)
        res[(h, a)] = dict(all=float(np.nanmean(P[:, m])), sup=float(np.nanmean(sup[:, m])),
                          inf=float(np.nanmean(inf[:, m])), g1=float(np.nanmean(g1[m])) if g1 is not None else np.nan,
                          n=int(m.sum()))

CHECK = [("V1", "V2"), ("V2", "V4"), ("V1", "V4"), ("A1", "MBelt"), ("A1", "LBelt"), ("A1", "PBelt")]
FRONT = [("6mp", "6ma"), ("4", "6mp"), ("4", "6ma"), ("6d", "6mp"), ("6a", "6ma"), ("8BL", "6ma"), ("8BL", "6mp"),
         ("8Ad", "6ma"), ("9-46d", "6ma"), ("46", "6ma"), ("6v", "6mp"), ("44", "6ma"), ("4", "6v"), ("6ma", "SCEF"), ("SCEF", "6mp")]
MARM = {("4", "6mp"): "マーモセット：4 → 補足 0.39・補足 → 4 0.65（4 が少し上）",
        ("4", "6ma"): "マーモセット：4 → 前補足 0.85（前補足が上）",
        ("6d", "6mp"): "マーモセット：6DR → 補足 0.16（6DR が上）／ 6DC → 補足 0.31",
        ("6a", "6ma"): "マーモセット：6DR → 前補足 0.86（前補足が上）",
        ("8BL", "6ma"): "マーモセット：8b → 前補足 0.71（前補足が上）",
        ("8BL", "6mp"): "マーモセット：8b → 補足 0.06（8b が上）",
        ("6v", "6mp"): "マーモセット：6Va → 補足 0.35（6Va が上）"}


def show(pairs, title, marm=False):
    for h in ("L", "R"):
        print(f"\n══ {title}（{'左' if h == 'L' else '右'}）  濃さ（全体・浅・深）と  差 ln(受け手 ÷ 送り手)：負 ＝ 送り手が濃い ＝ 上りの向き")
        for a, b in pairs:
            ra, rb = res.get((h, a)), res.get((h, b))
            if not ra or not rb:
                print(f"  {a} ⇄ {b}：区画が無い")
                continue
            dl = np.log(rb["all"] / ra["all"]) if ra["all"] > 0 and rb["all"] > 0 else np.nan
            up = (f"{a} → {b} が上り（{b} が上）" if dl < 0 else f"{b} → {a} が上り（{a} が上）") if np.isfinite(dl) else "―"
            s = (f"  {a:>6} {ra['all']:8.2f} {ra['sup']:8.2f} {ra['inf']:8.2f} G1 {ra['g1']:6.2f} ／ "
                 f"{b:>6} {rb['all']:8.2f} {rb['sup']:8.2f} {rb['inf']:8.2f} G1 {rb['g1']:6.2f}  差 {dl:+.3f} ⇒ {up}")
            print(s)
            if marm and (a, b) in MARM:
                print(f"           （参考 {MARM[(a, b)]}）")


show(CHECK, "確かめ：上下が分かっている組（左が下。差が負なら読み方が合う）")
show(FRONT, "補足運動野まわり", marm=True)
