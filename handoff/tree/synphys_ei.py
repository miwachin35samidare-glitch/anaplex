"""人の皮質の 興奮 → 抑制・抑制 → 興奮 の繋がる割合を、Allen の synphys の sqlite（前に使った synphys_r2.1_small.sqlite）から出す（arkhe で。CPU だけ・数秒）
  python3 synphys_ei.py [sqlite の場所]       （書かなければ ~ の下を探す。見つからなければ止まる。ダウンロードはしない）
  何のため：網の核・皮質の「抑制の体は体から 180本ずつ受ける」（torch_area INH_N_EXC、【C】。コードの注「人の抑制細胞の入力の数は見つからない」）を
    人の値と比べる（組み表_版112_案 §15）。網で比べられる形 ＝ 近くの体のうち何割が1つの抑制の体へ繋がるか
  出すもの（人だけ・species が human の subject）
    ① 表と列の一覧（列の名前が思っていた形と違ったら、ここを見て直す）
    ② 送り手の型 × 受け手の型（ex/in）ごとに：試した組・繋がった組・割合。細胞体の間の距離（横）で 0〜50・50〜100・100〜150・150〜250µm に分ける
    ③ 同じ層の中だけ（層が分かるとき）
  試した組の数え方【C】：送り手が ex なら n_ex_test_spikes ＞ 0、in なら n_in_test_spikes ＞ 0（aisynphys の pair_was_probed と同じ考え）。列が無ければ全部の組
"""
import sys, os, sqlite3, glob
p = sys.argv[1] if len(sys.argv) > 1 else None
if p is None:
    c = [f for f in glob.glob(os.path.expanduser("~/**/synphys*.sqlite"), recursive=True)]
    if not c:
        sys.exit("synphys の sqlite が ~ の下に見つからない（ダウンロードはしない。場所を引数で渡す）")
    p = sorted(c, key=os.path.getsize)[0]
print(f"読む：{os.path.basename(p)}（{os.path.getsize(p) / 1e9:.2f}GB）")
db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
cur = db.cursor()
tabs = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")]
cols = {t: [r[1] for r in cur.execute(f"pragma table_info('{t}')")] for t in tabs}
print("\n① 表と列（要るものだけ）")
for t in ("pair", "cell", "experiment", "slice", "subject", "cortical_location", "cortical_cell_location"):
    if t in cols:
        print(f"  {t}: {', '.join(cols[t])}")
need = ("pair", "cell", "experiment", "slice")
miss = [t for t in need if t not in cols]
SP = "su.species" if "subject" in cols and "species" in cols.get("subject", []) else "sl.species"   # 11-18：r2.1_small は species が slice にある
if miss:
    sys.exit(f"表が無い：{miss}（上の一覧を Claude に渡す）")
pc = cols["pair"]
dist = "lateral_distance" if "lateral_distance" in pc else ("distance" if "distance" in pc else None)
cls = "cell_class" if "cell_class" in cols["cell"] else ("cell_class_nonsynaptic" if "cell_class_nonsynaptic" in cols["cell"] else None)
if cls is None or "has_synapse" not in pc or dist is None:
    sys.exit("列が思っていた形でない（上の一覧を Claude に渡す）")
lay_t = next((t for t in ("cortical_location", "cortical_cell_location") if t in cols and "cortical_layer" in cols[t]), None)
q = f"""select pre.{cls}, post.{cls}, pr.has_synapse, pr.{dist},
         {'pr.n_ex_test_spikes' if 'n_ex_test_spikes' in pc else 'NULL'}, {'pr.n_in_test_spikes' if 'n_in_test_spikes' in pc else 'NULL'},
         {'lpre.cortical_layer, lpost.cortical_layer' if lay_t else 'NULL, NULL'}
  from pair pr join cell pre on pr.pre_cell_id = pre.id join cell post on pr.post_cell_id = post.id
  join experiment ex on pr.experiment_id = ex.id join slice sl on ex.slice_id = sl.id {'join subject su on sl.subject_id = su.id' if SP.startswith('su.') else ''}
  {f'left join {lay_t} lpre on lpre.cell_id = pre.id left join {lay_t} lpost on lpost.cell_id = post.id' if lay_t else ''}
  where (lower({SP}) like '%human%' or lower({SP}) like '%homo%') and pr.pre_cell_id != pr.post_cell_id"""
print("  species の値：", [r[0] for r in cur.execute(f"select distinct {SP.replace('su.', 'subject.').replace('sl.', 'slice.')} from {'subject' if SP.startswith('su.') else 'slice'}")])
print("  cell_class の値：", [r[0] for r in cur.execute(f"select distinct {cls} from cell")])
rows = cur.execute(q).fetchall()
print(f"\n人の組：{len(rows):,}")
BINS = ((0, 50e-6), (50e-6, 100e-6), (100e-6, 150e-6), (150e-6, 250e-6), (0, 1.0))


def table(rs, title):
    print(f"\n{title}")
    print(f"  {'送り → 受け':<10s}" + "".join(f"{f'{a * 1e6:.0f}〜{b * 1e6:.0f}µm' if b < 1 else '全部':>16s}" for a, b in BINS))
    for a_ in ("ex", "in"):
        for b_ in ("ex", "in"):
            s = []
            for lo, hi in BINS:
                pr_ = [r for r in rs if r[0] == a_ and r[1] == b_ and r[3] is not None and lo <= r[3] < hi
                       and ((r[4] is None and r[5] is None) or ((r[4] or 0) > 0 if a_ == "ex" else (r[5] or 0) > 0))]
                k = sum(1 for r in pr_ if r[2])
                s.append(f"{k:>4d}/{len(pr_):<5d}{(k / len(pr_) if pr_ else float('nan')):6.0%}")
            print(f"  {a_ + ' → ' + b_:<10s}" + "".join(f"{x:>16s}" for x in s))


table(rows, "② 送り手の型 × 受け手の型：繋がった組/試した組・割合（細胞体の横の距離ごと）")
if lay_t:
    same = [r for r in rows if r[6] is not None and r[6] == r[7]]
    for L_ in sorted({r[6] for r in same}):
        table([r for r in same if r[6] == L_], f"③ 同じ層の中：{L_}")
print("\n（終わり）")
