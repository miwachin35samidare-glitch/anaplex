"""版50：Group2 の履歴 tb を、行と同じ並び（領野ごとに 誤差・表現・抑制）で足す（~/anaplex で。印が1つでも無ければ何も書かずに止まる）
  python3 apply_v50.py
ずれ【測】コンテナ chk_tb（2領野・CPU、10-04）：領野 A の 4体だけ下から越えさせると
  A の表現の行の履歴 0（撃ったのに）／ B の誤差の行に A の発火が入る
  原因：cat([全領野の rep, 全領野の rep, 全領野の inh]) は種類ごとの並び。tb の行は領野ごと（off_recv・kind の作り）
  1領野の Area（torch_area.py）は cat([rep, rep, inh_spk]) で行と合っている ⇒ Group2 にまとめたときのずれ【C の落とし】
直し：行の印（is_err・is_rep・is_inh_r）で置く。学びの fired と同じ書き方
"""
import ast, pathlib, re, sys

def sub(path, old, new):
    p = pathlib.Path(path)
    s = p.read_text()
    if s.count(old) != 1:
        sys.exit(f"⚠️ {path} に印が {s.count(old)} 個（1 個のはず）。何も書いていない")
    return p, s.replace(old, new)

G_OLD = """            self.tb = self.tb * self.decay + torch.cat(
                [rep[self.x_index[self.is_err]], rep[self.x_index[self.is_rep]], inh_spk]
            ).unsqueeze(1) * (1 - self.decay)"""
G_NEW = """            # 版50：行（領野ごとに 誤差・表現・抑制）と同じ並びで足す
            #   ⟲ 版49 まで cat([全領野の rep, 全領野の rep, 全領野の inh])（種類ごとの並び）。
            #      先頭の領野の誤差の行のほかは、別の体の発火を履歴にしていた【測】chk_tb（10-04）
            now = torch.zeros(self.n_recv_all, device=self.dev)
            now[self.is_err] = rep[self.x_index[self.is_err]]
            now[self.is_rep] = rep[self.x_index[self.is_rep]]
            now[self.is_inh_r] = inh_spk
            self.tb = self.tb * self.decay + now.unsqueeze(1) * (1 - self.decay)"""
B_OLD = "VERSION = 49          # コードの版。網の作りを変えたら上げる"
B_NEW = "VERSION = 50          # コードの版。網の作りを変えたら上げる"
B_HIST_OLD = "#      口は発火を筋の形（【ヒト】Ito 2004 6.1Hz・ζ 0.69・15ms）でならした力 ≥ 0.20【C】で読む【み】10-03\n"
B_HIST_NEW = B_HIST_OLD + ("#  50  Group2 の履歴 tb を行と同じ並びで足す。版49 まで先頭の領野の誤差の行のほかは別の体の発火を履歴にしていた\n"
                           "#      （Group2 にまとめたときのずれ【C】。1領野の Area は合っていた。【測】chk_tb 10-04）\n")

pg, sg = sub("torch_group2.py", G_OLD, G_NEW)
pb, sb = sub("babble.py", B_OLD, B_NEW)
if sb.count(B_HIST_OLD) != 1:
    sys.exit("⚠️ babble.py の履歴の印が見つからない。何も書いていない")
sb = sb.replace(B_HIST_OLD, B_HIST_NEW)
ast.parse(sg); ast.parse(sb)
pg.write_text(sg); pb.write_text(sb)
print("✅ 版50 を当てた（torch_group2.py・babble.py）。state.pt は版49 のものなので使えない（0から）")
