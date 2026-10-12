"""measure_check.py の録音を「学ばずに聞かせるだけ」にする（~/anaplex で python3 fix_check_rec.py measure_check.py。印が無ければ何も書かずに止まる）"""
import sys,pathlib
p=pathlib.Path(sys.argv[1]); s=p.read_text()
old='st["tag"] = "録音"; B.run(b, c0, e0, False, True)'
assert s.count(old)==1
s=s.replace(old,'st["tag"] = "録音"; B.run(b, c0, e0, False, False)   # 10-09：学ばない（聞かせるだけ）')
oh='⚠️ 模型の耳は録音と全く同じ音を聞く（人は骨からも聞く）⇒ 自分と録音の差は写しと蹴りからしか出ない（【み】10-07）\n'
assert s.count(oh)==1
s=s.replace(oh,oh+'⟲ 10-09 録音は学ばずに聞かせるだけにした（【み】「赤ちゃんが録音を延々と聞かされる状況って現実には無い」）。\n   10-07〜10-09 の健診（版51〜53）は録音も学んだ網の値。喃語の本体（babble.py・day.py）は録音を聞かない\n')
p.write_text(s)
