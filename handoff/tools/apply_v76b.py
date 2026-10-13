"""版76b：版76 の torch.compile を止める（GPU で起動できない。値・網は版76 のまま。スイッチを消した部分もそのまま）
（~/anaplex で。版76 の上に当てる。印が合わなければ何も書かずに止まる）
  python3 apply_v76b.py
【測】10-14 arkhe measure_speed75（版76）：_flush_core の Triton の核が起動できない（driver error 401）。
      そのあと GPU が使えない状態になり、まとめない形にも戻れなかった（版76 の逃げ道は GPU では役に立たない作り。Claude の誤り）
【測】probe_compile：A（そのまま）・B（Triton 側の起動）・C（B ＋ 選ばない）・D（大きさを決める）の4つとも _flush_core で ✗
⇒ 辺の計算は版75 と同じく、まとめない形で回す。_sum_core・_flush_core・_learn_core の関数はそのまま使う（式は同じ）
   どの計算が起動できないかは probe_compile2.py で分けて、分かったら版77 で戻す
"""
import ast, pathlib, sys

OLD = ("def _run(fn, *a):\n"
       "    try:\n"
       "        return _cc(fn)(*a)\n"
       "    except Exception as ex:                                   # まとめられないときは元の形で（1回だけ知らせる）\n"
       "        if _CC.get(fn) is not fn:\n"
       "            print(f\"⚠️ torch.compile が {fn.__name__} で動かない（{type(ex).__name__}）。まとめずに回す\", flush=True)\n"
       "            _CC[fn] = fn\n"
       "        return fn(*a)\n")
NEW = ("def _run(fn, *a):\n"
       "    \"\"\"版76b：まとめない形で回す（GPU で Triton の核が起動できない【測】probe_compile 10-14。直したら戻す）\"\"\"\n"
       "    return fn(*a)\n")

if __name__ == "__main__":
    sys.path.insert(0, ".")
    import babble
    if babble.VERSION != 76:
        sys.exit(f"⚠️ babble.py が版76 ではない（版 {babble.VERSION}）")
    p = pathlib.Path("torch_group2.py"); s = p.read_text()
    if s.count(OLD) != 1:
        sys.exit(f"⚠️ torch_group2.py に印が {s.count(OLD)} 個（1 個のはず）。何も書いていない")
    s = s.replace(OLD, NEW); ast.parse(s); p.write_text(s)
    print("✅ 版76b を当てた（torch_group2）。torch.compile を止めた（版76 のスイッチを消した部分・値はそのまま）")
