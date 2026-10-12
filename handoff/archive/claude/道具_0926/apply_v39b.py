"""版39 → 版39b（arkhe の ~/anaplex で  python3 apply_v39b.py ）

恒常性の目標を、受け手の種類ごとの【ヒト】のふだんの率に（【み】09-27「普段の率を目標とみなす」「Bで様子見」）
  前は全領野・全種類に皮質の 0.45%/歩 を1つだけ掛けていた
  皮質（聴覚・視覚・連合・高次）  表現 0.45% ／ 誤差 0.45% ／ 抑制 2Hz ＝ 2.13%
      表現 【ヒト】Peyrache 2012（中側頭回 L2/3）RS 0.5Hz 未満。今の値のまま
      誤差 人に誤差専用の細胞は無い。上側頭回の普通の錐体が誤差のように振る舞う ⇒ 表現と同じ【C 当てはめ】
      抑制 【ヒト】Peyrache 2012 FS 約2Hz
  視床    表現 18.8Hz ＝ 20.1%（【ヒト】運動視床 Vim。聴覚の視床 MGB は人・霊長類とも数値なし）
          誤差・抑制は値が無いので下方調整から外す（B）
  運動    話す運動野の値が無いので、領野ごと下方調整から外す（B）
  海馬    今のまま（眠りでは sleep_mode で下げない。目標の値は使われない）
起きているあいだは今までどおり下方調整は止まっているので、喃語の中身は変わらない ⇒ VERSION は 39 のまま（喃語800回の state.pt をそのまま使える）
印が1つでも見つからなければ何も書かずに止まる
"""
import ast
import shutil
import sys


def once(s, old, new, name):
    if s.count(old) != 1:
        sys.exit(f"⚠️ {name} の印が {s.count(old)} 回。何も書いていない")
    return s.replace(old, new)


F = {k: open(k, encoding="utf-8").read() for k in ("torch_group2.py", "torch_brain.py", "babble.py")}
if "VERSION = 39 " not in F["babble.py"]:
    sys.exit("⚠️ babble.py が版39 ではない。何も書いていない")
if "target_recv" in F["torch_group2.py"]:
    sys.exit("⚠️ もう版39b になっている。何も書いていない")

g = F["torch_group2.py"]
g = once(g, "        self.rate_ema = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)\n",
         "        self.rate_ema = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)\n"
         "        # 恒常性の目標を受け手ごとに（版39b。torch_brain が種類ごとの【ヒト】の値を入れる）\n"
         "        self.target_recv = torch.full((self.n_recv_all,), float(self.scaling_target), device=dev)\n"
         "        self.scale_skip = torch.zeros(self.n_recv_all, dtype=torch.bool, device=dev)   # 人の値が無い受け手は下げない\n",
         "group2 の rate_ema")
g = once(g, "self.rate_ema / self.scaling_target - 1", "self.rate_ema / self.target_recv - 1", "下方調整の式")
g = once(g, "                self.scale = self.scale * f\n",
         "                f = torch.where(self.scale_skip, torch.ones_like(f), f)   # 版39b：人の値が無い受け手\n"
         "                self.scale = self.scale * f\n", "scale の掛け算")

t = F["torch_brain.py"]
k = "            self.g.mod_send[self._send_rows(area, first, cnt)] = True\n"
t = once(t, k, k +
         "        # 恒常性の目標を受け手の種類ごとに（版39b）【み】09-27「普段の率を目標とみなす」「Bで様子見」\n"
         "        #   皮質  表現・誤差 0.45%（【ヒト】Peyrache 2012 RS 0.5Hz 未満。誤差は表現と同じ【C 当てはめ】）／ 抑制 2Hz（FS）\n"
         "        #   視床  表現 18.8Hz（【ヒト】運動視床 Vim。MGB は人・霊長類とも数値なし）。誤差・抑制は値なし ⇒ 下げない\n"
         "        #   運動  話す運動野の値なし ⇒ 領野ごと下げない ／ 海馬は眠りでは sleep_mode で下げない（今のまま）\n"
         "        from torch_area import DT_MS as _DT\n"
         "        hz = lambda v: v * _DT / 1000.0\n"
         "        gg = self.g\n"
         "        for i, nm in enumerate(NAMES):\n"
         "            lo, hi, nc = int(gg.off_recv[i]), int(gg.off_recv[i + 1]), self.areas[nm].n_ch\n"
         "            er, rp, ih = slice(lo, lo + nc), slice(lo + nc, lo + 2 * nc), slice(lo + 2 * nc, hi)\n"
         "            if nm == \"視床\":\n"
         "                gg.target_recv[rp] = hz(18.8)\n"
         "                gg.scale_skip[er] = True\n"
         "                gg.scale_skip[ih] = True\n"
         "            elif nm == \"運動\":\n"
         "                gg.scale_skip[lo:hi] = True\n"
         "            elif nm in (\"聴覚\", \"視覚\", \"連合\", \"高次\"):\n"
         "                gg.target_recv[ih] = hz(2.0)\n"
         "        gg.rate_ema = gg.target_recv.clone()     # 0から組むときは目標から始める（state.pt を読めば上書きされる）\n",
         "brain の調整する側の後")

b = F["babble.py"]
k = "#      上からの戻り3本と指令の写しは表現を撃たせない（予測 p には入る）／ 驚きの基準が 0 から始まる不具合を直した\n"
b = once(b, k, k +
         "#  39b 恒常性の目標を受け手の種類ごとの【ヒト】のふだんの率に（皮質の抑制 2Hz・視床の表現 18.8Hz。値の無い\n"
         "#      視床の誤差・抑制と運動は下げない）。起きているあいだは変わらないので VERSION は 39 のまま\n",
         "版の履歴")

out = {"torch_group2.py": g, "torch_brain.py": t, "babble.py": b}
for v in out.values():
    ast.parse(v)
for k, v in out.items():
    shutil.copy(k, k + ".v39")
    open(k, "w", encoding="utf-8").write(v)
hip = "is_hippo, torch.ones_like" in g
print("版39b にした（元は *.v39 に残した）")
print(f"眠りで海馬を下げない行（sleep_mode）: {'ある' if hip else '⚠️ 無い（海馬も眠りで下がる）'}")
print("眠りの様子を見るときは  python3 watch_sleep.py")
