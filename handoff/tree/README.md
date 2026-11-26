# handoff/tree — 容器の版の木の写し（いま渡した最後の形）

- 版：**116b**（11-26 02:4x に渡し、07:2x の出力で当たったと確かめた）
  - torch_brain・torch_group2：apply_v116b ／ babble・測る道具の門：apply_v116 ／ life：apply_guard109d
  - 11-26 に足した道具：measure_tc116（版116・116b で回る）・check_v116b
- 新しいスレッドの木の作り方：リポジトリ直下の .py 以外（prior_v*.npz・synapse_model_human.json ほか）を写し、その上にこのフォルダの .py を写す
  - 11-26 18:0x：この作り方で作った木で check_v116b run が全部 ✓（20歩・容器）
  - 版116 の木：その木で handoff/tools/apply_v116b_back.py ／ 版115b の木：さらに apply_v116_back.py
  - 比べ：`ANAPLEX_V116B=0 python3 check_v116b.py run a.npz`（版116b の木）と版116 の木の `check_v116b.py run b.npz` を `check_v116b.py cmp a.npz b.npz` で（11-26 に重さ・発火が同じ ✓）
- arkhe に新しい apply を渡したら、当てたあとの .py をここにも写す（渡した形とここが同じであること）
- ⚠️ 本物は arkhe の ~/anaplex。食い違えば arkhe が正しい
