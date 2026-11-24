# handoff/tree — 容器の版の木の写し（いま渡した最後の形）

- 版：**116**（torch_brain・babble・測る道具は apply_v116・life は apply_guard109d）
- 作り方（11-23）：リポジトリ直下（版66）に apply_v67〜v97 を当て、v98 から先は各 apply の FILES を sha の鎖の順に書いた。check_v115b 40 が全部 ✓
- 新しいスレッドの木：リポジトリ直下の .py 以外（prior_v*.npz・synapse_model_human.json）を写し、その上にこのフォルダの .py を写す
  - 版115 の木（check_v115b の比べ先）：その写しで apply_v115b_back を当てる
- arkhe に新しい apply を渡したら、当てたあとの .py をここにも写す（渡した形とここが同じであること）
- ⚠️ 本物は arkhe の ~/anaplex。食い違えば arkhe が正しい
