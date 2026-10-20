"""切り分け2：揺れは境（外側手綱核への入力）が規則的に撃つことから来るか ⇒ V2 境を 33Hz のポアソンに（外側手綱核は今の先生のまま）"""
from dbg_lib import *
th=make(); th.pois={"境": 33.0}
print("V2 境ポアソン：外側手綱核", rate(th,"外側手綱核"), flush=True)
print("   黒質 i0", sn_i0(th), flush=True)
volley(th, "V2 境ポアソン")
