import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""准备端到端微调的新数据：把 300 张新图拷入 vm_dump/new300，
生成独立的 new300_gt.json 真值文件（不污染共享 ground_truth_all.json）。"""
import os, json, shutil

SRC = os.path.join(_VD, "new300")
GT_OUT = os.path.join(_HERE, "_sampling_out", "new300_gt.json")
recs = json.load(open(os.path.join(_HERE, "_sampling_out", "final_labels_300.json"),
                      encoding="utf-8"))
gt = {}
missing = 0
for r in recs:
    fn = r["file"]
    if not os.path.exists(os.path.join(SRC, fn)):
        print("MISSING", fn)
        missing += 1
        continue
    gt[fn[:-4]] = r["truth"]
json.dump(gt, open(GT_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("真值写入", GT_OUT, "共", len(gt), "张，missing", missing)
from collections import Counter
lc = Counter(len(v) for v in gt.values())
print("新真值长度分布:", dict(lc))