import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""生成人工核对真值表 CSV：按"位4高危 + 低置信"排序，供用户逐行核对。
真值一列初始为模型候选，人工确认/修正后回填即可。
"""
import os, sys, json, csv
sys.path.insert(0, _HERE)

recs = json.load(open(os.path.join(_HERE, "_sampling_out", "auto_labeled.json"), encoding="utf-8"))
HOT = set("cgouwxyz")

def clean(p):
    return p[:-1] if p.endswith("{") else p

def sortkey(r):
    c = clean(r["pred"])
    pos4 = c[3] if len(c) >= 4 else ""
    hot = pos4 in HOT
    low = r["minconf"] < 0.999
    return (0 if hot else 1, 0 if low else 1, r["file"])

rows = []
for r in sorted(recs, key=sortkey):
    c = clean(r["pred"])
    pos4 = c[3] if len(c) >= 4 else ""
    rows.append({
        "文件": r["file"],
        "位数": len(c),
        "模型候选真值": c,
        "位4字符": pos4,
        "位4高危": "是" if pos4 in HOT else "",
        "最低置信": r["minconf"],
        "人工真值(请核对修正)": "",   # 人工回填列
        "备注": "难例" if r["minconf"] < 0.999 else "",
    })

out = os.path.join(_HERE, "_sampling_out", "标注核对表.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

hot_count = sum(1 for r in rows if r["位4高危"] == "是")
print(f"核对表已生成: {out}")
print(f"共 {len(rows)} 张 | 位4高危候选 {hot_count} 张 | 红色索引图对应低置信难例")
print("人工真值列留空，核对后回填")