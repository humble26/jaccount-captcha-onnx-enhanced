# -*- coding: utf-8 -*-
"""应用人工核对修正，生成最终 300 张真值表。
真值规则：高置信(>=0.999)且未被人工修正 -> 采信 pred；
低置信但人工未指出的 -> pred（用户逐张核对后默认认可）；
人工指出的 -> 用人工真值覆盖。
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 归档后本文件位于 <归档>/02-重构研究/analysis/，输入输出都在同级 _sampling_out/。
# （原来写死的 E:\harness\重构研究\analysis\... 在归档剪切移动后已不存在。）
SRC = os.environ.get("AUTOLABEL_IN") or os.path.join(HERE, "_sampling_out", "auto_labeled.json")
recs = json.load(open(SRC, encoding="utf-8"))

# 人工核对修正：{文件名: 人工真值}
CORR = {
    "c078.png": "ijjca",
    "c215.png": "mxpk",
    "c269.png": "ltfbm",
}

HOT = set("cgouwxyz")

def clean(p):
    """兼容用旧版 _autolabel.py 产出的数据：旧版没处理第 5 头的 blank，
    4 位码的 pred 末尾会多一个 '{'（chr(123)）。
    新版 _autolabel.py 已在源头修掉，这里保留只为能重跑历史文件。"""
    return p[:-1] if p.endswith("{") else p

final_recs = []
n_high, n_low, n_corr = 0, 0, 0
for r in recs:
    fn = r["file"]
    pred = clean(r["pred"])
    # 确定真值
    if fn in CORR:
        truth = CORR[fn]
        n_corr += 1
    else:
        truth = pred
    pos4 = truth[3] if len(truth) >= 4 else ""
    final_recs.append({
        "file": fn,
        "truth": truth,
        "length": len(truth),
        "pos4": pos4,
        "pos4_hot": pos4 in HOT,
        "source": "corr" if fn in CORR else ("high" if r["minconf"] >= 0.999 else "low"),
        "confidence": r["minconf"],
    })
    if r["minconf"] >= 0.999:
        n_high += 1
    else:
        n_low += 1

out = os.environ.get("FINAL_LABELS_OUT") or os.path.join(HERE, "_sampling_out", "final_labels_300.json")
json.dump(final_recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"最终真值表: {len(final_recs)} 张 -> {out}")
print(f"来源: 高置信采信={n_high} 低置信={n_low} 人工修正={n_corr}")
from collections import Counter
c4 = Counter(r["pos4"] for r in final_recs if len(r["pos4"]) == 1)
print("位4真值分布:", dict(sorted(c4.items())))
hot = [r for r in final_recs if r["pos4_hot"]]
print(f"位4高危真值: {len(hot)} 张")
hc = Counter(r["pos4"] for r in hot)
print("位4高危字符分布:", dict(sorted(hc.items())))
# 4/5 位分布
l4 = sum(1 for r in final_recs if r["length"] == 4)
l5 = sum(1 for r in final_recs if r["length"] == 5)
print(f"4位={l4} 5位={l5}")