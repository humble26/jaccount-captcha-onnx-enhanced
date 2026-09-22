# -*- coding: utf-8 -*-
"""构建统一的 520 张真值集（220 张人工标注 + 300 张已人工核对的自标注）。

为什么需要：300 张新样本的真值一直单独放在 new300_gt.json，没有并入
ground_truth_all.json，导致无法一次性评估全部 520 张。而「冻结的统一回归集」
是这批数据最有价值的形态 —— 任何模型 / 预处理 / 脚本改动都能一键回归。

输出：_sampling_out/unified_gt_520.json
{
  "gt":    { key: "code" },
  "index": { key: { "rel": "<基准目录下的相对路径>", "base": "<基准名>", "split": "tune|hold|new" } },
  "meta":  { ... }
}
图片只记相对路径（相对某个基准目录），换机器或换位置后按 base 重新拼接即可。
"""
import os
import json
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # <归档>/02-重构研究
WS = os.environ.get("PROD_WS") or r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
OUT = os.environ.get("UNIFIED_GT_OUT") or os.path.join(HERE, "_sampling_out", "unified_gt_520.json")

GT_220 = os.path.join(WS, "vm_dump", "ground_truth_all.json")
NEW300 = os.path.join(HERE, "_sampling_out", "new300_gt.json")
NEW_SRC_REL = "vm_dump/new300"

gt220 = json.load(open(GT_220, encoding="utf-8"))
gt300 = json.load(open(NEW300, encoding="utf-8"))

gt, index = {}, {}
dup = []

for split, dirs in (("tune", ["samples", "samples2"]), ("hold", ["holdout"])):
    for d in dirs:
        p = os.path.join(WS, "vm_dump", d)
        for f in sorted(os.listdir(p)):
            if not f.endswith(".png"):
                continue
            k = os.path.splitext(f)[0]
            if k not in gt220:
                continue
            if k in gt:
                dup.append(k)
            gt[k] = gt220[k]
            index[k] = {"rel": "vm_dump/%s/%s" % (d, f), "base": "PROD_WS", "split": split}

new_dir = os.path.join(ROOT, "new300")
for f in sorted(os.listdir(new_dir)):
    if not f.endswith(".png"):
        continue
    k = os.path.splitext(f)[0]
    if k not in gt300:
        continue
    if k in gt:
        dup.append(k)
    gt[k] = gt300[k]
    index[k] = {"rel": "%s/%s" % (NEW_SRC_REL.replace("\\", "/"), f), "base": "PROD_WS", "split": "new"}

meta = {
    "总张数": len(gt),
    "来源": {"220 张人工标注（调参+留出）": len([1 for v in index.values() if v["split"] in ("tune", "hold")]),
             "300 张新样本（自标注 + 人工核对 3 张）": len([1 for v in index.values() if v["split"] == "new"])},
    "分段": {s: len([1 for v in index.values() if v["split"] == s]) for s in ("tune", "hold", "new")},
    "长度分布": {str(L): len([1 for v in gt.values() if len(v) == L]) for L in (4, 5)},
    "bases": {"PROD_WS": r"<工作区>（含 vm_dump/）", "ARCHIVE": r"<本归档根目录>"},
    "生成时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "重复 key": dup,
    "说明": "tune/hold 的真值来自人工逐张辨认；new 的真值来自生产模型自标注 + 人工核对 "
            "（3 张修正 c078/c215/c269），并经 48 张盲测与 33 张位4高危全量核对验证。",
}

json.dump({"gt": gt, "index": index, "meta": meta},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("已写出", OUT)
for k, v in meta.items():
    print("  %-28s %s" % (k, v))
