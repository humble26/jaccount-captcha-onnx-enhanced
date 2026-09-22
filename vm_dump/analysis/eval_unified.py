# -*- coding: utf-8 -*-
"""在统一的 520 张回归集上评估任意 ONNX 模型。

用法:
  python eval_unified.py                        # 评估生产模型
  python eval_unified.py <model.onnx>           # 评估指定模型
  python eval_unified.py <model.onnx> <out.json>

⚠ 读数的正确姿势（重要）：
  * tune(120) 与 hold(100) 的真值是**人工逐张辨认**的，其准确率可作为泛化参考；
  * new(300) 的真值来自**生产模型自标注 + 人工核对**，模型在它上面的「准确率」本质是
    **自洽性**（真值就是模型的输出），不能当泛化指标用；它适合做的是
    「改动是否破坏了原有行为」的回归检查。
  报告整体数字时请分开列，不要合并成一个"520 张准确率"。
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # <归档>/02-重构研究
WS = os.environ.get("PROD_WS") or r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
UNIFIED = os.environ.get("UNIFIED_GT") or os.path.join(HERE, "_sampling_out", "unified_gt_520.json")
CH = "abcdefghijklmnopqrstuvwxyz"

model = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("PROD_ONNX")
                                               or os.path.join(WS, "vm_dump", "nn_model.onnx"))
out_json = sys.argv[2] if len(sys.argv) > 2 else None

data = json.load(open(UNIFIED, encoding="utf-8"))
gt, index = data["gt"], data["index"]


def abs_path(entry):
    rel = entry["rel"].replace("/", os.sep)
    if entry["base"] == "PROD_WS":
        return os.path.join(WS, rel)
    # ARCHIVE 的 rel 形如 02-重构研究/sampled/new_captchas/c000.png
    parts = rel.split(os.sep)
    if parts and parts[0] == os.path.basename(ROOT):
        rel = os.sep.join(parts[1:])
    return os.path.join(ROOT, rel)


def prep(p):
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32).reshape(1, 1, 40, 110)


so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
names = sorted([o.name for o in sess.get_outputs()], key=int)
iname = sess.get_inputs()[0].name

rows = []
for k, entry in index.items():
    p = abs_path(entry)
    if not os.path.exists(p):
        rows.append({"key": k, "split": entry["split"], "missing": True})
        continue
    zo = sess.run(names, {iname: prep(p)})
    idx = [int(np.argmax(z[0])) for z in zo]
    text = "".join(CH[c] for c in idx[:4])
    if idx[4] != 26:
        text += CH[idx[4]]
    rows.append({"key": k, "split": entry["split"], "pred": text,
                 "truth": gt[k], "ok": text == gt[k], "len_ok": len(text) == len(gt[k])})

print("模型:", os.path.basename(model), "| 样本", len(rows))
print()
print("%-8s %6s %8s %8s %8s" % ("分段", "张数", "整串对", "准确率", "长度同时对"))
print("-" * 46)
res = {}
for split, label in (("tune", "tune"), ("hold", "hold"), ("new", "new"), (None, "合计")):
    sub = [r for r in rows if (split is None or r["split"] == split)]
    sub = [r for r in sub if not r.get("missing")]
    if not sub:
        continue
    ok = sum(1 for r in sub if r["ok"])
    lok = sum(1 for r in sub if r["len_ok"])
    res[split or "all"] = {"n": len(sub), "ok": ok, "acc": ok / len(sub), "len_ok": lok}
    print("%-8s %6d %8d %7.2f%% %8d" % (label, len(sub), ok, 100 * ok / len(sub), lok))

miss = [r["key"] for r in rows if r.get("missing")]
if miss:
    print()
    print("⚠ 找不到图片:", len(miss), miss[:10])

print()
print("误判样本（tune/hold 为真实误判；new 只可能是 3 张人工修正过的）：")
for r in rows:
    if not r.get("ok") and not r.get("missing"):
        print("   [%-4s] %-8s 预测=%-7s 真值=%-7s" % (r["split"], r["key"], r["pred"], r["truth"]))

print()
print("⚠ 提醒：new 段的「准确率」是自洽性指标（真值由该模型自标注），不能当泛化准确率引用。")

if out_json:
    json.dump({"model": model, "summary": res, "rows": rows},
              open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写出", out_json)
