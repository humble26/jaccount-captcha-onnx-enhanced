import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""真值字符在 top-K 中的排位 —— 判断"重排序/候选集"策略是否有救

关键问题：模型是"自信地错"（真值排到第 5、第 8 名之后，任何后处理都救不了），
还是"犹豫地错"（真值稳居 top2~top3，只要有更好的打分就能翻盘）？
"""
import os, json
import numpy as np
import onnxruntime as rt
from PIL import Image

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"

gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
h = json.load(open(os.path.join(VD, "holdout_eval.json"), encoding="utf-8"))
pred_base = h["pred"]["base"]

paths = {}
for d in ["samples", "samples2", "holdout"]:
    dd = os.path.join(VD, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)

def prep(p):
    a = np.asarray(Image.open(p).convert("L")).astype(np.float64)
    return (np.round(a) >= 156).astype(np.float32).reshape(1, 1, 40, 110)

so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
names = sorted([o.name for o in sess.get_outputs()], key=lambda x: int(x))

err_keys = sorted([k for k in gt if k in pred_base and gt[k] != pred_base[k]])
print("=" * 84)
print("错误样本：真值字符在该位的 top-K 排位")
print("=" * 84)
for k in err_keys:
    t = gt[k]
    out = sess.run(names, {iname: prep(paths[k])})
    print(f"\n[{k}] 真值={t}  预测={pred_base[k]}  长度={len(t)}")
    for i in range(len(t)):
        v = out[i][0]
        order = np.argsort(v)[::-1]
        e = np.exp(v - v.max()); pr = e / e.sum()
        tv = CH.index(t[i]) if t[i] in CH else -1
        rank = int(np.where(order == tv)[0][0]) + 1 if tv >= 0 else -1
        top3 = ", ".join(f"{CH[c]}({pr[c]*100:.2f}%)" for c in order[:3])
        flag = "   <<< 错位" if i < len(pred_base[k]) and t[i] != pred_base[k][i] else ""
        print(f"   位{i+1}: 真值={t[i]} 排位=#{rank}   top3: {top3}{flag}")

print()
print("=" * 84)
print("全体统计：真值字符的排位分布（只统计有真值的位）")
print("=" * 84)
ranks = []
for k, t in gt.items():
    if k not in paths:
        continue
    out = sess.run(names, {iname: prep(paths[k])})
    for i in range(len(t)):
        v = out[i][0]
        order = np.argsort(v)[::-1]
        tv = CH.index(t[i]) if t[i] in CH else -1
        if tv >= 0:
            ranks.append(int(np.where(order == tv)[0][0]) + 1)
ranks = np.array(ranks)
print(f"总位数 {len(ranks)}")
for r in range(1, 8):
    c = int((ranks == r).sum())
    print(f"  真值排第 {r} 名: {c:4d} 张 ({c/len(ranks)*100:6.2f}%)")
print(f"  真值排第 8 名及以后: {int((ranks > 7).sum())} 张 ({(ranks>7).sum()/len(ranks)*100:.3f}%)")
print(f"  -> 若能用完美打分在 top3 内重排，理论上限 = {int((ranks<=3).sum())}/{len(ranks)} = {(ranks<=3).sum()/len(ranks)*100:.2f}%")
print(f"  -> 若只在 top2 内重排，理论上限 = {int((ranks<=2).sum())}/{len(ranks)} = {(ranks<=2).sum()/len(ranks)*100:.2f}%")

print()
print("=" * 84)
print("4 字符码的 blank 行为检查（第 5 头是否稳定报 blank）")
print("=" * 84)
n4 = n5 = 0
bad4 = []
for k, t in gt.items():
    if k not in paths:
        continue
    out = sess.run(names, {iname: prep(paths[k])})
    v5 = out[4][0]
    is_blank = int(np.argmax(v5)) == 26
    if len(t) == 4:
        n4 += 1
        if not is_blank:
            bad4.append((k, int(np.argmax(v5))))
    elif len(t) == 5:
        n5 += 1
print(f"4 字符码 {n4} 张，其中第 5 头未报 blank 的: {len(bad4)} 张 {bad4[:8]}")
print(f"5 字符码 {n5} 张")
