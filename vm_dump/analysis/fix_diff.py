import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""定位差异：为什么权威脚本 215/220，我的调用 3.64%？

差异候选：
  1. 输出顺序：sess.run(None) 按图定义顺序 vs 我按名字数值排序
  2. 输入阈值方向：base_input 用 (g >= 156) —— 与我 arch_probe2 的 (<156) 相反！
"""
import os, json
import numpy as np
import onnxruntime as rt
from PIL import Image

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"

gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
paths = {}
for d in ["samples", "samples2", "holdout"]:
    dd = os.path.join(VD, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)
keys = [k for k in gt if k in paths]

so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), so, providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name
order_natural = [o.name for o in sess.get_outputs()]
order_sorted = sorted(order_natural, key=lambda x: int(x))
print("图定义顺序:", order_natural)
print("名字排序  :", order_sorted)

def decode_ge156(k):
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    a = (g >= 156).astype(np.float32)
    out = sess.run(None, {IN: a[None, None, ...]})
    txt = ""
    for t in out:
        i = int(np.argmax(t, 1)[0])
        if i >= 26: continue
        txt += CH[i]
    return txt

def decode_lt156(k):
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    a = (g < 156).astype(np.float32)
    out = sess.run(None, {IN: a[None, None, ...]})
    txt = ""
    for t in out:
        i = int(np.argmax(t, 1)[0])
        if i >= 26: continue
        txt += CH[i]
    return txt

for name, fn in [("(g>=156) 权威用法", decode_ge156), ("(g<156)  我arch_probe2", decode_lt156)]:
    ok = sum(1 for k in keys if fn(k) == gt[k])
    print(f"\n{name}: {ok}/{len(keys)} = {ok/len(keys)*100:.2f}%")
    for k in sorted(keys)[:3]:
        print(f"    {k}: 真值={gt[k]}  预测={fn(k)}")

print()
print("=" * 70)
print("修正后重跑：各输出头的置信统计")
print("=" * 70)
stats = {}
for k in keys:
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    a = (g >= 156).astype(np.float32)
    out = sess.run(None, {IN: a[None, None, ...]})
    for i, t in enumerate(out):
        nm = order_natural[i]
        v = t[0]
        e = np.exp(v - v.max()); p = e / e.sum()
        st = stats.setdefault(nm, {"conf": [], "n": 0})
        st["conf"].append(float(p[np.argmax(v)]))
        st["n"] += 1
print(f"{'头':8s} {'均值置信':>10s} {'最低置信':>10s} {'P1':>8s} {'P5':>8s}")
for nm in order_natural:
    c = np.array(stats[nm]["conf"])
    print(f"{nm:8s} {c.mean()*100:9.2f}% {c.min()*100:9.2f}% {np.percentile(c,1)*100:7.2f}% {np.percentile(c,5)*100:7.2f}%")

# 关键：头部与位置的对应 —— 用真值逐位验证
print()
print("各输出头 → 位置的对应验证（哪一头的 argmax 与真值第几位吻合）")
for i, nm in enumerate(order_natural):
    hit = 0; tot = 0
    for k in keys:
        g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
        a = (g >= 156).astype(np.float32)
        out = sess.run(None, {IN: a[None, None, ...]})
        t = gt[k]
        if i >= len(t): continue
        tot += 1
        idx = int(np.argmax(out[i][0]))
        if idx < 26 and CH[idx] == t[i]:
            hit += 1
    print(f"  头{nm} (序号{i}) 作为「第{i+1}位」时准确率: {hit}/{tot} = {hit/max(tot,1)*100:.2f}%")
