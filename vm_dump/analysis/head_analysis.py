import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""深挖：为什么错误全部集中在第 4 位？

假设 A：模型输出头与字符位置的对应关系错位（例如第 4 个头其实对应别的槽）
假设 B：模型对第 4 槽（或某个 x 区间）的训练不足
假设 C：第 4 个头与第 5 个头的分工导致边界问题 ——
       前 4 个头是 26 类、第 5 个头是 27 类（含 blank），
       模型可能是"按 4 位训练 + 第 5 位补位"，第 4 位是两者的交界

用 logit 数据检验：如果第 4 头的 logit 分布与其他头不同，假设 C 就很可疑。
"""
import os, json
import numpy as np

WS = _REPO

print("=" * 80)
print("检查 1：logit_check.json 的字段语义")
print("=" * 80)
lc = json.load(open(os.path.join(WS, "vm_dump", "logit_check.json"), encoding="utf-8"))
print(f"条数 {len(lc)}")
print("首条:", json.dumps(lc[0], ensure_ascii=False, indent=1))

print()
print("=" * 80)
print("检查 2：模型输出头 218~222 的维度（从 ONNX 解析）")
print("=" * 80)

try:
    import onnx
    m = onnx.load(os.path.join(WS, "vm_dump", "nn_model.onnx"))
    g = m.graph
    print("图输入:")
    for i in g.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"   {i.name}  {dims}")
    print("图输出:")
    for o in g.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"   {o.name}  {dims}")
except ImportError:
    print("onnx 未安装，改用 onnxruntime 读取")
    import onnxruntime as rt
    s = rt.InferenceSession(os.path.join(WS, "vm_dump", "nn_model.onnx"),
                            providers=["CPUExecutionProvider"])
    print("输入:")
    for i in s.get_inputs():
        print(f"   {i.name}  shape={i.shape}  type={i.type}")
    print("输出:")
    for o in s.get_outputs():
        print(f"   {o.name}  shape={o.shape}  type={o.type}")

print()
print("=" * 80)
print("检查 3：用真模型跑 220 张，看每个头的 argmax 分布与置信度")
print("=" * 80)

import onnxruntime as rt
from PIL import Image

sess = rt.InferenceSession(os.path.join(WS, "vm_dump", "nn_model.onnx"),
                           providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
onames = [o.name for o in sess.get_outputs()]
print("输入名", iname, " 输出名", onames)

gt = json.load(open(os.path.join(WS, "vm_dump", "ground_truth_all.json"), encoding="utf-8"))
h = json.load(open(os.path.join(WS, "vm_dump", "holdout_eval.json"), encoding="utf-8"))
pred_base = h["pred"]["base"]

paths = {}
for d in ["vm_dump\\samples", "vm_dump\\samples2", "vm_dump\\holdout"]:
    dd = os.path.join(WS, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)

CH = "abcdefghijklmnopqrstuvwxyz"

def prep(p):
    im = Image.open(p).convert("L")
    a = np.asarray(im).astype(np.float64)
    g = np.round(a)
    return (g >= 156).astype(np.float32).reshape(1, 1, 40, 110)

# 按输出名数值排序
names = sorted(onames, key=lambda x: int(x))
print("排序后的输出名:", names)

head_stats = {n: {"conf": [], "blank": 0, "n": 0, "top2gap": []} for n in names}

mismatch = []
for k, t in gt.items():
    if k not in paths:
        continue
    x = prep(paths[k])
    out = sess.run(names, {iname: x})
    for n, o in zip(names, out):
        v = o[0]
        n_cls = v.shape[0]
        order = np.argsort(v)[::-1]
        top, second = order[0], order[1]
        # softmax 置信
        e = np.exp(v - v.max()); pr = e / e.sum()
        head_stats[n]["conf"].append(float(pr[top]))
        head_stats[n]["top2gap"].append(float(v[top] - v[second]))
        head_stats[n]["n"] += 1
        if top >= 26:
            head_stats[n]["blank"] += 1

print()
print(f"{'头':6s} {'类别数':>5s} {'平均置信':>9s} {'最低置信':>9s} {'blank数':>8s} {'top2差值均值':>12s}")
for n in names:
    st = head_stats[n]
    if not st["n"]:
        continue
    # 需要类别数：218-221 是 26，222 是 27
    nc = 26 if n in ("218", "219", "220", "221") else 27
    print(f"{n:6s} {nc:5d} {np.mean(st['conf'])*100:8.2f}% {np.min(st['conf'])*100:8.2f}% "
          f"{st['blank']:8d} {np.mean(st['top2gap']):12.3f}")

print()
print("=" * 80)
print("检查 4：第 4 位错误的样本，其第 4 头 top1 vs top2 的差距")
print("=" * 80)

err_keys = [k for k in gt if k in pred_base and gt[k] != pred_base[k]]
print(f"错误样本: {err_keys}")
for k in err_keys:
    if k not in paths:
        continue
    x = prep(paths[k])
    out = sess.run(names, {iname: x})
    t = gt[k]
    print(f"\n  [{k}] 真值={t}  预测={pred_base[k]}")
    for i, n in enumerate(names):
        v = out[i][0]
        order = np.argsort(v)[::-1]
        top, second = order[0], order[1]
        e = np.exp(v - v.max()); pr = e / e.sum()
        truth_ch = t[i] if i < len(t) else "?"
        mark = "  <-- 错" if (i < len(pred_base[k]) and i < len(t) and t[i] != pred_base[k][i]) else ""
        got = CH[top] if top < 26 else "<blank>"
        alt = CH[second] if second < 26 else "<blank>"
        print(f"     头{n} 位置{i+1}: top1={got}({pr[top]*100:5.1f}%)  top2={alt}({pr[second]*100:5.1f}%)"
              f"  差值={v[top]-v[second]:6.3f}  真值字符={truth_ch}{mark}")
