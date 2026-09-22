import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""逐张量比较原生 ONNX 与回写 ONNX 的权重差异。

目的：报告称「微调 npz 累计绝对差 1.76e-4 / 最大单权重差 3.63e-05」，
但 onnxruntime 实测两者 logits 最大绝对差达 0.68 —— 量级不自洽，需要定论。
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper

W = _VD
MODEL_NATIVE = os.path.join(W, 'nn_model.onnx')
MODEL_E2E = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\_e2e_RT_backport\nn_model_e2e.onnx'

da = onnx.load(MODEL_NATIVE)
db = onnx.load(MODEL_E2E)
wa = {i.name: numpy_helper.to_array(i) for i in da.graph.initializer}
wb = {i.name: numpy_helper.to_array(i) for i in db.graph.initializer}

print('原生 initializer %d 个 | 回写 %d 个' % (len(wa), len(wb)))
onlya = set(wa) - set(wb)
onlyb = set(wb) - set(wa)
print('仅原生有:', onlya)
print('仅回写有:', onlyb)
print()

rows = []
for k in sorted(set(wa) & set(wb)):
    A = wa[k].astype(np.float64)
    B = wb[k].astype(np.float64)
    if A.shape != B.shape:
        print('SHAPE MISMATCH', k, A.shape, B.shape)
        continue
    d = np.abs(A - B)
    scale = max(float(np.abs(A).max()), 1e-12)
    rows.append((k, A.size, float(d.max()), float(d.mean()), scale,
                 float(d.max() / scale)))

print('%-26s %8s %12s %12s %12s %10s' % ('张量', '元素数', 'max|Δ|', 'mean|Δ|', 'max|w|', '相对'))
for k, n, mx, mn, sc, rel in sorted(rows, key=lambda r: -r[2])[:25]:
    print('%-26s %8d %12.3e %12.3e %12.3e %9.2e' % (k, n, mx, mn, sc, rel))

print()
allmx = max(r[2] for r in rows)
allmean = float(np.mean([r[3] for r in rows]))
tot = sum(r[2] for r in rows)
print('全局 max|Δ|            = %.3e   （报告称 3.63e-05）' % allmx)
print('逐张量 max|Δ| 之和     = %.3e   （报告称「累计绝对差 1.76e-4」）' % tot)
print('逐张量 mean|Δ| 的平均  = %.3e' % allmean)
print('改动最大的张量数（max|Δ|>1e-6）: %d / %d' % (sum(1 for r in rows if r[2] > 1e-6), len(rows)))
print('改动最大的张量数（max|Δ|>1e-3）: %d / %d' % (sum(1 for r in rows if r[2] > 1e-3), len(rows)))

# 相对改动量：微调是否真的"几乎没动"
rels = sorted(rows, key=lambda r: -r[5])
print()
print('相对改动（max|Δ| / max|w|）最大的 10 个张量：')
for k, n, mx, mn, sc, rel in rels[:10]:
    print('  %-26s 相对 %.3e  (max|Δ|=%.3e, max|w|=%.3e)' % (k, rel, mx, sc))
print('相对改动中位数: %.3e' % float(np.median([r[5] for r in rows])))

# 头层（直接决定 argmax 的地方）
print()
print('5 个输出头权重差异：')
for i in range(1, 6):
    for t in ('weight', 'bias'):
        k = 'linear%d.%s' % (i, t)
        if k in wa:
            A = wa[k].astype(np.float64); B = wb[k].astype(np.float64)
            d = np.abs(A - B)
            print('  %-16s max|Δ|=%.3e  max|w|=%.3e  相对=%.2e'
                  % (k, d.max(), np.abs(A).max(), d.max() / max(np.abs(A).max(), 1e-12)))
