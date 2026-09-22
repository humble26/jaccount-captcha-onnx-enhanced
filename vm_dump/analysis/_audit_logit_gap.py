import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""解释「权重差 3.6e-5 却导致 logits 差 0.68」这一矛盾。

三条线索：
 1) 两个 ONNX 的 initializer dtype 是否一致（float32 vs float64 混算会改数值路径）
 2) 图结构是否完全一致（节点数 / 算子类型分布）
 3) 用 NumPy float64 跑同一批图：权重差造成的 logits 差到底有多大
    （float64 无累积舍入，可把「权重更新效应」与「float32 数值噪声」分开）
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper
from PIL import Image
import onnxruntime as ort

W = _VD
A_PATH = os.path.join(W, 'nn_model.onnx')
B_PATH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\_e2e_RT_backport\nn_model_e2e.onnx'
ANALYSIS = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis'

print('== 1) initializer dtype ==')
for tag, p in (('原生', A_PATH), ('回写', B_PATH)):
    m = onnx.load(p)
    from collections import Counter
    c = Counter(str(numpy_helper.to_array(i).dtype) for i in m.graph.initializer)
    print('  %s: %s' % (tag, dict(c)))
    print('      graph 节点数 %d | 输入 %s | 输出 %s'
          % (len(m.graph.node),
             [i.name for i in m.graph.input],
             [o.name for o in m.graph.output]))
    ops = Counter(n.op_type for n in m.graph.node)
    print('      算子分布:', dict(sorted(ops.items())))

print()
print('== 2) 图是否逐节点一致 ==')
ma, mb = onnx.load(A_PATH), onnx.load(B_PATH)
sa = [(n.op_type, tuple(n.input), tuple(n.output)) for n in ma.graph.node]
sb = [(n.op_type, tuple(n.input), tuple(n.output)) for n in mb.graph.node]
print('  节点序列完全一致:', sa == sb)
if sa != sb:
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x != y:
            print('   首个差异 @%d: %s  vs  %s' % (i, x, y))
            break

print()
print('== 3) float64 纯 NumPy 下，两套权重的 logits 差 ==')
import sys
sys.path.insert(0, ANALYSIS)
import resnet20_np as RN

Wa = {k: v.astype(np.float64) for k, v in RN.load_onnx_weights(A_PATH).items()}
Wb = {k: v.astype(np.float64) for k, v in RN.load_onnx_weights(B_PATH).items()}


def prep(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float64).reshape(1, 1, 40, 110)


def decode_head(logits):
    idx = [int(np.argmax(z[0])) for z in logits]
    s = ''.join(chr(ord('a') + c) for c in idx[:4])
    if idx[4] != 26:
        s += chr(ord('a') + idx[4])
    return s


files = []
for d in ('samples', 'samples2', 'holdout', 'new300'):
    dd = os.path.join(W, d)
    if os.path.isdir(dd):
        files += sorted(os.path.join(dd, f) for f in os.listdir(dd) if f.endswith('.png'))
files = files[::5]

# onnxruntime 对照
so = ort.SessionOptions(); so.log_severity_level = 3
oa = ort.InferenceSession(A_PATH, so, providers=['CPUExecutionProvider'])
ob = ort.InferenceSession(B_PATH, so, providers=['CPUExecutionProvider'])
names = sorted([o.name for o in oa.get_outputs()], key=int)
iname = oa.get_inputs()[0].name

md_np64, md_ort, ndiff64, ndiff_ort = [], [], 0, 0
for f in files:
    x = prep(f)
    la = RN.forward_heads(RN.forward_fast(x, Wa)['feat'], Wa)
    lb = RN.forward_heads(RN.forward_fast(x, Wb)['feat'], Wb)
    d64 = max(float(np.abs(np.asarray(p) - np.asarray(q)).max()) for p, q in zip(la, lb))
    md_np64.append(d64)
    if decode_head(la) != decode_head(lb):
        ndiff64 += 1
    x32 = x.astype(np.float32)
    za = oa.run(names, {iname: x32}); zb = ob.run(names, {iname: x32})
    dort = max(float(np.abs(p - q).max()) for p, q in zip(za, zb))
    md_ort.append(dort)
    if decode_head(za) != decode_head(zb):
        ndiff_ort += 1

print('  样本数: %d' % len(files))
print('  float64 NumPy 路径: logits max|Δ| 中位=%.3e 最大=%.3e | 预测不同 %d 张'
      % (float(np.median(md_np64)), max(md_np64), ndiff64))
print('  onnxruntime 路径 : logits max|Δ| 中位=%.3e 最大=%.3e | 预测不同 %d 张'
      % (float(np.median(md_ort)), max(md_ort), ndiff_ort))

print()
print('== 4) logits 自身量级（判断 0.45 是相对还是绝对大）==')
x = prep(files[0]).astype(np.float32)
z = oa.run(names, {iname: x})
for i, zz in enumerate(z):
    print('  头%d logits: min=%.2f max=%.2f 幅度=%.2f' % (i + 1, zz.min(), zz.max(), zz.max() - zz.min()))
print('  -> 若 logits 幅度达数十，则 0.4 的绝对差为百分比级 float32 噪声')
