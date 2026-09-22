import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""决定性检验：回写验证用的是「改动最小」的 step 25 权重（best_hold4），
而不是 step 300 的最终权重。若换成最终权重回写后预测发生变化，
则「回写无价值」这一结论依赖于选择偏差。

步骤：
  1) 比较两个 npz 的权重差异（step 25 vs step 300）
  2) 用 step 300 权重回写一份 ONNX
  3) 520 张上对比：原生 / 回写(step25，官方那份) / 回写(step300)
"""
import os
import json
import numpy as np
import onnx
from onnx import numpy_helper
from PIL import Image
import onnxruntime as ort

W = _VD
ARCH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究'
E2E_OUT = os.path.join(ARCH, 'analysis', '_e2e_out')
NATIVE = os.path.join(W, 'nn_model.onnx')
OFFICIAL = os.path.join(ARCH, '_e2e_RT_backport', 'nn_model_e2e.onnx')
MINE = os.path.join(W, 'audit_sheets', 'nn_model_step300.onnx')


def load_npz(p):
    d = np.load(p)
    meta = json.load(open(p + '.meta.json', encoding='utf-8'))
    return {k: d['w%d' % i] for i, k in enumerate(meta['keys'])}, meta


P25 = os.path.join(E2E_OUT, 'e2e_final_extra_best_hold4.npz')
P300 = os.path.join(E2E_OUT, 'e2e_final_extra.npz')
w25, m25 = load_npz(P25)
w300, m300 = load_npz(P300)

print('== 1) 两个 npz 的差异（step25 vs step300）==')
rows = []
for k in w25:
    if k not in w300:
        print('  仅 step25 有:', k); continue
    a = np.abs(w25[k].astype(np.float64) - w300[k].astype(np.float64))
    rows.append((k, float(a.max()), float(a.mean())))
rows.sort(key=lambda r: -r[1])
print('  %-26s %12s %12s' % ('张量', 'max|Δ|', 'mean|Δ|'))
for k, mx, mn in rows[:12]:
    print('  %-26s %12.3e %12.3e' % (k, mx, mn))
print('  全局 max|Δ| = %.3e | 逐张量 max|Δ| 之和 = %.3e' % (max(r[1] for r in rows), sum(r[1] for r in rows)))

# 原生 → step300 的差异
nw = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in onnx.load(NATIVE).graph.initializer}
r2 = [(k, float(np.abs(nw[k] - w300[k].astype(np.float64)).max())) for k in w300 if k in nw]
r2.sort(key=lambda x: -x[1])
print()
print('  原生 → step300: 全局 max|Δ| = %.3e | 之和 = %.3e' % (max(x[1] for x in r2), sum(x[1] for x in r2)))
print('  原生 → step25 : 全局 max|Δ| = %.3e | 之和 = %.3e'
      % (max(float(np.abs(nw[k] - w25[k].astype(np.float64)).max()) for k in w25 if k in nw),
         sum(float(np.abs(nw[k] - w25[k].astype(np.float64)).max()) for k in w25 if k in nw)))

print()
print('== 2) 用 step300 权重回写 ONNX ==')
m = onnx.load(NATIVE)
ini = {i.name: i for i in m.graph.initializer}
rep = 0
for k, v in w300.items():
    if k not in ini:
        continue
    arr = np.asarray(v).astype(np.float32)
    if numpy_helper.to_array(ini[k]).shape != arr.shape:
        print('  形状不一致，跳过', k); continue
    ini[k].CopyFrom(numpy_helper.from_array(arr, k))
    rep += 1
onnx.save(m, MINE)
print('  已回写 %d 个张量 -> %s (%d bytes)' % (rep, MINE, os.path.getsize(MINE)))

print()
print('== 3) 三模型预测对比（520 张）==')
so = ort.SessionOptions(); so.log_severity_level = 3
s_nat = ort.InferenceSession(NATIVE, so, providers=['CPUExecutionProvider'])
s_off = ort.InferenceSession(OFFICIAL, so, providers=['CPUExecutionProvider'])
s_mine = ort.InferenceSession(MINE, so, providers=['CPUExecutionProvider'])
names = sorted([o.name for o in s_nat.get_outputs()], key=int)
iname = s_nat.get_inputs()[0].name


def prep(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32).reshape(1, 1, 40, 110)


def dec(zo):
    idx = [int(np.argmax(z[0])) for z in zo]
    s = ''.join(chr(97 + c) for c in idx[:4])
    if idx[4] != 26:
        s += chr(97 + idx[4])
    return s


files = []
for d in ('samples', 'samples2', 'holdout', 'new300'):
    dd = os.path.join(W, d)
    files += sorted(os.path.join(dd, f) for f in os.listdir(dd) if f.endswith('.png'))

n = {'off': 0, 'mine': 0}
maxd = {'off': 0.0, 'mine': 0.0}
changes = []
for f in files:
    x = prep(f)
    za = s_nat.run(names, {iname: x})
    zb = s_off.run(names, {iname: x})
    zc = s_mine.run(names, {iname: x})
    pa, pb, pc = dec(za), dec(zb), dec(zc)
    maxd['off'] = max(maxd['off'], max(float(np.abs(p - q).max()) for p, q in zip(za, zb)))
    maxd['mine'] = max(maxd['mine'], max(float(np.abs(p - q).max()) for p, q in zip(za, zc)))
    if pb != pa:
        n['off'] += 1
    if pc != pa:
        n['mine'] += 1
        changes.append((os.path.basename(f), pa, pc))

print('  样本数 %d' % len(files))
print('  回写(step25  官方): 预测变化 %d 张 | logits max|Δ| = %.3e' % (n['off'], maxd['off']))
print('  回写(step300 本次): 预测变化 %d 张 | logits max|Δ| = %.3e' % (n['mine'], maxd['mine']))
for c in changes[:15]:
    print('     %s 原生=%s step300=%s' % c)
