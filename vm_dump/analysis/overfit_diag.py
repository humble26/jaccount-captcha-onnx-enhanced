import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""诊断：64 维特征是否过拟合？—— 决定"扩容量"是收益还是加剧过拟合

现象：原生头在位1/2/3/5 都是 100%，但用同一份 64 维特征在调参集上重训头，
      在留出集上只有 34%~72%。差距巨大。

解释假设：
  H1  特征对调参集过拟合（模型记住了训练样本），泛化到留出集时 64 维表示不可迁移
  H2  调参集与留出集存在分布差异（采集时间/字体渲染不同）
  H3  二者兼有

判据：比较调参集内部与留出集内部的"特征可分性"，以及跨集迁移性。
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

def ge(k):
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    return (g >= 156).astype(np.float32)

import onnx
m = onnx.load(os.path.join(VD, "nn_model.onnx"))
FEAT = "/Reshape_output_0"
if not any(o.name == FEAT for o in m.graph.output):
    vi = onnx.helper.ValueInfoProto(); vi.name = FEAT
    m.graph.output.append(vi)
TMP = os.path.join(VD, "_tmp_feat7.onnx"); onnx.save(m, TMP)
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(TMP, so, providers=["CPUExecutionProvider"])
ON = [o.name for o in sess.get_outputs()]; FI = ON.index(FEAT)

F = {}
for k in keys:
    F[k] = sess.run(ON, {"input.1": ge(k)[None, None]})[FI][0].astype(np.float64)

TUNE = sorted([k for k in keys if not k.startswith("h")])
HOLD = sorted([k for k in keys if k.startswith("h")])

print("=" * 88)
print("H1/H2 判别：64 维特征的集间可分性")
print("=" * 88)
print("  若特征对调参集过拟合，则调参集内部可分性 >> 留出集内部可分性")

def loo_acc(ids, pos):
    s = [k for k in ids if len(gt[k]) > pos]
    X = np.array([F[k] for k in s]); Y = np.array([gt[k][pos] for k in s])
    cls = sorted(set(Y))
    mu = {c: X[Y == c].mean(axis=0) for c in cls}
    ok = 0
    for i in range(len(X)):
        d = {c: np.linalg.norm(X[i] - mu[c]) for c in cls}
        ok += int(min(d, key=d.get) == Y[i])
    return ok / len(X), len(X)

print(f"  {'位置':6s} {'调参集内部':>14s} {'留出集内部':>14s}")
for pos in range(5):
    a, na = loo_acc(TUNE, pos)
    b, nb = loo_acc(HOLD, pos)
    print(f"  位{pos+1}: {a*100:9.2f}% ({na:3d}) {b*100:9.2f}% ({nb:3d})")

print()
print("=" * 88)
print("H2 判别：两集原始像素是否同分布")
print("=" * 88)
def stats(ids):
    ms = []
    for k in ids:
        a = ge(k)
        ms.append([a.mean(), a.std()])
    return np.array(ms)

st, sh = stats(TUNE), stats(HOLD)
print(f"  调参集 前景占比: {st[:,0].mean()*100:.2f}% ± {st[:,0].std()*100:.2f}%")
print(f"  留出集 前景占比: {sh[:,0].mean()*100:.2f}% ± {sh[:,0].std()*100:.2f}%")
print(f"  -> 差异 {abs(st[:,0].mean()-sh[:,0].mean())*100:.2f} 个百分点")

# 特征分布距离
Ft = np.array([F[k] for k in TUNE]); Fh = np.array([F[k] for k in HOLD])
print(f"  调参集特征范数均值 {np.linalg.norm(Ft,axis=1).mean():.2f}")
print(f"  留出集特征范数均值 {np.linalg.norm(Fh,axis=1).mean():.2f}")
# 中心距离（用调参集类均值预测留出集）
def cross_eval(pos):
    tr = [k for k in TUNE if len(gt[k]) > pos]
    te = [k for k in HOLD if len(gt[k]) > pos]
    Xtr = np.array([F[k] for k in tr]); Ytr = np.array([gt[k][pos] for k in tr])
    Xte = np.array([F[k] for k in te]); Yte = np.array([gt[k][pos] for k in te])
    mu = {c: Xtr[Ytr == c].mean(axis=0) for c in set(Ytr)}
    ok = 0
    for i in range(len(Xte)):
        d = {c: np.linalg.norm(Xte[i] - mu[c]) for c in mu}
        ok += int(min(d, key=d.get) == Yte[i])
    return ok / len(Xte)

print()
print("  跨集迁移（调参集类均值 -> 预测留出集）:")
for pos in range(5):
    a = cross_eval(pos)
    b, _ = loo_acc(HOLD, pos)
    print(f"    位{pos+1}: 跨集 {a*100:6.2f}%   集内 LOO {b*100:6.2f}%   差距 {(b-a)*100:+6.2f}%")

print()
print("=" * 88)
print("结论")
print("=" * 88)
print("  若「调参集内部 >> 留出集内部」或「跨集 << 集内」=> 特征过拟合/分布漂移")
print("  这意味着：单纯加宽 64 维向量会【加剧】过拟合，而不是提升泛化。")
print("  正确的方向是【增加数据】+ 【正则/增广】，而非【扩容量】。")
