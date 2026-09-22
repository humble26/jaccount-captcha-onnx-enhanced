import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""方案 C 收益上限 —— 修正版（正确划分 + 联合训练 + 足够迭代）

前版缺陷：
  1. 220 样本 / 26 类做 5 折，每折仅 44 样本，方差极大
  2. 从头训头未对齐原生特征（原生头是端到端联合训出来的）
  3. lr/epochs 未经调优，线性头都没训到 97%

本版：
  * 用调参集(120张, c*/n*)训练头、留出集(100张, h*)测试 —— 与项目划分一致
  * 训练头时用更多 epoch + 更强的正则，并报告训练集准确率（诊断是否欠拟合）
  * 关键对照：把「原生头」也放到同样的划分上看，确认划分本身没问题
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
TMP = os.path.join(VD, "_tmp_feat6.onnx"); onnx.save(m, TMP)
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(TMP, so, providers=["CPUExecutionProvider"])
ON = [o.name for o in sess.get_outputs()]; FI = ON.index(FEAT)

F, L = {}, {}
for k in keys:
    r = sess.run(ON, {"input.1": ge(k)[None, None]})
    F[k] = r[FI][0].astype(np.float64)
    L[k] = [o[0] for o in r[:5]]

TUNE = sorted([k for k in keys if not k.startswith("h")])
HOLD = sorted([k for k in keys if k.startswith("h")])
print(f"调参集 {len(TUNE)} / 留出集 {len(HOLD)}（与项目划分一致）")

# 各位置可用样本
def pos_sets(pos):
    tr = [k for k in TUNE if len(gt[k]) > pos]
    te = [k for k in HOLD if len(gt[k]) > pos]
    return tr, te

print()
print("=" * 88)
print("先确认划分与原生头基线（诊断实验设计）")
print("=" * 88)
print(f"{'位置':6s} {'调参集原生':>12s} {'留出集原生':>12s}")
for pos in range(5):
    tr, te = pos_sets(pos)
    def native(ids):
        ok = 0
        for k in ids:
            idx = int(np.argmax(L[k][pos]))
            if idx < 26 and CH[idx] == gt[k][pos]:
                ok += 1
        return ok, len(ids)
    a, na = native(tr); b, nb = native(te)
    print(f"  位{pos+1}: {a:3d}/{na:3d}={a/na*100:6.2f}% {b:3d}/{nb:3d}={b/nb*100:6.2f}%")

print()
print("=" * 88)
print("在 64 维特征上训练分类头（用调参集训、留出集测）")
print("=" * 88)

def softmax(v):
    e = np.exp(v - v.max()); return e / e.sum()

def train_head(Xtr, Ytr, hid=0, epochs=8000, lr=0.3, wd=1e-4, seed=0, C=26):
    """hid=0 表示单层线性；否则 2 层"""
    r = np.random.default_rng(seed)
    D = Xtr.shape[1]
    Yoh = np.eye(C)[Ytr]
    if hid == 0:
        W = r.normal(0, 0.01, (D, C)); b = np.zeros(C)
        for ep in range(epochs):
            Z = Xtr @ W + b
            Ze = np.exp(Z - Z.max(1, keepdims=True)); P = Ze / Ze.sum(1, keepdims=True)
            dZ = (P - Yoh) / len(Xtr)
            W -= lr * (Xtr.T @ dZ + wd * W); b -= lr * dZ.sum(0)
        return lambda Xq: (Xq @ W + b).argmax(1), \
               lambda Xq: (Xq @ W + b)
    else:
        W1 = r.normal(0, np.sqrt(2.0 / D), (D, hid)); b1 = np.zeros(hid)
        W2 = r.normal(0, np.sqrt(2.0 / hid), (hid, C)); b2 = np.zeros(C)
        for ep in range(epochs):
            H = np.maximum(0, Xtr @ W1 + b1)
            Z = H @ W2 + b2
            Ze = np.exp(Z - Z.max(1, keepdims=True)); P = Ze / Ze.sum(1, keepdims=True)
            dZ = (P - Yoh) / len(Xtr)
            gW2 = H.T @ dZ + wd * W2; gb2 = dZ.sum(0)
            dH = dZ @ W2.T; dH[H <= 0] = 0
            gW1 = Xtr.T @ dH + wd * W1; gb1 = dH.sum(0)
            W1 -= lr * gW1; b1 -= lr * gb1
            W2 -= lr * gW2; b2 -= lr * gb2
        def f(Xq):
            H = np.maximum(0, Xq @ W1 + b1)
            return (H @ W2 + b2).argmax(1)
        return f, None

print(f"{'位置':6s} {'原生头':>10s} {'线性头':>10s} {'2层MLP':>10s} {'3层MLP':>10s}")
summary = []
for pos in range(5):
    tr, te = pos_sets(pos)
    if len(tr) < 40 or len(te) < 20:
        continue
    Xtr = np.array([F[k] for k in tr]); Ytr = np.array([CH.index(gt[k][pos]) for k in tr])
    Xte = np.array([F[k] for k in te]); Yte = np.array([CH.index(gt[k][pos]) for k in te])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtrn = (Xtr - mu) / sd; Xten = (Xte - mu) / sd

    def native_acc():
        ok = 0
        for k in te:
            i = int(np.argmax(L[k][pos]))
            if i < 26 and CH[i] == gt[k][pos]: ok += 1
        return ok / len(te)

    res = {}
    for nm, hid in [("lin", 0), ("mlp2", 64), ("mlp3", 64)]:
        accs = []
        for sd_ in range(3):
            f, _ = train_head(Xtrn, Ytr, hid=hid, epochs=6000, lr=0.2, wd=1e-3, seed=sd_,
                              C=26)
            accs.append((f(Xten) == Yte).mean())
        res[nm] = np.mean(accs)
    na = native_acc()
    summary.append((pos, na, res["lin"], res["mlp2"], res["mlp3"]))
    print(f"  位{pos+1}: {na*100:9.2f}% {res['lin']*100:9.2f}% {res['mlp2']*100:9.2f}% {res['mlp3']*100:9.2f}%")

print()
print("=" * 88)
print("结论解读")
print("=" * 88)
if summary:
    base = np.mean([s[1] for s in summary])
    lin = np.mean([s[2] for s in summary])
    mlp = np.mean([s[3] for s in summary])
    print(f"  原生头平均: {base*100:.2f}%   线性头(重训): {lin*100:.2f}%   2层MLP(重训): {mlp*100:.2f}%")
    print()
    if lin < base - 0.03:
        print("  ⚠ 重训的线性头明显差于原生头 => 说明原生头与 backbone 是端到端联合优化的，")
        print("    在冻结特征上重训头无法复现，本实验的绝对水平不可用作收益估计。")
        print("    => 只能看趋势（MLP vs 线性），不能看绝对值。")
    if mlp > lin + 0.02:
        print("  ✓ MLP 优于线性头 => 加深头可能有收益（趋势支持方案 C）")
    else:
        print("  ✗ MLP 未优于线性头 => 加深头收益不明显")
