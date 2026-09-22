import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""方案 C 收益上限估算 —— 在 64 维特征上做"扩容量 + 加深头"的等效实验

既然瓶颈是"64 维向量 + 单层线性头"，那么上限可以这样估：
  用当前 64 维特征，训练一个比单层线性更强的分类器（多层/非线性），
  看第 4 位准确率能从 97.73% 提升到多少。

注意：这不等价于重训整个网络（那会更强），但能给出一个有依据的下限估计。
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
TMP = os.path.join(VD, "_tmp_feat5.onnx"); onnx.save(m, TMP)
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(TMP, so, providers=["CPUExecutionProvider"])
ON = [o.name for o in sess.get_outputs()]; FI = ON.index(FEAT)

F = {}; L = {}
for k in keys:
    r = sess.run(ON, {"input.1": ge(k)[None, None]})
    F[k] = r[FI][0]
    L[k] = [o[0] for o in r[:5]]

print("=" * 88)
print("方案 C 收益上限估计 —— 在固定 64 维特征上换更强的分类器")
print("=" * 88)

# 用 onnxruntime 的输出顺序：218..222
def softmax(v):
    e = np.exp(v - v.max()); return e / e.sum()

# 基线：模型原生头
base_hit4 = 0; tot = 0
for k in keys:
    if len(gt[k]) <= 3: continue
    tot += 1
    idx = int(np.argmax(L[k][3]))
    if idx < 26 and CH[idx] == gt[k][3]: base_hit4 += 1
print(f"  基线（模型原生 linear4 头）: {base_hit4}/{tot} = {base_hit4/tot*100:.2f}%")

# 更强分类器：用 64 维特征训练 2 层 MLP（交替优化，简单梯度下降）
sel = [k for k in keys if len(gt[k]) > 3]
X = np.array([F[k] for k in sel], dtype=np.float64)
Y = np.array([CH.index(gt[k][3]) for k in sel])
print(f"  样本 {X.shape}，类别 {len(set(Y))}")

# 标准化
Xm, Xs = X.mean(0), X.std(0) + 1e-8
Xn = (X - Xm) / Xs

rng = np.random.default_rng(0)

def train_mlp(Xtr, Ytr, hid=128, epochs=3000, lr=0.05, seed=0):
    r = np.random.default_rng(seed)
    D = Xtr.shape[1]; C = 26
    W1 = r.normal(0, np.sqrt(2.0 / D), (D, hid))
    b1 = np.zeros(hid)
    W2 = r.normal(0, np.sqrt(2.0 / hid), (hid, C))
    b2 = np.zeros(C)
    Yoh = np.eye(C)[Ytr]
    for ep in range(epochs):
        H = np.maximum(0, Xtr @ W1 + b1)
        Z = H @ W2 + b2
        Ze = np.exp(Z - Z.max(1, keepdims=True)); P = Ze / Ze.sum(1, keepdims=True)
        dZ = (P - Yoh) / len(Xtr)
        gW2 = H.T @ dZ; gb2 = dZ.sum(0)
        dH = dZ @ W2.T
        dH[H <= 0] = 0
        gW1 = Xtr.T @ dH; gb1 = dH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2

def pred_mlp(par, Xq):
    W1, b1, W2, b2 = par
    H = np.maximum(0, Xq @ W1 + b1)
    return (H @ W2 + b2).argmax(1)

# 5 折交叉验证
idx_all = np.arange(len(sel))
rng.shuffle(idx_all)
K = 5
folds = np.array_split(idx_all, K)
accs = []
for fi in range(K):
    te = folds[fi]
    tr = np.concatenate([folds[j] for j in range(K) if j != fi])
    par = train_mlp(Xn[tr], Y[tr], hid=128, epochs=2500, lr=0.05, seed=fi)
    p = pred_mlp(par, Xn[te])
    accs.append((p == Y[te]).mean())
print(f"  2 层 MLP (64->128->26) 5 折交叉验证: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

# 对照：线性头（逻辑回归）
def train_lin(Xtr, Ytr, epochs=3000, lr=0.1, seed=0):
    D = Xtr.shape[1]; C = 26
    W = np.zeros((D, C)); b = np.zeros(C)
    Yoh = np.eye(C)[Ytr]
    for ep in range(epochs):
        Z = Xtr @ W + b
        Ze = np.exp(Z - Z.max(1, keepdims=True)); P = Ze / Ze.sum(1, keepdims=True)
        dZ = (P - Yoh) / len(Xtr)
        W -= lr * (Xtr.T @ dZ); b -= lr * dZ.sum(0)
    return W, b

accs_lin = []
for fi in range(K):
    te = folds[fi]
    tr = np.concatenate([folds[j] for j in range(K) if j != fi])
    W, b = train_lin(Xn[tr], Y[tr], seed=fi)
    p = (Xn[te] @ W + b).argmax(1)
    accs_lin.append((p == Y[te]).mean())
print(f"  单层线性头（复现模型结构）5 折: {np.mean(accs_lin)*100:.2f}% ± {np.std(accs_lin)*100:.2f}%")

# 直接把 5 个头的 logit 拼接再分类（集成）
Xall = np.array([np.concatenate([softmax(L[k][i]) for i in range(5)]) for k in sel])
print(f"  5 头 logit 拼接特征 {Xall.shape}")

Xalln = (Xall - Xall.mean(0)) / (Xall.std(0) + 1e-8)
accs_cat = []
for fi in range(K):
    te = folds[fi]
    tr = np.concatenate([folds[j] for j in range(K) if j != fi])
    par = train_mlp(Xalln[tr], Y[tr], hid=64, epochs=2500, lr=0.05, seed=fi)
    p = pred_mlp(par, Xalln[te])
    accs_cat.append((p == Y[te]).mean())
print(f"  5 头 logit + MLP 5 折: {np.mean(accs_cat)*100:.2f}% ± {np.std(accs_cat)*100:.2f}%")

print()
print("=" * 88)
print("解读")
print("=" * 88)
print(f"  原生头        : {base_hit4/tot*100:.2f}%")
print(f"  线性头(重训)  : {np.mean(accs_lin)*100:.2f}%")
print(f"  2层MLP(重训)  : {np.mean(accs)*100:.2f}%")
print(f"  5头logit+MLP  : {np.mean(accs_cat)*100:.2f}%")
print()
print("  若 MLP 显著优于线性 => 加深头有收益，方案 C 有效")
print("  若差不多          => 64 维特征本身已饱和，加深头无用，需回到 backbone")
