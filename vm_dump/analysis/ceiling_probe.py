import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""天花板判定：这 5 个错误到底还能不能修？

方法：把模型倒数第二层（penultimate feature）挖出来，对第 4 头做线性探测。
如果连"用全 220 张的标注重新拟合一个线性分类器"都修不好，说明特征层面已经不可分
—— 那就真的是这份数据 + 这个模型的天花板，除非重训。

同时验证一个实用替代方案：把错误样本的 top2/top3 全部当候选输出，
配合「置信度低就请求人工/换一张」（jAccount 本身支持换验证码）。
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
onames = [o.name for o in sess.get_outputs()]
names = sorted(onames, key=lambda x: int(x))
keys = [k for k in gt if k in paths]

print("=" * 86)
print("实验 A：把第 4 位的 26 类 logit 单独看 —— 真值字符的 logit 值有多低？")
print("=" * 86)
err_keys = sorted([k for k in keys if pred_base.get(k) != gt[k]])
for k in err_keys:
    t = gt[k]
    out = sess.run(names, {iname: prep(paths[k])})
    i = 3
    if i >= len(t):
        continue
    v = out[i][0]
    tv = CH.index(t[i])
    order = np.argsort(v)[::-1]
    top1 = order[0]
    print(f"  [{k}] 第4位 真值={t[i]}  logit={v[tv]:+.3f} (rank #{int(np.where(order==tv)[0][0])+1}) | "
          f"top1={CH[top1]} logit={v[top1]:+.3f} | 差={v[top1]-v[tv]:+.3f}")

print()
print("=" * 86)
print("实验 B：全体样本第 4 位 —— 真值 logit 与 top1 logit 的差距分布")
print("=" * 86)
gaps, gaps_err = [], []
for k in keys:
    t = gt[k]
    out = sess.run(names, {iname: prep(paths[k])})
    i = 3
    if i >= len(t):
        continue
    v = out[i][0]
    tv = CH.index(t[i])
    g = float(np.max(v) - v[tv])
    (gaps_err if pred_base.get(k) != gt[k] else gaps).append(g)
gaps = np.array(gaps); gaps_err = np.array(gaps_err)
print(f"  正确样本 (n={len(gaps)}) 第4位 logit 差距: 均值 {gaps.mean():.3f}  中位 {np.median(gaps):.3f}  最大 {gaps.max():.3f}")
print(f"  错误样本 (n={len(gaps_err)}) 第4位 logit 差距: 均值 {gaps_err.mean():.3f}  中位 {np.median(gaps_err):.3f}  最小 {gaps_err.min():.3f}")
print(f"  -> 分界点：正确样本最大差距 {gaps.max():.2f} vs 错误样本最小差距 {gaps_err.min():.2f}")

print()
print("=" * 86)
print("实验 C：把一个「完美线性头」装在第 4 位 —— 上限能到多少？（仅测第 4 位）")
print("=" * 86)


def feats(k):
    """取模型第 4 头输入前的特征：用 onnx 图中间张量。这里退而用 logit 本身做非线性特征。"""
    out = sess.run(names, {iname: prep(paths[k])})
    return out[3][0]


X, Y = [], []
for k in keys:
    t = gt[k]
    if len(t) <= 3:
        continue
    X.append(feats(k))
    Y.append(CH.index(t[3]))
X = np.array(X); Y = np.array(Y)
print(f"  样本 {len(X)} 条，特征维 {X.shape[1]}（第4头 logit）")

# 最近类中心分类器（Leave-one-out）
Xp = np.exp(X - X.max(axis=1, keepdims=True))
Xp = Xp / Xp.sum(axis=1, keepdims=True)
cent = np.zeros((26, Xp.shape[1]))
for c in range(26):
    idx = np.where(Y == c)[0]
    if len(idx):
        cent[c] = Xp[idx].mean(axis=0)
d = ((Xp[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
pred_nc = d.argmin(axis=1)
acc_nc = (pred_nc == Y).mean()
print(f"  最近类中心(对 logit 做) 第4位准确率: {acc_nc*100:.2f}%")

# softmax 后取最大
pred_sm = Xp.argmax(axis=1)
print(f"  直接 softmax argmax 第4位准确率: {(pred_sm == Y).mean()*100:.2f}%")

# 关键：正确样本与错误样本各自的类内可分性
print()
print("  第 4 位各字符的样本数（用于判断混淆对样本是否太少）:")
from collections import Counter
cnt = Counter(CH[y] for y in Y)
print("   ", dict(sorted(cnt.items())))

print()
print("=" * 86)
print("实验 D：错误样本的混淆对在训练数据里的对照样本量")
print("=" * 86)
pairs = [("x", "o"), ("c", "o"), ("y", "u"), ("w", "g"), ("g", "z")]
for a, b in pairs:
    ca, cb = cnt.get(a, 0), cnt.get(b, 0)
    print(f"  {a}->{b}  对照: 真值 {a} 出现 {ca} 次, 误判目标 {b} 出现 {cb} 次")
