import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""架构瓶颈的定量证明 + 突破方案可行性实测

已确认的事实：
  * backbone 是 ResNet-20 变体（21 Conv），最后 AveragePool 全局池化 -> Reshape 成 (1,64)
  * 5 个输出头（linear1~linear5，各 [26 or 27, 64]）**全部并联在同一个 64 维向量上**
  * 即：模型没有"位置专用特征"，而是把整张图压成 1 个 64 维向量，再靠 5 个线性头解码 5 个字符

架构级问题清单：
  1. 全局池化抹掉了空间信息 —— 而验证码的本质是"每个字符在特定位置"
  2. 64 维向量要承载 5 个独立字符（5 x log2(26) = 23.5 bits）
  3. 5 个头共享同一个特征，无法为"第 4 位"学专用表征

本脚本量化证明 #1 和 #2，并实测"切分式架构"能否突破。
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

def strokes(k):
    a = np.asarray(Image.open(paths[k]).convert("L")).astype(np.float64)
    return (np.round(a) < 156).astype(np.float32)

print("=" * 88)
print("【证明 1】全局池化抹掉空间信息 —— 第 4 位的信息在 64 维向量里还剩多少？")
print("=" * 88)

# 构建带中间输出的模型
import onnx
m = onnx.load(os.path.join(VD, "nn_model.onnx"))
g = m.graph
FEAT = "/Reshape_output_0"
vi = onnx.helper.ValueInfoProto(); vi.name = FEAT
if not any(o.name == FEAT for o in g.output):
    g.output.append(vi)
TMP = os.path.join(VD, "_tmp_feat2.onnx")
onnx.save(m, TMP)
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(TMP, so, providers=["CPUExecutionProvider"])
iname = "input.1"
onames = [o.name for o in sess.get_outputs()]
feat_idx = onames.index(FEAT)

def run(k):
    out = sess.run(onames, {iname: strokes(k).reshape(1, 1, 40, 110)})
    return {n: o for n, o in zip(onames, out)}

# 收集 64 维特征 + 第 4 位真值
F, Y4, K4 = [], [], []
for k in keys:
    if len(gt[k]) <= 3:
        continue
    r = run(k)
    F.append(r[FEAT][0]); Y4.append(gt[k][3]); K4.append(k)
F = np.array(F); Y4 = np.array(Y4)
print(f"  特征矩阵 {F.shape}（220 样本 x 64 维）")

# 特征里第 4 位字符的可分性：线性探测（留一法逻辑回归风格 —— 用类均值 + 协方差白化）
from collections import Counter
cnt4 = Counter(Y4)
print(f"  第 4 位字符种类 {len(cnt4)}  样本数分布 min={min(cnt4.values())} max={max(cnt4.values())}")

# 类均值
mu = np.array([F[Y4 == c].mean(axis=0) for c in sorted(set(Y4))])
classes = sorted(set(Y4))
print(f"  类均值矩阵 {mu.shape}")

# 总类均值 + 类内散度 Sw / 类间散度 Sb（Fisher 判别比）
overall = F.mean(axis=0)
Sw = np.zeros((64, 64)); Sb = np.zeros((64, 64))
for i, c in enumerate(classes):
    Xi = F[Y4 == c]
    d = Xi - mu[i]
    Sw += d.T @ d
    n = len(Xi)
    dm = (mu[i] - overall).reshape(-1, 1)
    Sb += n * (dm @ dm.T)
print(f"  类内散度 Sw trace {np.trace(Sw):.2f}   类间散度 Sb trace {np.trace(Sb):.2f}")
# Fisher 比：可用线性变换达到的最大类间/类内比
try:
    ev, _ = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    ev = np.sort(np.real(ev))[::-1]
    print(f"  Fisher 判别特征值 top10: {np.round(ev[:10], 3)}")
    print(f"  有效判别维数（特征值 >1 的个数）: {(ev>1).sum()}  <- 越高说明特征里第4位信息越可分")
except Exception as e:
    print("  特征值分解失败:", e)

# 更直接：用 64 维特征做最近类中心（留一），看准确率
print()
print("  用 64 维特征做最近类中心分类（留一法）:")
preds_lc = []
for i in range(len(F)):
    others = [j for j in range(len(F)) if j != i and Y4[j] == Y4[i]]
    # 用同类其他样本算均值
    mui = F[others].mean(axis=0) if others else F[i]
    d_to_true = np.linalg.norm(F[i] - mui)
    best, bestd = None, 1e18
    for c in classes:
        if c == Y4[i]:
            continue
        dd = np.linalg.norm(F[i] - mu[classes.index(c)])
        if dd < bestd:
            bestd, best = dd, c
    preds_lc.append(Y4[i] if d_to_true < bestd else best)
acc_lc = np.mean([p == t for p, t in zip(preds_lc, Y4)])
print(f"    特征空间最近类中心准确率: {acc_lc*100:.2f}%")

# 用模型自己的 logit
r4_acc = 0
for i, k in enumerate(K4):
    r = run(k)
    v = r["221"][0]
    r4_acc += int(CH[int(np.argmax(v))] == Y4[i])
print(f"    模型第4头 argmax 准确率: {r4_acc/len(K4)*100:.2f}%")

print()
print("=" * 88)
print("【证明 2】64 维向量同时承载 5 个字符 —— 容量是否饱和？")
print("=" * 88)
# 用 PCA 看 64 维里各主成分与各位置字符的相关性
Fc = F - F.mean(axis=0)
U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
print(f"  奇异值 top12: {np.round(S[:12], 2)}")
print(f"  方差解释比 top12: {np.round((S**2/np.sum(S**2))[:12], 4)}")
print(f"  累计方差解释（前 8 主成分）: {(S[:8]**2).sum()/np.sum(S**2)*100:.2f}%")
print(f"  累计方差解释（前 16 主成分）: {(S[:16]**2).sum()/np.sum(S**2)*100:.2f}%")
print("  -> 若少数主成分就能解释绝大部分方差，说明 64 维严重冗余；")
print("     若需要很多主成分，说明 64 维被 5 个位置的信息挤满。")

# 各位置的字符在第 4 位之外是否也影响同一个 64 维向量
print()
print("  各字符位置对 64 维特征的可预测性（用最近类中心，留一）:")
for pos in range(5):
    Xp, Yp = [], []
    for k in keys:
        if len(gt[k]) <= pos:
            continue
        r = run(k)
        Xp.append(r[FEAT][0]); Yp.append(gt[k][pos])
    Xp = np.array(Xp); Yp = np.array(Yp)
    cls = sorted(set(Yp))
    muP = {c: Xp[Yp == c].mean(axis=0) for c in cls}
    ok = 0
    for i in range(len(Xp)):
        oth = [j for j in range(len(Xp)) if j != i and Yp[j] == Yp[i]]
        mi = Xp[oth].mean(axis=0) if oth else Xp[i]
        dt = np.linalg.norm(Xp[i] - mi)
        bo = min((np.linalg.norm(Xp[i] - muP[c]), c) for c in cls if c != Yp[i])
        ok += int(dt < bo[0])
    print(f"    位{pos+1}: 样本 {len(Xp):3d}  特征空间可分率 {ok/len(Xp)*100:6.2f}%")
