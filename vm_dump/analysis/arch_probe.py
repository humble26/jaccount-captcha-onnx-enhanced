import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""架构突破可能性评估 —— 从信息论与实测两条线并进

问题：当前 ResNet-20 + 40x110 二值图 的架构，是否存在"信息瓶颈"？
      这类瓶颈靠换骨架能突破，靠调参不能。

三条证据链：
  A. 输入信息量核算 —— 5 个字符位，每位需要 log2(26) 位信息。当前输入像素够不够承载？
  B. 像素级可分性 —— 那 5 个错误样本的第 4 位，在原始像素空间与正确样本是否可分？
     若像素空间可分而模型 logit 不可分 => 是"模型容量/结构"问题 => 换骨架有救
     若像素空间也不可分 => 是"数据/分辨率"问题 => 换骨架没救，只能改采集
  C. 头部拓扑分析 —— 当前是 5 个独立 26 类 softmax（多任务硬切分），
     是否有更优的架构范式？（CTC / 序列解码 / 共享-特化混合）
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

so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
names = sorted([o.name for o in sess.get_outputs()], key=lambda x: int(x))
keys = [k for k in gt if k in paths]

print("=" * 88)
print("【A】输入信息量核算")
print("=" * 88)
print(f"  输入张量: [1, 1, 40, 110] = {40*110} 个二值像素")
print(f"  二值图理论信息上限: {40*110} bits (每像素 1 bit)")
print(f"  但要识别的目标: 5 字符 x log2(26) = {5*np.log2(26):.2f} bits")
print(f"  4 字符码:        4 x log2(26) = {4*np.log2(26):.2f} bits")
print(f"  -> 朴素结论: 4400 bits 输入 vs ~23.5 bits 目标，输入冗余 {4400/23.5:.0f} 倍")
print()
# 实际有效像素：字符只占一部分
masks = []
for k in keys:
    a = np.asarray(Image.open(paths[k]).convert("L")).astype(np.float64)
    masks.append((np.round(a) >= 156))
masks = np.array(masks)
print(f"  但真实笔画墨迹占比: 均值 {masks.mean()*100:.2f}%  ->  有效像素约 {masks.mean()*4400:.0f} 个/张")
print(f"  每张图有效像素承载 {5*np.log2(26)/max(masks.mean()*4400,1):.3f} bits/像素（远未饱和）")
print()
# 40x110 是"偏高太矮" —— 打印每行/列的平均墨迹，看纵向是否被压扁
row_prof = masks.mean(axis=(0, 2))   # 40 行
col_prof = masks.mean(axis=(0, 1))   # 110 列
print("  行方向（高度 40）墨迹分布，看纵向信息密度:")
nz = np.where(row_prof > 0.001)[0]
print(f"    有墨迹的行: {nz.min()}~{nz.max()} (共 {len(nz)} 行)  峰值行墨迹 {row_prof.max()*100:.1f}%")
print(f"    行墨迹标准差 {row_prof.std():.4f}  列墨迹标准差 {col_prof.std():.4f}")
print(f"    -> 列方向变化远大于行方向: 列 std / 行 std = {col_prof.std()/max(row_prof.std(),1e-9):.2f}")
print("       说明水平方向（字符排布）信息丰富，垂直方向（字形笔画）被严重压缩")

print()
print("=" * 88)
print("【B】像素级可分性检验 —— 决定换骨架有没有救")
print("=" * 88)

# 取第 4 槽区域（40x110 图中，5字符码第4槽中心 x=68.8，4字符码第4槽中心 x=74.9）
def slot_crop(k):
    a = np.asarray(Image.open(paths[k]).convert("L")).astype(np.float64)
    b = (np.round(a) >= 156).astype(np.float64)
    t = gt[k]
    cx = 68.8 if len(t) == 5 else 74.9
    # 取一列字符宽（约 18 列）
    lo, hi = int(cx - 11), int(cx + 11)
    return b[:, max(0, lo):min(110, hi)]

# 第 4 位真值字符
X4, Y4 = [], []
for k in keys:
    t = gt[k]
    if len(t) <= 3:
        continue
    c = slot_crop(k)
    if c.shape[1] < 22:
        pad = np.zeros((40, 22 - c.shape[1]))
        c = np.hstack([c, pad])
    X4.append(c.ravel())
    Y4.append(t[3])
X4 = np.array(X4); Y4 = np.array(Y4)
print(f"  第 4 位字符样本 {len(X4)} 条，特征维 {X4.shape[1]} (40 x 22 二值像素)")

# 各类质心与类内散度
cent = {}
for c in set(Y4):
    idx = np.where(Y4 == c)[0]
    cent[c] = X4[idx].mean(axis=0)
within = np.mean([np.linalg.norm(X4[i] - cent[Y4[i]]) for i in range(len(X4))])
print(f"  平均类内距离（到自身类质心）: {within:.3f}")
print()
others = {}
for c in set(Y4):
    oth = [np.linalg.norm(cent[c] - cent[d]) for d in set(Y4) if d != c]
    others[c] = (np.mean(oth), min(oth))
print("  易混淆字符的类间距离（与最接近的类）:")
for c in ["x", "o", "c", "y", "u", "w", "g", "z"]:
    if c in others:
        print(f"    {c}: 平均类间 {others[c][0]:8.3f}  最近类间距 {others[c][1]:8.3f}")

# 关键判定：错误样本第 4 槽，与真值类质心 vs 与预测类质心的距离
print()
print("  错误样本在第 4 槽的像素距离（决定像素空间是否可分）:")
err_keys = sorted([k for k in keys if pred_base.get(k) != gt[k]])
for k in err_keys:
    t = gt[k]
    if len(t) <= 3:
        continue
    c = slot_crop(k)
    if c.shape[1] < 22:
        c = np.hstack([c, np.zeros((40, 22 - c.shape[1]))])
    v = c.ravel()
    tv = t[3]
    pv = pred_base[k][3]
    dt = np.linalg.norm(v - cent[tv]) if tv in cent else -1
    dp = np.linalg.norm(v - cent[pv]) if pv in cent else -1
    verdict = "像素空间可分 (距真值更近)" if dt < dp else "像素空间不可分 (距误判类更近)"
    print(f"    [{k}] 真值={tv} 预测={pv}  到真值类质心 {dt:8.3f}  到误判类质心 {dp:8.3f}  -> {verdict}")

print()
print("  对照：正确样本的距离差分布（真值距离 - 最近其他类距离）")
margins = []
for k in keys:
    t = gt[k]
    if len(t) <= 3 or k in err_keys:
        continue
    c = slot_crop(k)
    if c.shape[1] < 22:
        c = np.hstack([c, np.zeros((40, 22 - c.shape[1]))])
    v = c.ravel()
    tv = t[3]
    dt = np.linalg.norm(v - cent[tv])
    do = min([np.linalg.norm(v - cent[d]) for d in cent if d != tv])
    margins.append(do - dt)   # 正数=距真值更近
margins = np.array(margins)
print(f"    正确样本 margin: 均值 {margins.mean():.3f}  中位 {np.median(margins):.3f}  最小 {margins.min():.3f}")
print(f"    margin <= 0 的正确样本数: {(margins<=0).sum()} / {len(margins)}")

print()
print("=" * 88)
print("【C】头部拓扑分析")
print("=" * 88)
try:
    import onnx
    m = onnx.load(os.path.join(VD, "nn_model.onnx"))
    g = m.graph
    print("  输出头:")
    for o in g.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}  {dims}")
    # 找最后的 Gemm/MatMul 层，看共享结构
    import collections
    ops = collections.Counter(n.op_type for n in g.node)
    print()
    print(f"  算子统计 (共 {len(g.node)} 节点):")
    for op, c in ops.most_common():
        print(f"    {op:16s} {c}")
    # 找出 5 个头各自的最后线性层，看是否共享 backbone
    gemms = [n for n in g.node if n.op_type in ("Gemm", "MatMul")]
    print()
    print(f"  线性层 (Gemm/MatMul) 共 {len(gemms)} 个 —— 若为 5 个独立输出层则说明头是并联独立")
    for n in gemms:
        print(f"    {n.name or '(unnamed)'}  in={list(n.input)}  out={list(n.output)}")
except Exception as e:
    print("  onnx 解析失败:", e)
