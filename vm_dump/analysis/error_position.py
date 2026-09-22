import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""验证一个惊人的规律：5 个错误是否全部集中在第 4 个字符位置？

如果成立，说明问题不是"模型整体不行"，而是某个特定位置（或某种特定
图结构）系统性地偏移。那就有很明确的改进方向。
"""
import os, json
import numpy as np
from PIL import Image
from collections import Counter

WS = _REPO

print("=" * 80)
print("验证 1：全部 220 张的错误，按字符位置分布")
print("=" * 80)

h = json.load(open(os.path.join(WS, "vm_dump", "holdout_eval.json"), encoding="utf-8"))
gt = h["gt"]; pred = h["pred"]["base"]
common = sorted(set(gt) & set(pred))

pos_counter = Counter()          # 出错的位置计数
pos_total = Counter()            # 每个位置出现的总次数（分母）
per_pos_err = {}                 # 位置 -> 错误样本

for k in common:
    t, p = gt[k], pred[k]
    for i, ch in enumerate(t):
        pos_total[i] += 1
    if t != p and len(t) == len(p):
        for i, (a, b) in enumerate(zip(t, p)):
            if a != b:
                pos_counter[i] += 1
                per_pos_err.setdefault(i, []).append((k, a, b))

print("\n位置（1-based）  出错数  该位置样本数  位置错误率")
for i in range(5):
    n = pos_total[i]
    e = pos_counter[i]
    if n:
        print(f"    第 {i+1} 位          {e}       {n:3d}        {e/n*100:5.2f}%")
    else:
        print(f"    第 {i+1} 位          {e}         0            -")

print("\n各位置错误样本：")
for i in sorted(per_pos_err):
    print(f"  第 {i+1} 位: {per_pos_err[i]}")

print()
print("=" * 80)
print("验证 2：第 4 位在图上的 x 坐标范围（是否与其他位置不同）")
print("=" * 80)

# 收集所有样本的字符块位置
paths = {}
for d in ["vm_dump\\samples", "vm_dump\\samples2", "vm_dump\\holdout"]:
    dd = os.path.join(WS, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)

# 对 4 字符 与 5 字符分别统计各"槽位"的中心 x
slot_centers = {}
for k, t in gt.items():
    if k not in paths:
        continue
    im = Image.open(paths[k]).convert("L")
    a = np.asarray(im).astype(np.float64)
    ink = 1 - (np.round(a) >= 156).astype(np.uint8)
    cp = ink.sum(axis=0)
    blocks = []
    inside = False
    for x in range(len(cp)):
        if cp[x] > 0 and not inside:
            s = x; inside = True
        elif cp[x] == 0 and inside:
            blocks.append((s, x - 1)); inside = False
    if inside:
        blocks.append((s, len(cp) - 1))
    if len(blocks) != len(t):
        continue
    for i, (s, e) in enumerate(blocks):
        slot_centers.setdefault((len(t), i), []).append((s + e) / 2)

print("\n按(字符数, 槽位)看字符块中心 x 的分布：")
for L in (4, 5):
    print(f"\n  --- {L} 字符验证码 ---")
    for i in range(L):
        v = slot_centers.get((L, i), [])
        if v:
            print(f"    槽位 {i+1}: 中心 x 均值 {np.mean(v):6.1f}  范围 [{min(v):.0f}, {max(v):.0f}]  样本 {len(v)}")

print()
print("=" * 80)
print("验证 3：错误样本与正确样本的笔画密度对比（第 4 位）")
print("=" * 80)

err_keys = {k for k in common if gt[k] != pred[k]}
print(f"\n{'样本':6s} {'真值':7s} {'预测':7s} {'错/对':4s} {'第4位块宽':>9s} {'第4位密度':>9s}")
for k in common:
    t, p = gt[k], pred[k]
    if k not in paths or len(t) != len(p):
        continue
    im = Image.open(paths[k]).convert("L")
    a = np.asarray(im).astype(np.float64)
    ink = 1 - (np.round(a) >= 156).astype(np.uint8)
    cp = ink.sum(axis=0)
    blocks = []
    inside = False
    for x in range(len(cp)):
        if cp[x] > 0 and not inside:
            s = x; inside = True
        elif cp[x] == 0 and inside:
            blocks.append((s, x - 1)); inside = False
    if inside:
        blocks.append((s, len(cp) - 1))
    if len(blocks) != len(t):
        continue
    s, e = blocks[3]
    sub = ink[:, s:e + 1]
    dens = sub.sum() / sub.size * 100
    flag = "错" if k in err_keys else "对"
    # 只打印错的 + 少量对的作对照
    if k in err_keys or k in ("c00", "c01", "c03", "c04"):
        print(f"{k:6s} {t:7s} {p:7s} {flag:4s} {e-s+1:9d} {dens:8.1f}%")

# 统计整体
d4_err, d4_ok = [], []
for k in common:
    t, p = gt[k], pred[k]
    if k not in paths or len(t) != len(p):
        continue
    im = Image.open(paths[k]).convert("L")
    a = np.asarray(im).astype(np.float64)
    ink = 1 - (np.round(a) >= 156).astype(np.uint8)
    cp = ink.sum(axis=0)
    blocks = []
    inside = False
    for x in range(len(cp)):
        if cp[x] > 0 and not inside:
            s = x; inside = True
        elif cp[x] == 0 and inside:
            blocks.append((s, x - 1)); inside = False
    if inside:
        blocks.append((s, len(cp) - 1))
    if len(blocks) != len(t):
        continue
    s, e = blocks[3]
    sub = ink[:, s:e + 1]
    dens = sub.sum() / sub.size * 100
    (d4_err if k in err_keys else d4_ok).append(dens)

print(f"\n第 4 位笔画密度：错误样本均值 {np.mean(d4_err):.1f}%   正确样本均值 {np.mean(d4_ok):.1f}%")
