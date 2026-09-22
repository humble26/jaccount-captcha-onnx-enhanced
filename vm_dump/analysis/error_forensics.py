import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""分析 5 个错误样本的图像特征：它们是不是真的"模糊到不可分"？

目的：回答"还能不能提升"这个问题。如果错误样本的笔画已经糊成一团，
那就是模型/数据的上限；如果笔画清晰、只是某些位置与背景粘连，
那还有预处理或后处理的空间。
"""
import os, sys
import numpy as np
from PIL import Image

WS = _REPO

CASES = [
    ("c12",  r"vm_dump\samples\c12.png",  "riixo", "riioo", "x->o"),
    ("n088", r"vm_dump\samples2\n088.png", "mwoc",  "mwoo",  "c->o"),
    ("n051", r"vm_dump\samples2\n051.png", "ffryu", "ffruu", "y->u"),
    ("h030", r"vm_dump\holdout\h030.png", "ijtwg", "ijtgg", "w->g"),
    ("h040", r"vm_dump\holdout\h040.png", "ifjgz", "ifjzz", "g->z"),
]

print("=" * 78)
print("错误样本图像特征分析")
print("=" * 78)

for name, rel, truth, pred, diff in CASES:
    p = os.path.join(WS, rel)
    if not os.path.exists(p):
        print(f"\n[{name}] 文件不存在: {p}")
        continue
    im = Image.open(p).convert("L")
    a = np.asarray(im).astype(np.float64)
    h, w = a.shape

    # 用与识别一致的阈值 156 二值化
    binary = (np.round(a) >= 156).astype(np.uint8)   # 1=背景(白), 0=笔画(黑)
    ink = 1 - binary                                  # 1=笔画

    # 每列的笔画像素数 -> 可以看字符的横向分布与重叠情况
    col_profile = ink.sum(axis=0)
    # 找字符之间的空白列（分隔）
    blank_cols = [x for x in range(w) if col_profile[x] == 0]
    # 找"连通"的字符块
    blocks = []
    start = None
    for x in range(w):
        if col_profile[x] > 0:
            if start is None:
                start = x
        else:
            if start is not None:
                blocks.append((start, x - 1))
                start = None
    if start is not None:
        blocks.append((start, w - 1))

    # 平均笔画粗细：用笔画像素数 / 字符块数估算
    total_ink = int(ink.sum())

    print(f"\n[{name}]  真值={truth}  预测={pred}  错位={diff}")
    print(f"  图像尺寸 {w}x{h}   笔画像素 {total_ink} ({total_ink/(w*h)*100:.1f}%)")
    print(f"  空白列数 {len(blank_cols)}   连通块数 {len(blocks)}")
    print(f"  连通块: {blocks}")
    print(f"  预期字符数 {len(truth)}  vs  连通块数 {len(blocks)}"
          + ("   <- 字符粘连！" if len(blocks) < len(truth) else ""))

    # 与真值长度比较：如果块数少于字符数，说明字符互相粘连
    # 定位出错的那个位置：找到 diff 里第几个字符
    pos = None
    for i, (t, q) in enumerate(zip(truth, pred)):
        if t != q:
            pos = i
            break
    print(f"  出错位置: 第 {pos+1} 个字符（0-based {pos}）")

    # 估计出错位置对应哪个连通块（按字符等宽近似）
    if pos is not None and blocks:
        est_x = int(pos * w / len(truth))
        nearest = min(blocks, key=lambda b: abs((b[0] + b[1]) / 2 - est_x))
        bw = nearest[1] - nearest[0] + 1
        print(f"  该位置附近块 {nearest}  宽度 {bw}px  "
              f"（平均字符宽 {w/len(truth):.1f}px）")
        # 该块的笔画密度
        sub = ink[:, nearest[0]:nearest[1] + 1]
        dens = sub.sum() / sub.size * 100
        print(f"  该块笔画密度 {dens:.1f}%")

print()
print("=" * 78)
print("总体：字符块数与字符数的关系（全 220 张）")
print("=" * 78)

import json
gt = json.load(open(os.path.join(WS, "vm_dump", "ground_truth_all.json"), encoding="utf-8"))
# 收集所有样本路径
paths = {}
for d, pref in [("vm_dump\\samples", "c"), ("vm_dump\\samples2", "n"), ("vm_dump\\holdout", "h")]:
    dd = os.path.join(WS, d)
    if not os.path.isdir(dd):
        continue
    for f in os.listdir(dd):
        if f.endswith(".png"):
            paths[os.path.splitext(f)[0]] = os.path.join(dd, f)

stats = {"total": 0, "glued": 0, "split": 0, "exact": 0}
glued_list, split_list = [], []
for k, t in gt.items():
    if k not in paths:
        continue
    im = Image.open(paths[k]).convert("L")
    a = np.asarray(im).astype(np.float64)
    ink = 1 - (np.round(a) >= 156).astype(np.uint8)
    cp = ink.sum(axis=0)
    blocks = 0
    inside = False
    for x in range(len(cp)):
        if cp[x] > 0 and not inside:
            blocks += 1; inside = True
        elif cp[x] == 0:
            inside = False
    stats["total"] += 1
    if blocks < len(t):
        stats["glued"] += 1; glued_list.append((k, t, blocks))
    elif blocks > len(t):
        stats["split"] += 1; split_list.append((k, t, blocks))
    else:
        stats["exact"] += 1

print(f"  {stats}")
print(f"  字符粘连（块数 < 字符数）: {stats['glued']} 张")
print(f"  字符断裂（块数 > 字符数）: {stats['split']} 张")
print(f"\n  粘连样本示例: {glued_list[:15]}")
print(f"  断裂样本示例: {split_list[:15]}")
