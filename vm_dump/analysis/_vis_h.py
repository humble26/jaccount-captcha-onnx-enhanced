# -*- coding: utf-8 -*-
"""可视化 h030(位4=w被误判g) 与 h040(位4=g被误判z) 的字形，逐字符切片。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_e2e as T

keys, x, trg = T.load_split("hold")
for k in ["h030", "h040"]:
    i = keys.index(k)
    lab = [T.CH[trg[p][i]] if trg[p][i] < 26 else "." for p in range(5)]
    print("=" * 70)
    print(f"{k}  真值={' '.join(lab)}  shape={x[i,0].shape}")
    binm = (x[i, 0] >= 0.5).astype(int)
    # 5 个字符切成 5 列(每字符约22px, 宽110)
    w = binm.shape[1] // 5
    cols = []
    for p in range(5):
        seg = binm[:, p * w:(p + 1) * w]
        cols.append(seg)
    for r in range(0, binm.shape[0], 2):
        line = "  ".join("".join("#" if v else "." for v in col[r]) for col in cols)
        print(line)
    print("  字符:", "      ".join(lab))