import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""检查模型 AveragePool kernel=(10,25) 之外被丢弃的右边缘是否含笔画。

背景：conv 总 stride=4 -> layer3 输出特征图 10x28；AveragePool 只覆盖 :10,:25，
最后 3 列（对应原始 x 约 100-110）从未参与识别。
若那里是纯白边距，则该结构瑕疵无实害；若有笔画，则是被丢弃的信息。
"""
import os
import numpy as np
from PIL import Image

W = _VD

sets = []
for d in ('samples', 'samples2', 'holdout', 'new300'):
    dd = os.path.join(W, d)
    if os.path.isdir(dd):
        sets += [os.path.join(dd, f) for f in sorted(os.listdir(dd)) if f.endswith('.png')]
print('样本 %d 张' % len(sets))

ink_right = []     # x>=100 的笔画像素数
ink_left = []      # x<=9
col_ink = np.zeros(110, dtype=np.int64)
for p in sets:
    g = np.round(np.asarray(Image.open(p).convert('L'), dtype=np.float64))
    ink = (g < 156)                       # 笔画为深色
    col_ink += ink.sum(axis=0)
    ink_right.append(int(ink[:, 100:].sum()))
    ink_left.append(int(ink[:, :10].sum()))

col_ink = col_ink / len(sets)
print()
print('每列平均笔画像素（按 10 列分组）:')
for a in range(0, 110, 10):
    seg = col_ink[a:a + 10]
    print('  x=%3d-%3d : %s' % (a, a + 9, '  '.join('%.1f' % v for v in seg)))
print()
print('x>=100（被池化窗口丢弃的右边缘，对应特征图第 26-28 列）:')
print('  含笔画的样本数: %d / %d' % (sum(1 for v in ink_right if v > 0), len(ink_right)))
print('  笔画像素中位=%.1f 最大=%d' % (float(np.median(ink_right)), max(ink_right)))
print('x<=9（左边缘，未被丢弃）:')
print('  含笔画的样本数: %d / %d' % (sum(1 for v in ink_left if v > 0), len(ink_left)))
print('  笔画像素中位=%.1f 最大=%d' % (float(np.median(ink_left)), max(ink_left)))
print()
# 有笔画的列范围
nz = [i for i, v in enumerate(col_ink) if v > 0.01]
print('平均每列 >0.01 笔画的横向范围: x=%d ~ %d' % (min(nz), max(nz)))
print('-> 特征图 10x28 中，被丢弃的是第 26-28 列，对应原始 x 约 100-110')
