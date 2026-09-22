import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""单张复查 c031.png：盲测里它是我唯一读错的候选（我读 clef，真值 iclef）。

放大到 8 倍，并同时给出：
  - 原始灰度图
  - 生产预处理后的二值图（g>=156 -> 1）
  - 每列是否有前景像素的投影（判断第 1 个字符位到底有没有笔画）
这样可以判断：图上到底有没有开头那个 i。
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw

W = _VD
NEW = os.path.join(W, 'new300')
OUT = os.path.join(W, 'audit_sheets')
A = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'

fn = 'c031.png'
p = os.path.join(NEW, fn)
im = Image.open(p).convert('L')
g = np.asarray(im, dtype=np.float64)
bw = (np.round(g) >= 156)

print('图尺寸:', im.size, '| 前景像素:', int(bw.sum()), '| 前景占比 %.2f%%' % (100 * bw.mean()))

# 列投影：哪些列有前景
cols = bw.sum(axis=0)
runs = []
inrun = False
for i, v in enumerate(cols):
    if v > 0 and not inrun:
        start = i; inrun = True
    elif v == 0 and inrun:
        runs.append((start, i - 1, int(cols[start:i].sum()))); inrun = False
if inrun:
    runs.append((start, len(cols) - 1, int(cols[start:].sum())))
print('前景列区间（列起点, 列终点, 像素数）:')
for a, b, s in runs:
    print('   列 %3d-%3d  (%d px)' % (a, b, s))
print('-> 笔画横向分段数 = %d，可粗判字符个数' % len(runs))

SC = 8
big = im.convert('RGB').resize((im.width * SC, im.height * SC), Image.NEAREST)
big2 = Image.fromarray(((1 - bw) * 255).astype('uint8')).convert('RGB')
big2 = big2.resize((big2.width * SC, big2.height * SC), Image.NEAREST)

canvas = Image.new('RGB', (big.width + 20, big.height * 2 + 120), (255, 255, 255))
d = ImageDraw.Draw(canvas)
d.text((6, 4), '%s  8x  original' % fn, fill=(0, 0, 0), font_size=20)
canvas.paste(big, (10, 28))
d.text((6, big.height + 34), '%s  8x  binary (g>=156)' % fn, fill=(0, 0, 0), font_size=20)
canvas.paste(big2, (10, big.height + 58))
# 列投影条
y0 = big.height * 2 + 70
d.text((6, y0 - 20), 'column foreground profile', fill=(0, 0, 0), font_size=18)
for i, v in enumerate(cols):
    h = int(v / max(1, cols.max()) * 40)
    d.rectangle([10 + i * SC, y0 + 42 - h, 10 + i * SC + SC - 2, y0 + 42], fill=(60, 90, 200))
canvas.save(os.path.join(OUT, 'check_c031.png'))
print('已生成', os.path.join(OUT, 'check_c031.png'), canvas.size)

# 模型原始记录
auto = {r['file']: r for r in json.load(open(os.path.join(A, 'auto_labeled.json'), encoding='utf-8'))}
fin = {r['file']: r for r in json.load(open(os.path.join(A, 'final_labels_300.json'), encoding='utf-8'))}
print()
print('自动标注记录:', auto[fn])
print('最终真值记录:', fin[fn])
