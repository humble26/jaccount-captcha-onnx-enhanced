import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""生成联络表，供独立视觉核对自标注真值。

核心疑问：300 张新样本的真值里 297 张等于模型自己的预测（只有 3 张人工改过）。
于是报告「位4真值∈{w,g,z} 的样本全部高置信正确、无任何 w→g/g→z 难例」是自证的：
模型把 w 看成 g，真值就写成 g，永远不会留下误判痕迹。

本脚本生成两张联络表：
  sheetB_selfcal.png —— 含 3 张已知人工修正样本（auto 错 / truth 对）+ 最低置信样本
                       用途：校准我自己的视觉判断是否与当初的人工核对一致
  sheetA_hot4.png    —— 位4∈{w,g,z} 的全部样本（按置信升序）
                       用途：独立核对这些真值是否真的无误判
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw

W = _VD
NEW = os.path.join(W, 'new300')
A = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'
OUT = os.path.join(W, 'audit_sheets')
os.makedirs(OUT, exist_ok=True)

auto = {r['file']: r for r in json.load(open(os.path.join(A, 'auto_labeled.json'), encoding='utf-8'))}
final = {r['file']: r for r in json.load(open(os.path.join(A, 'final_labels_300.json'), encoding='utf-8'))}

SCALE = 4
PAD = 8
LABEL_H = 34
BG = (255, 255, 255)
INK = (0, 0, 0)
RED = (190, 30, 30)


def tile(fn, note='', note_color=INK):
    im = Image.open(os.path.join(NEW, fn)).convert('L').convert('RGB')
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
    canvas = Image.new('RGB', (im.width + PAD * 2, im.height + LABEL_H + PAD), BG)
    canvas.paste(im, (PAD, PAD))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, im.height + PAD + 4), fn, fill=INK)
    if note:
        d.text((PAD, im.height + PAD + 17), note, fill=note_color)
    return canvas


def sheet(items, cols, path, title):
    """items: list of (fn, note, color)"""
    tiles = [tile(*it) for it in items]
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new('RGB', (tw * cols, th * rows + 26), (245, 246, 248))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), title, fill=(20, 20, 20))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(t, (c * tw, 26 + r * th))
    canvas.save(path)
    print('已生成 %s  %dx%d  %d 张' % (path, canvas.width, canvas.height, len(tiles)))


# ---------- sheetB：自我校准 + 最低置信 ----------
selfcal = ['c078.png', 'c215.png', 'c269.png']
items = []
for fn in selfcal:
    a, f = auto[fn], final[fn]
    items.append((fn, 'model=%s -> human=%s' % (a['pred'], f['truth']), RED))

low = sorted([r for r in final.values() if r['source'] == 'low'], key=lambda r: r['confidence'])
for r in low[:9]:
    fn = r['file']
    items.append((fn, 'model/truth=%s conf=%.3f' % (r['truth'], r['confidence']), INK))
sheet(items, 3, os.path.join(OUT, 'sheetB_selfcal.png'),
      'B: 3 known human corrections (red) + 9 lowest-confidence, model truth shown')

# ---------- sheetA：位4 in {w,g,z} ----------
hot = sorted([r for r in final.values() if r['pos4'] in ('w', 'g', 'z')],
             key=lambda r: r['confidence'])
items = []
for r in hot:
    fn = r['file']
    items.append((fn, 'pos4=%s truth=%s conf=%.3f' % (r['pos4'], r['truth'], r['confidence']), INK))
sheet(items, 3, os.path.join(OUT, 'sheetA_hot4.png'),
      'A: all samples whose labelled 4th char in {w,g,z}, sorted by confidence asc')

print()
print('位4∈{w,g,z} 共 %d 张，置信最低 5 张：' % len(hot))
for r in hot[:5]:
    print('   %s truth=%s conf=%.4f source=%s' % (r['file'], r['truth'], r['confidence'], r['source']))
