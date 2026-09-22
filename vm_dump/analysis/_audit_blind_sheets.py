import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""生成盲测联络表：随机抽 48 张新样本，图上只写文件名，不写模型预测。

目的：独立测量「自标注真值」的错误率。
  - 300 张中 297 张真值 = 模型自身预测，若模型在这批图上判错，真值就被写成错的，
    且因为"真值==预测"，训练/评估里永远看不到这个错误（自我确认）。
  - 唯一能测出它的是外部独立读数：人（我）看图，不看预测，再与真值比。
已排除此前已核对过的 45 张（低置信 12 + 位4高危 33）。
"""
import os
import json
import random
from PIL import Image, ImageDraw

W = _VD
NEW = os.path.join(W, 'new300')
A = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'
OUT = os.path.join(W, 'audit_sheets')
os.makedirs(OUT, exist_ok=True)

final = {r['file']: r for r in json.load(open(os.path.join(A, 'final_labels_300.json'), encoding='utf-8'))}
done = set(['c078.png', 'c215.png', 'c269.png'])
done |= set(sorted([r['file'] for r in final.values() if r['source'] == 'low'],
                   key=lambda x: final[x]['confidence'])[:9])
done |= set(r['file'] for r in final.values() if r['pos4'] in ('w', 'g', 'z'))

pool = sorted(f for f in final if f not in done)
rnd = random.Random(20260922)
pick = rnd.sample(pool, 48)
pick.sort()
json.dump(pick, open(os.path.join(OUT, 'blind_pick.json'), 'w', encoding='utf-8'), indent=1)

SCALE, PAD, LABEL_H = 4, 8, 20


def tile(fn):
    im = Image.open(os.path.join(NEW, fn)).convert('L').convert('RGB')
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
    canvas = Image.new('RGB', (im.width + PAD * 2, im.height + LABEL_H + PAD), (255, 255, 255))
    canvas.paste(im, (PAD, PAD))
    ImageDraw.Draw(canvas).text((PAD, im.height + PAD + 3), fn, fill=(0, 0, 0))
    return canvas


def sheet(items, path, title, cols=3):
    tiles = [tile(f) for f in items]
    tw = max(t.width for t in tiles); th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new('RGB', (tw * cols, th * rows + 24), (245, 246, 248))
    ImageDraw.Draw(canvas).text((6, 5), title, fill=(20, 20, 20))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(t, (c * tw, 24 + r * th))
    canvas.save(path)
    print('已生成 %s  %dx%d  %d 张' % (path, canvas.width, canvas.height, len(tiles)))


sheet(pick[:24], os.path.join(OUT, 'sheetC1_blind.png'), 'C1 blind set (no prediction shown) - read the code yourself')
sheet(pick[24:], os.path.join(OUT, 'sheetC2_blind.png'), 'C2 blind set (no prediction shown) - read the code yourself')

print()
print('盲测清单（%d 张）：' % len(pick))
for i in range(0, len(pick), 8):
    print('  ', ' '.join(pick[i:i + 8]))
