import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""独立复核：新增 300 张与既有样本集之间是否存在重复（数据泄漏）。

若留出集样本混进训练用的 300 张，则"留出集 98%"就不再是干净的泛化估计。
比较两种粒度：文件字节 md5、解码后灰度像素 md5（跨 png/jpg 格式）。
"""
import os
import glob
import hashlib
from collections import defaultdict
from PIL import Image

W = _VD

SETS = {
    'new300': os.path.join(W, 'new300', '*.png'),
    'holdout': os.path.join(W, 'holdout', '*.png'),
    'hold_A_original': os.path.join(W, 'hold_A_original', '*.jpg'),
    'hold_B_bin156': os.path.join(W, 'hold_B_bin156', '*.png'),
    'samples2': os.path.join(W, 'samples2', '*.png'),
    'samples': os.path.join(W, 'samples', '*.png'),
    'all_A_original': os.path.join(W, 'all_A_original', '*.jpg'),
    'all_B_bin156': os.path.join(W, 'all_B_bin156', '*.png'),
}

byte_map = defaultdict(list)   # md5 -> [(set, file)]
px_map = defaultdict(list)


def px_key(p):
    im = Image.open(p).convert('L')
    return hashlib.md5(im.tobytes()).hexdigest()


for name, pat in SETS.items():
    files = sorted(glob.glob(pat))
    if not files:
        print('  (空) %s' % name)
        continue
    for f in files:
        with open(f, 'rb') as fh:
            b = fh.read()
        byte_map[hashlib.md5(b).hexdigest()].append((name, os.path.basename(f)))
        try:
            px_map[px_key(f)].append((name, os.path.basename(f)))
        except Exception as e:
            print('  px err', f, e)
    print('  %-16s %d 张' % (name, len(files)))

print()
print('== 字节级重复（不同集合之间）==')
cross = 0
for k, v in byte_map.items():
    sets = set(x[0] for x in v)
    if len(sets) > 1:
        cross += 1
        print('  ', v)
print('  跨集合重复组数:', cross)

print()
print('== 像素级重复（跨格式也算）==')
cross2 = 0
for k, v in px_map.items():
    sets = set(x[0] for x in v)
    if len(sets) > 1:
        cross2 += 1
        if cross2 <= 25:
            print('  ', v)
print('  跨集合重复组数:', cross2)

print()
print('== 重点：new300 与 holdout / hold_A_original 是否有重叠 ==')
hit = 0
for k, v in px_map.items():
    sets = set(x[0] for x in v)
    if 'new300' in sets and (sets & {'holdout', 'hold_A_original', 'hold_B_bin156'}):
        hit += 1
        print('  !', v)
print('  命中:', hit)
