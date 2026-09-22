import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""盘点当前项目状态：交付目录、E 盘根、new300 样本属性、vm_dump 顶层结构。"""
import os
import datetime
import struct

W = _REPO


def show(base, limit=60):
    print('== %s ==' % base)
    try:
        items = sorted(os.listdir(base))
    except Exception as e:
        print('  err', e)
        print()
        return
    for n in items[:limit]:
        p = os.path.join(base, n)
        try:
            t = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M')
        except Exception:
            t = '--'
        if os.path.isdir(p):
            cnt = sum(len(f) for _, _, f in os.walk(p))
            print('  [D] %-40s %s (%d files)' % (n, t, cnt))
        else:
            print('  [F] %-40s %s %d B' % (n, t, os.path.getsize(p)))
    print()


show('E:\\harness')
show('E:\\')

print('== vm_dump 顶层 ==')
for n in sorted(os.listdir(os.path.join(W, 'vm_dump'))):
    p = os.path.join(W, 'vm_dump', n)
    if os.path.isdir(p):
        fs = os.listdir(p)
        cnt = sum(len(f) for _, _, f in os.walk(p))
        t = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M')
        print('  [D] %-24s %s (%d files)  sample=%s' % (n, t, cnt, fs[:3]))
    else:
        print('  [F] %-24s %d B' % (n, os.path.getsize(p)))

# --- new300 样本属性（尺寸 / 真实格式）---
print()
print('== new300 样本属性 ==')


def png_or_jpeg(path):
    with open(path, 'rb') as f:
        head = f.read(32)
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', head[16:24])
        return 'PNG %dx%d' % (w, h)
    if head[:2] == b'\xff\xd8':
        # 扫 SOF 拿尺寸
        with open(path, 'rb') as f:
            data = f.read()
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            m = data[i + 1]
            if m in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack('>HH', data[i + 5:i + 9])
                return 'JPEG %dx%d' % (w, h)
            if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            seg = struct.unpack('>H', data[i + 2:i + 4])[0]
            i += 2 + seg
    return 'UNKNOWN'


d = os.path.join(W, 'vm_dump', 'new300')
if os.path.isdir(d):
    fs = sorted(os.listdir(d))
    print('  文件数:', len(fs))
    from collections import Counter
    kinds = Counter()
    for f in fs[:400]:
        kinds[png_or_jpeg(os.path.join(d, f))] += 1
    for k, v in kinds.most_common():
        print('   %-16s %d' % (k, v))
    print('  第一批首个:', fs[0] if fs else '-', '| 最后:', fs[-1] if fs else '-')
else:
    print('  new300 不存在')
