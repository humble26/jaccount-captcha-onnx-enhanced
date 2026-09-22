import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""1) 校验两份 300 张拷贝是否逐字节一致（训练读的是哪一份？）
   2) 核对训练轨迹 meta.json 里记录的数字是否与报告一致
   3) 检查归档后脚本里的硬编码路径是否已失效
"""
import os
import json
import hashlib

W = _VD
ARCH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究'
A_DUP = os.path.join(W, 'new300')
A_ORIG = os.path.join(ARCH, 'sampled')

print('== 1) 两份 300 张拷贝的一致性 ==')


def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


print('  vm_dump/new300 存在:', os.path.isdir(A_DUP), '|', len(os.listdir(A_DUP)) if os.path.isdir(A_DUP) else 0, '项')
print('  归档 sampled    存在:', os.path.isdir(A_ORIG), '|', len(os.listdir(A_ORIG)) if os.path.isdir(A_ORIG) else 0, '项')
if os.path.isdir(A_ORIG):
    sub = [n for n in os.listdir(A_ORIG) if os.path.isdir(os.path.join(A_ORIG, n))]
    print('  sampled 下的子目录:', sub)

for label, base in (('vm_dump/new300', A_DUP), ('归档 sampled', A_ORIG)):
    if not os.path.isdir(base):
        continue
    files = [f for f in os.listdir(base) if f.endswith('.png')]
    h = {f: md5(os.path.join(base, f)) for f in files}
    print('  %-16s %d 张, 合并哈希 %s' % (label, len(h),
          hashlib.md5(''.join(sorted(h.values())).encode()).hexdigest()[:16]))

if os.path.isdir(A_DUP) and os.path.isdir(A_ORIG):
    fa = {f for f in os.listdir(A_DUP) if f.endswith('.png')}
    fb = {f for f in os.listdir(A_ORIG) if f.endswith('.png')}
    print('  文件名集合相同:', fa == fb, '| 仅A有 %d, 仅B有 %d' % (len(fa - fb), len(fb - fa)))
    same = diff = 0
    diffs = []
    for f in sorted(fa & fb):
        if md5(os.path.join(A_DUP, f)) == md5(os.path.join(A_ORIG, f)):
            same += 1
        else:
            diff += 1
            diffs.append(f)
    print('  逐字节相同 %d 张，不同 %d 张' % (same, diff))
    if diffs:
        print('   不同清单:', diffs[:10])

print()
print('== 2) 训练轨迹（_e2e_out/*.meta.json）==')
OUT = os.path.join(ARCH, 'analysis', '_e2e_out')
for n in sorted(os.listdir(OUT)):
    if n.endswith('.meta.json'):
        m = json.load(open(os.path.join(OUT, n), encoding='utf-8'))
        hist = m.get('hist', [])
        print('  %-40s 记录了 %d 个评估点' % (n, len(hist)))
        for h in hist:
            print('      step %4d loss=%.4f tune位4=%s hold位4=%s%s'
                  % (h.get('step', -1), h.get('loss', -1),
                     ('%.2f%%' % (h['acc4_train'] * 100)) if 'acc4_train' in h else '-',
                     ('%.2f%%' % (h['acc4_hold'] * 100)) if 'acc4_hold' in h else '-',
                     '  ★best' if h.get('best') else ''))
        if 'best_hold_acc4' in m:
            print('      best_hold_acc4 = %.4f' % m['best_hold_acc4'])
        print()

print('== 3) 归档后脚本里的硬编码路径是否还存在 ==')
paths = [
    r'E:\harness\重构研究',
    r'E:\harness\重构研究\analysis',
    r'E:\harness\重构研究\sampled\new_captchas',
    r'E:\harness\重构研究\analysis\_sampling_out',
]
for p in paths:
    print('  %-52s 存在=%s' % (p, os.path.exists(p)))
