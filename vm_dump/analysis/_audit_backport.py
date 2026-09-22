import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""1) 递归校验两份 300 张是否逐字节一致
   2) 打印回写/复验脚本与结果，确认回写用的是哪一版权重
"""
import os
import hashlib

W = _VD
ARCH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究'
DUP = os.path.join(W, 'new300')
ORIG = os.path.join(ARCH, 'sampled', 'new_captchas')


def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


print('== 1) 两份 300 张逐字节比对 ==')
print('  A =', DUP, '存在', os.path.isdir(DUP))
print('  B =', ORIG, '存在', os.path.isdir(ORIG))
if os.path.isdir(DUP) and os.path.isdir(ORIG):
    fa = sorted(f for f in os.listdir(DUP) if f.endswith('.png'))
    fb = sorted(f for f in os.listdir(ORIG) if f.endswith('.png'))
    print('  A %d 张 | B %d 张 | 文件名集合相同 %s' % (len(fa), len(fb), set(fa) == set(fb)))
    same = diff = 0
    diffs = []
    for f in fa:
        if f in set(fb):
            if md5(os.path.join(DUP, f)) == md5(os.path.join(ORIG, f)):
                same += 1
            else:
                diff += 1; diffs.append(f)
    print('  逐字节相同 %d | 不同 %d' % (same, diff))
    if diffs:
        print('  不同清单前 20:', diffs[:20])

print()
print('== 2) 回写脚本与复验结果 ==')
for rel in ('analysis/_backport_onnx.py', 'analysis/_eval_backport_onnx.py',
            'analysis/_eval_backport_result.txt', '_e2e_RT_backport/_rt_compare_result.txt'):
    p = os.path.join(ARCH, rel)
    print('-' * 70)
    print('###', rel, '(存在 %s)' % os.path.exists(p))
    if os.path.exists(p):
        print(open(p, encoding='utf-8', errors='replace').read())
