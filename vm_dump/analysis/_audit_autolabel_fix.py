import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""验证 _autolabel.py 的修复：新输出 vs 归档旧输出 vs 最终真值。"""
import os
import json
from collections import Counter

ARCH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'
NEW = _os.path.join(_VD, "audit_sheets", "autolabel_new.json")

old = {r['file']: r for r in json.load(open(os.path.join(ARCH, 'auto_labeled.json'), encoding='utf-8'))}
new = {r['file']: r for r in json.load(open(NEW, encoding='utf-8'))}
fin = {r['file']: r for r in json.load(open(os.path.join(ARCH, 'final_labels_300.json'), encoding='utf-8'))}

print('旧 %d 张 | 新 %d 张 | 最终真值 %d 张' % (len(old), len(new), len(fin)))
print()

print('== 1) blank 泄漏是否消除 ==')
print('  旧 pred 含 "{": %d 张' % sum(1 for r in old.values() if '{' in r['pred']))
print('  新 pred 含 "{": %d 张' % sum(1 for r in new.values() if '{' in r['pred']))
print()

print('== 2) 长度分布 ==')
print('  旧（原样，未 clean）:', dict(sorted(Counter(len(r['pred']) for r in old.values()).items())))
print('  旧（clean 后）      :', dict(sorted(Counter(len(r['pred'][:-1]) if r['pred'].endswith('{') else len(r['pred'])
                                                    for r in old.values()).items())))
print('  新                  :', dict(sorted(Counter(len(r['pred']) for r in new.values()).items())))
print('  最终真值            :', dict(sorted(Counter(r['length'] for r in fin.values()).items())))
print()

print('== 3) 新 pred 与「旧 clean 后」是否一致 ==')
diff = []
for f, r in new.items():
    o = old[f]['pred']
    o_clean = o[:-1] if o.endswith('{') else o
    if r['pred'] != o_clean:
        diff.append((f, o, o_clean, r['pred']))
print('  不一致 %d 张' % len(diff))
for d in diff[:10]:
    print('     %s  旧=%s -> clean=%s  新=%s' % d)
print()

print('== 4) minconf 口径差异（新只统计实际字符位，应 >= 旧）==')
lo = 0
for f, r in new.items():
    if r['minconf'] < old[f]['minconf'] - 1e-9:
        lo += 1
print('  新 < 旧 的样本数（应为 0）:', lo)
旧low = sum(1 for r in old.values() if r['minconf'] < 0.999)
新low = sum(1 for r in new.values() if r['minconf'] < 0.999)
print('  判为「低置信」的张数: 旧 %d -> 新 %d' % (旧low, 新low))
print('  按 0.999 阈值受影响（跨过阈值）的样本:',
      sum(1 for f, r in new.items() if (r['minconf'] < 0.999) != (old[f]['minconf'] < 0.999)))
print()

print('== 5) 新 pred 与最终真值的一致性 ==')
same = sum(1 for f, r in new.items() if r['pred'] == fin[f]['truth'])
print('  一致 %d / %d' % (same, len(new)))
print('  不一致（应为 3 张人工修正）:')
for f, r in new.items():
    if r['pred'] != fin[f]['truth']:
        print('     %-10s 模型=%s  人工真值=%s' % (f, r['pred'], fin[f]['truth']))
print()

print('== 6) 位4 高危统计 ==')
hc = Counter(r['pos4'] for r in new.values() if r['pos4_hot'])
print('  新:', dict(sorted(hc.items())), '共', sum(hc.values()))
print('  旧:', dict(sorted(Counter(r['pos4'] for r in old.values() if r['pos4_hot']).items())))
