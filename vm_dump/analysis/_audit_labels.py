# -*- coding: utf-8 -*-
"""独立复核归档研究线的 300 张标注数据。

要验证的怀疑：
1) _autolabel.py 未处理第 5 头的 blank(26 类) —— 4 位码末位会变成 '{'，
   且 minconf 会把 blank 位的置信一起算进 min，与生产脚本口径不一致。
2) 真值主要来自模型自标注（source=high/low 都等于 pred），只有 3 张人工修正。
"""
import os
import json
from collections import Counter

A = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'
auto = json.load(open(os.path.join(A, 'auto_labeled.json'), encoding='utf-8'))
final = json.load(open(os.path.join(A, 'final_labels_300.json'), encoding='utf-8'))
gt = json.load(open(os.path.join(A, 'new300_gt.json'), encoding='utf-8'))

print('auto_labeled: %d 条 | final_labels: %d 条 | new300_gt: %d 条'
      % (len(auto), len(final), len(gt)))
print()

print('== 1) 自动标注的原始 pred 里是否出现 blank 泄漏（chr(26+a)="{"）==')
brace = [r for r in auto if '{' in r['pred']]
print('  含 "{" 的样本数: %d / %d' % (len(brace), len(auto)))
brace_end = [r for r in brace if r['pred'].endswith('{')]
brace_mid = [r for r in brace if not r['pred'].endswith('{')]
print('  其中 "{" 在末尾: %d（应被 clean() 截掉）; 在中间: %d（clean() 处理不到）'
      % (len(brace_end), len(brace_mid)))
if brace_mid:
    for r in brace_mid[:8]:
        print('     MIDDLE:', r['file'], repr(r['pred']), r['confs'])
print()

print('== 2) minconf 口径：4 位码（末尾 blank）的 min 是否被 blank 位拖低 ==')
bad = []
for r in auto:
    pred = r['pred']
    confs = r['confs']
    if pred.endswith('{'):
        # 真实长度 4；生产口径 min 应取前 4 位
        prod_min = min(confs[:4])
        saved_min = r['minconf']
        if abs(prod_min - saved_min) > 1e-9:
            bad.append((r['file'], saved_min, prod_min, confs))
print('  受影响样本数: %d' % len(bad))
for f, s, p, c in bad[:10]:
    print('     %s  记录 min=%.5f  生产口径 min=%.5f  confs=%s'
          % (f, s, p, [round(x, 5) for x in c]))
print()

print('== 3) 因口径差异被错误打成「低置信难例」的样本 ==')
n_false_low = sum(1 for f, s, p, c in bad if s < 0.999 and p >= 0.999)
print('  实际 min<0.999 但生产口径>=0.999 的样本: %d' % n_false_low)
print()

print('== 4) 真值来源构成 ==')
print('  来源分布:', dict(Counter(r['source'] for r in final)))
lens = Counter(r['length'] for r in final)
print('  长度分布:', dict(sorted(lens.items())))
print('  低置信样本里的长度分布:',
      dict(Counter(r['length'] for r in final if r['source'] == 'low')))
print()

print('== 5) 真值与模型预测是否 100% 同源 ==')
same = 0
diff = []
for r in final:
    a = next(x for x in auto if x['file'] == r['file'])
    cleaned = a['pred'][:-1] if a['pred'].endswith('{') else a['pred']
    if cleaned == r['truth']:
        same += 1
    else:
        diff.append((r['file'], cleaned, r['truth'], r['source']))
print('  与自动标注一致: %d / %d' % (same, len(final)))
print('  不一致（即真正被人工改过的）: %d' % len(diff))
for d in diff:
    print('     %s  auto=%s  truth=%s  source=%s' % d)
print()

print('== 6) 位4 高危字符分布（真值）==')
hot = set('cgouwxyz')
c4 = Counter(r['pos4'] for r in final)
print('  位4 全字符分布:', dict(sorted(c4.items())))
hc = Counter(r['pos4'] for r in final if r['pos4_hot'])
print('  高危小计:', dict(sorted(hc.items())), '共', sum(hc.values()))
print()

print('== 7) 是否存在「模型误判被写成真值」的痕迹 ==')
print('  说明：真值==预测，故按构造不可能出现误判记录；')
print('  可检的是有多少样本本来是低置信（模型自己都不确定）却仍被采信为真值：')
low = [r for r in final if r['source'] == 'low']
print('  低置信仍采信为真值: %d 张（其中人工修正 3 张）' % len(low))
print('  低置信样本示例:')
for r in sorted(low, key=lambda x: x['confidence'])[:12]:
    print('     %-10s truth=%-7s pos4=%s conf=%.5f' % (r['file'], r['truth'], r['pos4'], r['confidence']))
