# -*- coding: utf-8 -*-
"""统计含个人路径的「写法模式」，为批量替换提供依据。

只统计，不修改。
"""
import os
import re
import collections

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANA = os.path.join(WS, 'vm_dump', 'analysis')
EXT = os.path.join(WS, 'extension-src')

pat = re.compile(r'.*g1507.*')
forms = collections.Counter()
byfile = collections.defaultdict(list)

for base in (ANA, EXT, WS):
    for r, ds, fs in os.walk(base):
        if any(x in r for x in ('.git', '.venv', 'node_modules', 'extension-build', 'audit_sheets')):
            continue
        # 只扫顶层与 analysis/extension-src，避免把 vm_dump 里的样本目录也走一遍
        if base == WS and r != WS:
            continue
        for f in fs:
            if not f.lower().endswith(('.py', '.js', '.json')):
                continue
            p = os.path.join(r, f)
            try:
                t = open(p, encoding='utf-8', errors='ignore').read()
            except Exception:
                continue
            for line in t.splitlines():
                if 'g1507' not in line:
                    continue
                byfile[os.path.relpath(p, WS)].append(line.strip())
                # 归一化：去掉变量名与尾部差异，留下路径字面量形态
                m = re.findall(r'''["']?[A-Za-z]:[\\/]+Users[\\/]+g1507[^"'\s]*''', line)
                for x in m:
                    forms[x] += 1
                if not m:
                    forms['<无路径字面量: %s>' % line.strip()[:70]] += 1

print('含 g1507 的文件 %d 个' % len(byfile))
print()
print('=== 路径字面量形态统计 ===')
for k, v in forms.most_common(40):
    print('  x%-4d %s' % (v, k))
print()
print('=== 文件与所在行（前 30 个文件）===')
for i, (f, lines) in enumerate(sorted(byfile.items())):
    if i >= 30:
        print('  ... 另有 %d 个文件' % (len(byfile) - 30))
        break
    print('  %s' % f)
    for l in lines[:4]:
        print('       %s' % l[:120])
