# -*- coding: utf-8 -*-
"""提交前隐私/路径扫描：在待提交文件与已入库文本文件里找个人路径与敏感串。

只扫文本类文件（跳过图片/二进制）。输出按"命中模式 -> 文件"聚合。
"""
import os
import re
import subprocess

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_EXT = ('.py', '.js', '.json', '.md', '.txt', '.html', '.htm', '.css', '.yml', '.yaml',
            '.user.js', '.ps1', '.bat', '.sh')

PATTERNS = [
    ('个人目录 C:\\Users\\g1507', re.compile(r'C:\\\\Users\\\\g1507|C:/Users/g1507', re.I)),
    ('用户名 g1507', re.compile(r'g1507')),
    ('E:\\harness 本地路径', re.compile(r'E:\\\\harness|E:/harness')),
    ('邮箱', re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')),
    ('疑似学号/长数字串', re.compile(r'\b\d{9,12}\b')),
    ('手机号样式', re.compile(r'\b1[3-9]\d{9}\b')),
]


def run_git(args):
    r = subprocess.run(['git'] + args, cwd=WS, capture_output=True)
    return (r.stdout or b'').decode('utf-8', 'replace')


def files_from_git():
    """(待提交/修改文件集合, 已入库文件集合)"""
    pending = []
    for l in run_git(['status', '--porcelain']).splitlines():
        if not l.strip():
            continue
        p = l[3:].strip().strip('"')
        if p.endswith('/'):
            for r, ds, fs in os.walk(os.path.join(WS, p)):
                for f in fs:
                    pending.append(os.path.relpath(os.path.join(r, f), WS))
        else:
            pending.append(p)
    tracked = [f.strip() for f in run_git(['ls-files']).splitlines() if f.strip()]
    return pending, tracked


def scan(files, label):
    hits = {}
    for rel in files:
        if not rel.lower().endswith(TEXT_EXT):
            continue
        p = os.path.join(WS, rel)
        if not os.path.exists(p):
            continue
        try:
            t = open(p, encoding='utf-8', errors='ignore').read()
        except Exception:
            continue
        for name, pat in PATTERNS:
            m = pat.findall(t)
            if m:
                hits.setdefault(name, []).append((rel, len(m), str(m[0])[:60]))
    print('=' * 90)
    print(label)
    print('=' * 90)
    if not hits:
        print('  未命中任何模式 ✓')
        return
    for name, items in hits.items():
        print('  [%s] %d 个文件' % (name, len(items)))
        for rel, n, ex in sorted(items)[:12]:
            print('     %-58s x%-3d 例: %s' % (rel, n, ex))
        if len(items) > 12:
            print('     ... 另有 %d 个' % (len(items) - 12))
    print()


pending, tracked = files_from_git()
print('待提交/修改文件 %d 个 | 已入库文件 %d 个' % (len(pending), len(tracked)))
print()
scan(pending, '① 待提交文件')
scan(tracked, '② 已入库文件（已在历史中）')
