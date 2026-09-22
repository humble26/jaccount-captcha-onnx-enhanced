import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""查看归档目录结构与归档快照。"""
import os
import datetime
import json

A = r'E:\harness\jAccount验证码识别-项目归档'
W = _REPO


def tree(base, depth=2, prefix=''):
    entries = sorted(os.listdir(base))
    for n in entries:
        p = os.path.join(base, n)
        if os.path.isdir(p):
            cnt = sum(len(f) for _, _, f in os.walk(p))
            t = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M')
            print('%s[D] %-34s %s (%d)' % (prefix, n, t, cnt))
            if depth > 1:
                tree(p, depth - 1, prefix + '    ')
        else:
            t = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M')
            print('%s[F] %-34s %s %d B' % (prefix, n, t, os.path.getsize(p)))


print('===== 归档目录树（深度 2）=====')
tree(A, 2)

snap = r'E:\harness\_arch_snapshot.json'
print()
print('===== _arch_snapshot.json =====')
try:
    d = json.load(open(snap, encoding='utf-8'))
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                print('%s: %s(%d)' % (k, type(v).__name__, len(v)))
                if isinstance(v, list) and v:
                    print('   前 5 项:', json.dumps(v[:5], ensure_ascii=False)[:600])
            else:
                print('%s: %s' % (k, str(v)[:300]))
    else:
        print(type(d), str(d)[:500])
except Exception as e:
    print('err', e)

# 归档目录里的 md/txt/html 文件（可能有说明）
print()
print('===== 归档内的文档文件 =====')
for r, ds, fs in os.walk(A):
    for f in fs:
        if f.lower().endswith(('.md', '.txt', '.html', '.json')):
            p = os.path.join(r, f)
            print('  %-70s %d B' % (os.path.relpath(p, A), os.path.getsize(p)))
