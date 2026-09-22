import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""运行指定目录下的 node / python 测试脚本并汇总结果。"""
import os
import subprocess
import sys

BASE = _REPO
ANA = os.path.join(BASE, 'vm_dump', 'analysis')
NODE = _NODE_BIN
PY = _PY_BIN

targets = sys.argv[1:]
if not targets:
    targets = ['bug_hunt_retry.js', 'bug_hunt_flow.js', 'race_retry_observer.js',
               'test_retry.js', 'consistency_check.py', 'final_check.py']

env = dict(os.environ)
env['PYTHONIOENCODING'] = 'utf-8'
env['NODE_DISABLE_COLORS'] = '1'

for t in targets:
    p = t if os.path.isabs(t) else os.path.join(ANA, t)
    if not os.path.exists(p):
        print('=== %s : MISSING ===' % t)
        continue
    cmd = [NODE, p] if t.endswith('.js') else [PY, p]
    try:
        r = subprocess.run(cmd, cwd=ANA, capture_output=True, timeout=300,
                           env=env)
    except subprocess.TimeoutExpired:
        print('=== %s : TIMEOUT ===' % t)
        continue
    out = (r.stdout or b'').decode('utf-8', 'replace')
    err = (r.stderr or b'').decode('utf-8', 'replace')
    lines = [l for l in out.splitlines() if l.strip()]
    tail = lines[-6:] if len(lines) > 6 else lines
    fails = [l for l in out.splitlines() if l.strip().startswith('✗')]
    print('=== %s  exit=%d ===' % (t, r.returncode))
    for l in tail:
        print('   ', l)
    if fails:
        print('  -- failures --')
        for l in fails[:12]:
            print('   ', l)
    if err.strip():
        e = [l for l in err.splitlines() if l.strip() and 'dirname' not in l
             and 'cd: null' not in l]
        if e:
            print('  -- stderr --')
            for l in e[:6]:
                print('   ', l)
    print()
