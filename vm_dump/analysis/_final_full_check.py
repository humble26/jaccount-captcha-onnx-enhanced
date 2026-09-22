import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""一键全量检验：把项目里所有验证跑一遍，按类别汇总。

用途：任何改动后（改脚本 / 换模型 / 换阈值）跑这一条命令，就知道有没有破坏什么。
用法：python _final_full_check.py
"""
import os
import re
import subprocess
import sys

WS = _REPO
ANA = os.path.join(WS, 'vm_dump', 'analysis')
NODE = _NODE_BIN
VENV_PY = os.path.join(WS, '.venv', 'Scripts', 'python.exe')
PY = _PY_BIN
USERJS = os.path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js')

env = dict(os.environ)
env['PYTHONIOENCODING'] = 'utf-8'
env['NODE_DISABLE_COLORS'] = '1'


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd or ANA, capture_output=True, timeout=900, env=env)
    out = (r.stdout or b'').decode('utf-8', 'replace')
    err = (r.stderr or b'').decode('utf-8', 'replace')
    return r.returncode, out, err


def parse_zh_pass(out):
    """解析 '通过 X / 失败 Y' 或 '结果：全部通过 ✓ (n/n)'"""
    m = re.search(r'通过\s*(\d+)\s*/\s*失败\s*(\d+)', out)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)\s*/\s*(\d+)\s*\)', out.split('结果：')[-1]) if '结果：' in out else None
    if m:
        return int(m.group(1)), int(m.group(2)) - int(m.group(1))
    return None, None


TESTS = [
    # (显示名, 命令, 失败关键字, 说明)
    ('bug_hunt_retry.js', [NODE, 'bug_hunt_retry.js'], None, 'findRefreshButton / 换图等待语义 / 配置健全性'),
    ('bug_hunt_flow.js', [NODE, 'bug_hunt_flow.js'], None, '整段真实 recognize()：失败路径 / 重试竞态 / title'),
    ('bug_hunt_v2.js', [NODE, 'bug_hunt_v2.js'], None, '换图后结果归属 / 异步换图 / AbortError / 闸门'),
    ('race_retry_observer.js', [NODE, 'race_retry_observer.js'], None, '换图自激环路收敛 + 闸门不误伤手动换图'),
    ('test_retry.js', [NODE, 'test_retry.js', USERJS], None, '重试循环行为（含阈值边界 5.999 / 6.000 / 6.001）'),
    ('consistency_check.py', [PY, 'consistency_check.py'], None, '两版同构性：版本号 / 参数 / 判据 / 防回归锁'),
    ('final_check.py', [PY, 'final_check.py'], None, '交付包总校验：条目 / 版本 / 关键符号'),
    ('_verify_margin_impl.js', [NODE, '_verify_margin_impl.js'], None,
     '把 520 张 logits 喂给真实 postprocess，与 numpy 参考逐张对比'),
    ('verify_extension.js', [NODE, os.path.join(WS, 'extension-src', 'verify_extension.js')], None,
     '扩展构建产物端到端（220 张，与 Python 参考比对）'),
    ('web_verify.js', [NODE, 'web_verify.js'], None,
     '油猴版真实 onnxruntime-web 路径端到端（220 张）'),
]

print('=' * 96)
print('全量检验')
print('=' * 96)
print('%-24s %-12s %s' % ('项目', '结果', '说明'))
print('-' * 96)

all_ok = True
rows = []
for name, cmd, failkw, desc in TESTS:
    rc, out, err = run(cmd)
    p, f = parse_zh_pass(out)
    if p is not None:
        ok = (f == 0)
        label = '%d/%d' % (p, p + f)
    else:
        # 自定义输出：找关键字
        if 'text 不一致' in out:
            d = [int(x) for x in re.findall(r'text 不一致\s*:\s*(\d+)', out)]
            dm = [float(x) for x in re.findall(r'minMargin 不一致\s*:\s*(\d+)', out)]
            ok = all(v == 0 for v in d) and all(v == 0 for v in dm) and bool(d)
            label = '一致' if ok else '不一致'
        elif '结论: 全部通过' in out or '结论: ORT-web 路径与参考实现完全一致' in out:
            ok = True
            label = '通过'
        else:
            ok = (rc == 0)
            label = 'exit=%d' % rc
    if not ok:
        all_ok = False
    rows.append((name, label, desc, ok, out, err))
    print('%-24s %-12s %s' % (name, ('✓ ' if ok else '✗ ') + label, desc))

print('-' * 96)

# 额外：语法检查
syn = []
for f in (USERJS, os.path.join(WS, 'extension-src', 'app.js'),
          os.path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js')):
    rc, out, err = run([NODE, '--check', f])
    syn.append((os.path.basename(f), rc == 0))
print('语法检查：' + '  '.join('%s %s' % (n, '✓' if ok else '✗') for n, ok in syn))
if not all(ok for _, ok in syn):
    all_ok = False

# 额外：关键交付物 md5（工作区 vs 归档）
import hashlib


def md5(p):
    try:
        return hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    except Exception:
        return 'ERR'


A = r'E:\harness\jAccount验证码识别-项目归档\01-交付物'
pairs = [
    ('油猴脚本', USERJS, os.path.join(A, '增强版-油猴脚本', 'jaccount-captcha-onnx-enhanced.user.js')),
    ('CHANGELOG', os.path.join(WS, 'CHANGELOG.md'), os.path.join(A, '增强版-油猴脚本', 'CHANGELOG.md')),
    ('复审报告', os.path.join(WS, '复审报告-2026-09-22.md'), os.path.join(A, '增强版-油猴脚本', '复审报告-2026-09-22.md')),
    ('content.js', os.path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js'),
     os.path.join(A, '浏览器扩展版', 'jaccount-captcha-extension', 'content.js')),
    ('manifest', os.path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'manifest.json'),
     os.path.join(A, '浏览器扩展版', 'jaccount-captcha-extension', 'manifest.json')),
    ('test_retry.js', os.path.join(ANA, 'test_retry.js'), os.path.join(A, '增强版-油猴脚本', '测试脚本', 'test_retry.js')),
    ('consistency_check.py', os.path.join(ANA, 'consistency_check.py'),
     os.path.join(A, '增强版-油猴脚本', '测试脚本', 'consistency_check.py')),
]
print()
print('工作区 vs 归档交付物 md5：')
for n, a, b in pairs:
    same = md5(a) == md5(b)
    if not same:
        all_ok = False
    print('  %-22s %s | %s  %s' % (n, md5(a), md5(b), '一致' if same else '★不一致'))

print()
print('=' * 96)
print('总判定：%s' % ('全部通过 ✓ 可以使用' if all_ok else '存在失败 ✗ 需要排查'))
print('=' * 96)

# 失败详情
for name, label, desc, ok, out, err in rows:
    if ok:
        continue
    print()
    print('--- %s 失败详情 ---' % name)
    for l in out.splitlines():
        if l.strip().startswith('✗') or '失败' in l or 'MISMATCH' in l or '不一致' in l:
            print('   ', l)
    for l in err.splitlines()[:6]:
        print('   [stderr]', l)

sys.exit(0 if all_ok else 1)
