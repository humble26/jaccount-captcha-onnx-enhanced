# -*- coding: utf-8 -*-
"""批量修复归档内脚本的硬编码路径（E:\\harness\\重构研究\\... -> 相对本文件定位）。

背景：归档是「剪切移动」，原来写死的 E:\\harness\\重构研究\\ 已不存在，
      这些脚本因此全部无法重跑。本脚本把这类字面量改写成基于 __file__ 的相对定位。

安全措施：
  - 只匹配**带引号的字符串字面量**（不会误伤注释里出现的同一路径 —— 注释里没有引号包裹）
  - 改完对每个文件做 ast.parse 语法校验；语法不过就还原该文件
  - 打印每个文件的改动处数
"""
import os
import re
import ast

ANA = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis'
ROOT_NAME = '重构研究'

# 匹配 r'E:\harness\重构研究...' / "E:\harness\重构研究..."（含可选的 r 前缀与单双引号）
LIT = re.compile(r"""(r?)(["'])E:\\harness\\重构研究([^"']*)\2""")
# 正斜杠变体（dbg_layers.py 里那处）
LIT_FWD = re.compile(r"""(["'])E:/harness/重构研究([^"']*)\1""")

INJECT = ('import os as _os\n'
          '_ARCH_HERE = _os.path.dirname(_os.path.abspath(__file__))\n'
          '_ARCH_ROOT = _os.path.dirname(_ARCH_HERE)\n')

BASE_INJECTED = False


def expr_for(suffix):
    """把 \\analysis\\_sampling_out\\auto_labeled.json 这样的后缀转成相对定位表达式。"""
    parts = [p for p in suffix.replace('/', '\\').split('\\') if p]
    if parts and parts[0] == 'analysis':
        base, rest = '_ARCH_HERE', parts[1:]
    else:
        base, rest = '_ARCH_ROOT', parts
    if not rest:
        return base
    inner = ', '.join('"%s"' % p for p in rest)
    return 'os.path.join(%s, %s)' % (base, inner)


def needs_inject(src):
    """是否需要补注入基准变量定义。

    ⚠ 这里有个必须记住的坑：最初写的是 `'_ARCH_HERE' not in src`，
    而替换后的文本里**已经出现** `_ARCH_HERE`（作为表达式的一部分），
    于是被误判成"已注入"，实际根本没注入 → 运行时 NameError。
    必须检查"定义"（`_ARCH_HERE = `）而不是"出现"。
    """
    used = ('_ARCH_HERE' in src) or ('_ARCH_ROOT' in src)
    defined = ('_ARCH_HERE = ' in src)
    return used and not defined


def inject_helper(src):
    lines = src.splitlines(keepends=True)
    idx = None
    for i, l in enumerate(lines):
        if re.match(r'^(import |from )', l):
            idx = i
            break
    if idx is None:
        idx = 1 if lines and lines[0].startswith('#') else 0
    return ''.join(lines[:idx]) + INJECT + ''.join(lines[idx:])


changed_files = []
for name in sorted(os.listdir(ANA)):
    if not name.endswith('.py'):
        continue
    p = os.path.join(ANA, name)
    src0 = open(p, encoding='utf-8').read()
    n1 = len(LIT.findall(src0)) + len(LIT_FWD.findall(src0))
    src1 = LIT.sub(lambda m: expr_for(m.group(3)), src0)
    src1 = LIT_FWD.sub(lambda m: expr_for(m.group(2)), src1)
    injected = False
    if needs_inject(src1):
        src1 = inject_helper(src1)
        injected = True
    if n1 == 0 and not injected:
        continue
    if src1 == src0:
        continue
    try:
        ast.parse(src1)
    except SyntaxError as e:
        print('  ✗ %-30s 语法校验失败，跳过（%s）' % (name, e))
        continue
    open(p, 'w', encoding='utf-8').write(src1)
    changed_files.append((name, n1, injected))
    print('  ✓ %-30s 改写路径 %d 处%s' % (name, n1, '，并补上路径基准定义' if injected else ''))

print()
print('共处理 %d 个文件（改写路径 %d 处）'
      % (len(changed_files), sum(c for _, c, _ in changed_files)))

# 复查：还有没有残留
left = []
for name in sorted(os.listdir(ANA)):
    if not name.endswith('.py'):
        continue
    src = open(os.path.join(ANA, name), encoding='utf-8').read()
    hits = LIT.findall(src) + LIT_FWD.findall(src)
    if hits:
        left.append((name, len(hits)))
print()
if left:
    print('⚠ 仍有残留：')
    for n, c in left:
        print('   %-30s %d 处' % (n, c))
else:
    print('全部 .py 已无 E:\\harness\\重构研究 形式的硬编码路径 ✓')

# 全部 .py 语法总检
bad = []
for name in sorted(os.listdir(ANA)):
    if not name.endswith('.py'):
        continue
    try:
        ast.parse(open(os.path.join(ANA, name), encoding='utf-8').read())
    except SyntaxError as e:
        bad.append((name, str(e)))
print()
print('语法总检：%s' % ('全部通过 ✓' if not bad else '存在问题 ✗'))
for n, e in bad:
    print('   %s: %s' % (n, e))
