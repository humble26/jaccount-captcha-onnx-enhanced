import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""以归档目录为源，同步本轮改动并重新打包两个交付 zip。

背景：归档采用「剪切移动」，E:\\harness 下原来的散装交付目录已不存在，
      因此打包源改为 归档/01-交付物/ 下的两个文件夹（归档是唯一完整副本）。
"""
import os
import shutil
import zipfile

WS = _REPO
ARCH = r'E:\harness\jAccount验证码识别-项目归档\01-交付物'
MONKEY_DIR = os.path.join(ARCH, '增强版-油猴脚本')
EXT_DIR = os.path.join(ARCH, '浏览器扩展版')
HARNESS = r'E:\harness'
MONKEY_ZIP = os.path.join(HARNESS, 'jAccount验证码识别-ResNet增强版.zip')
EXT_ZIP = os.path.join(HARNESS, 'jAccount验证码识别-浏览器扩展版.zip')

print('=' * 78)
print('1) 同步工作区最新文件到归档交付物')
print('=' * 78)
sync = [
    (os.path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'),
     os.path.join(MONKEY_DIR, 'jaccount-captcha-onnx-enhanced.user.js')),
    (os.path.join(WS, 'CHANGELOG.md'), os.path.join(MONKEY_DIR, 'CHANGELOG.md')),
    (os.path.join(WS, '复审报告-2026-09-22.md'), os.path.join(MONKEY_DIR, '复审报告-2026-09-22.md')),
]
for f in ('bug_hunt_flow.js', 'bug_hunt_retry.js', 'bug_hunt_v2.js',
          'race_retry_observer.js', 'test_retry.js', 'consistency_check.py'):
    sync.append((os.path.join(WS, 'vm_dump', 'analysis', f),
                 os.path.join(MONKEY_DIR, '测试脚本', f)))
for s, d in sync:
    shutil.copy2(s, d)
    print('  %-52s %8d B' % (os.path.relpath(d, ARCH), os.path.getsize(d)))

build = os.path.join(WS, 'extension-build', 'jaccount-captcha-extension')
dst_ext = os.path.join(EXT_DIR, 'jaccount-captcha-extension')
if os.path.exists(dst_ext):
    shutil.rmtree(dst_ext)
shutil.copytree(build, dst_ext)
n_ext = sum(len(fs) for _, _, fs in os.walk(dst_ext))
print('  %-52s %8d 个文件' % ('浏览器扩展版/jaccount-captcha-extension/', n_ext))

print()
print('=' * 78)
print('2) 文档里的版本号')
print('=' * 78)
ver_edits = [
    (os.path.join(MONKEY_DIR, '安装说明.html'), '4.5.1', '4.5.2'),
    (os.path.join(MONKEY_DIR, 'README.txt'), '4.5.1', '4.5.2'),
    (os.path.join(EXT_DIR, '安装说明.html'), '1.0.7', '1.0.8'),
]
for p, old, new in ver_edits:
    t = open(p, encoding='utf-8').read()
    c = t.count(old)
    if c:
        open(p, 'w', encoding='utf-8').write(t.replace(old, new))
    print('  %-44s %s -> %s（%d 处）' % (os.path.relpath(p, ARCH), old, new, c))


def pack(src, zip_path, top):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for r, ds, fs in os.walk(src):
            ds.sort()
            for f in sorted(fs):
                full = os.path.join(r, f)
                arc = top + '/' + os.path.relpath(full, src).replace('\\', '/')
                z.write(full, arc)
    zz = zipfile.ZipFile(zip_path)
    bad = zz.testzip()
    print()
    print('%s' % os.path.basename(zip_path))
    print('  %d 条目 | %.2f KB | testzip=%s' % (len(zz.namelist()), os.path.getsize(zip_path) / 1024, bad))
    for n in zz.namelist():
        print('     ', n)


print()
print('=' * 78)
print('3) 打包')
print('=' * 78)
pack(MONKEY_DIR, MONKEY_ZIP, 'jAccount验证码识别-ResNet增强版')
pack(EXT_DIR, EXT_ZIP, 'jAccount验证码识别-浏览器扩展版')
