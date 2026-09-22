import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""同步当前版本的油猴脚本到 E:\\harness 并重新打包。

版本号从脚本头部自动读取，避免像 4.4.0 那次一样：脚本已升到新版本、
同步脚本里却还硬编码着旧版本号字面量，导致安装说明与脚本对不上。
"""
import os, re, shutil, zipfile, hashlib

WS = _REPO
SRC = os.path.join(WS, "jaccount-captcha-onnx-enhanced.user.js")
PKG = r"E:\harness\jAccount验证码识别-ResNet增强版"
ZIP = PKG + ".zip"

# 0) 从脚本头部读出真实版本号
src_text = open(SRC, encoding="utf-8").read()
m = re.search(r"@version\s+([\d.]+)", src_text)
if not m:
    raise SystemExit("未能从脚本头部读取 @version")
VER = m.group(1)
print(f"脚本版本: {VER}")

# 1) 同步脚本
dst = os.path.join(PKG, "jaccount-captcha-onnx-enhanced.user.js")
shutil.copy2(SRC, dst)
a = hashlib.md5(open(SRC, "rb").read()).hexdigest()
b = hashlib.md5(open(dst, "rb").read()).hexdigest()
print(f"脚本已同步  md5 {a[:12]}  {'一致 ✓' if a == b else '不一致 ✗'}")
if a != b:
    raise SystemExit("同步后 md5 不一致，中止打包")

# 2) 版本号写进安装说明（正则替换任意 x.y.z）
inst = os.path.join(PKG, "安装说明.html")
if os.path.exists(inst):
    s = open(inst, encoding="utf-8").read()
    s2 = re.sub(r"版本 \d+\.\d+\.\d+", f"版本 {VER}", s)
    if s2 != s:
        open(inst, "w", encoding="utf-8").write(s2)
        print(f"安装说明版本号已更新为 {VER}")
    else:
        print("安装说明版本号无需改动")
else:
    print("未找到安装说明.html，跳过")

# 3) 重新打包
if os.path.exists(ZIP):
    os.remove(ZIP)
base = os.path.basename(PKG)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(PKG):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            arc = os.path.join(base, os.path.relpath(full, PKG)).replace("\\", "/")
            z.write(full, arc)

print(f"\n压缩包: {ZIP}  ({os.path.getsize(ZIP)/1024:.1f} KB)")
with zipfile.ZipFile(ZIP) as z:
    print("完整性:", "通过 ✓" if z.testzip() is None else "损坏 ✗")
    for n in z.namelist():
        info = z.getinfo(n)
        print(f"  {info.file_size/1024:8.1f} KB  {n}")

# 4) 校验包内脚本与工作区一致
with zipfile.ZipFile(ZIP) as z:
    member = f"{base}/jaccount-captcha-onnx-enhanced.user.js"
    inner = z.read(member)
    ok = hashlib.md5(inner).hexdigest() == a
    print("\n包内脚本 md5:", hashlib.md5(inner).hexdigest()[:12],
          "->", "与工作区一致 ✓" if ok else "不一致 ✗")
    if not ok:
        raise SystemExit("包内脚本与工作区不一致，打包有问题")
