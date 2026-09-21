"""同步 4.4.1 到 E:\\harness 并重新打包油猴版。"""
import os, shutil, zipfile, hashlib

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
SRC = os.path.join(WS, "jaccount-captcha-onnx-enhanced.user.js")
PKG = r"E:\harness\jAccount验证码识别-ResNet增强版"
ZIP = PKG + ".zip"

# 1) 同步脚本
dst = os.path.join(PKG, "jaccount-captcha-onnx-enhanced.user.js")
shutil.copy2(SRC, dst)
a = hashlib.md5(open(SRC, "rb").read()).hexdigest()
b = hashlib.md5(open(dst, "rb").read()).hexdigest()
print(f"脚本已同步  md5 {a[:12]}  {'一致 ✓' if a == b else '不一致 ✗'}")

# 2) 版本号写进安装说明
inst = os.path.join(PKG, "安装说明.html")
s = open(inst, encoding="utf-8").read()
import re
s2 = re.sub(r"版本 4\.\d+\.\d+", "版本 4.4.1", s)
if s2 != s:
    open(inst, "w", encoding="utf-8").write(s2)
    print("安装说明版本号已更新为 4.4.1")
else:
    print("安装说明版本号无需改动")

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
    print("\n包内脚本 md5:", hashlib.md5(inner).hexdigest()[:12],
          "->", "与工作区一致 ✓" if hashlib.md5(inner).hexdigest() == a else "不一致 ✗")
