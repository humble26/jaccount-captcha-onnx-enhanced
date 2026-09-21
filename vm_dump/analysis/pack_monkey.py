import os, zipfile

PKG = r"E:\harness\jAccount验证码识别-ResNet增强版"
ZIP = r"E:\harness\jAccount验证码识别-ResNet增强版.zip"
base = os.path.basename(PKG)

if os.path.exists(ZIP):
    os.remove(ZIP)

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(PKG):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            arc = os.path.join(base, os.path.relpath(full, PKG)).replace("\\", "/")
            z.write(full, arc)

print(f"压缩包: {ZIP}  ({os.path.getsize(ZIP)/1024:.1f} KB)")
print()
with zipfile.ZipFile(ZIP) as z:
    print("完整性:", "通过" if z.testzip() is None else "损坏")
    for n in z.namelist():
        info = z.getinfo(n)
        print(f"  {info.file_size/1024:8.1f} KB  {n}")
