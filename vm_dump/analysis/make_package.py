import os, shutil, zipfile, json, hashlib

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
DST = r"E:\harness\jAccount验证码识别-ResNet增强版"
ZIP = r"E:\harness\jAccount验证码识别-ResNet增强版.zip"

# 1) 复制交付物
pairs = [
    (os.path.join(WS, "jaccount-captcha-onnx-enhanced.user.js"),
     os.path.join(DST, "jaccount-captcha-onnx-enhanced.user.js")),
    (os.path.join(WS, "验证码识别实测报告.html"),
     os.path.join(DST, "验证码识别实测报告.html")),
]
for src, dst in pairs:
    shutil.copy2(src, dst)
    print(f"复制 {os.path.basename(dst):<44} {os.path.getsize(dst)/1024:8.1f} KB")

# 2) 给不习惯点 HTML 的人一个纯文本入口
readme = """jAccount 验证码自动识别 · ResNet(ONNX) 增强版
=================================================

怎么装：
  双击打开同目录下的「安装说明.html」，按里面的 5 步做即可。

  简要版：
    1. 浏览器装一个用户脚本管理器（推荐 Violentmonkey）
    2. 点管理器图标 →「打开控制台」→ 左上角「+」→「从文件安装」
    3. 选中 jaccount-captcha-onnx-enhanced.user.js → 点「安装」
    4. 打开 jAccount 登录页就能用了

注意：
  如果你以前装过其他 jAccount 验证码脚本，先把旧的禁用掉，
  两个同时跑会互相覆盖输入框。

识别准确率 97.7%（220 张真实验证码实测），首次加载约 12MB / 2 秒，
之后走浏览器缓存。全部在本机推理，不上传任何数据。
"""
open(os.path.join(DST, "README.txt"), "w", encoding="utf-8").write(readme)
print(f"写入 README.txt")

# 3) 打包（用 Python zipfile，显式写 UTF-8 文件名，避免 Compress-Archive 的中文名乱码问题）
if os.path.exists(ZIP):
    os.remove(ZIP)
base = os.path.basename(DST)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(DST):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            arc = os.path.join(base, os.path.relpath(full, DST)).replace("\\", "/")
            z.write(full, arc)
print(f"\n打包完成: {ZIP}  ({os.path.getsize(ZIP)/1024:.0f} KB)")

# 4) 核对
print("\n=== 文件夹内容 ===")
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
    print(f"  {os.path.getsize(p)/1024:8.1f} KB  {h}  {f}")

print("\n=== 压缩包内条目 ===")
with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    print("  完整性检查:", "通过" if bad is None else f"损坏: {bad}")
    for i in z.infolist():
        name = i.filename.encode("cp437").decode("utf-8", "replace") if not (i.flag_bits & 0x800) else i.filename
        print(f"  {i.file_size/1024:8.1f} KB  {name}")

# 5) 与工作区原件比对，确保复制后内容一致
print("\n=== 与工作区原件一致性 ===")
for name in ("jaccount-captcha-onnx-enhanced.user.js", "验证码识别实测报告.html"):
    a = hashlib.md5(open(os.path.join(WS, name), "rb").read()).hexdigest()
    b = hashlib.md5(open(os.path.join(DST, name), "rb").read()).hexdigest()
    print(f"  {'一致 ✓' if a == b else '不一致 ✗'}  {name}")
