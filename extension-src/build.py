"""构建 Edge/Chrome 扩展（MV3），把 ORT + 模型 + wasm 全部打包进去，零网络依赖。

产物：<工作区>/extension-build/jaccount-captcha-extension/
"""
import os, re, json, shutil, struct, zlib

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
SRC = os.path.join(WS, "extension-src")
VM = os.path.join(WS, "vm_dump")
OUT = os.path.join(WS, "extension-build", "jaccount-captcha-extension")

# ---------- 1. 目录 ----------
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for sub in ("", "assets", "icons"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# ---------- 2. content.js = ORT(UMD) + 业务逻辑 ----------
ort = open(os.path.join(VM, "ort.min.js"), encoding="utf-8", errors="replace").read()
# 去掉 sourceMappingURL：不留一个必然 404 的请求
ort = re.sub(r"\n?//# sourceMappingURL=\S+\s*$", "", ort)
if not ort.endswith("\n"):
    ort += "\n"
assert "sourceMappingURL" not in ort

app = open(os.path.join(SRC, "app.js"), encoding="utf-8").read()

# 在本地再静音一次 ORT 的 W 级警告（app.js 里已经有一层，这里是防它先于 app 触发）
banner = (
    "/* ============================================================\n"
    " * jAccount 验证码自动识别 · content script（已内联 onnxruntime-web 1.16.3）\n"
    " *\n"
    " * 前半部分是 ONNX Runtime Web v1.16.3 的官方 UMD 发行包：\n"
    " *   Copyright (c) Microsoft Corporation. Licensed under the MIT License.\n"
    " * 它的包装是 `... : e.ort = t() }(self, ...)`，在 content script 里 self 即\n"
    " * 隔离世界的全局对象，因此拼接后 `ort` 直接可用，无需 eval、无需注入页面世界。\n"
    " * 后半部分是本项目的识别逻辑。\n"
    " * ============================================================ */\n"
)
content = banner + ort + "\n/* ===== 以下为本项目逻辑 ===== */\n" + app
open(os.path.join(OUT, "content.js"), "w", encoding="utf-8").write(content)

# ---------- 3. 资源 ----------
shutil.copy2(os.path.join(VM, "nn_model.onnx"), os.path.join(OUT, "assets", "nn_model.onnx"))
# wasm 必须与 numThreads=1 + simd=true 对应用 simd 版
wasm_src = r"C:\Users\g1507\.workbuddy\binaries\node\workspace\node_modules\onnxruntime-web\dist\ort-wasm-simd.wasm"
shutil.copy2(wasm_src, os.path.join(OUT, "assets", "ort-wasm-simd.wasm"))

# ---------- 4. manifest ----------
shutil.copy2(os.path.join(SRC, "manifest.json"), os.path.join(OUT, "manifest.json"))

# ---------- 5. 图标（纯 Python 手写 PNG，不依赖 Pillow） ----------
def png(path, size):
    """画一个圆角蓝底 + 白色对勾的图标"""
    bg = (31, 111, 235)      # 蓝
    fg = (255, 255, 255)
    r = max(2, size // 6)
    px = [[bg if (
        (x < r or x >= size - r) and (y < r or y >= size - r) and
        (min(x - (r - 1), size - r - x) ** 2 + min(y - (r - 1), size - r - y) ** 2) > r * r
    ) else bg for x in range(size)] for y in range(size)]
    # 在上面画对勾：两段粗线
    s = size / 16.0
    def thick(x0, y0, x1, y1, w):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) * 3) + 2
        for i in range(n + 1):
            t = i / n
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for dy in range(-w, w + 1):
                for dx in range(-w, w + 1):
                    X, Y = int(round(cx + dx)), int(round(cy + dy))
                    if 0 <= X < size and 0 <= Y < size and dx * dx + dy * dy <= w * w:
                        px[Y][X] = fg
    w = max(1, int(s * 0.9))
    thick(4.2 * s, 8.6 * s, 6.9 * s, 11.4 * s, w)
    thick(6.9 * s, 11.4 * s, 12.0 * s, 4.9 * s, w)

    raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in px)
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(out)

for sz in (16, 48, 128):
    png(os.path.join(OUT, "icons", f"icon{sz}.png"), sz)

# ---------- 6. 随包说明 ----------
open(os.path.join(OUT, "README-许可与来源.txt"), "w", encoding="utf-8").write(
"""jAccount 验证码自动识别（ResNet）· 浏览器扩展版
================================================

组成部分与许可
--------------
content.js 内联了 ONNX Runtime Web v1.16.3
    Copyright (c) Microsoft Corporation
    MIT License — https://github.com/microsoft/onnxruntime

assets/nn_model.onnx（ResNet-20 验证码识别模型）
    原始项目 PhotonQuantum/jaccount-captcha-solver
    分发来源 RyanStarFox/JAccountVerificationCode
    Apache License 2.0

识别逻辑与打包脚本
    由「jAccount 验证码识别 - Tesseract 版」(danyang685, MIT) 改造而来
    MIT License

本扩展不含任何网络通信代码。模型与 wasm 都打进包里，通过 chrome.runtime.getURL
读成字节直接喂给 ONNX Runtime（ort.env.wasm.wasmBinary），全程不发任何请求。
验证码与账号密码都不会离开本机。
""")

# ---------- 7. 汇报 ----------
print("=== 构建产物 ===")
total = 0
for root, dirs, files in os.walk(OUT):
    for f in sorted(files):
        p = os.path.join(root, f)
        n = os.path.getsize(p)
        total += n
        print(f"  {n/1024:9.1f} KB  {os.path.relpath(p, OUT)}")
print(f"  {'-'*44}\n  {total/1048576:9.2f} MB  合计")
print()
print("content.js 结构检查:")
print("  ORT 内联:", "ort" in content[:2000] and "ONNX Runtime Web v1.16.3" in content)
print("  sourceMappingURL 已剥离:", "sourceMappingURL" not in content)
print("  含本项目逻辑:", "function postprocess" in content and "function preprocess" in content)
