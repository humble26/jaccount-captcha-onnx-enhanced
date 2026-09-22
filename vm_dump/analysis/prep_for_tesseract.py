import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""为 Tesseract 对比实验准备两组图片，像素处理与 userscript 中完全一致。

A 组 = 原脚本喂给 Tesseract 的形态：原始尺寸 110x40 的 JPEG（canvas.toDataURL('image/jpeg')）
B 组 = 新脚本兜底路径的形态：4 倍上采样 + OTSU 二值化 + 24px 白边，PNG 无损
"""
import os
from PIL import Image

W = _VD
SD = os.path.join(W, "samples")
A = os.path.join(W, "tess_A_original")
B = os.path.join(W, "tess_B_prepared")
os.makedirs(A, exist_ok=True)
os.makedirs(B, exist_ok=True)

IN_W, IN_H, SCALE, PAD = 110, 40, 4, 24


def otsu(gray):
    hist = gray.histogram()[:256]
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_b = w_b = 0
    best, thr = -1.0, 156
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


files = sorted(f for f in os.listdir(SD) if f.endswith(".png"))
for fn in files:
    src = Image.open(os.path.join(SD, fn)).convert("RGB")

    # A 组：原始尺寸 JPEG = canvas.toDataURL('image/jpeg')（默认质量 0.92）
    src.convert("L").convert("RGB").save(os.path.join(A, fn.replace(".png", ".jpg")),
                                         quality=92, subsampling=2)

    # B 组：与 buildTesseractCanvas 等价
    w, h = IN_W * SCALE, IN_H * SCALE
    up = src.convert("L").resize((w, h), Image.LANCZOS)
    thr = otsu(up)
    bw = up.point(lambda v: 255 if v >= thr else 0, "L")

    canvas = Image.new("L", (w + PAD * 2, h + PAD * 2), 255)
    canvas.paste(bw, (PAD, PAD))
    canvas.save(os.path.join(B, fn))

print(f"A 组（原脚本形态, 110x40 JPEG）: {len(os.listdir(A))} 张 -> {A}")
print(f"B 组（新脚本形态, 4x+OTSU+白边）: {len(os.listdir(B))} 张 -> {B}")
print("A 组示例尺寸:", Image.open(os.path.join(A, files[0].replace('.png', '.jpg'))).size)
print("B 组示例尺寸:", Image.open(os.path.join(B, files[0])).size)
