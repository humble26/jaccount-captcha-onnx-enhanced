"""为 Tesseract 消融实验生成多组图像变体（与 userscript 里可能的处理方式一一对应）"""
import os
from PIL import Image

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
SD = os.path.join(W, "samples")
ROOT = os.path.join(W, "ablation")
os.makedirs(ROOT, exist_ok=True)

LUT156 = [0] * 156 + [255] * 100


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
        b = w_b * w_f * (m_b - m_f) ** 2
        if b > best:
            best, thr = b, t
    return thr


def pad(im, p):
    out = Image.new("L", (im.width + p * 2, im.height + p * 2), 255)
    out.paste(im, (p, p))
    return out


variants = {}
files = sorted(f for f in os.listdir(SD) if f.endswith(".png"))

for fn in files:
    rgb = Image.open(os.path.join(SD, fn)).convert("RGB")
    gray = rgb.convert("L")

    # raw：原尺寸灰度 PNG（无损，替代 JPEG 的对照）
    variants.setdefault("raw", []).append((fn, gray))

    # bin156：原尺寸 + 官方 156 阈值二值化
    variants.setdefault("bin156", []).append((fn, gray.point(LUT156, "L")))

    # up2 / up3：2x / 3x LANCZOS 灰度上采样
    variants.setdefault("up2", []).append(
        (fn, gray.resize((220, 80), Image.LANCZOS)))
    variants.setdefault("up3", []).append(
        (fn, gray.resize((330, 120), Image.LANCZOS)))

    # up2bin：2x 上采样后再二值化
    variants.setdefault("up2bin", []).append(
        (fn, gray.resize((220, 80), Image.LANCZOS).point(LUT156, "L")))

    # up4otsupad：4x + OTSU + 24px 白边（= 我当前脚本的做法）
    up4 = gray.resize((440, 160), Image.LANCZOS)
    variants.setdefault("up4otsupad", []).append(
        (fn, pad(up4.point(lambda v: 255 if v >= otsu(up4) else 0, "L"), 24)))

for name, items in variants.items():
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    for fn, im in items:
        im.save(os.path.join(d, fn))
    print(f"{name:<12} {len(items)} 张  尺寸示例={items[0][1].size}")

print("\n变体总数:", len(variants))
