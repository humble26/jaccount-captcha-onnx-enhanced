"""把可疑字符和已知的 i / j 字形并排放大，判定 n062 首字符到底是 i 还是 j。"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
OUT = os.path.join(W, "glyph_check.png")
ZOOM = 16

# (样本id, 字符序号0起, 该位置已知真值字符)
CASES = [
    ("n062", 0, "?"),   # 待判定
    ("n009", 3, "i"), ("n028", 3, "i"), ("n071", 4, "i"), ("n060", 1, "i"),
    ("n083", 0, "j"), ("n026", 3, "j"), ("n063", 1, "j"), ("n091", 2, "j"),
]


def load(sid):
    p = os.path.join(W, "samples", sid + ".png")
    if not os.path.exists(p):
        p = os.path.join(W, "samples2", sid + ".png")
    if not os.path.exists(p):
        p = os.path.join(W, "holdout", sid + ".png")
    return np.asarray(Image.open(p).convert("L"))


def segments(gray, thr=156):
    ink = (gray < thr)
    cols = ink.any(axis=0)
    segs, start = [], None
    for c, v in enumerate(cols):
        if v and start is None:
            start = c
        elif not v and start is not None:
            if c - start >= 2:
                segs.append((start, c))
            start = None
    if start is not None and len(cols) - start >= 2:
        segs.append((start, len(cols)))
    return segs


try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
except Exception:
    font = small = ImageFont.load_default(size=20)

COLS = 5
cw, ch = 190, 200
rows = (len(CASES) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * cw, rows * ch), "white")
d = ImageDraw.Draw(sheet)

for k, (sid, idx, expect) in enumerate(CASES):
    gray = load(sid)
    segs = segments(gray)
    if idx >= len(segs):
        d.text((10 + (k % COLS) * cw, 10 + (k // COLS) * ch), f"{sid} 无第{idx}段", fill="red", font=font)
        continue
    x0, x1 = segs[idx]
    ink = (gray < 156)
    ys = np.where(ink[:, x0:x1].any(axis=1))[0]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = Image.fromarray(gray[y0:y1, x0:x1]).convert("RGB")
    crop = crop.resize((crop.width * ZOOM, crop.height * ZOOM), Image.NEAREST)
    if crop.width > cw - 20:
        crop = crop.resize((cw - 20, int(crop.height * (cw - 20) / crop.width)), Image.LANCZOS)

    bx, by = (k % COLS) * cw + 8, (k // COLS) * ch + 8
    d.rectangle([bx, by, bx + cw - 16, by + ch - 16], outline=(215, 215, 215))
    title = f"{sid} #{idx}  期望={expect}"
    d.text((bx + 6, by + 4), title, fill=(160, 0, 0) if expect == "?" else (20, 20, 20), font=font)
    d.text((bx + 6, by + 30), f"字符宽 {x1-x0}px  高 {y1-y0}px", fill=(130, 130, 130), font=small)
    sheet.paste(crop, (bx + 6, by + 54))

sheet.save(OUT)
print("saved:", OUT, sheet.size)
print("\n各样本的分段情况（用于判断切分是否正确）:")
for sid, idx, expect in CASES:
    g = load(sid)
    segs = segments(g)
    print(f"  {sid}  共 {len(segs)} 段  {segs}   期望第{idx}段={expect}")
