"""生成盲标对照图：只标序号，不写任何模型预测结果，避免自我暗示。"""
import os
from PIL import Image, ImageDraw

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
SD = os.path.join(W, "samples")
files = sorted(f for f in os.listdir(SD) if f.endswith(".png"))

SCALE = 3
COLS, ROWS_PAD = 2, 270
CELL_W, CELL_H = 110 * SCALE + 60, 40 * SCALE + 60

rows = (len(files) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * CELL_W, rows * CELL_H), "white")
d = ImageDraw.Draw(sheet)

for i, fn in enumerate(files):
    im = Image.open(os.path.join(SD, fn)).convert("RGB")
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
    c, r = i % COLS, i // COLS
    x = c * CELL_W + 50
    y = r * CELL_H + 40
    sheet.paste(im, (x, y))
    d.text((x - 40, y + 40), f"{i:02d}", fill=(0, 0, 0))
    d.rectangle([x - 4, y - 4, x + im.width + 3, y + im.height + 3], outline=(200, 200, 200))

d.line([(CELL_W, 0), (CELL_W, sheet.height)], fill=(220, 220, 220), width=1)

out = os.path.join(W, "blind_sheet.png")
sheet.save(out)
print("saved:", out, sheet.size)
