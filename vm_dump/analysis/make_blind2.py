import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""为 samples2 的 100 张新样本生成盲标对照图（只标序号，不写任何预测结果）"""
import os
from PIL import Image, ImageDraw, ImageFont

W = _VD
SD = os.path.join(W, "samples2")
OUT = os.path.join(W, "blind2")
os.makedirs(OUT, exist_ok=True)

SCALE = 3.5
COLS, PER_SHEET = 2, 20
PAD_X, PAD_Y, LABEL_H = 40, 20, 44

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
except Exception:
    font = ImageFont.load_default(size=30)

files = sorted(f for f in os.listdir(SD) if f.endswith(".png"))
cell_w = int(110 * SCALE) + PAD_X * 2
cell_h = int(40 * SCALE) + PAD_Y * 2 + LABEL_H

made = []
for s in range(0, len(files), PER_SHEET):
    chunk = files[s:s + PER_SHEET]
    rows = (len(chunk) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * cell_w, rows * cell_h), "white")
    d = ImageDraw.Draw(sheet)
    for k, fn in enumerate(chunk):
        im = Image.open(os.path.join(SD, fn)).convert("RGB")
        im = im.resize((int(im.width * SCALE), int(im.height * SCALE)), Image.LANCZOS)
        c, r = k % COLS, k // COLS
        x = c * cell_w + PAD_X
        y = r * cell_h + PAD_Y + LABEL_H
        d.text((x, y - LABEL_H + 4), fn.replace(".png", ""), fill=(0, 0, 0), font=font)
        d.rectangle([x - 6, y - 6, x + im.width + 5, y + im.height + 5], outline=(205, 205, 205))
        sheet.paste(im, (x, y))
    d.line([(cell_w, 0), (cell_w, sheet.height)], fill=(225, 225, 225), width=2)
    p = os.path.join(OUT, f"sheet_{s // PER_SHEET}.png")
    sheet.save(p)
    made.append((p, sheet.size, len(chunk)))
    print(f"{os.path.basename(p)}  {sheet.size}  {len(chunk)} 张  ({chunk[0]}..{chunk[-1]})")
