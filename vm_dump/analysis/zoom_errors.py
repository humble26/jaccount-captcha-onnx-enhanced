import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""把 ResNet 的 4 个错例放大 8 倍，用于人工复核真值标注是否准确。"""
import os, json
from PIL import Image, ImageDraw, ImageFont

W = _VD
OUT = os.path.join(W, "verify_errors.png")
CASES = [("c12", "riixo", "riioo"), ("n051", "ffryu", "ffruu"),
         ("n062", "iyrv", "jyrv"), ("n088", "mwoc", "mwoo")]
SCALE = 8

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
except Exception:
    font = small = ImageFont.load_default(size=24)

pad, lab = 24, 84
cw = 110 * SCALE + pad * 2
ch = 40 * SCALE + pad * 2 + lab
sheet = Image.new("RGB", (2 * cw, 2 * ch), "white")
d = ImageDraw.Draw(sheet)

for k, (sid, truth, got) in enumerate(CASES):
    p = os.path.join(W, "samples", sid + ".png")
    if not os.path.exists(p):
        p = os.path.join(W, "samples2", sid + ".png")
    im = Image.open(p).convert("RGB")
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
    c, r = k % 2, k // 2
    x, y = c * cw + pad, r * ch + pad + lab
    d.text((x, y - lab + 6), f"{sid}   我标的真值: {truth}   模型读出: {got}",
           fill=(180, 30, 30) if k == 0 else (20, 20, 20), font=font)
    d.text((x, y - 24), "(最近邻放大 8 倍，未做任何平滑)", fill=(140, 140, 140), font=small)
    d.rectangle([x - 3, y - 3, x + im.width + 2, y + im.height + 2], outline=(200, 200, 200))
    sheet.paste(im, (x, y))

sheet.save(OUT)
print("saved:", OUT, sheet.size)
