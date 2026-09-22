import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""把关键冲突样本做成放大对照图，供人工目测复核标注真值。
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

SRC = os.path.join(_VD, "new300")
FINAL = {r["file"]: r for r in json.load(open(
    os.path.join(_HERE, "_sampling_out", "final_labels_300.json"), encoding="utf-8"))}

# (文件名, 标注类型说明)
groups = {
    "A.人工修正": ["c078.png", "c215.png", "c269.png"],
    "B.低置信+Tess冲突": ["c235.png", "c163.png", "c227.png", "c277.png", "c079.png"],
    "C.高置信+Tess词级冲突": ["c006.png", "c108.png", "c152.png", "c177.png", "c284.png"],
    "D.高置信+Tess字符级冲突抽样": ["c000.png", "c007.png", "c016.png", "c049.png", "c102.png", "c016.png"],
}

SCALE = 4
COLS = 4
cell_w, cell_h = 110 * SCALE, 40 * SCALE
label_h = 26
font = ImageFont.truetype("arial.ttf", 20)

sheets = []
for title, files in groups.items():
    files = sorted(set(files))
    rows = (len(files) + COLS - 1) // COLS
    W = COLS * cell_w
    H = rows * (cell_h + label_h) + 34
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill="black", font=font)
    for i, fn in enumerate(files):
        if fn not in FINAL:
            continue
        r, c = divmod(i, COLS)
        x, y = c * cell_w, 34 + r * (cell_h + label_h)
        img = Image.open(os.path.join(SRC, fn)).convert("L").resize(
            (cell_w, cell_h), Image.NEAREST).convert("RGB")
        canvas.paste(img, (x, y))
        truth = FINAL[fn]["truth"]
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline="red")
        d.text((x, y + cell_h + 2), f"{fn[:-4]} [{truth}]", fill=(180, 0, 0), font=font)
    out = os.path.join(os.path.join(_HERE, "_sampling_out"),
                       f"qa_{title[0]}.png")
    canvas.save(out)
    print("saved", out)

# 另存一个完整低置信 42 张总览
low = sorted([r for r in json.load(open(
    os.path.join(_HERE, "_sampling_out", "auto_labeled.json"), encoding="utf-8"))
    if r["minconf"] < 0.999], key=lambda r: r["minconf"])
SCALE2 = 3
rows = (len(low) + COLS - 1) // COLS
cell_w2, cell_h2 = 110 * SCALE2, 40 * SCALE2
W = COLS * cell_w2
H = rows * (cell_h2 + label_h) + 34
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
d.text((8, 6), "E.全部低置信样本(标注真值需特别关注)", fill="black", font=font)
for i, r in enumerate(low):
    fn = r["file"]
    rr, c = divmod(i, COLS)
    x, y = c * cell_w2, 34 + rr * (cell_h2 + label_h)
    img = Image.open(os.path.join(SRC, fn)).convert("L").resize(
        (cell_w2, cell_h2), Image.NEAREST).convert("RGB")
    canvas.paste(img, (x, y))
    truth = FINAL.get(fn, {}).get("truth", "?")
    d.rectangle([x, y, x + cell_w2 - 1, y + cell_h2 - 1], outline="blue")
    d.text((x, y + cell_h2 + 2), f"{fn[:-4]}[{truth}] conf={r['minconf']:.3f}",
           fill=(0, 0, 180), font=font)
out = os.path.join(os.path.join(_HERE, "_sampling_out"), "qa_E_all_low.png")
canvas.save(out)
print("saved", out, "共", len(low), "张")