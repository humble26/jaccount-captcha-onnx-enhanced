import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""把抓取验证码 + 自动标注拼成 index 图，便于人工核验真值。"""
import os, sys, glob, json
from PIL import Image, ImageDraw
sys.path.insert(0, _HERE)

SRC = os.path.join(_VD, "new300")
recs = json.load(open(os.path.join(_HERE, "_sampling_out", "auto_labeled.json"), encoding="utf-8"))
byfile = {r["file"]: r for r in recs}

files = sorted(glob.glob(os.path.join(SRC, "*.png")))
COLS = 8; CW, CH = 110, 40
PAD = 40
os.makedirs(os.path.join(_HERE, "_sampling_out", "index"), exist_ok=True)

# 分批拼
for bi, start in enumerate(range(0, len(files), COLS * 10)):
    batch = files[start:start + COLS * 10]
    rows = (len(batch) + COLS - 1) // COLS
    W_ = COLS * CW + (COLS + 1) * PAD
    H_ = rows * (CH + PAD + 16)
    canvas = Image.new("RGB", (W_, H_), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 5), f"batch{bi+1} (image#{start})  <- 预测值（需人工核对）", fill="black")
    for n, f in enumerate(batch):
        r = n // COLS; c = n % COLS
        x = PAD + c * (CW + PAD); y = 25 + r * (CH + PAD + 16)
        im = Image.open(f).convert("RGB")
        canvas.paste(im, (x, y))
        rec = byfile.get(os.path.basename(f), {})
        pred = rec.get("pred", "???")
        minc = rec.get("minconf", 0)
        clr = "red" if minc < 0.999 else "black"
        d.text((x, y + CH), f"{os.path.basename(f)}={pred}", fill=clr)
    out = os.path.join(os.path.join(_HERE, "_sampling_out", "index"), f"index_{bi+1}.png")
    canvas.save(out)
    print(f"saved {out} ({len(batch)} imgs, {rows} rows)")