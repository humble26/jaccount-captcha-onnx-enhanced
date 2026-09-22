import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""生成全量 300 张核对索引图 + 难例单独索引图。3 行文字为底色。
pred_clean: 末尾'{'（blank类）当作4位验证码，去掉该字符。
"""
import os, sys, glob, json
from PIL import Image, ImageDraw
sys.path.insert(0, _HERE)

SRC = os.path.join(_VD, "new300")
recs = json.load(open(os.path.join(_HERE, "_sampling_out", "auto_labeled.json"), encoding="utf-8"))
byfile = {r["file"]: r for r in recs}
files = sorted(glob.glob(os.path.join(SRC, "*.png")))
os.makedirs(os.path.join(_HERE, "_sampling_out", "index"), exist_ok=True)

def clean(pred):
    if pred.endswith("{"):
        return pred[:-1]  # 4位验证码
    return pred

def build(imglist, title, fname, highlight_lowconf=True, max_rows=50):
    COLS = 7; CW, CH = 110, 40; PAD = 14; TH = 16
    rows = (len(imglist) + COLS - 1) // COLS
    rows = min(rows, max_rows)
    W_ = COLS * CW + (COLS + 1) * PAD
    H_ = 36 + rows * (CH + PAD + TH)
    canvas = Image.new("RGB", (W_, H_), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, fill="black")
    for n, f in enumerate(imglist[:rows*COLS]):
        r = n // COLS; c = n % COLS
        x = PAD + c * (CW + PAD); y = 30 + r * (CH + PAD + TH)
        im = Image.open(f).convert("RGB")
        canvas.paste(im, (x, y))
        rec = byfile.get(os.path.basename(f), {})
        pred = clean(rec.get("pred", "???"))
        mc = rec.get("minconf", 1)
        clr = "red" if (highlight_lowconf and mc < 0.999) else "black"
        d.text((x, y + CH), f"{os.path.basename(f)}={pred}", fill=clr)
    out = os.path.join(os.path.join(_HERE, "_sampling_out", "index"), fname)
    canvas.save(out)
    print(f"saved {out}  rows={rows} imgs={min(len(imglist), rows*COLS)}")

HOT = set("cgouwxyz")
# 全部按高危/难例优先排序，便于核对
def sortkey(r):
    hot = r["pos4"] in HOT
    low = r["minconf"] < 0.999
    return (0 if hot else 1, 0 if low else 1, r["file"])
files_sorted = sorted(files, key=lambda f: sortkey(byfile[os.path.basename(f)]))

build(files_sorted, "全量 300 张核对（红线=低置信难例；命高位4高危字符者已前置）", "index_all_sorted.png", max_rows=48)
# 难例单独图
low_files = [f for f in files if byfile[os.path.basename(f)]["minconf"] < 0.999]
build(low_files, f"低置信难例 {len(low_files)} 张（需重点核对）", "index_hard_only.png", highlight_lowconf=False, max_rows=10)