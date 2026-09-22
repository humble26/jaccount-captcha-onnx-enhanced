import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""c215 高倍放大核查：真值 mxpk（4字母）vs 疑似多字符。
二值化版本 + 原灰度版，放大 8 倍。"""
import os
from PIL import Image

f = os.path.join(_VD, "new300", "c215.png")
img = Image.open(f).convert("L")
w, h = img.size
big = img.resize((w * 8, h * 8), Image.NEAREST)
big.save(os.path.join(_HERE, "_sampling_out", "qa_c215_gray.png"))

# 二值化版（与模型输入一致）
b = img.point(lambda p: 255 if p >= 156 else 0)
bb = b.resize((w * 8, h * 8), Image.NEAREST)
bb.save(os.path.join(_HERE, "_sampling_out", "qa_c215_bin.png"))
print("saved qa_c215_gray.png / qa_c215_bin.png", "原尺寸", w, h)