import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""紧急排查：为什么本地推理只有 3.64%？—— 输入预处理与输出头映射复核

已知工具链（holdout_eval.py）能得到 215/220，说明正确的调用方式存在。
本脚本逐步复现 holdout_eval.py 的调用，找出差异点。
"""
import os, json
import numpy as np
import onnxruntime as rt
from PIL import Image

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"

print("先读 holdout_eval.py 看它是怎么调用的")
p = os.path.join(VD, "analysis", "holdout_eval.py")
src = open(p, encoding="utf-8").read()
print(src[:4000])
