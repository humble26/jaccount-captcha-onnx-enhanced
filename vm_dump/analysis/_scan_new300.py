import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""扫描新采集300张: 位4真值在w/g/z的样本, 模型预测是否误判为混淆方。"""
import sys, json, os
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_np import forward_fast, forward_heads, load_onnx_weights, CH
import train_e2e as T

W = {k: v.astype(np.float64) for k, v in load_onnx_weights(T.MODEL).items()}
newgt = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sampling_out", "new300_gt.json"), encoding="utf-8"))
SRC = os.path.join(_VD, "new300")


def preprocess(p):
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def softmax(z):
    ze = np.exp(z - z.max(-1, keepdims=True))
    return ze / ze.sum(-1, keepdims=True)


targets = ["w", "g", "z"]
key_list = [k for k, v in newgt.items() if len(v) >= 4 and v[3] in targets]
xs = [preprocess(os.path.join(SRC, k + ".png"))[None] for k in key_list]
x = np.stack(xs).astype(np.float64)
c = forward_fast(x, W)
lg = forward_heads(c["feat"], W)
pr = [softmax(z) for z in lg]

header = f"{'文件':<7}{'位4真值':<7}{'位4预测':<7}{'整串':<9}{'位4conf':<9}{'类型'}"
print(header)
print("-" * len(header.encode("gbk", errors="ignore")) if False else "-" * 60)
for i, k in enumerate(key_list):
    pred4 = CH[int(np.argmax(lg[3][i]))]
    t4 = newgt[k][3]
    conf4 = float(pr[3][i].max())
    chars = [CH[int(np.argmax(lg[p][i]))] if int(np.argmax(lg[p][i])) < 26 else "."
             for p in range(5)]
    full = "".join(chars)
    if full.endswith("."):
        full = full[:-1]
    wrong = pred4 != t4
    kind = ""
    if wrong and t4 == "w" and pred4 == "g":
        kind = "<<<< w->g难例"
    elif wrong and t4 == "g" and pred4 == "z":
        kind = "<<<< g->z难例"
    elif wrong:
        kind = "<<<< 位4误判"
    print(f"{k:<7}{t4:<7}{pred4:<7}{full:<9}{conf4:<9.4f}{kind}")