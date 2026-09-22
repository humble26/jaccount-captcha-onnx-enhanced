import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""对 holdout 推理置信度，验证"高置信阈值=100%正确"假设，
为新图自动采信提供依据。"""
import os, sys, json, glob
import numpy as np
from PIL import Image
sys.path.insert(0, _HERE)
from resnet20_np import forward_fast, forward_heads, load_onnx_weights

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
MODEL = os.path.join(WS, "vm_dump", "nn_model.onnx")
HOLD = os.path.join(WS, "vm_dump", "holdout")
GT = json.load(open(os.path.join(WS, "vm_dump", "holdout_eval.json"), encoding="utf-8"))["gt"]

def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)

def softmax(z):
    ze = np.exp(z - z.max(-1, keepdims=True))
    return ze / ze.sum(-1, keepdims=True)

W = {k: v.astype(np.float64) for k, v in load_onnx_weights(MODEL).items()}

files = sorted(glob.glob(os.path.join(HOLD, "*.png")))
recs = []
for f in files:
    name = os.path.basename(f)[:-4]
    x = preprocess(f).reshape(1, 1, 40, 110).astype(np.float64)
    c = forward_fast(x, W)
    logits = forward_heads(c["feat"], W)
    prob = [softmax(z) for z in logits]
    chars, confs = [], []
    for pos in range(5):
        pi = int(np.argmax(logits[pos][0]))
        chars.append(chr(ord('a') + pi))
        confs.append(float(prob[pos][0][pi]))
    pred = "".join(chars)
    # 去除 blank 后缀
    if pred.endswith("{"):
        pred = pred[:-1]
    minconf = min(confs[:len(pred)])
    recs.append({"name": name, "pred": pred, "gt": GT.get(name, "?"),
                 "minconf": round(minconf, 6), "confs": confs})

# 整体准确率
correct = sum(1 for r in recs if r["pred"] == r["gt"])
print(f"holdout 总数: {len(recs)}  正确: {correct}  整串准确率: {correct/len(recs)*100:.2f}%")

# 按 minconf 阈值分桶
for thr in [0.999, 0.995, 0.99, 0.95, 0.9]:
    high = [r for r in recs if r["minconf"] >= thr]
    low = [r for r in recs if r["minconf"] < thr]
    hc = sum(1 for r in high if r["pred"] == r["gt"])
    lc = sum(1 for r in low if r["pred"] == r["gt"])
    hr = hc/len(high)*100 if high else 0
    lr = lc/len(low)*100 if low else 0
    print(f"阈值 {thr}: 高置信 {len(high)}张 准确率 {hr:.2f}% | 低置信 {len(low)}张 准确率 {lr:.2f}%")

# 位4准确率
pos4_correct = sum(1 for r in recs if len(r["pred"])>=4 and len(r["gt"])>=4 and r["pred"][3]==r["gt"][3])
print(f"\n位4准确率: {pos4_correct}/{len(recs)} = {pos4_correct/len(recs)*100:.2f}%")

# 输出所有错误样本及置信度
print("\n错误样本:")
for r in recs:
    if r["pred"] != r["gt"]:
        print(f"  {r['name']}: pred={r['pred']} gt={r['gt']} minconf={r['minconf']}")