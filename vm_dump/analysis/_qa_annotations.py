import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""核对 300 张新样本标注质量：
1) 重跑模型标注，取各位置 top1/top2 的置信与 margin；
2) 用独立引擎 Tesseract(PSM7+白名单) 交叉识别；
3) 与 final_labels_300.json 当前真值对比，列出不一致与低置信样本。
"""
import os, sys, glob, json
import numpy as np
from PIL import Image
sys.path.insert(0, _HERE)
from resnet20_np import forward_fast, forward_heads, load_onnx_weights

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
MODEL = os.path.join(WS, "vm_dump", "nn_model.onnx")
SRC = os.path.join(_VD, "new300")
FINAL = os.path.join(_HERE, "_sampling_out", "final_labels_300.json")

def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)

def softmax(z):
    ze = np.exp(z - z.max(-1, keepdims=True))
    return ze / ze.sum(-1, keepdims=True)

W = {k: v.astype(np.float64) for k, v in load_onnx_weights(MODEL).items()}
files = sorted(glob.glob(os.path.join(SRC, "*.png")))

# Tesseract 独立交叉验证
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\g1507\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\bin\tesseract.cmd"
os.environ["TESSDATA_PREFIX"] = r"C:\Users\g1507\AppData\Local\Temp\tessdata"

def tess_ocr(path):
    img = Image.open(path).convert("L").resize((220, 80), Image.LANCZOS)
    txt = pytesseract.image_to_string(img, lang="eng", config="--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz")
    return "".join(ch for ch in txt.lower() if "a" <= ch <= "z")

recs = []
for i, f in enumerate(files):
    fn = os.path.basename(f)
    x = preprocess(f).reshape(1, 1, 40, 110).astype(np.float64)
    c = forward_fast(x, W)
    logits = forward_heads(c["feat"], W)
    prob = [softmax(z) for z in logits]
    chars, confs, margins = [], [], []
    for pos in range(5):
        z = logits[pos][0]
        order = np.argsort(z)[::-1]
        top1, top2 = int(order[0]), int(order[1])
        p1 = float(prob[pos][0][top1]); p2 = float(prob[pos][0][top2])
        chars.append(chr(ord('a') + top1))
        confs.append(p1)
        margins.append(p1 - p2)
    pred = "".join(chars)
    clean_pred = pred[:-1] if pred.endswith("{") else pred
    minconf = min(confs[:len(clean_pred)])
    min_margin = min(margins[:len(clean_pred)])
    recs.append({
        "file": fn, "pred": pred, "clean_pred": clean_pred,
        "confs": confs, "margins": margins,
        "minconf": minconf, "min_margin": min_margin,
        "tess": tess_ocr(f),
    })
    if i % 50 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)

# 载入当前真值
final = {r["file"]: r for r in json.load(open(FINAL, encoding="utf-8"))}

print(f"\n==== 总览: 共 {len(recs)} 张 ====")
from collections import Counter
src = Counter(r["source"] for r in final.values())
print(f"final_labels 真值来源: {dict(src)}")

# 1) 模型 pred 与最终真值的一致性
diff_truth = []
for r in recs:
    f = final.get(r["file"])
    if f and f["truth"] != r["clean_pred"]:
        diff_truth.append((r["file"], f["truth"], r["clean_pred"], r["minconf"]))
print(f"\n[1] 模型自动标注与最终真值不一致: {len(diff_truth)} 处 (应=人工修正数)")
for d in diff_truth:
    print(f"    {d[0]}: 真值={d[1]} 模型={d[2]} minconf={d[3]:.5f}")

# 2) Tesseract 与最终真值的一致性
diff_tess = []
for r in recs:
    f = final.get(r["file"])
    if f and f["truth"] != r["tess"]:
        diff_tess.append((r["file"], f["truth"], r["tess"], r["minconf"]))
print(f"\n[2] Tesseract 与最终真值不一致: {len(diff_tess)} 张")
for d in diff_tess:
    print(f"    {d[0]}: 真值={d[1]} Tesseract={d[2]} minconf={d[3]:.5f}")

# 3) 低置信样本
low = [r for r in recs if r["minconf"] < 0.999]
print(f"\n[3] 低置信(<0.999)样本: {len(low)} 张")
for r in sorted(low, key=lambda x: x["minconf"]):
    f = final.get(r["file"])
    print(f"    {r['file']}: 真值={f['truth'] if f else '?'} 模型={r['clean_pred']} "
          f"Tess={r['tess']} minconf={r['minconf']:.5f} margin={r['min_margin']:.5f}")

# 4) 高置信但 margin 极小(top1-top2)且跨引擎不一致 —— 危险信号
danger = [r for r in recs if r["minconf"] >= 0.999 and r["min_margin"] < 0.05]
print(f"\n[4] 高置信但存在近邻危险(margin<0.05)样本: {len(danger)} 张")
for r in sorted(danger, key=lambda x: x["min_margin"]):
    f = final.get(r["file"])
    print(f"    {r['file']}: 真值={f['truth'] if f else '?'} 模型={r['clean_pred']} "
          f"Tess={r['tess']} minmargin={r['min_margin']:.5f}")

# 5) 冲突矩阵：模型 vs Tesseract 不一致时重点人工看
conflict = []
for r in recs:
    f = final.get(r["file"])
    if f and r["clean_pred"] != r["tess"]:
        conflict.append((r["file"], f["truth"], r["clean_pred"], r["tess"], r["minconf"]))
print(f"\n[5] 模型与Tesseract冲突: {len(conflict)} 张 (需人工判定)")
for d in conflict:
    print(f"    {d[0]}: 真值={d[1]} 模型={d[2]} Tess={d[3]} minconf={d[4]:.5f}")