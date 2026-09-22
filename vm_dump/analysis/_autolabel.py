# -*- coding: utf-8 -*-
"""对抓取的验证码原图用生产模型自动标注，输出候选真值+置信度+位4高危统计。

真值是模型候选，需人工核对后才能作为正式真值。

⚠ 2026-09-22 晚修正两处缺陷（此前的版本已跑出过 auto_labeled.json，见文末说明）：

1) **未处理第 5 个输出头的 blank 类**。第 5 头是 27 类（26 字母 + index 26 的 blank 占位），
   原写法 `chr(ord('a') + pi)` 对 blank 会产出 `'{'`（chr(123)），于是 300 张里有 160 张的
   pred 末尾多一个 `{`（这 160 张其实是 4 位码）。当时靠下游 `_apply_labels.py` 的
   `clean()` 截掉末尾 `{` 打了补丁才没出事。现在按生产语义在源头处理：
   `argmax >= 26` 视为 blank，该位不存在，不产生字符。

2) **置信度口径与生产脚本不一致**。原 `minconf = min(confs) if len(pred) == 5 else confs[0]`
   里 `len(pred) == 5` 恒为真（pred 永远是 5 个字符的拼接），else 分支是死代码；
   更实质的问题是它把 **blank 位的置信度**也纳入了 min，而生产 postprocess 对 blank 是
   `continue`（不参与统计）。这会让 4 位码的 minconf 系统性偏低、被误判成"低置信难例"。
   本版只统计实际字符位。

路径已改为相对本文件定位，归档剪切移动后仍可直接运行。
"""
import os
import sys
import glob
import json
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # <归档>/02-重构研究
sys.path.insert(0, HERE)
from resnet20_np import forward_fast, forward_heads, load_onnx_weights

# 生产源在工作区；若工作区不存在，可改用环境变量 PROD_ONNX 指定
WS = os.environ.get("PROD_WS") or r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
MODEL = os.environ.get("PROD_ONNX") or os.path.join(WS, "vm_dump", "nn_model.onnx")
SRC = os.environ.get("CAPTCHA_SRC") or os.path.join(ROOT, "new300")
OUT = os.environ.get("AUTOLABEL_OUT") or os.path.join(HERE, "_sampling_out", "auto_labeled.json")

HOT = set("cgouwxyz")


def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def softmax(z):
    ze = np.exp(z - z.max(-1, keepdims=True))
    return ze / ze.sum(-1, keepdims=True)


W = {k: v.astype(np.float64) for k, v in load_onnx_weights(MODEL).items()}
files = sorted(glob.glob(os.path.join(SRC, "*.png")))
recs = []
for i, f in enumerate(files):
    x = preprocess(f).reshape(1, 1, 40, 110).astype(np.float64)
    c = forward_fast(x, W)
    logits = forward_heads(c["feat"], W)
    prob = [softmax(z) for z in logits]
    chars, confs = [], []
    for pos in range(5):
        pi = int(np.argmax(logits[pos][0]))
        # 第 5 个头多一个 blank 类：argmax 落在 26 表示"这一位不存在"（4 位码）。
        # 与生产 postprocess 的 `if (best >= CFG.numClasses) continue;` 语义一致。
        if pos == 4 and pi >= 26:
            continue
        chars.append(chr(ord('a') + pi))
        confs.append(float(prob[pos][0][pi]))
    pred = "".join(chars)
    # 只统计实际字符位；blank 不参与（生产脚本同样跳过）
    minconf = min(confs) if confs else 0.0
    recs.append({"file": os.path.basename(f), "pred": pred, "confs": confs,
                 "minconf": round(minconf, 5), "pos4": chars[3],
                 "pos4_hot": chars[3] in HOT})
    if i % 30 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(recs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n自动标注 {len(recs)} 张（真值是模型候选，需人工核对）")
c4 = Counter(r["pos4"] for r in recs)
print("位4预测字符分布:", dict(sorted(c4.items())))
hot4 = [r for r in recs if r["pos4_hot"]]
print(f"位4落入高危字符: {len(hot4)}/{len(recs)}")
lowc = [r for r in recs if r["minconf"] < 0.999]
print(f"低置信(难例候选): {len(lowc)} 张")
print("\n难例候选（低置信）:")
for r in lowc:
    print(f"  {r['file']}: pred={r['pred']} pos4={r['pos4']} minconf={r['minconf']}")

# 说明：已归档的 auto_labeled.json 是修正前跑出来的，其 pred 对 160 张 4 位码多带一个 '{'，
# 且 minconf 纳入了 blank 位。下游 _apply_labels.py 的 clean() 已把长度问题补上，
# 实测也没有任何样本因此被误分类（blank 位置信同样高于阈值），故**无需重跑**。
# 若将来重新标注，直接用本脚本即可得到正确形式。
