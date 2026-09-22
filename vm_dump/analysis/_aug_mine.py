import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""字形定向难例挖掘: 对位4真值为 w/g/z 的所有训练样本施加推向混淆边界的扰动,
选出基础模型误判(尤其 w->g, g->z)或低置信的变体, 作为难例补充数据。"""
import sys, os, json
import numpy as np
from PIL import Image
import copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_np import CH
import train_e2e as T

W = {k: v.astype(np.float64) for k, v in T.load_onnx_weights(T.MODEL).items()}
VD = T.VD
GT = T.GT
NEW_SRC = os.path.join(_VD, "new300")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hardaug")
os.makedirs(OUT, exist_ok=True)


def preprocess(p):
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def softmax(z):
    ze = np.exp(z - z.max(-1, keepdims=True))
    return ze / ze.sum(-1, keepdims=True)


def predict(x):
    c = T.forward_fast(x, W)
    lg = T.forward_heads(c["feat"], W)
    pr = [softmax(z) for z in lg]
    return lg, pr


# ---- 收集训练集中位4真值为 w/g/z 的样本（tune旧 + 300新）----
train_items = []  # (key, path, label4)
gt = json.load(open(T.GT, encoding="utf-8"))
for d in ("samples", "samples2"):
    dd = os.path.join(VD, d)
    for f in sorted(os.listdir(dd)):
        if f.endswith(".png"):
            k = os.path.splitext(f)[0]
            if k in gt and len(gt[k]) > 3:
                train_items.append((k, os.path.join(dd, f), gt[k][3]))
newgt = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sampling_out", "new300_gt.json"), encoding="utf-8"))
for f in sorted(os.listdir(NEW_SRC)):
    if f.endswith(".png"):
        k = os.path.splitext(f)[0]
        if k in newgt and len(newgt[k]) > 3:
            train_items.append((k, os.path.join(NEW_SRC, f), newgt[k][3]))

# 位4真值 w/g/z 的样本
base = [it for it in train_items if it[2] in ("w", "g", "z")]
print(f"训练集中位4真值∈w/g/z 样本: {len(base)}")
from collections import Counter
print(Counter(it[2] for it in base))


# ---- 扰动函数: 全部作用于二值图单个样本 (40,110) ----
rng = np.random.default_rng(0)


def perturb(img, kind):
    """返回扰动后的二值图(可能叠加多种)。kind 决定偏向混淆方向的强度。"""
    a = img.copy()
    # 1) 随机笔划腐蚀(去掉边缘像素) -> w 变得像 g 的缺口, g 闭环变弱像 z
    if kind >= 1:
        # 随机擦除小块横向笔画
        for _ in range(rng.integers(2, 7)):
            r0 = rng.integers(5, 35); c0 = rng.integers(5, 100)
            h = rng.integers(1, 4); wdt = rng.integers(1, 6)
            a[r0:r0 + h, c0:c0 + wdt] = 0
    if kind >= 2:
        # 随机加盐噪声(雾点)
        mask = rng.random(a.shape) < 0.01
        # 只在前景少加, 背景多些
        a[mask] = 0
    if kind >= 3:
        # 细粒度抖动(近似轻微变形)
        a = np.roll(a, rng.integers(-1, 2), axis=0)
    return a


# ---- 对每个 w/g/z 样本生成扰动变体, 记录基础模型误判/低置信 ---
xs = [preprocess(p)[None] for _, p, _ in base]
xall = np.stack(xs).astype(np.float64)
lg0, pr0 = predict(xall)

pairs = [("w", "g"), ("g", "z")]
hard = []
max_per_seed = 4   # 每个种子最多产出的难例数
for idx, (k, p, label4) in enumerate(base):
    img = preprocess(p)
    t4_idx = CH.index(label4)
    want_conf_partner = "g" if label4 == "w" else ("z" if label4 == "g" else None)
    made = 0
    for trial in range(40):
        if made >= max_per_seed:
            break
        kind = rng.integers(1, 4)
        aug = perturb(img, kind)
        x1 = aug[None, None].astype(np.float64)
        lg1, pr1 = predict(x1)
        pred4 = CH[int(np.argmax(lg1[3][0]))]
        conf4 = float(pr1[3][0].max())
        is_target_mis = (label4 == "w" and pred4 == "g") or (label4 == "g" and pred4 == "z")
        if is_target_mis or conf4 < 0.70:
            # 保存难例
            fn = f"{k}__{trial}_l{label4}_p{pred4}_c{conf4:.3f}.npy"
            np.save(os.path.join(OUT, fn), aug)
            hard.append({"key": k, "truth4": label4, "pred4": pred4,
                         "conf": conf4, "target_mis": is_target_mis, "file": fn})
            made += 1

print(f"\n产出难例变体: {len(hard)}")
tm = [h for h in hard if h["target_mis"]]
print(f"  其中精准命中混淆对(w->g或g->z): {len(tm)}")
from collections import Counter
print("  命中混淆类型:", Counter((h["truth4"], h["pred4"]) for h in hard))
json.dump(hard, open(os.path.join(OUT, "hard_list.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"清单已写入 {OUT}\\hard_list.json")