import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""验证方案 A 的核心假设：灰度输入是否比 1-bit 二值输入更好？

当前模型是在 (g>=156) 二值输入上训练的。直接喂灰度图理论上会 OOD（分布外），
但我们仍能测出：
  1. 灰度输入的直接表现（判断模型对灰度的容忍度）
  2. 灰度信息的"可判别性"上限 —— 即如果模型能用上灰度，理论上能到多少？

方法：用「灰度笔画切片」做最近类中心分类，对比「二值笔画切片」的可分性。
若灰度可分性显著高于二值 => 证明二值化是瓶颈，方案 A 有救。
"""
import os, json
import numpy as np
from PIL import Image

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"
gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
h = json.load(open(os.path.join(VD, "holdout_eval.json"), encoding="utf-8"))
pred_base = h["pred"]["base"]
paths = {}
for d in ["samples", "samples2", "holdout"]:
    dd = os.path.join(VD, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)
keys = [k for k in gt if k in paths]
err_keys = sorted([k for k in keys if pred_base.get(k) != gt[k]])

C4 = [5.5, 28.8, 51.7, 74.9]
C5 = [5.6, 26.6, 47.7, 68.8, 89.9]

def gray(k):
    return np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)

def slot_gray(k, idx):
    """灰度切片：保留原始灰度，归一化到 [0,1]"""
    g = gray(k)
    cs = C4 if len(gt[k]) == 4 else C5
    cx = cs[idx]
    lo, hi = max(0, int(cx - 10)), min(110, int(cx + 10))
    c = g[:, lo:hi] / 255.0
    if c.shape[1] < 20:
        c = np.hstack([np.ones((40, 20 - c.shape[1]), np.float32), c])
    return c[:, :20].ravel()

def slot_bin(k, idx):
    """二值切片：>=156 掩码（与模型训练一致）"""
    g = gray(k)
    cs = C4 if len(gt[k]) == 4 else C5
    cx = cs[idx]
    lo, hi = max(0, int(cx - 10)), min(110, int(cx + 10))
    c = (g[:, lo:hi] >= 156).astype(np.float32)
    if c.shape[1] < 20:
        c = np.hstack([np.ones((40, 20 - c.shape[1]), np.float32), c])
    return c[:, :20].ravel()

def slot_bin_inv(k, idx):
    """反相二值：(g<156) -> 笔画=1"""
    g = gray(k)
    cs = C4 if len(gt[k]) == 4 else C5
    cx = cs[idx]
    lo, hi = max(0, int(cx - 10)), min(110, int(cx + 10))
    c = (g[:, lo:hi] < 156).astype(np.float32)
    if c.shape[1] < 20:
        c = np.zeros((40, 20 - c.shape[1]), np.float32)
        c = np.hstack([c, np.zeros((40, 0), np.float32)])
        c = np.hstack([np.zeros((40, 20 - (c.shape[1] if c.shape[1]<=20 else 20)), np.float32), c])[:, :20]
    return c[:, :20].ravel()

def loo(X, Y):
    cls = sorted(set(Y))
    mu = {c: X[[i for i, y in enumerate(Y) if y == c]].mean(axis=0) for c in cls}
    ok = 0
    for i in range(len(X)):
        d = {c: np.linalg.norm(X[i] - mu[c]) for c in cls}
        ok += int(min(d, key=d.get) == Y[i])
    return ok / len(X)

print("=" * 88)
print("对比：灰度切片 vs 二值切片 vs 反相二值切片 的可分性（第 4 位）")
print("=" * 88)

sel = [k for k in keys if len(gt[k]) > 3]
Y4 = [gt[k][3] for k in sel]

for name, fn in [("灰度 (g/255)", slot_gray),
                 ("二值 (g>=156) 与模型一致", slot_bin),
                 ("反相二值 (g<156)", slot_bin_inv)]:
    X = np.array([fn(k, 3) for k in sel])
    acc = loo(X, Y4)
    print(f"  {name:28s}: 第4位可分率 {acc*100:6.2f}%")

print()
print("  各位置横向对比（灰度 vs 二值）:")
print(f"  {'位置':8s} {'灰度':>10s} {'二值':>10s} {'提升':>10s}")
for pos in range(5):
    s = [k for k in keys if len(gt[k]) > pos]
    if len(s) < 30:
        continue
    Y = [gt[k][pos] for k in s]
    Xg = np.array([slot_gray(k, pos) for k in s])
    Xb = np.array([slot_bin(k, pos) for k in s])
    ag, ab = loo(Xg, Y), loo(Xb, Y)
    print(f"  位{pos+1}: {ag*100:9.2f}% {ab*100:9.2f}% {(ag-ab)*100:+9.2f}%")

print()
print("=" * 88)
print("错误样本的灰度可分性（关键：灰度能否救回这 5 个）")
print("=" * 88)
Xg = np.array([slot_gray(k, 3) for k in sel])
mu_g = {c: Xg[[i for i, y in enumerate(Y4) if y == c]].mean(axis=0) for c in set(Y4)}
for k in err_keys:
    if len(gt[k]) <= 3:
        continue
    v = slot_gray(k, 3)
    tv, pv = gt[k][3], pred_base[k][3]
    dt, dp = np.linalg.norm(v - mu_g[tv]), np.linalg.norm(v - mu_g[pv])
    print(f"  [{k}] 真值={tv} 预测={pv}  灰度到真值 {dt:7.3f}  到误判 {dp:7.3f}  "
          f"-> {'灰度可分' if dt<dp else '灰度不可分'}  (margin {dp-dt:+.3f})")

print()
print("=" * 88)
print("灰度 vs 二值：错误样本的 margin 对比（margin 越大越可分）")
print("=" * 88)
Xb = np.array([slot_bin(k, 3) for k in sel])
mu_b = {c: Xb[[i for i, y in enumerate(Y4) if y == c]].mean(axis=0) for c in set(Y4)}
print(f"  {'样本':8s} {'灰度margin':>12s} {'二值margin':>12s} {'改善':>10s}")
for k in err_keys:
    if len(gt[k]) <= 3: continue
    tv, pv = gt[k][3], pred_base[k][3]
    vg, vb = slot_gray(k, 3), slot_bin(k, 3)
    mg = np.linalg.norm(vg - mu_g[pv]) - np.linalg.norm(vg - mu_g[tv])
    mb = np.linalg.norm(vb - mu_b[pv]) - np.linalg.norm(vb - mu_b[tv])
    print(f"  {k:8s} {mg:12.3f} {mb:12.3f} {mg-mb:+10.3f}")

print()
print("=" * 88)
print("直接测试：把灰度图喂给当前模型（判断 OOD 程度）")
print("=" * 88)
import onnxruntime as rt
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), so, providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name

def decode_with(arr):
    out = sess.run(None, {IN: arr[None, None]})
    s = ""
    for t in out:
        i = int(np.argmax(t[0]))
        s += CH[i] if i < 26 else ""
    return s

# 三种输入
variants = {
    "二值 (g>=156) 训练分布内": lambda k: (gray(k) >= 156).astype(np.float32),
    "灰度 g/255": lambda k: gray(k) / 255.0,
    "灰度 1-g/255（反相）": lambda k: 1.0 - gray(k) / 255.0,
}
for name, fn in variants.items():
    ok = sum(1 for k in keys if decode_with(fn(k)) == gt[k])
    print(f"  {name:28s}: {ok}/{len(keys)} = {ok/len(keys)*100:6.2f}%")

print()
print("  -> 若灰度输入表现接近二值，说明模型对输入分布不敏感；")
print("     但真正要用上灰度信息，仍需用灰度重新训练。")
print("     上面「灰度可分率 > 二值可分率」的对比才是有意义的证据。")
