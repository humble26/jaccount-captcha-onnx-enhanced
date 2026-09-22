import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""决定性实验：预处理变体集成（多视图投票）能否翻盘那 5 个错误？

思路来源：TTA 用几何变换（平移/缩放）已证明负收益。但错误样本的特点是
"真值稳居 top2~top3"，说明模型对该字符是有感知的，只是边界偏了。
换个角度：改变二值化阈值 / 加极轻的形态学处理，可能把边界推回来。

本脚本在 220 张上测试多种预处理变体单独 & 投票的效果，
并严格记录"改进数 / 退化数"（McNemar 式对比），避免只看总数被掩盖。
"""
import os, json
import numpy as np
import onnxruntime as rt
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

so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
names = sorted([o.name for o in sess.get_outputs()], key=lambda x: int(x))
keys = [k for k in gt if k in paths]


def gray(p):
    return np.asarray(Image.open(p).convert("L")).astype(np.float64)


def prep_thr(a, thr):
    return (np.round(a) >= thr).astype(np.float32).reshape(1, 1, 40, 110)


def run_variant(a, thr=156, shift=0, scale=None):
    """threshold 变体 + 可选行/列平移"""
    b = (np.round(a) >= thr).astype(np.float32)
    if scale is not None:
        # 行方向缩放（高度 40），用最近邻，结果严格裁/补回 40 行
        hs = max(1, int(round(40 * scale)))
        idx = np.clip((np.arange(hs) / scale).astype(int), 0, 39)
        b = b[idx, :]
        if b.shape[0] > 40:
            st = (b.shape[0] - 40) // 2
            b = b[st:st + 40, :]
        elif b.shape[0] < 40:
            need = 40 - b.shape[0]
            top = need // 2
            bot = need - top
            b = np.vstack([np.zeros((top, b.shape[1]), dtype=np.float32), b,
                           np.zeros((bot, b.shape[1]), dtype=np.float32)])
        assert b.shape[0] == 40, b.shape
    if shift:
        b = np.roll(b, shift, axis=1)
        if shift > 0:
            b[:, :shift] = 0
        else:
            b[:, shift:] = 0
    return sess.run(names, {iname: b.reshape(1, 1, 40, 110).copy()})


def decode(out):
    s = ""
    for o in out:
        i = int(np.argmax(o[0]))
        s += CH[i] if i < 26 else ""
    return s


VARIANTS = {
    "thr156(基准)": lambda a: run_variant(a, 156),
    "thr140": lambda a: run_variant(a, 140),
    "thr170": lambda a: run_variant(a, 170),
    "thr185": lambda a: run_variant(a, 185),
    "thr200": lambda a: run_variant(a, 200),
    "shift-1": lambda a: run_variant(a, 156, shift=-1),
    "shift+1": lambda a: run_variant(a, 156, shift=1),
    "scale0.97": lambda a: run_variant(a, 156, scale=0.97),
    "scale1.03": lambda a: run_variant(a, 156, scale=1.03),
    "thr130": lambda a: run_variant(a, 130),
    "thr170+shift-1": lambda a: run_variant(a, 170, shift=-1),
}

cache = {}
for name, fn in VARIANTS.items():
    preds = {}
    for k in keys:
        preds[k] = decode(fn(gray(paths[k])))
    cache[name] = preds

print("=" * 90)
print("各单一预处理变体在 220 张上的表现（对照 base 的正确集合）")
print("=" * 90)
print(f"{'变体':>16s} {'正确数':>7s} {'正确率':>8s} {'改进':>6s} {'退化':>6s} {'净':>5s}")
for name, preds in cache.items():
    correct = sum(1 for k in keys if preds[k] == gt[k])
    imp = sum(1 for k in keys if preds[k] == gt[k] and pred_base.get(k) != gt[k])
    deg = sum(1 for k in keys if preds[k] != gt[k] and pred_base.get(k) == gt[k])
    print(f"{name:>16s} {correct:7d} {correct/len(keys)*100:7.2f}% {imp:6d} {deg:6d} {imp-deg:5d}")

print()
print("=" * 90)
print("逐位多数投票（在 11 个变体上，每个字符位单独投票）")
print("=" * 90)


def vote(nameset, mode="majority"):
    out_pred = {}
    for k in keys:
        t = gt[k]
        s = ""
        for i in range(len(t)):
            votes = {}
            for nm in nameset:
                v = decode(cache[nm][k][: i + 1])  # 截断保证长度一致
                ch = cache[nm][k][i] if i < len(cache[nm][k]) else ""
                votes[ch] = votes.get(ch, 0) + 1
            if mode == "majority":
                s += max(votes.items(), key=lambda x: (x[1], x[0] == cache["thr156(基准)"][k][i:i+1]))[0]
            else:
                s += max(votes, key=lambda c: votes[c])
        out_pred[k] = s
    return out_pred


ALL = list(VARIANTS.keys())
BASE_ONLY = ["thr156(基准)"]
HYBRID = ["thr156(基准)", "thr140", "thr170", "scale0.97", "scale1.03"]

for label, ns in [("全体11变体", ALL), ("5变体混合", HYBRID)]:
    p = vote(ns)
    correct = sum(1 for k in keys if p[k] == gt[k])
    imp = sum(1 for k in keys if p[k] == gt[k] and pred_base.get(k) != gt[k])
    deg = sum(1 for k in keys if p[k] != gt[k] and pred_base.get(k) == gt[k])
    print(f"  {label:>14s}: 正确 {correct}/{len(keys)} = {correct/len(keys)*100:.2f}%  "
          f"改进 {imp} 退化 {deg} 净 {imp-deg:+d}")
    fixed = [k for k in keys if p[k] == gt[k] and pred_base.get(k) != gt[k]]
    broke = [k for k in keys if p[k] != gt[k] and pred_base.get(k) == gt[k]]
    print(f"      修复: {fixed}")
    print(f"      弄错: {broke}")

print()
print("=" * 90)
print("聚焦：仅对「第 4 位（含4字符码第4槽）低置信」的位做变体投票")
print("=" * 90)


def softmax_conf(out, i):
    v = out[i][0]
    e = np.exp(v - v.max()); pr = e / e.sum()
    o = np.argsort(v)[::-1]
    return float(pr[o[0]]), (CH[o[0]] if o[0] < 26 else ""), (CH[o[1]] if o[1] < 26 else "")


for thr_conf in [0.95, 0.99, 0.995]:
    p = dict(pred_base)
    touched = 0
    for k in keys:
        out = run_variant(gray(paths[k]), 156)
        t = gt[k]
        for i in range(len(t)):
            c, top1, top2 = softmax_conf(out, i)
            if c >= thr_conf:
                continue
            touched += 1
            # 该位走变体投票
            votes = {}
            for nm in ALL:
                ch = cache[nm][k][i] if i < len(cache[nm][k]) else ""
                votes[ch] = votes.get(ch, 0) + 1
            newch = max(votes, key=lambda c2: votes[c2])
            p[k] = p[k][:i] + newch + p[k][i + 1:]
    correct = sum(1 for k in keys if p[k] == gt[k])
    imp = sum(1 for k in keys if p[k] == gt[k] and pred_base.get(k) != gt[k])
    deg = sum(1 for k in keys if p[k] != gt[k] and pred_base.get(k) == gt[k])
    print(f"  置信阈值 {thr_conf*100:.1f}%: 触发 {touched} 位 | 正确 {correct}/{len(keys)} "
          f"= {correct/len(keys)*100:.2f}% | 改进 {imp} 退化 {deg} 净 {imp-deg:+d}")
    print(f"      修复: {[k for k in keys if p[k]==gt[k] and pred_base.get(k)!=gt[k]]}")
    print(f"      弄错: {[k for k in keys if p[k]!=gt[k] and pred_base.get(k)==gt[k]]}")
