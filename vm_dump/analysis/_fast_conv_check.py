# -*- coding: utf-8 -*-
"""校验向量化 im2col 卷积(forward_fast/backward_fast)与原版实现完全一致。
前向逐层对比；反向在真实 ResNet-20 上逐层对比原版 backward。
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_np import (conv2d, conv2d_backward, forward, backward,
                         forward_fast, backward_fast, forward_heads,
                         load_onnx_weights)

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
VD = os.path.join(WS, "vm_dump")
MODEL = os.path.join(VD, "nn_model.onnx")

def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)

def main():
    W = load_onnx_weights(MODEL)
    W = {k: v.astype(np.float64) for k, v in W.items()}
    imgs = [os.path.join(VD, "holdout", f) for f in sorted(os.listdir(os.path.join(VD, "holdout")))[:3]]
    x = np.stack([preprocess(p)[None] for p in imgs]).astype(np.float64)
    print("input", x.shape)

    # ---------- 1) 前向逐层对比 fast vs 原版 ----------
    cf = forward_fast(x, W)
    cs = forward(x, W)
    worst = 0
    for k in cf:
        d = np.abs(cf[k] - cs[k]).max()
        worst = max(worst, d)
        print(f"  forward {k}: max|diff| = {d:.3e}  {'OK' if d < 1e-9 else 'FAIL'}")
        assert d < 1e-9, k
    print(f"  [forward] 全部 {len(cf)} 层一致，worst={worst:.3e}  PASS")

    # ---------- 2) 反向逐层对比 fast vs 原版 backward ----------
    loss_grad = [np.random.default_rng(0).normal(0, 1, (x.shape[0], C)) for C in [26, 26, 26, 26, 27]]
    g_slow = backward(loss_grad, cs, W)
    g_fast = backward_fast(loss_grad, cf, W)
    worst = 0
    for k in sorted(g_slow.keys()):
        if k not in g_fast:
            print(f"  backward {k}: 缺失 in fast  FAIL")
            worst = float('inf'); continue
        d = np.abs(g_slow[k] - g_fast[k]).max()
        worst = max(worst, d)
        if d >= 1e-8:
            print(f"  backward {k}: max|diff| = {d:.3e}  FAIL")
    print(f"  [backward] 共 {len(g_slow)} 个权重张量，worst={worst:.3e}  "
          + ("PASS" if worst < 1e-8 else "FAIL"))
    return 0 if worst < 1e-8 else 1

if __name__ == "__main__":
    sys.exit(main())