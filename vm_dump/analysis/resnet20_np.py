# -*- coding: utf-8 -*-
"""纯 NumPy ResNet-20 前向/反向 —— 端到端重构的可训练骨架

前向已与生产 ONNX(onnxruntime) 逐层校验一致（误差 <= 2.3e-5，纯浮点舍入）。

结构（与 jAccount nn_model.onnx 完全一致）：
  conv1(1->16,k3,s1,p1) -> ReLU
  layer1: 3 blocks, 16ch, identity shortcut
  layer2: block0 为下采样(stride=2) + 1x1 shortcut；其余 identity
  layer3: block0 为下采样(stride=2) + 1x1 shortcut；其余 identity
  avgpool(kernel=10x25) -> 64 维全局向量
  5 个 Gemm 头：前 4 个 26 类，第 5 个 27 类（含 blank）

用法：作为模块 import；训练脚本在数据/头裁剪层面使用。
"""
import numpy as np

CH = "abcdefghijklmnopqrstuvwxyz"


def conv2d(x, wk, bk, pad, stride):
    N, C, H, W = x.shape
    O, Ck, K, _ = wk.shape
    ph, pw = pad
    sh, sw = stride
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    Oh = (H + 2 * ph - K) // sh + 1
    Ow = (W + 2 * pw - K) // sw + 1
    out = np.empty((N, O, Oh, Ow), dtype=np.float64)
    for oh in range(Oh):
        for ow in range(Ow):
            r = xp[:, :, oh * sh:oh * sh + K, ow * sw:ow * sw + K]
            out[:, :, oh, ow] = np.einsum("nchw,ochw->no", r, wk)
    return out + bk.reshape(1, O, 1, 1)


def relu(x):
    return np.maximum(0, x)


def avgpool(x, kh=10, kw=25):
    return x[:, :, :kh, :kw].mean(axis=(2, 3))


def load_onnx_weights(model_path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(model_path)
    w = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    return w


def forward(x, W):
    """返回 dict：激活缓存（反向传播用）。所有中间量都是 float64。"""
    c = {}
    # conv1
    c["x0"] = x
    c["h1"] = conv2d(x, W["onnx::Conv_224"], W["onnx::Conv_225"], (1, 1), (1, 1))
    c["a1"] = relu(c["h1"])
    # layer1
    c["l1_in"] = c["a1"]
    h = c["a1"]
    for i, (c1, c2) in enumerate([(227, 230), (233, 236), (239, 242)]):
        t = relu(conv2d(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"], (1, 1), (1, 1)))
        t2 = conv2d(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"], (1, 1), (1, 1))
        h = relu(t2 + h)
        c[f"l1_{i}_t"] = t; c[f"l1_{i}_t2"] = t2
        c[f"l1_{i}_out"] = h
    c["l1_out"] = h
    # layer2
    for i, (c1, c2, sc) in enumerate([(245, 248, 251), (254, 257, 0), (260, 263, 0)]):
        s1 = (2, 2) if i == 0 else (1, 1)
        t = relu(conv2d(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"], (1, 1), s1))
        t2 = conv2d(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"], (1, 1), (1, 1))
        s = h if sc == 0 else conv2d(h, W[f"onnx::Conv_{sc}"], W[f"onnx::Conv_{sc+1}"], (0, 0), (2, 2))
        h = relu(t2 + s)
        c[f"l2_{i}_t"] = t; c[f"l2_{i}_t2"] = t2; c[f"l2_{i}_s"] = s
        c[f"l2_{i}_out"] = h
    c["l2_out"] = h
    # layer3
    for i, (c1, c2, sc) in enumerate([(266, 269, 272), (275, 278, 0), (281, 284, 0)]):
        s1 = (2, 2) if i == 0 else (1, 1)
        t = relu(conv2d(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"], (1, 1), s1))
        t2 = conv2d(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"], (1, 1), (1, 1))
        s = h if sc == 0 else conv2d(h, W[f"onnx::Conv_{sc}"], W[f"onnx::Conv_{sc+1}"], (0, 0), (2, 2))
        h = relu(t2 + s)
        c[f"l3_{i}_t"] = t; c[f"l3_{i}_t2"] = t2; c[f"l3_{i}_s"] = s
        c[f"l3_{i}_out"] = h
    c["l3_out"] = h
    c["feat"] = avgpool(h).reshape(x.shape[0], -1)
    return c


def head_forward(feat, W, num):
    return feat @ W[f"linear{num}.weight"].T + W[f"linear{num}.bias"]


def forward_heads(feat, W, head_nums=("1", "2", "3", "4", "5")):
    return [head_forward(feat, W, n) for n in head_nums]


# =================================================================
#  原版逐位置卷积反向（与数值校验一致；训练用 fast 版）
# =================================================================

def conv2d_backward(gout, x, wk, pad, stride):
    """gout: (N,O,Oh,Ow) 上游梯度；返回 (gx, gwk, gbk)"""
    N, C, H, W = x.shape
    O, Ck, K, _ = wk.shape
    ph, pw = pad
    sh, sw = stride
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    gx = np.zeros_like(xp)
    gwk = np.zeros_like(wk)
    Oh, Ow = gout.shape[2], gout.shape[3]
    for oh in range(Oh):
        for ow in range(Ow):
            r = xp[:, :, oh * sh:oh * sh + K, ow * sw:ow * sw + K]
            g = gout[:, :, oh, ow]
            gwk += np.einsum("no,nchw->ochw", g, r)
            gx[:, :, oh * sh:oh * sh + K, ow * sw:ow * sw + K] += np.einsum("no,ochw->nchw", g, wk)
    if ph or pw:
        gx = gx[:, :, ph:ph + H, pw:pw + W]
    gbk = gout.sum(axis=(0, 2, 3))
    return gx, gwk, gbk


# =================================================================
#  向量化 im2col 卷积（端到端训练专用，替代逐位置循环）
# =================================================================

def _im2col(x, K, Oh, Ow, sh, sw):
    """把输入展开为 (N, C*K*K, Oh*Ow) 列矩阵。
    col[:, c*K*K+ci*K+cj, l] = x[:, c, oh*sh+ci, ow*sw+cj]  (l = oh*Ow+ow)
    """
    N, C = x.shape[0], x.shape[1]
    L = Oh * Ow
    col = np.empty((N, C * K * K, L), dtype=x.dtype)
    for c in range(C):
        base = c * K * K
        for ci in range(K):
            for cj in range(K):
                view = x[:, c, ci:ci + Oh * sh:sh, cj:cj + Ow * sw:sw]
                col[:, base + ci * K + cj, :] = view.reshape(N, L)
    return col


def conv2d_fast(x, wk, bk, pad, stride):
    """与 conv2d 等价，但用 im2col + matmul 向量化。"""
    N, C, H, W = x.shape
    O, Ck, K, _ = wk.shape
    ph, pw = pad; sh, sw = stride
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    Oh = (H + 2 * ph - K) // sh + 1
    Ow = (W + 2 * pw - K) // sw + 1
    col = _im2col(xp, K, Oh, Ow, sh, sw)                      # (N, CKK, L)
    Wm = wk.reshape(O, C * K * K)                              # (O, CKK)
    out = np.einsum("oc,ncl->nol", Wm, col)                   # (N, O, L)
    return out.reshape(N, O, Oh, Ow) + bk.reshape(1, O, 1, 1)


def conv2d_backward_fast(gout, x, wk, pad, stride):
    """返回 (gx, gwk, gbk)。gout:(N,O,Oh,Ow) 上游梯度。x 为未填充输入。"""
    N, C, H, W = x.shape
    O, Ck, K, _ = wk.shape
    ph, pw = pad; sh, sw = stride
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    Oh, Ow = gout.shape[2], gout.shape[3]
    L = Oh * Ow
    col = _im2col(xp, K, Oh, Ow, sh, sw)                      # (N, CKK, L)
    Gr = gout.reshape(N, O, L)
    Wm = wk.reshape(O, C * K * K)
    # gwk
    gwk = np.einsum("nol,ncl->oc", Gr, col).reshape(O, C, K, K)
    gbk = gout.sum(axis=(0, 2, 3))
    # gcol = dL/dcol，再散射回 xp
    gcol = np.einsum("nol,oc->ncl", Gr, Wm)                   # (N, CKK, L)
    gxp = np.zeros_like(xp)
    for c in range(C):
        base = c * K * K
        for ci in range(K):
            for cj in range(K):
                patch = gcol[:, base + ci * K + cj, :].reshape(N, Oh, Ow)
                sl_h = slice(ci, ci + Oh * sh, sh)
                sl_w = slice(cj, cj + Ow * sw, sw)
                gxp[:, c, sl_h, sl_w] += patch
    if ph or pw:
        gx = gxp[:, :, ph:ph + H, pw:pw + W]
    else:
        gx = gxp
    return gx, gwk, gbk


# =================================================================
#  原版完整 backward（与数值校验一致）+ 完整 fast 版网络
# =================================================================

def backward(loss_grad, c, W, head_nums=("1", "2", "3", "4", "5")):
    """loss_grad: list of dL/dlogit per head (1,C)。返回权重梯度 dict。
    卷积反传用逐位置原版实现（供数值校验对比；训练用 backward_fast）。
    """
    g = {}
    dfeat = np.zeros_like(c["feat"])
    for i, num in enumerate(head_nums):
        dlogit = loss_grad[i]
        g[f"linear{num}.weight"] = np.einsum("nc,nd->cd", dlogit, c["feat"])
        g[f"linear{num}.bias"] = dlogit.sum(0)
        dfeat += dlogit @ W[f"linear{num}.weight"]
    d3 = np.zeros_like(c["l3_out"])
    d3[:, :, :10, :25] = dfeat.reshape(c["feat"].shape[0], -1, 1, 1) / (10 * 25)
    for i in range(2, -1, -1):
        c1 = [266, 275, 281][i]; c2 = c1 + 3; sc = [272, 0, 0][i]
        t = c[f"l3_{i}_t"]; t2 = c[f"l3_{i}_t2"]; s = c[f"l3_{i}_s"]
        xin = c["l2_out"] if i == 0 else c[f"l3_{i-1}_out"]
        ds = d3 * (c[f"l3_{i}_out"] > 0)
        gx_s = None
        if sc != 0:
            gx_s, gwk_s, gbk_s = conv2d_backward(ds, xin, W[f"onnx::Conv_{sc}"], (0, 0), (2, 2))
            g[f"onnx::Conv_{sc}"] = gwk_s; g[f"onnx::Conv_{sc+1}"] = gbk_s
        gx_t2, gwk2, gbk2 = conv2d_backward(ds, t, W[f"onnx::Conv_{c2}"], (1, 1), (1, 1))
        gx_t = gx_t2 * (t > 0)
        s1 = (2, 2) if i == 0 else (1, 1)
        gx_prev, gwk1, gbk1 = conv2d_backward(gx_t, xin, W[f"onnx::Conv_{c1}"], (1, 1), s1)
        g[f"onnx::Conv_{c1}"] = gwk1; g[f"onnx::Conv_{c1+1}"] = gbk1
        g[f"onnx::Conv_{c2}"] = gwk2; g[f"onnx::Conv_{c2+1}"] = gbk2
        d3 = gx_prev + (gx_s if gx_s is not None else ds)
    d2 = d3
    for i in range(2, -1, -1):
        c1 = [245, 254, 260][i]; c2 = c1 + 3; sc = [251, 0, 0][i]
        t = c[f"l2_{i}_t"]; t2 = c[f"l2_{i}_t2"]; s = c[f"l2_{i}_s"]
        xin = c["l1_out"] if i == 0 else c[f"l2_{i-1}_out"]
        ds = d2 * (c[f"l2_{i}_out"] > 0)
        gx_s = None
        if sc != 0:
            gx_s, gwk_s, gbk_s = conv2d_backward(ds, xin, W[f"onnx::Conv_{sc}"], (0, 0), (2, 2))
            g[f"onnx::Conv_{sc}"] = gwk_s; g[f"onnx::Conv_{sc+1}"] = gbk_s
        gx_t2, gwk2, gbk2 = conv2d_backward(ds, t, W[f"onnx::Conv_{c2}"], (1, 1), (1, 1))
        gx_t = gx_t2 * (t > 0)
        s1 = (2, 2) if i == 0 else (1, 1)
        gx_prev, gwk1, gbk1 = conv2d_backward(gx_t, xin, W[f"onnx::Conv_{c1}"], (1, 1), s1)
        g[f"onnx::Conv_{c1}"] = gwk1; g[f"onnx::Conv_{c1+1}"] = gbk1
        g[f"onnx::Conv_{c2}"] = gwk2; g[f"onnx::Conv_{c2+1}"] = gbk2
        d2 = gx_prev + (gx_s if gx_s is not None else ds)
    d1 = d2
    for i in range(2, -1, -1):
        c1 = [227, 233, 239][i]; c2 = c1 + 3
        t = c[f"l1_{i}_t"]; t2 = c[f"l1_{i}_t2"]
        xin = c["a1"] if i == 0 else c[f"l1_{i-1}_out"]
        ds = d1 * (c[f"l1_{i}_out"] > 0)
        gx_t2, gwk2, gbk2 = conv2d_backward(ds, t, W[f"onnx::Conv_{c2}"], (1, 1), (1, 1))
        gx_t = gx_t2 * (t > 0)
        gx_prev, gwk1, gbk1 = conv2d_backward(gx_t, xin, W[f"onnx::Conv_{c1}"], (1, 1), (1, 1))
        g[f"onnx::Conv_{c1}"] = gwk1; g[f"onnx::Conv_{c1+1}"] = gbk1
        g[f"onnx::Conv_{c2}"] = gwk2; g[f"onnx::Conv_{c2+1}"] = gbk2
        d1 = gx_prev + ds
    gx_conv1, gwk0, gbk0 = conv2d_backward(d1 * (c["a1"] > 0), c["x0"], W["onnx::Conv_224"], (1, 1), (1, 1))
    g["onnx::Conv_224"] = gwk0; g["onnx::Conv_225"] = gbk0
    return g


# ---------- 完整 fast 前向 ----------

def forward_fast(x, W):
    """与 forward 等价，卷积用向量化 im2col。返回相同激活缓存 dict。"""
    c = {}
    c["x0"] = x
    c["h1"] = conv2d_fast(x, W["onnx::Conv_224"], W["onnx::Conv_225"], (1, 1), (1, 1))
    c["a1"] = relu(c["h1"])
    h = c["a1"]
    for i, (c1, c2) in enumerate([(227, 230), (233, 236), (239, 242)]):
        t = relu(conv2d_fast(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"], (1, 1), (1, 1)))
        t2 = conv2d_fast(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"], (1, 1), (1, 1))
        h = relu(t2 + h)
        c[f"l1_{i}_t"] = t; c[f"l1_{i}_t2"] = t2; c[f"l1_{i}_out"] = h
    c["l1_out"] = h
    for i, (c1, c2, sc) in enumerate([(245, 248, 251), (254, 257, 0), (260, 263, 0)]):
        s1 = (2, 2) if i == 0 else (1, 1)
        t = relu(conv2d_fast(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"], (1, 1), s1))
        t2 = conv2d_fast(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"], (1, 1), (1, 1))
        s = h if sc == 0 else conv2d_fast(h, W[f"onnx::Conv_{sc}"], W[f"onnx::Conv_{sc+1}"], (0, 0), (2, 2))
        h = relu(t2 + s)
        c[f"l2_{i}_t"] = t; c[f"l2_{i}_t2"] = t2; c[f"l2_{i}_s"] = s; c[f"l2_{i}_out"] = h
    c["l2_out"] = h
    for i, (c1, c2, sc) in enumerate([(266, 269, 272), (275, 278, 0), (281, 284, 0)]):
        s1 = (2, 2) if i == 0 else (1, 1)
        t = relu(conv2d_fast(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"], (1, 1), s1))
        t2 = conv2d_fast(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"], (1, 1), (1, 1))
        s = h if sc == 0 else conv2d_fast(h, W[f"onnx::Conv_{sc}"], W[f"onnx::Conv_{sc+1}"], (0, 0), (2, 2))
        h = relu(t2 + s)
        c[f"l3_{i}_t"] = t; c[f"l3_{i}_t2"] = t2; c[f"l3_{i}_s"] = s; c[f"l3_{i}_out"] = h
    c["l3_out"] = h
    c["feat"] = avgpool(h).reshape(x.shape[0], -1)
    return c


# ---------- 完整 fast 反向 ----------
# 只能复用 backward_fast_joint（forward_fast 的缓存结构一致）

def backward_fast(loss_grad, c, W, head_nums=("1", "2", "3", "4", "5")):
    """与 backward 等价，卷积反传用向量化 im2col。"""
    g = {}
    dfeat = np.zeros_like(c["feat"])
    for i, num in enumerate(head_nums):
        dlogit = loss_grad[i]
        g[f"linear{num}.weight"] = np.einsum("nc,nd->cd", dlogit, c["feat"])
        g[f"linear{num}.bias"] = dlogit.sum(0)
        dfeat += dlogit @ W[f"linear{num}.weight"]
    d3 = np.zeros_like(c["l3_out"])
    d3[:, :, :10, :25] = dfeat.reshape(c["feat"].shape[0], -1, 1, 1) / (10 * 25)
    for i in range(2, -1, -1):
        c1 = [266, 275, 281][i]; c2 = c1 + 3; sc = [272, 0, 0][i]
        t = c[f"l3_{i}_t"]; t2 = c[f"l3_{i}_t2"]; s = c[f"l3_{i}_s"]
        xin = c["l2_out"] if i == 0 else c[f"l3_{i-1}_out"]
        ds = d3 * (c[f"l3_{i}_out"] > 0)
        gx_s = None
        if sc != 0:
            gx_s, gwk_s, gbk_s = conv2d_backward_fast(ds, xin, W[f"onnx::Conv_{sc}"], (0, 0), (2, 2))
            g[f"onnx::Conv_{sc}"] = gwk_s; g[f"onnx::Conv_{sc+1}"] = gbk_s
        gx_t2, gwk2, gbk2 = conv2d_backward_fast(ds, t, W[f"onnx::Conv_{c2}"], (1, 1), (1, 1))
        gx_t = gx_t2 * (t > 0)
        s1 = (2, 2) if i == 0 else (1, 1)
        gx_prev, gwk1, gbk1 = conv2d_backward_fast(gx_t, xin, W[f"onnx::Conv_{c1}"], (1, 1), s1)
        g[f"onnx::Conv_{c1}"] = gwk1; g[f"onnx::Conv_{c1+1}"] = gbk1
        g[f"onnx::Conv_{c2}"] = gwk2; g[f"onnx::Conv_{c2+1}"] = gbk2
        d3 = gx_prev + (gx_s if gx_s is not None else ds)
    d2 = d3
    for i in range(2, -1, -1):
        c1 = [245, 254, 260][i]; c2 = c1 + 3; sc = [251, 0, 0][i]
        t = c[f"l2_{i}_t"]; t2 = c[f"l2_{i}_t2"]; s = c[f"l2_{i}_s"]
        xin = c["l1_out"] if i == 0 else c[f"l2_{i-1}_out"]
        ds = d2 * (c[f"l2_{i}_out"] > 0)
        gx_s = None
        if sc != 0:
            gx_s, gwk_s, gbk_s = conv2d_backward_fast(ds, xin, W[f"onnx::Conv_{sc}"], (0, 0), (2, 2))
            g[f"onnx::Conv_{sc}"] = gwk_s; g[f"onnx::Conv_{sc+1}"] = gbk_s
        gx_t2, gwk2, gbk2 = conv2d_backward_fast(ds, t, W[f"onnx::Conv_{c2}"], (1, 1), (1, 1))
        gx_t = gx_t2 * (t > 0)
        s1 = (2, 2) if i == 0 else (1, 1)
        gx_prev, gwk1, gbk1 = conv2d_backward_fast(gx_t, xin, W[f"onnx::Conv_{c1}"], (1, 1), s1)
        g[f"onnx::Conv_{c1}"] = gwk1; g[f"onnx::Conv_{c1+1}"] = gbk1
        g[f"onnx::Conv_{c2}"] = gwk2; g[f"onnx::Conv_{c2+1}"] = gbk2
        d2 = gx_prev + (gx_s if gx_s is not None else ds)
    d1 = d2
    for i in range(2, -1, -1):
        c1 = [227, 233, 239][i]; c2 = c1 + 3
        t = c[f"l1_{i}_t"]; t2 = c[f"l1_{i}_t2"]
        xin = c["a1"] if i == 0 else c[f"l1_{i-1}_out"]
        ds = d1 * (c[f"l1_{i}_out"] > 0)
        gx_t2, gwk2, gbk2 = conv2d_backward_fast(ds, t, W[f"onnx::Conv_{c2}"], (1, 1), (1, 1))
        gx_t = gx_t2 * (t > 0)
        gx_prev, gwk1, gbk1 = conv2d_backward_fast(gx_t, xin, W[f"onnx::Conv_{c1}"], (1, 1), (1, 1))
        g[f"onnx::Conv_{c1}"] = gwk1; g[f"onnx::Conv_{c1+1}"] = gbk1
        g[f"onnx::Conv_{c2}"] = gwk2; g[f"onnx::Conv_{c2+1}"] = gbk2
        d1 = gx_prev + ds
    gx_conv1, gwk0, gbk0 = conv2d_backward_fast(d1 * (c["a1"] > 0), c["x0"], W["onnx::Conv_224"], (1, 1), (1, 1))
    g["onnx::Conv_224"] = gwk0; g["onnx::Conv_225"] = gbk0
    return g
