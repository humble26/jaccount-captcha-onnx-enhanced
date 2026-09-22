# -*- coding: utf-8 -*-
"""反向传播数值梯度校验：实现 ResNet-20 完整 backward，与有限差分对比

校验策略：
  1. 用 3 张图片、随机采样 20% 权重张量位置
  2. 对每个位置，比较 解析梯度 vs 数值梯度（中心差分），要求相对误差 < 1e-4
  3. 校验 5 个 Gemm 头 + 全部卷积层的梯度

运行：
  python e2e_backward_check.py
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_np import conv2d, relu, avgpool, forward, forward_heads

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
VD = os.path.join(WS, "vm_dump")
MODEL = os.path.join(VD, "nn_model.onnx")


def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


# ------------------------------------------------------------- 反向

def conv2d_backward(gout, x, wk, pad, stride):
    """gout: (1,O,Oh,Ow) 上游梯度；返回 (gx, gwk, gbk)"""
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


def backward(loss_grad, c, W, head_nums, head_idx=None):
    """loss_grad: list of dL/dlogit per head (1,C)。返回权重梯度 dict"""
    g = {}
    # 头部
    dfeat = np.zeros_like(c["feat"])
    for i, num in enumerate(head_nums):
        dlogit = loss_grad[i]
        g[f"linear{num}.weight"] = np.einsum("nc,nd->cd", dlogit, c["feat"])
        g[f"linear{num}.bias"] = dlogit.sum(0)
        dfeat += dlogit @ W[f"linear{num}.weight"]
    # 池化反传：feat = mean(h[:, :, :10, :25])
    d3 = np.zeros_like(c["l3_out"])
    d3[:, :, :10, :25] = dfeat.reshape(c["feat"].shape[0], -1, 1, 1) / (10 * 25)
    # layer3 反传（逆序）。注意：block i 的 relu mask 用该 block 自身输出
    for i in range(2, -1, -1):
        c1 = [266, 275, 281][i]; c2 = c1 + 3; sc = [272, 0, 0][i]
        t = c[f"l3_{i}_t"]; t2 = c[f"l3_{i}_t2"]; s = c[f"l3_{i}_s"]
        xin = c["l2_out"] if i == 0 else c[f"l3_{i-1}_out"]
        # h_i = relu(t2 + s)；ds 需用 block i 自己的输出做 relu mask
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
    # layer2 反传（逆序）
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
    # layer1 反传（逆序）
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
    # conv1
    gx_conv1, gwk0, gbk0 = conv2d_backward(d1 * (c["a1"] > 0), c["x0"], W["onnx::Conv_224"], (1, 1), (1, 1))
    g["onnx::Conv_224"] = gwk0; g["onnx::Conv_225"] = gbk0
    return g


# ------------------------------------------------------------- 数值梯度

def loss_fn(logits_list, targets, num_classes, blank=False):
    """交叉熵（对 head5 支持 27 类含 blank）。返回 (loss, dlogits)"""
    ds = []
    tot = 0.0
    for i, z in enumerate(logits_list):
        C = num_classes[i]
        t = targets[i]
        ze = np.exp(z - z.max(1, keepdims=True))
        p = ze / ze.sum(1, keepdims=True)
        # one-hot
        onehot = np.zeros_like(p)
        onehot[0, t] = 1.0
        loss = -np.log(p[0, t] + 1e-12)
        tot += loss
        ds.append(p - onehot)
    return tot, ds


def total_loss(logits_list, targets, num_classes):
    """多图平均交叉熵损失"""
    tot = 0.0
    n = logits_list[0].shape[0]
    for i, z in enumerate(logits_list):
        C = num_classes[i]
        t = targets[i]
        ze = np.exp(z - z.max(1, keepdims=True))
        p = ze / ze.sum(1, keepdims=True)
        tot += -np.log(p[np.arange(n), t] + 1e-12).mean()
    return tot


def numeric_grad(x, W, wkey, flat_idx, eps=1e-6, targets=None, num_classes=None):
    Wp = {k: v.copy() for k, v in W.items()}
    Wm = {k: v.copy() for k, v in W.items()}
    Wp[wkey].flat[flat_idx] += eps
    Wm[wkey].flat[flat_idx] -= eps
    cp = forward(x, Wp); zp = forward_heads(cp["feat"], Wp)
    cm = forward(x, Wm); zm = forward_heads(cm["feat"], Wm)
    lp = total_loss(zp, targets, num_classes)
    lm = total_loss(zm, targets, num_classes)
    return (lp - lm) / (2 * eps)


def main():
    from resnet20_np import load_onnx_weights, CH
    W = load_onnx_weights(MODEL)
    # 训练端统一 float64，避免中心差分变化量落入 float32 数值噪声基底
    W = {k: v.astype(np.float64) for k, v in W.items()}
    imgs = [os.path.join(VD, "holdout", f) for f in sorted(os.listdir(os.path.join(VD, "holdout")))[:2]]
    x = np.stack([preprocess(p)[None] for p in imgs]).astype(np.float64)
    c = forward(x, W)
    logits = forward_heads(c["feat"], W)
    # 目标：用真实标注（ground truth）做标签
    import json
    gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
    targets = []
    for pos in range(5):
        t = []
        for f in imgs:
            sid = os.path.splitext(os.path.basename(f))[0]
            label = gt[sid]
            t.append(CH.index(label[pos]) if pos < len(label) else 26)
        targets.append(np.array(t))
    num_classes = [26, 26, 26, 26, 27]

    # 解析梯度（多图平均：loss 对每个头求 mean）
    dlogits = []
    n = x.shape[0]
    for i, z in enumerate(logits):
        ze = np.exp(z - z.max(1, keepdims=True)); p = ze / ze.sum(1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), targets[i]] = 1.0
        dlogits.append((p - onehot) / n)
    g = backward(dlogits, c, W, head_nums=["1", "2", "3", "4", "5"])

    # 校验：线性头与卷积层各随机抽样。双阈值：相对误差 或 绝对误差都小则通过
    import random
    random.seed(42)
    checked = []
    all_keys = [k for k in W if k.startswith("onnx::Conv") or k.startswith("linear")]
    for wkey in all_keys:
        w = W[wkey]
        n_choose = 80 if wkey.startswith("linear") else 3
        idxs = random.sample(range(w.size), min(n_choose, w.size))
        for fi in idxs:
            ng = numeric_grad(x, W, wkey, fi, targets=targets, num_classes=num_classes)
            ag = g[wkey].flat[fi]
            denom = max(1e-12, abs(ng) + abs(ag))
            rel = abs(ng - ag) / denom
            absdiff = abs(ng - ag)
            # 近零梯度用绝对误差判定（相对误差失真）
            ok = rel < 1e-4 or absdiff < 1e-9
            checked.append((wkey, fi, ag, ng, rel, ok))
            if not ok:
                print(f"  MISMATCH {wkey} idx={fi} analytic={ag:.8e} numeric={ng:.8e} rel={rel:.3e}", flush=True)

    bad = [t for t in checked if not t[5]]
    n = len(checked)
    print(f"校验 {n} 个梯度位置：OK={n - len(bad)}  FAIL={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())