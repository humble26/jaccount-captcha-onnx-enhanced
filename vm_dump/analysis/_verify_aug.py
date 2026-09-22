# -*- coding: utf-8 -*-
"""机制验证: 增广难例能否推动 h030/h040 位4 边界。
训练集 = 420 原始 + 75 增广难例。只跑 60 步, 观察 h030/h040 位4 是否翻转。
若翻转 -> 增广方向有效, 值得完整重训; 否则止损放弃。
"""
import os, json, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_np import (forward_fast, backward_fast, forward_heads,
                         load_onnx_weights, CH)
import train_e2e as T

HARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hardaug")

# 训练集: 420 原始
keys, x, targets = T.load_split("tune", extra=True)
n0 = x.shape[0]
print(f"原始训练集 {n0} 张")

# 加载 75 增广难例
hard = json.load(open(os.path.join(HARD, "hard_list.json"), encoding="utf-8"))
xa = np.array([np.load(os.path.join(HARD, h["file"])) for h in hard])[..., None]  # (N,40,110,1)? -> 需 (N,1,40,110)
# np.load 存的是 (40,110) float32 二值图; 需转 (N,1,40,110)
xa = np.moveaxis(xa, -1, 1)  # -> 不确定, 直接重构
xa = np.array([np.load(os.path.join(HARD, h["file"])) for h in hard])  # (N,40,110)
xa = xa[:, None, :, :].astype(np.float64)  # (N,1,40,110)
print(f"增广难例 {xa.shape[0]} 张")
lab_a = np.array([CH.index(h["truth4"]) if h["truth4"] in CH else 26 for h in hard])

# 合并: 位4真值用各自真值, 其它位用原图? 增广只有位4改变, 其它位用种子样本的真值有风险。
# 简化: 增广难例只用于位4, 其它位 target 直接用它自己的位4(即只影响位4损失, 其它头损失用占位让其不计权重)。
# 更稳妥: 增广难例所有5位 target 都设为忽略(用种子原图其它位真值? 我们没有种子配对)。
# 所以: 对增广样本, 位4 用真值, 其它4位用特殊标记26(blank好像不对)。改用 pos4-only 加权掩盖。
# 方案: 把增广样本位4真值放数组, 其它位给一个不会产生梯度的处理 -> 直接只在 pos4 头给增广样本损失。

# 组装成大数组 + 掩码标记哪些样本是"纯位4样本"(idx >= n0)
nA = n0 + xa.shape[0]
XB = np.concatenate([x, xa], axis=0)
# targets per head
T4 = np.concatenate([targets[3], lab_a])
# 其它头对原始用真值, 对增广用 seed 真值? 简化: 用各自 pos4; 但对增广样本其它头设 target=26(blank)
# 第5头27类(含blank), 前4头26类无blank. 设26会越界前4头. 因此对增广样本的非位4头, 用 seed 样本真值。
# 这里我们没有逐样本 seed 配对(multi-point), 但每个增广 h['key'] 记录了来源 key。
# 用来源 key 在原始集里查其各头真值。
orig_by_key = {keys[i]: [targets[p][i] for p in range(5)] for i in range(n0)}
newkeys = keys + [h["key"] + "#a" + str(i) for i, h in enumerate(hard)]
ta = np.array([[orig_by_key.get(h["key"], [26, 26, 26, CH.index(h["truth4"]) if h["truth4"] in CH else 26, 26])[p]
                for p in range(5)] for h in hard])
# 构造每头 target 数组
TT = []
for p in range(5):
    if p == 3:
        TT.append(np.concatenate([targets[3], ta[:, 3]]))
    else:
        TT.append(np.concatenate([targets[p], ta[:, p]]))

print("组装完成, 训练集", XB.shape)

# 留出集
hold_keys, hold_x, hold_t = T.load_split("hold")

W = {k: v.astype(np.float64) for k, v in
     T.filter_trainable({k2: v for k2, v in load_onnx_weights(T.MODEL).items()}).items()}
print("已从生产ONNX初始化权重")


def acc_and_h(weight):
    c = forward_fast(hold_x, weight)
    lg = forward_heads(c["feat"], weight)
    p4 = np.argmax(lg[3], 1)
    acc4 = float((p4 == hold_t[3]).mean())
    info = {}
    for kk in ["h030", "h040"]:
        i = hold_keys.index(kk)
        info[kk] = (CH[hold_t[3][i]], CH[int(p4[i])])
    return acc4, info


# 训练前的留出位4 (从原生权重)
a0, i0 = acc_and_h(W)
print(f"原生 留出位4={a0*100:.2f}%   h030位4={i0['h030'][0]}->{i0['h030'][1]}  h040位4={i0['h040'][0]}->{i0['h040'][1]}")

lr = 3e-6
rng = np.random.default_rng(42)
for step in range(1, 61):
    idx = rng.choice(nA, size=32, replace=False)
    xb, tb = XB[idx], [t[idx] for t in TT]
    c = forward_fast(xb, W)
    logits = forward_heads(c["feat"], W)
    loss = 0.0
    dlogits = []
    for pos in range(5):
        z = logits[pos]
        C = T.NUM_CLASSES[pos]
        ze = np.exp(z - z.max(1, keepdims=True)); p = ze / ze.sum(1, keepdims=True)
        target = tb[pos]
        l = -np.log(p[np.arange(len(z)), target] + 1e-12).mean()
        w = 2.0 if pos == 3 else 1.0
        loss += w * l
        oh = np.zeros_like(p); oh[np.arange(len(z)), target] = 1.0
        dlogits.append((p - oh) / len(z) * w)
    g = backward_fast(dlogits, c, W)
    for k, v in W.items():
        W[k] = v - lr * g[k]
    if step % 12 == 0 or step == 60:
        a, info = acc_and_h(W)
        print(f"step{step:3d} loss={loss:.4f}  留出位4={a*100:.2f}%  h030={info['h030'][0]}->{info['h030'][1]}  h040={info['h040'][0]}->{info['h040'][1]}", flush=True)

# 最终
a, info = acc_and_h(W)
print(f"\n60步后 留出位4={a*100:.2f}%  h030={info['h030'][0]}->{info['h030'][1]}  h040={info['h040'][0]}->{info['h040'][1]}")
print("验证结论:", "增广数据能推动边界 -> 值得完整重训" if (info['h030'][0] == info['h030'][1] and info['h040'][0] == info['h040'][1])
      else "增广难例未能修正h030/h040 -> 需换策略")