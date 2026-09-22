import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""架构突破的定量判定 —— 决定性实验

已确认的真实情况（看 mask_compare.png）：
  * 白底 + 灰色/黑色细笔画字符，笔画宽仅 1~2px，字形高 14~18px（画布 40px）
  * 字符有斜体倾斜，灰度渐变（同一字符左深右浅）
  * 模型：ResNet-20 backbone -> AveragePool 全局池化 -> Reshape (1,64) -> 5 个并联 Gemm 头
  * 权威结果 215/220 = 97.73%，全部错误在第 4 位

架构级要回答的两个问题：
  Q-A  「全局池化 + 5 并联头」是否丢了位置信息？—— 对比"保留空间"的方案
  Q-B  输入分辨率（40x110，字形仅 14~18px 高、笔画 1~2px）是否是硬瓶颈？
"""
import os, json
import numpy as np
import onnxruntime as rt
from PIL import Image

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"
gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
paths = {}
for d in ["samples", "samples2", "holdout"]:
    dd = os.path.join(VD, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)
keys = [k for k in gt if k in paths]

def img_ge(k):
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    return (g >= 156).astype(np.float32)

def ink(k):
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    return (g < 156).astype(np.float32)

# 中间特征模型
import onnx
m = onnx.load(os.path.join(VD, "nn_model.onnx"))
FEAT = "/Reshape_output_0"
if not any(o.name == FEAT for o in m.graph.output):
    vi = onnx.helper.ValueInfoProto(); vi.name = FEAT
    m.graph.output.append(vi)
TMP = os.path.join(VD, "_tmp_feat4.onnx"); onnx.save(m, TMP)
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(TMP, so, providers=["CPUExecutionProvider"])
IN = "input.1"
ON = [o.name for o in sess.get_outputs()]; FI = ON.index(FEAT)

err_keys = sorted([k for k in keys if json.load(open(os.path.join(VD,"holdout_eval.json"),encoding="utf-8"))["pred"]["base"].get(k) != gt[k]])
pred_base = json.load(open(os.path.join(VD, "holdout_eval.json"), encoding="utf-8"))["pred"]["base"]

print("=" * 88)
print("【Q-A】全局池化 vs 空间保留 —— 用中间特征验证位置信息是否被抹掉")
print("=" * 88)
print("  原理：若全局池化抹掉了位置信息，则「把第 4 槽单独挖掉/替换」")
print("       不应改变前 3 位的输出；反之若空间信息保留，则会影响。")
print("  更强的判据：把整张图沿横向切成 5 份，看第 4 头对「第 4 份」是否敏感。")

def run_full(k):
    return sess.run(ON, {IN: img_ge(k)[None, None]})

# 判据 1：把第 4 槽区域用背景填充，看各头输出变化
print()
print("  判据 1：抹掉第 4 槽区域（用背景 1.0 填充），各头 top1 变化")
C5 = [5.6, 26.6, 47.7, 68.8, 89.9]
C4 = [5.5, 28.8, 51.7, 74.9]
changed = {nm: 0 for nm in ON if nm != FEAT}
for k in keys:
    t = gt[k]
    cs = C4 if len(t) == 4 else C5
    a = img_ge(k)
    b = a.copy()
    if len(t) > 3:
        cx = cs[3]; lo, hi = max(0, int(cx - 11)), min(110, int(cx + 11))
        b[:, lo:hi] = 1.0
    o1 = sess.run(ON, {IN: a[None, None]}); o2 = sess.run(ON, {IN: b[None, None]})
    for i, nm in enumerate(ON):
        if nm == FEAT: continue
        if int(np.argmax(o1[i][0])) != int(np.argmax(o2[i][0])):
            changed[nm] += 1
for nm in ON:
    if nm == FEAT: continue
    print(f"    头{nm}: {changed[nm]:3d}/{len(keys)} 张输出改变")

# 判据 2：横向切分 —— 分别用小图像推理，看各头反应
print()
print("  判据 2：只保留第 4 槽（其余置背景），看第 4 头是否给出正确的第 4 位字符")
only4_hit = 0; only4_tot = 0
for k in keys:
    t = gt[k]
    if len(t) <= 3: continue
    cs = C4 if len(t) == 4 else C5
    a = img_ge(k).copy()
    cx = cs[3]; lo, hi = max(0, int(cx - 12)), min(110, int(cx + 12))
    b = np.ones_like(a); b[:, lo:hi] = a[:, lo:hi]
    o = sess.run(ON, {IN: b[None, None]})
    idx = int(np.argmax(o[3][0]))
    only4_tot += 1
    if idx < 26 and CH[idx] == t[3]:
        only4_hit += 1
print(f"    第 4 头在「只留第 4 槽」时准确率: {only4_hit}/{only4_tot} = {only4_hit/only4_tot*100:.1f}%")
print(f"    (对比：完整图时第 4 头准确率 215/220 = 97.7%)")

print()
print("=" * 88)
print("【Q-B】输入分辨率瓶颈 —— 笔画宽度实测")
print("=" * 88)
def stroke_width(k):
    b = ink(k)
    widths = []
    for c in range(110):
        col = b[:, c]
        if col.sum() == 0: continue
        # 连续段长度
        runs = []
        cur = 0
        for v in col:
            if v: cur += 1
            elif cur: runs.append(cur); cur = 0
        if cur: runs.append(cur)
        widths.extend(runs)
    return widths

allw = []
for k in keys:
    allw.extend(stroke_width(k))
allw = np.array(allw)
print(f"  竖向笔画宽度样本 {len(allw)} 个")
print(f"    均值 {allw.mean():.2f}px  中位 {np.median(allw):.1f}px")
for w in range(1, 6):
    print(f"    宽度 {w}px: {(allw==w).sum():5d} ({(allw==w).mean()*100:5.1f}%)")
print(f"    <=2px 占比: {(allw<=2).mean()*100:.1f}%")

# 字形高度
hs = []
for k in keys:
    b = ink(k)
    rows = np.where(b.any(axis=1))[0]
    if len(rows): hs.append(rows.max() - rows.min() + 1)
hs = np.array(hs)
print(f"  字形高度: 均值 {hs.mean():.1f}px  范围 {hs.min()}~{hs.max()}px (画布 40px)")
print(f"  字形宽度: ", end="")
ws = []
for k in keys:
    b = ink(k)
    cols = np.where(b.any(axis=0))[0]
    if len(cols): ws.append(cols.max() - cols.min() + 1)
ws = np.array(ws)
print(f"均值 {ws.mean():.1f}px  范围 {ws.min()}~{ws.max()}px")
nch = np.array([len(gt[k]) for k in keys])
print(f"  单字符平均宽度 ≈ {ws.mean()/nch.mean():.1f}px，高 {hs.mean():.0f}px")
print(f"  -> 当前 40x110 画布对字形来说{'偏小' if hs.mean()>34 else '尚有裕量'}")

print()
print("=" * 88)
print("【Q-C】灰度信息的损失 —— 当前只用 1-bit 二值图，信息丢了多少？")
print("=" * 88)
# 灰度版本 vs 二值版本的信息量
g_sum = 0; b_sum = 0
for k in keys[:50]:
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    inkp = (g < 156).astype(np.float32)
    # 灰度图的笔画区域包含了"深浅"信息（抗锯齿/渐变）
    # 量化：笔画区域内的灰度级数
    vals = g[inkp > 0]
    if len(vals):
        g_sum += len(np.unique(vals))
    b_sum += 1
print(f"  笔画区域内灰度级数（50 张均值）: {g_sum/50:.1f} 级")
print(f"  二值化后: 1 级（全部压成同一值）")
print(f"  -> 每个笔画像素损失 log2({g_sum/50:.0f}) ≈ {np.log2(max(g_sum/50,1)):.1f} bits 的灰度信息")
print(f"  -> 抗锯齿边缘（决定 x/o、c/o 这类细微差异的关键）被完全抹平")
