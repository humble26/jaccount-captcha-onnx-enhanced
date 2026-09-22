import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""架构瓶颈判定（修正版：使用权威预处理 g>=156）

已确认的架构事实：
  * ResNet-20 变体：21 Conv + 19 Relu + 9 Add + 1 AveragePool + 1 Reshape + 5 Gemm
  * AveragePool 全局池化 -> Reshape 成 (1,64)
  * 5 个输出头（Gemm，各 [26|27, 64]）**全部并联在同一个 64 维向量**上

核心判定问题：
  Q1  全局池化后，64 维向量里还能否区分第 4 位字符？（像素空间 vs 特征空间）
  Q2  如果特征空间不可分但像素空间可分 -> backbone 是瓶颈，换骨架有救
  Q3  如果两者都不可分 -> 分辨率/数据是瓶颈，换骨架无用
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
keys = [k for k in gt if k in paths]


def img(k):
    """权威预处理：>=156 为前景（黑底白字）"""
    g = np.asarray(Image.open(paths[k]).convert("L"), dtype=np.float32)
    return (g >= 156).astype(np.float32)


# 带中间特征的模型
import onnx
m = onnx.load(os.path.join(VD, "nn_model.onnx"))
FEAT = "/Reshape_output_0"
if not any(o.name == FEAT for o in m.graph.output):
    vi = onnx.helper.ValueInfoProto(); vi.name = FEAT
    m.graph.output.append(vi)
TMP = os.path.join(VD, "_tmp_feat3.onnx")
onnx.save(m, TMP)
so = rt.SessionOptions(); so.log_severity_level = 3
sess = rt.InferenceSession(TMP, so, providers=["CPUExecutionProvider"])
IN = "input.1"
ONAMES = [o.name for o in sess.get_outputs()]
FI = ONAMES.index(FEAT)

print("=" * 88)
print("【Q1】64 维全局特征里，各位置字符的可分性（留一最近类中心）")
print("=" * 88)
FEATS = np.array([sess.run(ONAMES, {IN: img(k)[None, None]})[FI][0] for k in keys])
print(f"  特征矩阵 {FEATS.shape}")

def loo_acc(F, Y):
    cls = sorted(set(Y))
    mu = {c: F[Y == c].mean(axis=0) for c in cls}
    ok = 0
    for i in range(len(F)):
        d = {c: np.linalg.norm(F[i] - mu[c]) for c in cls}
        ok += int(min(d, key=d.get) == Y[i])
    return ok / len(F)

for pos in range(5):
    Yp = [gt[k][pos] for k in keys if len(gt[k]) > pos]
    Xp = np.array([FEATS[i] for i, k in enumerate(keys) if len(gt[k]) > pos])
    print(f"  位{pos+1}: {len(Yp):3d} 样本 -> 64维特征可分率 {loo_acc(Xp, Yp)*100:6.2f}%")

print()
print("=" * 88)
print("【Q1b】对照：原始像素空间里同样做（用正确的 >=156 掩码）")
print("=" * 88)
def slot(k, idx):
    C4 = [5.5, 28.8, 51.7, 74.9]
    C5 = [5.6, 26.6, 47.7, 68.8, 89.9]
    cs = C4 if len(gt[k]) == 4 else C5
    cx = cs[idx]
    lo, hi = max(0, int(cx - 10)), min(110, int(cx + 10))
    c = img(k)[:, lo:hi]
    if c.shape[1] < 20:
        c = np.hstack([np.zeros((40, 20 - c.shape[1]), np.float32), c])
    return c[:, :20].ravel()

for pos in range(5):
    sel = [k for k in keys if len(gt[k]) > pos]
    if len(sel) < 30: continue
    X = np.array([slot(k, pos) for k in sel]); Y = [gt[k][pos] for k in sel]
    print(f"  位{pos+1}: {len(Y):3d} 样本 -> 40x20 像素可分率 {loo_acc(X, Y)*100:6.2f}%")

print()
print("=" * 88)
print("【Q2】第 4 位错误的像素可分性（修正掩码后重做）")
print("=" * 88)
err_keys = sorted([k for k in keys if pred_base.get(k) != gt[k]])
sel4 = [k for k in keys if len(gt[k]) > 3]
X4 = np.array([slot(k, 3) for k in sel4]); Y4 = [gt[k][3] for k in sel4]
mu4 = {c: X4[[i for i, y in enumerate(Y4) if y == c]].mean(axis=0) for c in set(Y4)}
print("  错误样本第 4 槽像素距离:")
for k in err_keys:
    if len(gt[k]) <= 3: continue
    v = slot(k, 3)
    tv, pv = gt[k][3], pred_base[k][3]
    dt, dp = np.linalg.norm(v - mu4[tv]), np.linalg.norm(v - mu4[pv])
    print(f"    [{k}] 真值={tv} 预测={pv}  到真值 {dt:7.3f}  到误判 {dp:7.3f}  "
          f"-> {'像素可分' if dt<dp else '像素不可分'}")

print()
print("  对照：第 4 位正确样本的 margin 分布")
mg = []
for k in sel4:
    if k in err_keys: continue
    v = slot(k, 3); tv = gt[k][3]
    dt = np.linalg.norm(v - mu4[tv])
    do = min(np.linalg.norm(v - mu4[c]) for c in mu4 if c != tv)
    mg.append(do - dt)
mg = np.array(mg)
print(f"    正确样本 margin: 均值 {mg.mean():.3f} 中位 {np.median(mg):.3f} 最小 {mg.min():.3f}")
print(f"    margin<=0 的: {(mg<=0).sum()}/{len(mg)} ({(mg<=0).mean()*100:.1f}%)")

print()
print("=" * 88)
print("【Q3】分辨率瓶颈：把第 4 槽纵向放大 2 倍后，像素可分性是否改善？")
print("=" * 88)
print("  目的：检验「40px 高度不够，笔画细节被压扁」这一假设")
def slot_up(k, idx, f=2):
    s = slot(k, idx).reshape(40, 20)
    up = np.repeat(np.repeat(s, f, axis=0), f, axis=1)
    return up.ravel()

X4u = np.array([slot_up(k, 3) for k in sel4])
mu4u = {c: X4u[[i for i, y in enumerate(Y4) if y == c]].mean(axis=0) for c in set(Y4)}
mg_u = []
for k in sel4:
    if k in err_keys: continue
    v = slot_up(k, 3); tv = gt[k][3]
    dt = np.linalg.norm(v - mu4u[tv])
    do = min(np.linalg.norm(v - mu4u[c]) for c in mu4u if c != tv)
    mg_u.append(do - dt)
mg_u = np.array(mg_u)
print(f"  放大后正确样本 margin: 均值 {mg_u.mean():.3f} 中位 {np.median(mg_u):.3f} 最小 {mg_u.min():.3f}")
print(f"  放大后 margin<=0 的: {(mg_u<=0).sum()}/{len(mg_u)} ({(mg_u<=0).mean()*100:.1f}%)")
print(f"  (纯上采样不增信息，仅作对照；改善即说明尺度不变性有问题)")

print()
print("=" * 88)
print("【Q4】字符像素高度实测 —— 40px 里字形占多少")
print("=" * 88)
hs, ws = [], []
for k in keys:
    b = img(k)
    rows = np.where(b.any(axis=1))[0]
    if len(rows):
        hs.append(rows.max() - rows.min() + 1)
    cols = np.where(b.any(axis=0))[0]
    if len(cols):
        ws.append(cols.max() - cols.min() + 1)
hs, ws = np.array(hs), np.array(ws)
print(f"  字形纵向跨度: 均值 {hs.mean():.1f}px  范围 {hs.min()}~{hs.max()}px  (画布 40px)")
print(f"  字形横向跨度: 均值 {ws.mean():.1f}px  范围 {ws.min()}~{ws.max()}px  (画布 110px)")
print(f"  -> 字形只占画面高度的 {hs.mean()/40*100:.0f}%，横向 {ws.mean()/110*100:.0f}%")
print(f"  -> 单字符平均宽约 {ws.mean()/4.5:.1f}px（4~5字符），高 {hs.mean():.0f}px")
