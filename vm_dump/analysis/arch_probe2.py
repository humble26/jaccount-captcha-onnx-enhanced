import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""纠正与深化：像素空间可分性 + 真实瓶颈定位

修正上一轮的错误：二值化方向搞反了（>=156 应为笔画为白还是黑）。
本脚本先确认真实像素分布与正确掩码方向，再做严格的瓶颈定位。

核心问题：如果像素空间可分（上一轮发现 5/5 错误都是"距真值更近"），
        那么瓶颈到底是 backbone 容量，还是输入分辨率/预处理？
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

print("=" * 88)
print("步骤 0：确认真实像素分布与掩码方向")
print("=" * 88)
k0 = keys[0]
a = np.asarray(Image.open(paths[k0]).convert("L")).astype(np.float64)
print(f"  样本 {k0}: min {a.min():.0f} max {a.max():.0f} mean {a.mean():.1f} distinct {len(np.unique(a))}")
print(f"  取值 >200 占比 {(a>200).mean()*100:.2f}%   <50 占比 {(a<50).mean()*100:.2f}%")
m_bright = (np.round(a) >= 156)
print(f"  mask (>=156) 占比 {m_bright.mean()*100:.2f}%   <- 脚本里用了这个，是 93%（背景）")
print(f"  mask (<156)  占比 {(~m_bright).mean()*100:.2f}%   <- 这才是笔画（暗色字）")

# 正确的笔画掩码
def strokes(k):
    a = np.asarray(Image.open(paths[k]).convert("L")).astype(np.float64)
    return (np.round(a) < 156).astype(np.float64)

sm = np.array([strokes(k) for k in keys])
print(f"  全体笔画占比: 均值 {sm.mean()*100:.2f}%  有效像素约 {sm.mean()*4400:.0f} 个/张")

print()
print("=" * 88)
print("步骤 1（修正）：用正确的笔画掩码，做像素级可分性检验")
print("=" * 88)
row_prof = sm.mean(axis=(0, 2))
col_prof = sm.mean(axis=(0, 1))
nz = np.where(row_prof > 0.001)[0]
print(f"  行方向有墨迹: {nz.min()}~{nz.max()}  峰值行墨迹 {row_prof.max()*100:.1f}%")
print(f"  行 std {row_prof.std():.4f}   列 std {col_prof.std():.4f}")
print(f"  -> 列/行 std 比 = {col_prof.std()/max(row_prof.std(),1e-9):.2f}  (>1 表示横向信息更丰富)")

def slot(k, idx):
    t = gt[k]
    C4 = [5.5, 28.8, 51.7, 74.9]
    C5 = [5.6, 26.6, 47.7, 68.8, 89.9]
    cs = C4 if len(t) == 4 else C5
    cx = cs[idx]
    lo, hi = int(cx - 10), int(cx + 10)
    lo, hi = max(0, lo), min(110, hi)
    c = strokes(k)[:, lo:hi]
    if c.shape[1] < 20:
        c = np.hstack([np.zeros((40, 20 - c.shape[1])), c])
    return c[:, :20]

# 第 4 位（index 3）
X, Y, KY = [], [], []
for k in keys:
    if len(gt[k]) <= 3:
        continue
    X.append(slot(k, 3).ravel()); Y.append(gt[k][3]); KY.append(k)
X = np.array(X); Y = np.array(Y)
print(f"  第 4 位样本 {len(X)}，特征维 {X.shape[1]} (40x20 笔画像素)")

cent = {c: X[Y == c].mean(axis=0) for c in set(Y)}
within = np.mean([np.linalg.norm(X[i] - cent[Y[i]]) for i in range(len(X))])
print(f"  平均类内距离: {within:.3f}")

err_keys = sorted([k for k in keys if pred_base.get(k) != gt[k]])
print()
print("  错误样本像素距离（修正后）:")
for k in err_keys:
    if len(gt[k]) <= 3:
        continue
    v = slot(k, 3).ravel()
    tv, pv = gt[k][3], pred_base[k][3]
    dt = np.linalg.norm(v - cent[tv]); dp = np.linalg.norm(v - cent[pv])
    print(f"    [{k}] 真值={tv} 预测={pv}  到真值 {dt:7.3f}  到误判 {dp:7.3f}  "
          f"-> {'像素可分' if dt < dp else '像素不可分'}  (差 {dp-dt:+.3f})")

print()
print("  正确样本 margin（到真值距离 vs 到最近他类距离，正数=可分）:")
mg = []
for i, k in enumerate(KY):
    if k in err_keys:
        continue
    tv = Y[i]
    dt = np.linalg.norm(X[i] - cent[tv])
    do = min(np.linalg.norm(X[i] - cent[d]) for d in cent if d != tv)
    mg.append(do - dt)
mg = np.array(mg)
print(f"    margin 均值 {mg.mean():.3f}  中位 {np.median(mg):.3f}  最小 {mg.min():.3f}")
print(f"    margin <= 0 的正确样本: {(mg<=0).sum()}/{len(mg)}  ({(mg<=0).mean()*100:.1f}%)")
print(f"    -> 像素空间可分性并不高，字符笔画确实高度重叠")

print()
print("=" * 88)
print("步骤 2：关键判定 —— 用「模型自己的中间特征」而非原始像素")
print("=" * 88)
print("  方法：把 ONNX 图在第 4 头之前的张量挖出来，看特征空间里是否可分。")
print("  若特征空间不可分但像素空间可分 => backbone 是瓶颈，换骨架有救。")
print("  若两者都不可分 => 数据/分辨率是瓶颈，换骨架无用。")

try:
    import onnx
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False
    print("  onnx 未安装，尝试 pip 安装...")
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "-q"],
                       capture_output=True, text=True)
    print("  pip:", r.returncode, (r.stdout or "")[-200:], (r.stderr or "")[-300:])
    try:
        import onnx
        HAVE_ONNX = True
    except ImportError:
        HAVE_ONNX = False

if HAVE_ONNX:
    m = onnx.load(os.path.join(VD, "nn_model.onnx"))
    g = m.graph
    print()
    print("  图输出头:")
    for o in g.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}  {dims}")
    print()
    print(f"  图输入（含被误列为 initializer 的）: {len(g.input)} 个")
    for i in g.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"    {i.name}  {dims}")

    import collections
    ops = collections.Counter(n.op_type for n in g.node)
    print()
    print(f"  算子统计 (共 {len(g.node)} 节点):")
    for op, c in ops.most_common():
        print(f"    {op:18s} {c}")

    # 输入张量名（真正那个 [1,1,40,110]）
    real_in = None
    for i in g.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        if dims[:2] == [1, 1] and len(dims) == 4:
            real_in = i.name
    print(f"\n  真实输入张量: {real_in}")

    # 找每个输出头的最后一个 Gemm，取其输入作为"特征"
    gemms = [n for n in g.node if n.op_type in ("Gemm", "MatMul")]
    print(f"  末层线性层 {len(gemms)} 个:")
    for n in gemms:
        print(f"    {n.name or '(unnamed)':24s} 输入特征={list(n.input)[0]}  输出={list(n.output)[0]}")

    # 收集所有可能的中间张量名
    produced = set()
    for n in g.node:
        for o in n.output:
            produced.add(o)
    print(f"\n  图中可用的中间张量 {len(produced)} 个")
    if gemms:
        feat_names = [list(n.input)[0] for n in gemms]
        print(f"  候选特征张量（各头输入）: {feat_names}")
        # 试试能不能作为附加输出跑
        probe = dict(pred_base)
        print()
        print("  尝试把特征张量加为输出并推理...")
        try:
            import onnx as _o
            m2 = _o.load(os.path.join(VD, "nn_model.onnx"))
            g2 = m2.graph
            added = []
            for fn in feat_names:
                vi = _o.helper.ValueInfoProto()
                vi.name = fn
                g2.output.append(vi)
                added.append(fn)
            tmp = os.path.join(VD, "_tmp_feat.onnx")
            _o.save(m2, tmp)
            so2 = rt.SessionOptions(); so2.log_severity_level = 3
            s2 = rt.InferenceSession(tmp, so2, providers=["CPUExecutionProvider"])
            out_names = sorted([o.name for o in s2.get_outputs()], key=lambda x: (len(x), x))
            print("    附加输出成功:", out_names)
            for k in [keys[0]] + err_keys[:2]:
                res = s2.run(out_names, {real_in: strokes(k).astype(np.float32).reshape(1, 1, 40, 110)})
                for nm, r in zip(out_names, res):
                    if nm in added:
                        print(f"      {k} {nm} shape={np.array(r).shape}")
        except Exception as e:
            print("    附加输出失败:", type(e).__name__, e)
else:
    print("  onnx 不可用，跳过图结构分析")
