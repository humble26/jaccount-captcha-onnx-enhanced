import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""验证「第 4 位弱」的可修复性 —— 三条候选路径的可行性探测

路径 1（切分重识别）：把第 4 个字符按列投影切出来，单独送模型第 4 头再判一次。
        若不改善 => 说明第 4 头的判别边界本身就错，切分无用。
路径 2（邻字符联合约束）：4 位码的第 5 头输出是否被浪费？检查第 5 头的分布。
路径 3（混淆对置信度门槛）：统计 5 个错误样本的 top1 置信分布 vs 正确样本，
        判断"阈值筛掉 + 人工兜底"能否把 97.7% 提到 100% 且不损失太多自动化率。
路径 4（结构性缺陷影响量化）：对比 ORT 默认优化等级 vs 关闭全部优化 的预测差异。
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

def prep_arr(p):
    im = Image.open(p).convert("L")
    a = np.asarray(im).astype(np.float64)
    return np.round(a)

def prep(p):
    g = prep_arr(p)
    return (g >= 156).astype(np.float32).reshape(1, 1, 40, 110)

so = rt.SessionOptions()
so.log_severity_level = 3
sess = rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), so,
                           providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
names = sorted([o.name for o in sess.get_outputs()], key=lambda x: int(x))

print("=" * 84)
print("路径 3：置信度门槛能否筛出错误？（用于「低置信转人工」策略评估）")
print("=" * 84)

rows = []
for k, t in gt.items():
    if k not in paths:
        continue
    pr = sess.run(names, {iname: prep(paths[k])})
    # 每位 softmax 置信
    per = []
    for i in range(len(t)):
        v = pr[i][0]
        e = np.exp(v - v.max()); p = e / e.sum()
        per.append(float(p[np.argmax(v)]))
    minconf = min(per) if per else 0.0
    ok = (pred_base.get(k) == t)
    rows.append((k, t, pred_base.get(k), ok, minconf, per))

rows.sort(key=lambda r: r[4])
print(f"{'样本':6s} {'真值':8s} {'预测':8s} {'正确':4s} {'最低位置信':>10s}")
for k, t, p, ok, mc, per in rows[:14]:
    print(f"{k:6s} {t:8s} {str(p):8s} {'OK' if ok else 'XX':4s} {mc*100:9.2f}%")

err_mc = [r[4] for r in rows if not r[3]]
ok_mc = [r[4] for r in rows if r[3]]
print()
print(f"正确样本最低位置信: 均值 {np.mean(ok_mc)*100:.2f}%  最小 {np.min(ok_mc)*100:.2f}%")
print(f"错误样本最低位置信: 均值 {np.mean(err_mc)*100:.2f}%  最大 {np.max(err_mc)*100:.2f}%")
print()
for thr in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
    caught = sum(1 for r in rows if (not r[3]) and r[4] < thr)
    flagged = sum(1 for r in rows if r[4] < thr)
    n = len(rows)
    auto = n - flagged
    auto_acc = sum(1 for r in rows if r[4] >= thr and r[3]) / auto * 100 if auto else 0
    print(f"  阈值 {thr*100:6.1f}%  ->  转人工 {flagged:3d}/{n} ({flagged/n*100:5.1f}%)  "
          f"自动通过 {auto:3d} 张准确率 {auto_acc:6.2f}%  错误拦截 {caught}/{len(err_mc)}")

print()
print("=" * 84)
print("路径 1：按列投影切出第 4 个字符，单独看过不了第 4 头")
print("=" * 84)

# 4 字符样本的槽位中心（前一轮测得）：5.5 / 28.8 / 51.7 / 74.9
C4 = [5.5, 28.8, 51.7, 74.9]
C5 = [5.6, 26.6, 47.7, 68.8, 89.9]

def col_proj_rows(p):
    g = prep_arr(p)
    b = (g >= 156).astype(np.float64)
    return b.sum(axis=0)  # 每列笔画数

err_keys = [k for k in gt if k in pred_base and gt[k] != pred_base[k]]
ok_keys = [k for k in gt if k in pred_base and gt[k] == pred_base[k]]

print(f"{'样本':6s} {'长度':4s} {'第4位判定窗口内的笔画重心':>24s} {'槽位理论中心':>12s}")
for k in err_keys:
    cp = col_proj_rows(paths[k])
    t = gt[k]
    centers = C4 if len(t) == 4 else C5
    idx = 3
    lo = int((centers[idx-1] + centers[idx]) / 2) if idx > 0 else 0
    hi = int((centers[idx] + centers[idx+1]) / 2) if idx + 1 < len(centers) else 110
    win = cp[max(0, lo):min(110, hi)]
    mass = win.sum()
    com = (np.arange(len(win)) * win).sum() / mass + max(0, lo) if mass > 0 else -1
    print(f"{k:6s} {len(t):4d} {com:24.1f} {centers[idx]:12.1f}")

print()
print("=" * 84)
print("路径 4：onnxruntime 优化等级对预测的影响（结构性缺陷是否真的拖后腿）")
print("=" * 84)

def build(level):
    o = rt.SessionOptions()
    o.graph_optimization_level = level
    o.log_severity_level = 3
    return rt.InferenceSession(os.path.join(VD, "nn_model.onnx"), o,
                               providers=["CPUExecutionProvider"])

lv = {
    "DISABLE_ALL": rt.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "BASIC": rt.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "EXTENDED": rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ALL": rt.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
res = {}
for lname, lvval in lv.items():
    try:
        s = build(lvval)
        preds, tp, tn = {}, 0, 0
        for k, t in gt.items():
            if k not in paths:
                continue
            out = s.run(names, {iname: prep(paths[k])})
            got = "".join(CH[int(np.argmax(o[0]))] if int(np.argmax(o[0])) < 26 else "" for o in out)
            preds[k] = got
            tn += 1
            if got == t:
                tp += 1
        res[lname] = preds
        print(f"  {lname:12s} 准确率 {tp}/{tn} = {tp/tn*100:.2f}%")
    except Exception as e:
        print(f"  {lname:12s} 失败: {e}")

if "DISABLE_ALL" in res and "ALL" in res:
    diff = [k for k in res["ALL"] if res["ALL"][k] != res["DISABLE_ALL"][k]]
    print(f"  ALL 与 DISABLE_ALL 预测不一致的样本数: {len(diff)}")
    for k in diff[:10]:
        print(f"     {k}: ALL={res['ALL'][k]}  NONE={res['DISABLE_ALL'][k]}  真值={gt[k]}")
