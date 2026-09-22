import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""用回写后的 ONNX(onnxruntime) 在 留出集+调参集 完整评估, 验证回写有效性。
对比原生 ONNX, 确认: ①留出集零退化  ②调参集旧难例被修正  ③无新引入错误。
结果写入 ./_eval_backport_result.txt
"""
import os, sys, json
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
real_err = sys.stderr
sys.stderr = open(os.devnull, "w")

import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_e2e as T
import onnxruntime as ort

CH = T.CH
NEW = os.path.join(_VD, "_e2e_RT_backport", "nn_model_e2e.onnx")
S0 = T.MODEL
OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_eval_backport_result.txt")
L = []


def preprocess(p):
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def run_session(path, xs):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    i0 = sess.get_inputs()[0].name
    preds = []
    for i in range(xs.shape[0]):
        r = sess.run(None, {i0: xs[i:i+1].astype(np.float32)})
        preds.append([int(np.argmax(o[0])) for o in r])
    return preds


def whole_ok(lab, pr):
    return all(pr[p] == lab[p] for p in range(len(lab)))


def load_set(split):
    gt = json.load(open(T.GT, encoding="utf-8"))
    keys = []
    pools = ["holdout"] if split == "hold" else ["samples", "samples2"]
    for d in pools:
        for f in sorted(os.listdir(os.path.join(T.VD, d))):
            if f.endswith(".png"):
                k = os.path.splitext(f)[0]
                if k in gt:
                    keys.append((k, os.path.join(T.VD, d, f)))
    xs = [preprocess(p)[None] for _, p in keys]
    x = np.stack(xs).astype(np.float32)
    labs = [[CH.index(gt[k][p]) if p < len(gt[k]) else 26 for p in range(5)] for k, _ in keys]
    return [k for k, _ in keys], x, labs


for split in ["hold", "tune"]:
    keys, x, labs = load_set(split)
    p0 = run_session(S0, x)
    pn = run_session(NEW, x)
    whole0 = sum(1 for i in range(len(keys)) if whole_ok(labs[i], p0[i]))
    wholen = sum(1 for i in range(len(keys)) if whole_ok(labs[i], pn[i]))
    p4_0 = sum(1 for i in range(len(keys)) if p0[i][3] == labs[i][3])
    p4_n = sum(1 for i in range(len(keys)) if pn[i][3] == labs[i][3])
    L.append(f"===== {split} 集 {len(keys)} 张 =====")
    L.append(f"  原生: 整串 {whole0}/{len(keys)}={whole0/len(keys)*100:.2f}%  位4 {p4_0}/{len(keys)}")
    L.append(f"  回写: 整串 {wholen}/{len(keys)}={wholen/len(keys)*100:.2f}%  位4 {p4_n}/{len(keys)}")
    changed = [i for i in range(len(keys)) if p0[i] != pn[i]]
    L.append(f"  预测发生变化的样本 {len(changed)} 张:")
    for i in changed:
        o = "".join(CH[p] if p < 26 else "." for p in p0[i])
        n = "".join(CH[p] if p < 26 else "." for p in pn[i])
        lab = "".join(CH[l] if l < 26 else "." for l in labs[i])
        if whole_ok(labs[i], pn[i]) and not whole_ok(labs[i], p0[i]):
            flag = "修正!"
        elif whole_ok(labs[i], p0[i]) and not whole_ok(labs[i], pn[i]):
            flag = "退化!"
        else:
            flag = "变化"
        L.append(f"    {keys[i]}: 真值={lab} 原生={o} 回写={n}  [{flag}]")

sys.stderr = real_err
open(OUT_TXT, "w", encoding="utf-8").write("\n".join(L))
print("done")