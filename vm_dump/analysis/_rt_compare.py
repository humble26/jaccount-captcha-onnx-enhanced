import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
import os, sys, io
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
orig_err = sys.stderr
sys.stderr = open(os.devnull, "w")
import onnxruntime as ort
import numpy as np

OUT = os.path.join(_VD, "_e2e_RT_backport", "nn_model_e2e.onnx")
S0 = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump\nn_model.onnx"
lines = []
try:
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    sess0 = ort.InferenceSession(S0, providers=["CPUExecutionProvider"])
    i0 = sess.get_inputs()[0].name
    np.random.seed(0)
    for trial in range(3):
        x = np.asarray(np.random.random((1, 1, 40, 110)) > 0.45, dtype=np.float32)
        y = sess.run(None, {i0: x})
        y0 = sess0.run(None, {i0: x})
        assert len(y) == len(y0)
        md = max(float(np.abs(a - b).max()) for a, b in zip(y, y0))
        lines.append(f"trial{trial}: 输出数={len(y)} shapes={[a.shape for a in y]} maxdiff_vs_原生={md:.3e}")
    lines.append("回写模型通过 onnxruntime 加载与推理, 结构一致。")
except Exception as e:
    lines.append("ERR: " + repr(e))
sys.stderr = orig_err
out = "\n".join(lines)
open(os.path.join(_VD, "_e2e_RT_backport", "_rt_compare_result.txt"), "w", encoding="utf-8").write(out)
print(out)