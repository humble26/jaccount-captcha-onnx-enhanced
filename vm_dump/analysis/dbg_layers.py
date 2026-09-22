import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""定位 nn.py 前向在第几层偏离 onnxruntime（一次性调试）"""
import os
import numpy as np
from PIL import Image

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
VD = os.path.join(WS, "vm_dump")
MODEL = os.path.join(VD, "nn_model.onnx")
HOLD_L = ["/Relu_output_0",
          "/layer1/layer1.2/Relu_1_output_0",
          "/layer2/layer2.2/Relu_1_output_0",
          "/layer3/layer3.2/Relu_1_output_0",
          "/Reshape_output_0"]


def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def main():
    import onnx
    from onnx import numpy_helper
    m = onnx.load(MODEL)
    w = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    conv = {}
    for n in m.graph.node:
        if n.op_type == "Conv":
            a = {z.name: z for z in n.attribute}
            conv[n.input[1]] = (
                tuple(a["pads"].ints) if "pads" in a else (0, 0, 0, 0),
                tuple(a["strides"].ints) if "strides" in a else (1, 1))
    for name in HOLD_L:
        vi = onnx.helper.ValueInfoProto(); vi.name = name
        m.graph.output.append(vi)
    tmp = os.path.join(VD, "_tmp_dbg.onnx")
    onnx.save(m, tmp)
    try:
        import onnxruntime as rt
        so = rt.SessionOptions(); so.log_severity_level = 3
        sess = rt.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        onames = [o.name for o in sess.get_outputs()]
        fi = onames.index("/Reshape_output_0")

        f = "h000.png"
        x = preprocess(os.path.join(VD, "holdout", f))[None, None]
        r = sess.run(onames, {iname: x})
        for i, nm in enumerate(onames):
            o = r[i]
            print(f"{nm:58s} shape={tuple(o.shape)} dtype={o.dtype} "
                  f"min={o.min():.4f} max={o.max():.4f} mean={o.mean():.4f}")
        ref_feat = r[fi]
        # 我们 numpy 的 feat
        sys_p = os.path.join(_HERE, "e2e_forward_check.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("e2e", sys_p)
        # 会重复 import onnx；直接复制 forward 逻辑太重。改为打印 reference 供人工对比
        np.save(os.path.join(VD, "_ref_feat.npy"), ref_feat)
        print("ref feat saved")
    finally:
        os.remove(tmp)


if __name__ == "__main__":
    main()