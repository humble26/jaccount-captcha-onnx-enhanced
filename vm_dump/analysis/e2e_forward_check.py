# -*- coding: utf-8 -*-
"""端到端重构：纯 NumPy ResNet-20 前向实现，与生产 ONNX(onnxruntime) 逐输出校验

用途：为「端到端重训」打地基。只有 numpy 前向能 1:1 复现生产模型输出，
后续用同一套结构做反向传播训练才有意义。

运行：
  python e2e_forward_check.py
"""
import os, json, sys
import numpy as np
from PIL import Image

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
VD = os.path.join(WS, "vm_dump")
MODEL = os.path.join(VD, "nn_model.onnx")
CH = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------- ONNX 权重装载
def load_weights(model_path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(model_path)
    g = m.graph
    w = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    conv_pads = {}   # weight-name -> (pad_top, pad_left)
    conv_strides = {}
    for n in g.node:
        if n.op_type == "Conv":
            wkey = n.input[1] if len(n.input) > 1 else n.name
            attrs = {a.name: a for a in n.attribute}
            pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
            strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
            conv_pads[wkey] = (pads[0], pads[1])
            conv_strides[wkey] = (strides[0], strides[1])
    return w, conv_pads, conv_strides


# ---------------------------------------------------------------- 算子

def conv2d(x, wk, bk, pad, stride):
    """x: (1,C,H,W); wk: (O,C,K,K); 显式窗口卷积，正确支持 stride/pad"""
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
            r = xp[:, :, oh * sh:oh * sh + K, ow * sw:ow * sw + K]   # (1,C,K,K)
            out[:, :, oh, ow] = np.einsum("nchw,ochw->no", r, wk)
    return (out + bk.reshape(1, O, 1, 1)).astype(np.float32)


def relu(x):
    return np.maximum(0, x)


def avgpool(x, kh=10, kw=25):
    """AveragePool kernel_shape=[10,25]（模型导出即固定，忽略右侧多余列）"""
    return x[:, :, :kh, :kw].mean(axis=(2, 3))


def preprocess_py(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def onnx_heads(m):
    return [ini.name for ini in m.graph.initializer if ini.name.startswith("linear") and ini.name.endswith(".weight")]


def main():
    import onnx
    W, conv_pads, conv_strides = load_weights(MODEL)
    m = onnx.load(MODEL)
    heads = sorted([i.name for i in m.graph.initializer
                    if i.name.startswith("linear") and i.name.endswith(".weight")],
                   key=lambda s: int(s[len("linear"):-len(".weight")]))
    head_bias = {}
    for h in heads:
        num = h[len("linear"):-len(".weight")]
        head_bias[num] = f"linear{num}.bias"
    import onnx

    # ---- 扩展模型输出：让 onnxruntime 暴露中间激活 ----
    ACT_NAMES = ["/Relu_output_0",
                 "/layer1/layer1.2/Relu_1_output_0",
                 "/layer2/layer2.2/Relu_1_output_0",
                 "/layer3/layer3.2/Relu_1_output_0",
                 "/Reshape_output_0"]
    for name in ACT_NAMES:
        vi = onnx.helper.ValueInfoProto(); vi.name = name
        m.graph.output.append(vi)
    tmp = os.path.join(VD, "_tmp_e2e.onnx")
    onnx.save(m, tmp)
    try:
        import onnxruntime as rt
        so = rt.SessionOptions(); so.log_severity_level = 3
        sess = rt.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        onames = [o.name for o in sess.get_outputs()]
        idx = {o.name: i for i, o in enumerate(sess.get_outputs())}

        hold = os.path.join(VD, "holdout")
        imgs = sorted([f for f in os.listdir(hold) if f.endswith(".png")])[:6]
        gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))

        all_errs = {n: 0.0 for n in ACT_NAMES}
        for f in imgs:
            x = preprocess_py(os.path.join(hold, f))[None, None]
            ref = sess.run(onames, {iname: x})
            acts = {}
            out_np, feat = forward_np(x, W, conv_pads, conv_strides, heads, head_bias, acts=acts)
            print(f"--- {os.path.splitext(f)[0]} ---")
            for n in ACT_NAMES:
                r = np.asarray(ref[idx[n]])
                a = np.asarray(acts[n])
                err = float(np.abs(r - a).max())
                all_errs[n] = max(all_errs[n], err)
                if n == "/Reshape_output_0":
                    print("      ref[:12] =", np.round(r[0, :12], 4))
                    print("      np [:12] =", np.round(a[0, :12], 4))
                print(f"    {n:60s} max|Δ|={err:.3e}")
            # 5 头 logit 对齐
            for i, num in enumerate(sorted(head_bias)):
                err = float(np.abs(np.asarray(ref[idx["218" if num == "1" else "219" if num == "2" else "220" if num == "3" else "221" if num == "4" else "222"]]) - out_np[i]).max())
                all_errs[f"head{num}"] = max(all_errs.get(f"head{num}", 0.0), err)
        print("=" * 70)
        bad = [k for k, v in all_errs.items() if v > 1e-3]
        for k, v in all_errs.items():
            print(f"  {k:64s} max|Δ| = {v:.3e}")
        print("FAIL 于:", bad if bad else "（无，全部一致）")
        return 1 if bad else 0
    finally:
        os.remove(tmp)


def forward_np(x, W, conv_pads, conv_strides, heads, head_bias, acts=None):
    P = conv_pads; S = conv_strides
    h = relu(conv2d(x, W["onnx::Conv_224"], W["onnx::Conv_225"], P["onnx::Conv_224"], S["onnx::Conv_224"]))
    if acts is not None:
        acts["/Relu_output_0"] = h
    shortcut = h
    for c1, c2 in [(227, 230), (233, 236), (239, 242)]:
        t = relu(conv2d(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"],
                        P[f"onnx::Conv_{c1}"], S[f"onnx::Conv_{c1}"]))
        t = conv2d(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"],
                   P[f"onnx::Conv_{c2}"], S[f"onnx::Conv_{c2}"])
        h = relu(t + shortcut); shortcut = h
    if acts is not None:
        acts["/layer1/layer1.2/Relu_1_output_0"] = h
    for c1, c2, sc in [(245, 248, 251), (254, 257, 0), (260, 263, 0)]:
        t = relu(conv2d(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"],
                        P[f"onnx::Conv_{c1}"], S[f"onnx::Conv_{c1}"]))
        t = conv2d(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"],
                   P[f"onnx::Conv_{c2}"], S[f"onnx::Conv_{c2}"])
        if sc == 0:
            s = h
        else:
            s = conv2d(h, W[f"onnx::Conv_{sc}"], W[f"onnx::Conv_{sc+1}"],
                       P[f"onnx::Conv_{sc}"], S[f"onnx::Conv_{sc}"])
        h = relu(t + s)
    if acts is not None:
        acts["/layer2/layer2.2/Relu_1_output_0"] = h
    for c1, c2, sc in [(266, 269, 272), (275, 278, 0), (281, 284, 0)]:
        t = relu(conv2d(h, W[f"onnx::Conv_{c1}"], W[f"onnx::Conv_{c1+1}"],
                        P[f"onnx::Conv_{c1}"], S[f"onnx::Conv_{c1}"]))
        t = conv2d(t, W[f"onnx::Conv_{c2}"], W[f"onnx::Conv_{c2+1}"],
                   P[f"onnx::Conv_{c2}"], S[f"onnx::Conv_{c2}"])
        if sc == 0:
            s = h
        else:
            s = conv2d(h, W[f"onnx::Conv_{sc}"], W[f"onnx::Conv_{sc+1}"],
                       P[f"onnx::Conv_{sc}"], S[f"onnx::Conv_{sc}"])
        h = relu(t + s)
    if acts is not None:
        acts["/layer3/layer3.2/Relu_1_output_0"] = h
    feat = avgpool(h)
    if acts is not None:
        acts["/Reshape_output_0"] = feat.reshape(1, -1)
    out = []
    for num in sorted(head_bias):
        out.append(feat @ W[f"linear{num}.weight"].T + W[head_bias[num]])
    return out, feat


if __name__ == "__main__":
    sys.exit(main())