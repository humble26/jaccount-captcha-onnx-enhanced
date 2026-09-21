"""导出 220 张样本的模型输入张量 + Python 参考输出，供 Node 里跑真实的 onnxruntime-web 比对。"""
import os, json, struct, warnings
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)
W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
OUT = os.path.join(W, "webcheck")
os.makedirs(OUT, exist_ok=True)

gt = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))
IDS = sorted(gt)
sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name


def path(sid):
    for s in ("samples", "samples2", "holdout"):
        p = os.path.join(W, s, sid + ".png")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(sid)


def decode(outs):
    txt = ""
    for t in outs:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        txt += chr(ord("a") + a)
    return txt


buf = bytearray()
expected = {}
raw0 = None
for k, sid in enumerate(IDS):
    rgb = np.asarray(Image.open(path(sid)).convert("RGB"), dtype=np.float64)
    f = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    arr = (np.round(f) >= 156).astype(np.float32)          # 与 JS 的 Math.round 路径一致
    assert arr.shape == (40, 110)
    buf += arr.tobytes()
    outs = sess.run(None, {IN: arr[None, None, ...]})
    expected[sid] = decode(outs)
    if k == 0:
        raw0 = [np.asarray(o, dtype=np.float32).ravel().tolist() for o in outs]

open(os.path.join(OUT, "inputs.bin"), "wb").write(bytes(buf))
json.dump(IDS, open(os.path.join(OUT, "ids.json"), "w", encoding="utf-8"))
json.dump(expected, open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8"), ensure_ascii=False)
json.dump([[o.name for o in sess.get_outputs()], raw0],
          open(os.path.join(OUT, "raw0.json"), "w", encoding="utf-8"))

hit = sum(1 for i in IDS if expected[i] == gt[i])
print(f"参考实现(Python onnxruntime): {hit}/{len(IDS)} = {hit/len(IDS)*100:.1f}%")
print(f"输入张量已导出: {len(buf)} bytes = {len(IDS)} x {len(buf)//len(IDS)} floats")
print("输出张量名:", [o.name for o in sess.get_outputs()])
print("样本0 的 logits 已存，用于逐数值比对")
