import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""准确度提升方案的可测性实验，全部在 120 张带真值样本上评估。

测试项：
  1. 二值化阈值扫描（官方用 156）
  2. 保留灰度不二值化
  3. TTA：对二值图做小幅平移后取 logits 平均（3×1 / 1×3 / 3×3 / 5×5）
  4. 全对比（把 120 张按图像哈希去重后看是否有重复样本）
"""
import os, json, warnings, hashlib
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)

W = _VD
GT = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))
IDS = sorted(GT)

sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name


def load_gray(sid):
    p = os.path.join(W, "samples", sid + ".png")
    if not os.path.exists(p):
        p = os.path.join(W, "samples2", sid + ".png")
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)


GRAY = {i: load_gray(i) for i in IDS}


def binarize(g, thr):
    return (g >= thr).astype(np.float32)


def shift(img, dx, dy, fill=1.0):
    out = np.full_like(img, fill)
    h, w = img.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0:ys1, xs0:xs1] = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def run_one(arr):
    return sess.run(None, {IN: arr[None, None, ...]})


def decode(outs):
    """新脚本逻辑：每个头按自身类别数 argmax，>=26 视为 blank 跳过"""
    txt = ""
    for t in outs:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        txt += chr(ord("a") + a)
    return txt


def acc(preds):
    return sum(1 for i in IDS if preds[i] == GT[i])


def report(name, preds):
    k = acc(preds)
    errs = [i for i in IDS if preds[i] != GT[i]]
    print(f"  {name:<34} {k:>3}/{len(IDS)}  {k/len(IDS)*100:5.1f}%   错例 {errs}")
    return k


print("=" * 84)
print("1) 二值化阈值扫描")
print("=" * 84)
thr_res = {}
for thr in [110, 130, 145, 150, 156, 162, 175, 190, 210]:
    preds = {i: decode(run_one(binarize(GRAY[i], thr))) for i in IDS}
    thr_res[thr] = report(f"threshold = {thr}", preds)

print("\n" + "=" * 84)
print("2) 不二值化，直接用灰度/255")
print("=" * 84)
report("grayscale / 255", {i: decode(run_one(GRAY[i] / 255.0)) for i in IDS})
report("(1 - grayscale/255) 反相", {i: decode(run_one(1.0 - GRAY[i] / 255.0)) for i in IDS})

print("\n" + "=" * 84)
print("3) TTA：平移后 logits 平均")
print("=" * 84)
BASE = {i: binarize(GRAY[i], 156) for i in IDS}


def tta(dxs, dys):
    preds = {}
    for i in IDS:
        acc_logits = None
        for dy in dys:
            for dx in dxs:
                outs = run_one(shift(BASE[i], dx, dy))
                if acc_logits is None:
                    acc_logits = [o.astype(np.float32).copy() for o in outs]
                else:
                    for k2, o in enumerate(outs):
                        acc_logits[k2] += o
        preds[i] = decode(acc_logits)
    return preds


report("TTA 1x3 (dx=-1,0,1)", tta([-1, 0, 1], [0]))
report("TTA 3x1 (dy=-1,0,1)", tta([0], [-1, 0, 1]))
report("TTA 3x3", tta([-1, 0, 1], [-1, 0, 1]))
report("TTA 5x3 (dx -2..2)", tta([-2, -1, 0, 1, 2], [-1, 0, 1]))

print("\n" + "=" * 84)
print("4) 样本重复性检查（同图不同 id 会让准确率虚高）")
print("=" * 84)
h = {}
for i in IDS:
    key = hashlib.md5(GRAY[i].tobytes()).hexdigest()
    h.setdefault(key, []).append(i)
dup = {k: v for k, v in h.items() if len(v) > 1}
print(f"  唯一图像 {len(h)} 张 / 样本 {len(IDS)} 张，重复组 {len(dup)}")
for k, v in list(dup.items())[:5]:
    print("   ", v)

json.dump({"threshold": thr_res}, open(os.path.join(W, "tune_result.json"), "w", encoding="utf-8"), indent=1)
print("\n完成")
