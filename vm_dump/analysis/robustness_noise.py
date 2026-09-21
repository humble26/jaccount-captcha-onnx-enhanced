"""解码器差异的鲁棒性测试。

验证码是 4:2:0 子采样的 JPEG，浏览器（Chrome）与 PIL（libjpeg）的色度上采样算法不同，
解出的 RGB 可能有几个单位的差异。这里用 ±k 的均匀随机噪声模拟这种差异，
看阈值判定和最终识别结果会不会被影响。
"""
import os, json, warnings
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)
W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
gt = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))

sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name


def path(sid):
    for s in ("samples", "samples2", "holdout"):
        p = os.path.join(W, s, sid + ".png")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(sid)


RGB = {sid: np.asarray(Image.open(path(sid)).convert("RGB"), dtype=np.int16) for sid in gt}
IDS = sorted(gt)


def bin_from_rgb(rgb):
    f = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return (np.round(f) >= 156).astype(np.float32)


def decode(outs):
    txt = ""
    for t in outs:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        txt += chr(ord("a") + a)
    return txt


def run(arr):
    return decode(sess.run(None, {IN: arr[None, None, ...]}))


base = {sid: run(bin_from_rgb(RGB[sid])) for sid in IDS}
base_hit = sum(1 for i in IDS if base[i] == gt[i])
print(f"基线（无扰动）: {base_hit}/{len(IDS)} = {base_hit/len(IDS)*100:.1f}%")

rng = np.random.default_rng(20260921)
for k in (1, 2, 4, 8):
    hits, flips, diffs = [], [], []
    for trial in range(5):
        h = 0
        flip = 0
        diff = 0
        for sid in IDS:
            noise = rng.integers(-k, k + 1, size=RGB[sid].shape, dtype=np.int16)
            noisy = np.clip(RGB[sid] + noise, 0, 255).astype(np.int16)
            b0 = bin_from_rgb(RGB[sid])
            b1 = bin_from_rgb(noisy)
            flip += int((b0 != b1).sum())
            p = run(b1)
            if p == gt[sid]:
                h += 1
            if p != base[sid]:
                diff += 1
        hits.append(h)
        flips.append(flip)
        diffs.append(diff)
    print(f"RGB 噪声 ±{k}: 准确率 {min(hits)}~{max(hits)}/{len(IDS)} "
          f"({min(hits)/len(IDS)*100:.1f}%~{max(hits)/len(IDS)*100:.1f}%)  "
          f"平均像素翻转 {np.mean(flips):.0f}/220张 ({np.mean(flips)/220:.2f}/张)  "
          f"平均结果改变 {np.mean(diffs):.0f}/220 张")
