import urllib.request, os, io
import numpy as np
import onnxruntime as rt
from PIL import Image

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
MODEL = os.path.join(W, "nn_model.onnx")
IMG = os.path.join(W, "captcha.jpg")

HDR = {"User-Agent": "Mozilla/5.0"}
for base in ["https://cdn.jsdelivr.net/gh/PhotonQuantum/jaccount-captcha-solver@master/captcha.jpg",
             "https://raw.githubusercontent.com/PhotonQuantum/jaccount-captcha-solver/master/captcha.jpg"]:
    try:
        req = urllib.request.Request(base, headers=HDR)
        with urllib.request.urlopen(req, timeout=25) as r:
            b = r.read()
        open(IMG, "wb").write(b)
        print("样例验证码已下载:", len(b), "bytes from", base)
        break
    except Exception as e:
        print("FAIL", base, e)

pil = Image.open(IMG)
print("样例图片尺寸:", pil.size, "模式:", pil.mode)

# ---------- 官方预处理 ----------
LUT = [0] * 156 + [1] * 100
img_rec = pil.convert("L").point(LUT, "1")
arr = np.array(img_rec, dtype=np.float32)
print("预处理后形状:", arr.shape, "取值:", sorted(set(arr.flatten().tolist())))

sess = rt.InferenceSession(MODEL)
print("输入:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("输出:", [(o.name, o.shape) for o in sess.get_outputs()])

out = sess.run(None, {sess.get_inputs()[0].name: arr[None, None, ...]})

print("\n=== 各输出头 argmax ===")
for o in sess.get_outputs():
    t = out[[x.name for x in sess.get_outputs()].index(o.name)]
    n = t.shape[1]
    a = int(np.argmax(t, 1)[0])
    print(f"  {o.name}: shape={t.shape} classes={n} argmax={a}"
          + ("  -> BLANK(该位不存在)" if a >= 26 else f"  -> '{chr(ord('a')+a)}'"))

# ---------- 官方后处理 ----------
official = ""
for t in out:
    asc = int(np.argmax(t, 1))
    if asc < 26:
        official += chr(ord("a") + asc)
print("\n官方 ocr.py 结果      :", official)

# ---------- 新脚本逻辑（等价复刻） ----------
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

new_txt = ""
confs = []
for t in out:
    n = t.shape[1]
    a = int(np.argmax(t, 1)[0])
    if a >= 26:
        continue
    p = float(softmax(t[0][:n])[a])
    confs.append(p)
    new_txt += chr(ord("a") + a)
print("新版脚本逻辑结果      :", new_txt, " 最低置信度 %.1f%%" % (min(confs) * 100 if confs else 0))

# ---------- 旧脚本逻辑（对所有输出只读前 26 类）----------
old_txt = ""
for t in out:
    d = t[0]
    old_txt += chr(ord("a") + int(np.argmax(d[:26])))
print("旧版脚本逻辑结果(错误) :", old_txt)

print("\n期望答案 (README): gbmke")
print("官方==新版 ?", official == new_txt, "| 新版==期望 ?", new_txt == "gbmke")
