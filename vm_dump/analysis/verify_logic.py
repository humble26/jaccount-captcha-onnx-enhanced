import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import os, io, urllib.request, socket
import numpy as np
import onnxruntime as rt
from PIL import Image, ImageDraw

socket.setdefaulttimeout(8)
W = _VD
MODEL = os.path.join(W, "nn_model.onnx")

# ---------- 1. 再试一次真实验证码（带 Referer） ----------
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
       "Referer": "https://jaccount.sjtu.edu.cn/jaccount/jalogin",
       "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}
try:
    req = urllib.request.Request("https://jaccount.sjtu.edu.cn/jaccount/captcha", headers=HDR)
    with urllib.request.urlopen(req, timeout=8) as r:
        b = r.read()
    print("captcha endpoint ->", len(b), "bytes")
    if b[:2] == b"\xff\xd8" or b[:8] == b"\x89PNG\r\n\x1a\n":
        open(os.path.join(W, "real_captcha.png"), "wb").write(b)
        print("  saved real captcha!")
except Exception as e:
    print("captcha endpoint FAIL:", type(e).__name__, e)

sess = rt.InferenceSession(MODEL)
in_name = sess.get_inputs()[0].name
out_meta = sess.get_outputs()
print("\ninput:", in_name, sess.get_inputs()[0].shape)
print("outputs:", [(o.name, o.shape) for o in out_meta])

# ---------- 2. 构造 110x40 测试图，验证张量流程 ----------
img = Image.new("L", (110, 40), 255)
ImageDraw.Draw(img).text((8, 12), "abcde", fill=0)
LUT = [0] * 156 + [1] * 100
arr = np.array(img.point(LUT, "1"), dtype=np.float32)
print("\n输入张量 shape:", arr[None, None, ...].shape, "值域:", set(arr.flatten().tolist()))

out = sess.run(None, {in_name: arr[None, None, ...]})
print("实际输出 shape:", [t.shape for t in out])

# ---------- 3. 官方逻辑 vs 新脚本逻辑 vs 旧脚本逻辑 ----------
official = "".join(chr(ord('a') + int(np.argmax(t, 1))) for t in out if int(np.argmax(t, 1)) < 26)

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

new_txt, confs = "", []
for t in out:
    n = t.shape[1]
    a = int(np.argmax(t, 1)[0])
    if a >= 26:
        continue
    confs.append(float(softmax(t[0][:n])[a]))
    new_txt += chr(ord('a') + a)

old_txt = "".join(chr(ord('a') + int(np.argmax(t[0][:26]))) for t in out)

print("\n官方 ocr.py :", official)
print("新版脚本    :", new_txt, " 长度:", len(new_txt))
print("旧版脚本    :", old_txt, " 长度:", len(old_txt))
print("new == official ?", new_txt == official)

# ---------- 4. 确认 blank 类的语义：第5个头是否为 27 类 ----------
for o, t in zip(out_meta, out):
    n = t.shape[1]
    a = int(np.argmax(t, 1)[0])
    print(f"  {o.name}: {n} classes, argmax={a}" + ("  <-- 落在 blank 上，跳过" if a >= 26 else ""))
