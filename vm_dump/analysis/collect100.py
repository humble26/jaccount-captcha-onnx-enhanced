"""追加采集 100 张真实验证码，并对每张同时跑 ResNet 的「新版逻辑」与「原版逻辑」，供后续盲标比对。"""
import os, time, json, socket, warnings
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)
socket.setdefaulttimeout(10)

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
SD = os.path.join(W, "samples2")
os.makedirs(SD, exist_ok=True)

sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
LUT = [0] * 156 + [1] * 100
HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
    "Referer": "https://jaccount.sjtu.edu.cn/jaccount/jalogin",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(pil):
    arr = np.array(pil.convert("L").point(LUT, "1"), dtype=np.float32)
    out = sess.run(None, {in_name: arr[None, None, ...]})

    new_txt, probs = "", []
    for t in out:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        probs.append(float(softmax(t[0][:n])[a]))
        new_txt += chr(ord("a") + a)

    # 原版逻辑：一律前 26 类 argmax + raw logit 启发式
    old_raw, first4 = "", []
    for i, t in enumerate(out):
        d = t[0]
        j = int(np.argmax(d[:26]))
        old_raw += chr(ord("a") + j)
        if i < 4:
            first4.append(float(d[:26].max()))
    last = float(out[4][0][:26].max())
    avg = sum(first4) / 4
    ratio = last / avg if avg else 0
    old_final = old_raw[:4] if (last < 10 or ratio < 0.6) else old_raw

    return new_txt, old_final, round(min(probs) * 100, 1) if probs else 0.0


def fetch():
    import urllib.request
    url = f"https://jaccount.sjtu.edu.cn/jaccount/captcha?_={int(time.time()*1000)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADER), timeout=10) as r:
        return r.read()


N = 100
rows, fails = [], 0
for i in range(N):
    try:
        b = fetch()
        if not (b[:2] == b"\xff\xd8" or b[:8] == b"\x89PNG\r\n\x1a\n"):
            fails += 1
            continue
        fn = f"n{i:03d}.png"
        p = os.path.join(SD, fn)
        open(p, "wb").write(b)
        pil = Image.open(p)
        new_txt, old_txt, conf = predict(pil)
        rows.append({"id": fn.replace(".png", ""), "resnet_new": new_txt,
                     "resnet_old": old_txt, "minConf": conf, "size": list(pil.size)})
        if (i + 1) % 20 == 0:
            print(f"  已采集 {len(rows)}/{i+1} ...", flush=True)
    except Exception as e:
        fails += 1
        print(f"  [{i:03d}] 失败 {type(e).__name__}: {e}", flush=True)
    time.sleep(0.3)

json.dump(rows, open(os.path.join(W, "samples2_resnet.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n完成：成功 {len(rows)} 张，失败 {fails} 张 -> {SD}")
sizes = {tuple(r['size']) for r in rows}
print("图片尺寸集合:", sizes)
