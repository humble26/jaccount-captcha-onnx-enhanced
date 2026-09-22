import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""复核：原脚本后处理里的 lastOne < 10 / ratio < 0.6 两个条件，
用的到底是 softmax 概率还是原始 logit？这决定了它的启发式是否有效。"""
import os, warnings, json
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)

W = _VD
sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
LUT = [0] * 156 + [1] * 100
OUTS = [o.name for o in sess.get_outputs()]

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

rows = []
for fn in sorted(os.listdir(os.path.join(W, "samples"))):
    if not fn.endswith(".png"):
        continue
    pil = Image.open(os.path.join(W, "samples", fn))
    arr = np.array(pil.convert("L").point(LUT, "1"), dtype=np.float32)
    out = sess.run(None, {in_name: arr[None, None, ...]})

    # 新版逻辑：每个头按自身类别数 argmax，命中 >=26 视为 blank
    new_txt, probs = "", []
    for t in out:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        probs.append(float(softmax(t[0][:n])[a]))
        new_txt += chr(ord("a") + a)

    # 原脚本逻辑：一律只读前 26 类，取原始 logit（未过 softmax）
    raw_last, raw_first4 = [], []
    old_txt = ""
    for i, t in enumerate(out):
        d = t[0]
        j = int(np.argmax(d[:26]))
        old_txt += chr(ord("a") + j)
        (raw_last if i == 4 else raw_first4).append(float(d[:26].max()))
    last_one = raw_last[0]
    first4_avg = sum(raw_first4) / 4
    ratio = last_one / first4_avg if first4_avg else 0

    # 第 5 个头：blank 类的原始 logit
    t5 = out[4][0]
    blank_logit = float(t5[26]) if t5.shape[0] > 26 else None
    max_over_27 = float(t5.max())
    argmax27 = int(np.argmax(t5))

    trigger = (last_one < 10) or (ratio < 0.6)
    old_final = old_txt[:4] if trigger else old_txt

    rows.append(dict(file=fn, new=new_txt, old_raw=old_txt,
                     lastLogit=round(last_one, 2), first4Avg=round(first4_avg, 2),
                     ratio=round(ratio, 3), trigger=trigger, old_final=old_final,
                     blankLogit=round(blank_logit, 2) if blank_logit is not None else None,
                     max27=round(max_over_27, 2), argmax27=argmax27,
                     minProb=round(min(probs) * 100, 1) if probs else 0))

print(f"{'file':<8}{'新版':<8}{'第5头maxlogit':>14}{'前4均值':>10}{'ratio':>8}{'触发截断':>9}{'旧最终':>8}{'blank logit':>13}{'27类argmax':>11}")
for r in rows:
    print(f"{r['file']:<8}{r['new']:<8}{r['lastLogit']:>14}{r['first4Avg']:>10}"
          f"{r['ratio']:>8}{str(r['trigger']):>9}{r['old_final']:>8}"
          f"{str(r['blankLogit']):>13}{r['argmax27']:>11}")

print("\n---- 汇总 ----")
print("样本数:", len(rows))
print("新版 4 位:", sum(1 for r in rows if len(r['new']) == 4),
      " 5 位:", sum(1 for r in rows if len(r['new']) == 5))
print("原脚本启发式触发截断(判为4位)的样本数:", sum(1 for r in rows if r['trigger']))
print("原脚本最终结果 == 新版结果 的样本数:",
      sum(1 for r in rows if r['old_final'] == r['new']))
print("原脚本最终结果 != 新版结果 的样本数:",
      sum(1 for r in rows if r['old_final'] != r['new']))
print("第5头 blank logit > 字母 max logit 的样本数:",
      sum(1 for r in rows if r['argmax27'] == 26))
json.dump(rows, open(os.path.join(W, "logit_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
