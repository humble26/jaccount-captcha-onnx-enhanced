import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""1) 错误样本的 softmax 置信度是否与正确样本可分（决定"低置信重试"是否可行）
   2) 冷启动下载预算：ort.min.js / wasm / 模型 各自多大"""
import os, json, warnings, urllib.request
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)
W = _VD

gt = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))
EV = json.load(open(os.path.join(W, "holdout_eval.json"), encoding="utf-8"))
base = EV["pred"]["base"]

sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name


def path_of(sid):
    for sub in ("samples", "samples2", "holdout"):
        p = os.path.join(W, sub, sid + ".png")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(sid)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


rows = []
for sid in sorted(gt):
    g = np.asarray(Image.open(path_of(sid)).convert("L"), dtype=np.float32)
    arr = (g >= 156).astype(np.float32)
    outs = sess.run(None, {IN: arr[None, None, ...]})
    probs = []
    for t in outs:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        probs.append(float(softmax(t[0][:n])[a]))
    rows.append({"id": sid, "pred": base[sid], "truth": gt[sid],
                 "minP": min(probs) if probs else 0.0,
                 "avgP": sum(probs) / len(probs) if probs else 0.0,
                 "ok": base[sid] == gt[sid]})

good = [r for r in rows if r["ok"]]
bad = [r for r in rows if not r["ok"]]
print(f"总样本 {len(rows)}  正确 {len(good)}  错误 {len(bad)}")
print("\n=== 错例（含占位判定）===")
print(f"{'id':<7}{'真值':<8}{'识别':<8}{'最低概率':>10}{'平均概率':>10}")
for r in bad:
    print(f"{r['id']:<7}{r['truth']:<8}{r['pred']:<8}{r['minP']*100:>9.2f}%{r['avgP']*100:>9.2f}%")

print("\n=== 正确样本的置信度分布 ===")
gp = sorted(r["minP"] for r in good)
for q in (0, 5, 10, 25, 50):
    k = gp[int(len(gp) * q / 100)]
    print(f"  {q:>2}% 分位: {k*100:.2f}%")
print(f"  最低值: {gp[0]*100:.2f}%")

print("\n=== 若用「最低概率 < 阈值 则重新取图重识别」能筛掉多少错例？===")
print(f"{'阈值':>8}{'被标记样本':>12}{'其中错误':>10}{'漏掉的错误':>12}{'标记率':>9}")
for thr in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999):
    flagged = [r for r in rows if r["minP"] < thr]
    caught = [r for r in flagged if not r["ok"]]
    missed = [r for r in bad if r["minP"] >= thr]
    print(f"{thr*100:>7.1f}%{len(flagged):>12}{len(caught):>10}{len(missed):>12}{len(flagged)/len(rows)*100:>8.1f}%")

print("\n=== 冷启动下载预算 ===")
H = {"User-Agent": "Mozilla/5.0"}
base_url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/"
total = 0
for f in ["ort.min.js", "ort-wasm.wasm", "ort-wasm-simd.wasm",
          "ort-wasm-threaded.wasm", "ort-wasm-simd-threaded.wasm"]:
    try:
        with urllib.request.urlopen(urllib.request.Request(base_url + f, headers=H), timeout=40) as r:
            n = len(r.read())
        total += n
        print(f"  {f:<32}{n/1024:>9.1f} KB")
    except Exception as e:
        print(f"  {f:<32} 取不到 ({type(e).__name__})")
print(f"  {'模型 nn_model.onnx':<32}{os.path.getsize(os.path.join(W,'nn_model.onnx'))/1024:>9.1f} KB")
print(f"\n  实际只需 ort.min.js + 一个 simd wasm + 模型："
      f"{(570149 + 0)/1024:.0f} KB + wasm + 1106 KB")

json.dump(rows, open(os.path.join(W, "confidence_analysis.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n已写入 confidence_analysis.json")
