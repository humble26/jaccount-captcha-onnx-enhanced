import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""独立复验归档研究线的两个核心声称：

声称 A：「纯 NumPy ResNet-20 前向与生产 ONNX 逐层一致」
        -> 用 onnxruntime 与 resnet20_np.forward_fast 在同一批图上看预测是否一致
声称 B：「回写微调的 ONNX 与原生 ONNX 预测 100% 相同，无交付价值」
        -> 用 onnxruntime 跑两个模型，逐张对比预测串

不依赖训练脚本，不依赖中间 npz。
"""
import os
import sys
import glob
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

W = _VD
ARCH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究'
ANALYSIS = os.path.join(ARCH, 'analysis')
BACKPORT = os.path.join(ARCH, '_e2e_RT_backport')

MODEL_NATIVE = os.path.join(W, 'nn_model.onnx')
MODEL_E2E = os.path.join(BACKPORT, 'nn_model_e2e.onnx')

SETS = {
    'samples(20)': glob.glob(os.path.join(W, 'samples', '*.png')),
    'samples2(100)': glob.glob(os.path.join(W, 'samples2', '*.png')),
    'holdout(100)': glob.glob(os.path.join(W, 'holdout', '*.png')),
    'new300': glob.glob(os.path.join(W, 'new300', '*.png')),
}

for k in SETS:
    SETS[k] = sorted(SETS[k])


def preprocess(path):
    g = np.asarray(Image.open(path).convert('L'), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32).reshape(1, 1, 40, 110)


def decode(logits_list):
    """与生产脚本一致：前 4 头 26 类，第 5 头 27 类（26=blank 则截断）"""
    chars = []
    for i, z in enumerate(logits_list):
        chars.append(int(np.argmax(z[0])))
    out = ''.join(chr(ord('a') + c) for c in chars[:4])
    if chars[4] != 26:
        out += chr(ord('a') + chars[4])
    return out


# ---------------- 会话 ----------------
print('OR T', ort.__version__)
so = ort.SessionOptions()
so.log_severity_level = 3
sess_native = ort.InferenceSession(MODEL_NATIVE, so, providers=['CPUExecutionProvider'])
sess_e2e = ort.InferenceSession(MODEL_E2E, so, providers=['CPUExecutionProvider'])

out_names = sorted([o.name for o in sess_native.get_outputs()], key=lambda s: int(s))
print('原生输出名(按位置序):', out_names)
print('回写输出名(按位置序):', sorted([o.name for o in sess_e2e.get_outputs()], key=lambda s: int(s)))
in_name = sess_native.get_inputs()[0].name
print('输入名:', in_name, sess_native.get_inputs()[0].shape)
print()

# ---------------- 声称 B ----------------
print('=' * 72)
print('声称 B：回写 ONNX vs 原生 ONNX 的预测差异')
print('=' * 72)
diff_total = 0
recs = []
for name, files in SETS.items():
    if not files:
        print('  (空) %s' % name)
        continue
    n_diff = 0
    detail = []
    for f in files:
        x = preprocess(f)
        za = sess_native.run(out_names, {in_name: x})
        zb = sess_e2e.run(out_names, {in_name: x})
        pa, pb = decode(za), decode(zb)
        # 同时比 logits 最大绝对差
        md = max(float(np.abs(a - b).max()) for a, b in zip(za, zb))
        recs.append((name, os.path.basename(f), pa, pb, md))
        if pa != pb:
            n_diff += 1
            detail.append((os.path.basename(f), pa, pb, md))
    diff_total += n_diff
    mds = [r[4] for r in recs if r[0] == name]
    print('  %-14s %3d 张 | 预测不同 %d 张 | logits 最大绝对差 max=%.3e  中位=%.3e'
          % (name, len(files), n_diff, max(mds), float(np.median(mds))))
    for d in detail[:10]:
        print('      DIFF %s  原生=%s 回写=%s  dmax=%.3e' % d)
print('  ==> 合计预测不同: %d 张' % diff_total)
print()

# ---------------- 声称 A ----------------
print('=' * 72)
print('声称 A：NumPy 前向 vs onnxruntime 前向（预测一致性）')
print('=' * 72)
try:
    sys.path.insert(0, ANALYSIS)
    import resnet20_np as RN
    print('  已导入 resnet20_np')
    Wd = {k: v.astype(np.float64) for k, v in RN.load_onnx_weights(MODEL_NATIVE).items()}
    print('  ONNX initializer 总数:', len(Wd))
    keep = [k for k in Wd if k.startswith('onnx::Conv') or k.startswith('linear')]
    drop = [k for k in Wd if k not in keep]
    print('  filter_trainable 保留:', len(keep), '| 丢弃:', len(drop))
    print('  丢弃清单:', drop)
    print('  保留中 linear 类:', sorted(k for k in keep if k.startswith('linear')))
    bad = 0
    n = 0
    for name, files in SETS.items():
        sub = files[::7]          # 抽样，NumPy 前向较慢
        for f in sub:
            x = preprocess(f).astype(np.float64)
            c = RN.forward_fast(x, Wd)
            lg = RN.forward_heads(c['feat'], Wd)
            pn = decode([np.asarray(z) for z in lg])
            zo = sess_native.run(out_names, {in_name: preprocess(f)})
            po = decode(zo)
            n += 1
            if pn != po:
                bad += 1
                if bad <= 10:
                    print('      MISMATCH %s  numpy=%s onnx=%s' % (os.path.basename(f), pn, po))
    print('  抽样 %d 张，预测不一致 %d 张' % (n, bad))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('  NumPy 路线复验未完成:', e)
