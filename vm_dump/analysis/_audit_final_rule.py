import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""确定最终换图判据：单判据 vs 带兜底的组合判据。

评估约束（必须同时满足）：
  - 调参集 120 张拦下其 3 个错误
  - 留出集 100 张拦下其 2 个错误
在此前提下比较自动化率（越高越好）与误伤（越低越好），并给出 300 张上的换图率（用户体感）。
"""
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

W = _VD
A = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'
MODEL = os.path.join(W, 'nn_model.onnx')
gt220 = json.load(open(os.path.join(W, 'ground_truth_all.json'), encoding='utf-8'))
gt300 = json.load(open(os.path.join(A, 'new300_gt.json'), encoding='utf-8'))

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
names = sorted([o.name for o in sess.get_outputs()], key=int)
iname = sess.get_inputs()[0].name
CH = 'abcdefghijklmnopqrstuvwxyz'


def prep(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32).reshape(1, 1, 40, 110)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def analyze(dirs, gm):
    out = []
    for d in dirs:
        dd = os.path.join(W, d)
        for f in sorted(os.listdir(dd)):
            if not f.endswith('.png'):
                continue
            k = os.path.splitext(f)[0]
            if k not in gm:
                continue
            z = sess.run(names, {iname: prep(os.path.join(dd, f))})
            chars, confs, margins = [], [], []
            for i, zz in enumerate(z):
                v = zz[0]
                o = np.argsort(-v)
                b = int(o[0])
                if i == 4 and b == 26:
                    chars.append(None); continue
                pr = softmax(v)
                chars.append(CH[b]); confs.append(float(pr[b]))
                margins.append(float(v[o[0]] - v[o[1]]))
            text = ''.join(c for c in chars if c)
            out.append({'key': k, 'ok': text == gm[k], 'text': text, 'truth': gm[k],
                        'conf': min(confs), 'margin': min(margins)})
    return out


TUNE = analyze(['samples', 'samples2'], gt220)
HOLD = analyze(['holdout'], gt220)
NEW = analyze(['new300'], gt300)
ALL220 = TUNE + HOLD

RULES = [
    ('现行 conf<0.999', lambda r: r['conf'] < 0.999),
    ('C  margin<6', lambda r: r['margin'] < 6),
    ('C+ margin<6 或 conf<0.99', lambda r: r['margin'] < 6 or r['conf'] < 0.99),
    ('C++ margin<6 或 conf<0.95', lambda r: r['margin'] < 6 or r['conf'] < 0.95),
    ('D  margin<5 或 conf<0.99', lambda r: r['margin'] < 5 or r['conf'] < 0.99),
    ('E  margin<7 或 conf<0.99', lambda r: r['margin'] < 7 or r['conf'] < 0.99),
    ('F  margin<8 或 conf<0.99', lambda r: r['margin'] < 8 or r['conf'] < 0.99),
]


def stat(recs, fn):
    low = [r for r in recs if fn(r)]
    errs = [r for r in recs if not r['ok']]
    return (100 - 100 * len(low) / len(recs),                       # 自动化率
            sum(1 for r in low if not r['ok']), len(errs),          # 拦错 / 总错
            sum(1 for r in low if r['ok']),                         # 误伤
            100 * len(low) / len(recs))                             # 换图率


print('%-28s | %-26s | %-26s | %8s %8s' % ('策略', '调参集 120', '留出集 100', '220换图率', '300换图率'))
print('%-28s | %-26s | %-26s | %8s %8s' % ('', '自动化率 拦错 误伤', '自动化率 拦错 误伤', '', ''))
print('-' * 108)
for name, fn in RULES:
    a1, b1, t1, h1, r1 = stat(TUNE, fn)
    a2, b2, t2, h2, r2 = stat(HOLD, fn)
    _, _, _, _, r220 = stat(ALL220, fn)
    _, _, _, _, r300 = stat(NEW, fn)
    ok = (b1 == t1) and (b2 == t2)
    print('%-28s | %6.1f%% %4d/%-3d %4d        | %6.1f%% %4d/%-3d %4d        | %7.1f%% %7.1f%%  %s'
          % (name, a1, b1, t1, h1, a2, b2, t2, h2, r220, r300,
             'OK' if ok else '★未全拦'))

print()
print('判据合格线：两个子集都必须拦下全部错误（拦错 == 总错）。')
print()
print('补充：错误样本在两个信号上的具体取值')
for r in ALL220:
    if not r['ok']:
        print('   %-6s conf=%.5f  margin=%.3f' % (r['key'], r['conf'], r['margin']))
print()
print('信号相关性（220 张）：conf 与 margin 的 Spearman 近似')
c = np.array([r['conf'] for r in ALL220])
m = np.array([r['margin'] for r in ALL220])
rc = np.argsort(np.argsort(c)); rm = np.argsort(np.argsort(m))
print('   rank 相关系数 = %.4f' % float(np.corrcoef(rc, rm)[0, 1]))
