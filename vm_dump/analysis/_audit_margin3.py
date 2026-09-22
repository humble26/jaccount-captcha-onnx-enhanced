import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""在新增 300 张上量化换图率（用户实际体感），并检查数据分布漂移。

300 张的真值在 new300_gt.json（未并入 ground_truth_all.json），故单独加载。
真值质量已独立核对（48 张盲测 + 33 张位4高危，零错误），可用于换图率评估。
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


def scan(dirs):
    out = []
    for d in dirs:
        dd = os.path.join(W, d)
        for f in sorted(os.listdir(dd)):
            if f.endswith('.png'):
                out.append((os.path.splitext(f)[0], os.path.join(dd, f)))
    return out


def analyze(items, gtmap):
    recs = []
    for k, p in items:
        if k not in gtmap:
            continue
        z = sess.run(names, {iname: prep(p)})
        chars, confs, margins = [], [], []
        for i, zz in enumerate(z):
            v = zz[0]
            o = np.argsort(-v); b = int(o[0])
            if i == 4 and b == 26:
                chars.append(None); continue
            pr = softmax(v)
            chars.append(CH[b]); confs.append(float(pr[b]))
            margins.append(float(v[o[0]] - v[o[1]]))
        text = ''.join(c for c in chars if c)
        recs.append({'key': k, 'text': text, 'truth': gtmap[k], 'ok': text == gtmap[k],
                     'conf_min': min(confs), 'margin_min': min(margins),
                     'conf_p4': confs[3], 'len4': len(text) == 4})
    return recs


GROUPS = {
    '220 张人工标注 (tune+hold)': (scan(['samples', 'samples2', 'holdout']), gt220),
    '300 张新增（独立采集）': (scan(['new300']), gt300),
}

print('%-28s %6s %8s %10s %10s %10s' % ('数据组', '张数', '4位占比', '现行换图率', '候选A换图率', '候选C换图率'))
print('-' * 84)
for name, (items, gm) in GROUPS.items():
    r = analyze(items, gm)
    if not r:
        continue
    n4 = sum(1 for x in r if x['len4'])
    a = 100 * sum(1 for x in r if x['conf_min'] < 0.999) / len(r)
    b = 100 * sum(1 for x in r if x['conf_p4'] < 0.999) / len(r)
    c = 100 * sum(1 for x in r if x['margin_min'] < 6) / len(r)
    print('%-28s %6d %7.1f%% %9.1f%% %11.1f%% %11.1f%%' % (name, len(r), 100 * n4 / len(r), a, b, c))
    errs = [x for x in r if not x['ok']]
    print('   自检：整串错 %d 张（%s）' % (len(errs), ', '.join('%s %s->%s' % (e['key'], e['text'], e['truth']) for e in errs[:6])))
    print('   信号分布：conf_min 中位=%.5f  margin_min 中位=%.3f'
          % (float(np.median([x['conf_min'] for x in r])), float(np.median([x['margin_min'] for x in r]))))
    print()

print('=' * 84)
print('数据漂移检查：两组的关键统计是否一致（若差异大，说明采集分布变了）')
print('=' * 84)
rows = {}
for name, (items, gm) in GROUPS.items():
    r = analyze(items, gm)
    if r:
        rows[name] = r
if len(rows) == 2:
    k = list(rows)
    for key, label in (('conf_min', 'min 置信'), ('margin_min', 'min logit 间隔'), ('conf_p4', '第4位置信')):
        v1 = [x[key] for x in rows[k[0]]]
        v2 = [x[key] for x in rows[k[1]]]
        print('  %-16s 220张: 中位=%.5f 5%%分位=%.5f | 300张: 中位=%.5f 5%%分位=%.5f'
              % (label, float(np.median(v1)), float(np.percentile(v1, 5)),
                 float(np.median(v2)), float(np.percentile(v2, 5))))
