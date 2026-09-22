import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""在调参集(120) 与留出集(100) 上分别验证「只看第 4 位」的低置信判据。

现行生产策略：min(所有字符位的 softmax 置信) < 0.999 -> 换图
候选策略 A  ：只看第 4 位置信 < 0.999 -> 换图
候选策略 B  ：第 4 位 < 0.999 或 其他位 < 0.9999 -> 换图（保守版）
候选策略 C  ：min logit 间隔 < 6 -> 换图

评估门槛：两个集合必须都能拦下该集合内的全部错误，再看自动化率与误伤。
"""
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

W = _VD
MODEL = os.path.join(W, 'nn_model.onnx')
gt = json.load(open(os.path.join(W, 'ground_truth_all.json'), encoding='utf-8'))
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


def load(dirs):
    out = []
    for d in dirs:
        dd = os.path.join(W, d)
        for f in sorted(os.listdir(dd)):
            if f.endswith('.png'):
                k = os.path.splitext(f)[0]
                if k in gt:
                    out.append((k, os.path.join(dd, f)))
    return out


def analyze(items):
    recs = []
    for k, p in items:
        z = sess.run(names, {iname: prep(p)})
        chars, confs, margins = [], [], []
        for i, zz in enumerate(z):
            v = zz[0]
            o = np.argsort(-v)
            b = int(o[0])
            if i == 4 and b == 26:
                chars.append(None); continue
            pr = softmax(v)
            chars.append(CH[b])
            confs.append(float(pr[b]))
            margins.append(float(v[o[0]] - v[o[1]]))
        text = ''.join(c for c in chars if c)
        recs.append({
            'key': k, 'ok': text == gt[k], 'text': text, 'truth': gt[k],
            'conf_min': min(confs), 'margin_min': min(margins),
            'conf_p4': confs[3] if len(confs) >= 4 else 1.0,
            'conf_other_min': min([c for i, c in enumerate(confs) if i != 3]) if len(confs) > 1 else 1.0,
        })
    return recs


SUB = {
    '调参集 samples+samples2 (120)': load(['samples', 'samples2']),
    '留出集 holdout (100)': load(['holdout']),
    '新增 300（自标注真值，仅看换图率）': load(['new300']),
}

STRATS = [
    ('现行: 全局min<0.999 换图', lambda r: r['conf_min'] < 0.999),
    ('候选A: 位4<0.999 换图', lambda r: r['conf_p4'] < 0.999),
    ('候选B: 位4<0.999 或 其他位<0.9999', lambda r: r['conf_p4'] < 0.999 or r['conf_other_min'] < 0.9999),
    ('候选C: 全局margin<6 换图', lambda r: r['margin_min'] < 6),
    ('候选D: 位4margin<8 换图', lambda r: r['pos4_margin'] < 8 if 'pos4_margin' in r else r['margin_min'] < 8),
]

allrecs = {}
for name, items in SUB.items():
    recs = analyze(items)
    allrecs[name] = recs
    errs = [r for r in recs if not r['ok']]
    print('==== %s ====  错误 %d 张: %s'
          % (name, len(errs), ', '.join('%s(%s->%s)' % (r['key'], r['text'], r['truth']) for r in errs)))
    if '仅看换图率' in name:
        for sname, fn in STRATS[:3]:
            low = sum(1 for r in recs if fn(r))
            print('   %-34s 换图率 %.1f%%' % (sname, 100 * low / len(recs)))
        print()
        continue
    print('   %-34s %10s %8s %8s' % ('策略', '自动化率', '拦错', '误伤'))
    for sname, fn in STRATS:
        low = [r for r in recs if fn(r)]
        keep = len(recs) - len(low)
        blocked = sum(1 for r in low if not r['ok'])
        harm = sum(1 for r in low if r['ok'])
        print('   %-34s %9.1f%% %5d/%-3d %8d'
              % (sname, 100 * keep / len(recs), blocked, len(errs), harm))
    print()

print('=' * 78)
print('结论检查：两个有真值的集合是否都能 5/5 拦下全部错误')
print('=' * 78)
tot_err = 0; tot_block = {'现行': 0, '候选A': 0, '候选B': 0}
for name, recs in allrecs.items():
    if '仅看换图率' in name:
        continue
    errs = [r for r in recs if not r['ok']]
    tot_err += len(errs)
    for tag, fn in (('现行', STRATS[0][1]), ('候选A', STRATS[1][1]), ('候选B', STRATS[2][1])):
        tot_block[tag] += sum(1 for r in errs if fn(r))
print('  错误总数 %d | 现行拦下 %d | 候选A 拦下 %d | 候选B 拦下 %d'
      % (tot_err, tot_block['现行'], tot_block['候选A'], tot_block['候选B']))

print()
print('整个 220 张（合并）的自动化率对比：')
merged = allrecs['调参集 samples+samples2 (120)'] + allrecs['留出集 holdout (100)']
for sname, fn in STRATS[:4]:
    low = sum(1 for r in merged if fn(r))
    print('  %-34s 换图率 %.1f%%  自动化率 %.1f%%' % (sname, 100 * low / len(merged), 100 - 100 * low / len(merged)))
