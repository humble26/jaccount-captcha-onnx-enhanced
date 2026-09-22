import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""在 220 张人工标注集上比较两种「不确定性信号」的筛错能力。

现状（生产脚本）：min softmax 概率 < 0.999 -> 换图重试
候选（本实验）  ：min (top1_logit - top2_logit) < T -> 换图重试

动机：softmax 概率受 logit 整体尺度影响（温度效应），而 logit 间隔是更直接的
决策边界距离。若同样自动化率下 margin 能拦下更多错误，就是一个一行改动的产品优化。
注意：错误样本只有 5 个，统计力弱，结论只能作方向性判断。
"""
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

W = _VD
MODEL = os.path.join(W, 'nn_model.onnx')
GT = os.path.join(W, 'ground_truth_all.json')

gt = json.load(open(GT, encoding='utf-8'))
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


files = []
for d in ('samples', 'samples2', 'holdout'):
    dd = os.path.join(W, d)
    for f in sorted(os.listdir(dd)):
        if f.endswith('.png'):
            k = os.path.splitext(f)[0]
            if k in gt:
                files.append((k, os.path.join(dd, f)))

rows = []
for k, p in files:
    z = sess.run(names, {iname: prep(p)})
    chars = []
    confs = []      # 生产口径：blank 不计
    margins = []    # top1 - top2 logit，blank 位不计
    for i, zz in enumerate(z):
        v = zz[0]
        order = np.argsort(-v)
        b = int(order[0])
        if i == 4 and b == 26:      # 第 5 头的 blank
            chars.append(None)
            continue
        pr = softmax(v)
        chars.append(CH[b])
        confs.append(float(pr[b]))
        margins.append(float(v[order[0]] - v[order[1]]))
    text = ''.join(c for c in chars if c)
    truth = gt[k]
    pos4_conf = None
    pos4_margin = None
    if len(chars) >= 4 and chars[3] is not None:
        pos4_conf = confs[3]
        pos4_margin = margins[3]
    rows.append({
        'key': k, 'pred': text, 'truth': truth, 'ok': text == truth,
        'conf_min': min(confs), 'margin_min': min(margins),
        'pos4_conf': pos4_conf, 'pos4_margin': pos4_margin,
        'len_ok': len(text) == len(truth),
    })

n = len(rows)
errs = [r for r in rows if not r['ok']]
print('样本 %d 张 | 整串错误 %d 张' % (n, len(errs)))
print('错误样本：')
for r in errs:
    print('   %-6s 预测=%-6s 真值=%-6s conf=%.5f margin=%.3f 位4conf=%.5f 位4margin=%.3f'
          % (r['key'], r['pred'], r['truth'], r['conf_min'], r['margin_min'],
             r['pos4_conf'] if r['pos4_conf'] else -1,
             r['pos4_margin'] if r['pos4_margin'] else -1))
print()
srt = sorted(rows, key=lambda r: r['conf_min'])
print('conf_min 最低的 8 张：')
for r in srt[:8]:
    print('   %-6s conf=%.5f margin=%.3f ok=%s' % (r['key'], r['conf_min'], r['margin_min'], r['ok']))
srt2 = sorted(rows, key=lambda r: r['margin_min'])
print('margin_min 最低的 8 张：')
for r in srt2[:8]:
    print('   %-6s margin=%.3f conf=%.5f ok=%s' % (r['key'], r['margin_min'], r['conf_min'], r['ok']))

print()
print('=' * 78)
print('阈值扫描：达到相同自动化率时，谁能拦下更多错误')
print('=' * 78)
print('%-22s %10s %8s %8s %10s' % ('信号/阈值', '自动化率', '拦错', '误伤', '换图率'))
print('-' * 78)
for sig, name, ths in (
    ('conf_min', 'min softmax 概率', [0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.95]),
    ('margin_min', 'min logit 间隔', [12, 10, 8, 6, 5, 4, 3]),
):
    for t in ths:
        keep = [r for r in rows if r[sig] >= t]
        low = [r for r in rows if r[sig] < t]
        blocked = sum(1 for r in low if not r['ok'])
        harm = sum(1 for r in low if r['ok'])
        print('%-22s %9.1f%% %6d/%-3d %8d %9.1f%%'
              % ('%s >= %s' % (name, t), 100 * len(keep) / n, blocked, len(errs),
                 harm, 100 * len(low) / n))
    print('-' * 78)

print()
print('只看第 4 位信号的扫描（错误全部落在第 4 位）：')
print('%-22s %10s %8s %8s' % ('信号/阈值', '自动化率', '拦错', '误伤'))
print('-' * 60)
for sig, name, ths in (('pos4_conf', '位4 概率', [0.9999, 0.999, 0.99]),
                       ('pos4_margin', '位4 logit 间隔', [8, 6, 5, 4, 3])):
    for t in ths:
        keep = [r for r in rows if r[sig] is not None and r[sig] >= t]
        low = [r for r in rows if r[sig] is not None and r[sig] < t]
        blocked = sum(1 for r in low if not r['ok'])
        harm = sum(1 for r in low if r['ok'])
        print('%-22s %9.1f%% %6d/%-3d %8d'
              % ('%s >= %s' % (name, t), 100 * len(keep) / n, blocked, len(errs), harm))
