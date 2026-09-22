import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""① 用 onnxruntime 跑统一 520 张，导出 logits（供 JS 侧喂给真实 postprocess）
   ② 同时用 numpy 算出参考值 text / minMargin / minConfidence

产出（audit_sheets/）：
  logits_520.bin   每张固定 524 字节 = (26+26+26+26+27)*4，float32 小端，按 index 顺序
  logits_520.json  [{key, split, truth, off, ref_text, ref_minMargin, ref_minConf}]
"""
import os
import json
import struct
import numpy as np
from PIL import Image
import onnxruntime as ort

W = _VD
ARCH = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究'
UNIFIED = os.path.join(ARCH, 'analysis', '_sampling_out', 'unified_gt_520.json')
OUT_DIR = os.path.join(W, 'audit_sheets')
MODEL = os.path.join(W, 'nn_model.onnx')

data = json.load(open(UNIFIED, encoding='utf-8'))
gt, index = data['gt'], data['index']
CH = 'abcdefghijklmnopqrstuvwxyz'


def abs_path(entry):
    rel = entry['rel'].replace('/', os.sep)
    if entry['base'] == 'PROD_WS':
        return os.path.join(W, '..', rel)
    parts = rel.split(os.sep)
    if parts and parts[0] == '02-重构研究':
        rel = os.sep.join(parts[1:])
    return os.path.join(ARCH, rel)


so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
names = sorted([o.name for o in sess.get_outputs()], key=int)
iname = sess.get_inputs()[0].name
print('输出头:', names, '| 输入:', iname)


def prep(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32).reshape(1, 1, 40, 110)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


recs = []
blob = bytearray()
missing = 0
for k, entry in index.items():
    p = abs_path(entry)
    if not os.path.exists(p):
        missing += 1
        continue
    zo = sess.run(names, {iname: prep(p)})
    off = len(blob)
    for z in zo:
        blob += z[0].astype('<f4').tobytes()
    # numpy 参考值
    chars, confs, margins = [], [], []
    for i, z in enumerate(zo):
        v = z[0]
        o = np.argsort(-v)
        b = int(o[0])
        if i == 4 and b == 26:
            continue
        chars.append(CH[b])
        confs.append(float(softmax(v)[b]))
        margins.append(float(v[o[0]] - v[o[1]]))
    ref_text = ''.join(chars)
    recs.append({
        'key': k, 'split': entry['split'], 'truth': gt[k], 'off': off,
        'ref_text': ref_text,
        'ref_minMargin': min(margins),
        'ref_minConf': min(confs),
        'ref_len': len(ref_text),
        'ref_pos4': chars[3],
    })

json.dump(recs, open(os.path.join(OUT_DIR, 'logits_520.json'), 'w', encoding='utf-8'),
          ensure_ascii=False)
open(os.path.join(OUT_DIR, 'logits_520.bin'), 'wb').write(bytes(blob))
print('导出 %d 张 | 缺失 %d | bin %.1f KB | 每张固定 %d 字节'
      % (len(recs), missing, len(blob) / 1024, 131 * 4))
print()
print('numpy 参考（按 lowMargin=6 判据）：')
for split in ('tune', 'hold', 'new'):
    sub = [r for r in recs if r['split'] == split]
    if not sub:
        continue
    errs = [r for r in sub if r['ref_text'] != r['truth']]
    low = [r for r in sub if r['ref_minMargin'] < 6]
    blocked = sum(1 for r in low if r['ref_text'] != r['truth'])
    harm = sum(1 for r in low if r['ref_text'] == r['truth'])
    print('  %-5s %3d 张 | 整串错 %d | 换图率 %5.1f%% | 拦错 %d/%d | 误伤 %d'
          % (split, len(sub), len(errs), 100 * len(low) / len(sub),
             blocked, len(errs), harm))
allr = recs
low = [r for r in allr if r['ref_minMargin'] < 6]
print('  ALL   %3d 张 | 换图率 %.1f%%' % (len(allr), 100 * len(low) / len(allr)))
