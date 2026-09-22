import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""看清真相：验证码图像到底是什么样的？

关键矛盾：
  * 权威脚本用 (g >= 156) 得到 97.73%
  * 但 (g>=156) 的占比是 93%（大片为 1）
  * 而"字形跨度 40/40, 110/110"说明这个掩码覆盖全图

=> 说明图像是「浅色背景 + 深色字」，但阈值 156 落在背景一侧？
   还是图像本身是「白底黑字」而 156 太低？

本脚本打印统计、导出可视化图，一次看清。
"""
import os, json
import numpy as np
from PIL import Image

WS = _REPO
VD = os.path.join(WS, "vm_dump")
gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
paths = {}
for d in ["samples", "samples2", "holdout"]:
    dd = os.path.join(VD, d)
    if os.path.isdir(dd):
        for f in os.listdir(dd):
            if f.endswith(".png"):
                paths[os.path.splitext(f)[0]] = os.path.join(dd, f)

print("=" * 80)
print("图像统计（多张横跨 samples/holdout）")
print("=" * 80)
for k in ["c00", "c12", "h000", "h030", "n088", "n051"]:
    if k not in paths:
        continue
    a = np.asarray(Image.open(paths[k]).convert("L"))
    u, c = np.unique(a, return_counts=True)
    print(f"\n[{k}] 真值={gt.get(k)}  shape={a.shape} dtype={a.dtype}")
    print(f"  min={a.min()} max={a.max()} mean={a.mean():.1f} median={np.median(a):.0f}")
    print(f"  >=156 占比 {(a>=156).mean()*100:.2f}%   <156 占比 {(a<156).mean()*100:.2f}%")
    print(f"  >200 占比 {(a>200).mean()*100:.2f}%   <100 占比 {(a<100).mean()*100:.2f}%")
    top = sorted(zip(c, u), reverse=True)[:6]
    print(f"  最常见取值: " + ", ".join(f"{v}({n})" for n, v in top))

print()
print("=" * 80)
print("列投影（用 <156 = 深色笔画）看字符分布")
print("=" * 80)
import numpy as np
for k in ["c00", "c12"]:
    if k not in paths: continue
    a = np.asarray(Image.open(paths[k]).convert("L"))
    ink = a < 156
    cp = ink.sum(axis=0)
    nz = np.where(cp > 0)[0]
    print(f"\n[{k}] 真值={gt[k]}  笔画像素总数 {ink.sum()}")
    if len(nz):
        print(f"  有笔画列范围: {nz.min()}~{nz.max()}  共 {len(nz)} 列")
    # 按列打印压缩后的分布（每 5 列一个桶）
    buckets = [cp[i:i+5].sum() for i in range(0, 110, 5)]
    print("  每5列的笔画数: " + " ".join(f"{b:2d}" for b in buckets))
    rows = np.where(ink.any(axis=1))[0]
    if len(rows):
        print(f"  有笔画行范围: {rows.min()}~{rows.max()}  共 {len(rows)} 行")

print()
print("=" * 80)
print("导出可视化对照图（原图 + >=156掩码 + <156掩码）")
print("=" * 80)
sel = ["c00", "c12", "h030", "n088", "n051", "h040"]
tiles = []
for k in sel:
    if k not in paths: continue
    a = np.asarray(Image.open(paths[k]).convert("L")).astype(np.float32)
    m1 = (a >= 156).astype(np.float32) * 255
    m2 = (a < 156).astype(np.float32) * 255
    row = np.hstack([a, np.full((40, 4), 128, np.float32), m1,
                     np.full((40, 4), 128, np.float32), m2])
    tiles.append(row)
    tiles.append(np.full((4, row.shape[1]), 128, np.float32))
canvas = np.vstack(tiles).astype(np.uint8)
Image.fromarray(canvas).resize((canvas.shape[1] * 3, canvas.shape[0] * 3), Image.NEAREST).save(
    os.path.join(VD, "mask_compare.png"))
print(f"  已保存 {os.path.join(VD,'mask_compare.png')}  size={canvas.shape}")
print("  三列依次为: 原图 | >=156掩码 | <156掩码")

print()
print("=" * 80)
print("全体样本笔画占比统计（用 <156）")
print("=" * 80)
ratios = []
for k in paths:
    a = np.asarray(Image.open(paths[k]).convert("L"))
    ratios.append((a < 156).mean())
ratios = np.array(ratios)
print(f"  <156 占比: 均值 {ratios.mean()*100:.2f}%  中位 {np.median(ratios)*100:.2f}%  "
      f"范围 {ratios.min()*100:.2f}%~{ratios.max()*100:.2f}%")
print(f"  即有效笔画像素约 {ratios.mean()*4400:.0f} 个/张")
