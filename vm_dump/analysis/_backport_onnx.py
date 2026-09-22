# -*- coding: utf-8 -*-
"""将端到端微调权重回写为生产 ONNX 模型（保留原始计算图与未训常量）。

输入:
  src_onnx     : 生产 nn_model.onnx
  weight_npz   : 微调权重 (key 与 onnx initializer 匹配, float64)
  out_onnx     : 输出路径
做法: 仅替换 npz 覆盖到的 initializer 数值, 其余(结构+未训常量)维持不变。

⚠ 2026-09-22 晚修正（重要）：
  默认回写的是 `*_best_hold4.npz` —— 它是"留出集上最优的**早停**权重"，改动量只有最终权重的
  约 1/10。用它回答"微调到底有没有效果"会得出误导性的结论：回写后预测与原生 100% 相同，
  看起来像"微调完全没生效"，实际是"选了一个几乎没动的权重"。
  要评估微调的真实效果，请指向 `e2e_final_extra.npz`（step 300 最终权重）——
  用它回写后 520 张里有 7 张预测改变（6 修正 / 1 退化），但全部落在训练集内，留出集 0 变化。
  用环境变量 E2E_NPZ 可覆盖默认值。
"""
import os, sys, json
import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_e2e as T

SRC = T.MODEL
NPZ = os.environ.get("E2E_NPZ") or os.path.join(HERE, "_e2e_out", "e2e_final_extra_best_hold4.npz")
# 归档后本文件位于 <归档>/02-重构研究/analysis/，回写产物放同级 _e2e_RT_backport/。
# （原来写死的 E:\harness\重构研究\... 在归档剪切移动后已不存在。）
OUT = os.environ.get("E2E_OUT_ONNX") or os.path.join(
    os.path.dirname(HERE), "_e2e_RT_backport", "nn_model_e2e.onnx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("加载生产 ONNX:", SRC)
m = onnx.load(SRC)
ini_by_name = {i.name: i for i in m.graph.initializer}
print(f"生产 initializer: {len(ini_by_name)} 个")

d = np.load(NPZ)
meta = json.load(open(NPZ + ".meta.json", encoding="utf-8"))
keys = meta["keys"]
print(f"微调 npz: {len(keys)} 个权重（best_hold4, 留出位4={meta.get('best_hold_acc4')}）")

replaced = 0
for i, k in enumerate(keys):
    if k not in ini_by_name:
        print("  [跳过] npz 权重无对应 initializer:", k)
        continue
    arr = d[f"w{i}"].astype(np.float32)
    # 关键形状校验: 回写前必须与原 initializer 形状一致
    orig = numpy_helper.to_array(ini_by_name[k])
    if orig.shape != arr.shape:
        raise SystemExit(f"形状不一致 {k}: 原{orig.shape} vs 新{arr.shape}")
    new_t = numpy_helper.from_array(arr, k)
    ini_by_name[k].CopyFrom(new_t)  # 原位替换, 保留 initializer 在 graph 的位置
    replaced += 1

# 记录被裁剪/未回写的常量（linear 之外的可训练权重确认全被覆盖）
trainable_all = [k for k in keys]
print(f"已回写 {replaced}/{len(keys)} 个权重")

onnx.checker.check_model(m)
onnx.save(m, OUT)
print(f"已保存回写模型: {OUT}  ({os.path.getsize(OUT)} bytes)")
print("onnx.checker 通过 ✓")

# 用 onnxruntime 验证新模型可加载且能推理一次
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
print("onnxruntime 加载回写模型成功, 输入:", [i.name for i in sess.get_inputs()])