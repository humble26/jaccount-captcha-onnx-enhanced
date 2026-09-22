import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
import os
d = _os.path.join(_REPO, ".workbuddy", "memory")
os.makedirs(d, exist_ok=True)
p = os.path.join(d, '2026-09-21.md')

note = """

## 架构层面突破评估（承接模型天花板报告）

### 重要修正（必须记住）
- 权威预处理 base_input = (g >= 156)，**前景为 1**。此前 arch_probe2 误用 <156 导致该轮分析作废。
- 图像真实构成：白底(255) + 灰色渐变细笔画，背景 93.08%，笔画仅 6.92%（约 304 px/张）。
- 字形几何：高 16.9px（画布 40），宽 86.2px（画布 110），单字符 19.2x17px。画布裕量充足。
- 笔画宽度：48.6% 的笔画只有 1~2px 宽（14691 样本统计）。
- decode 用 sess.run(None) 按图定义顺序；输出名数值序恰与位置序一致（218 对应位1 ... 222 对应位5）。

### 架构拓扑（从 ONNX 图读出）
- ResNet-20 变体：21 Conv + 19 Relu + 9 Add + 1 AveragePool + 1 Reshape + 5 Gemm，共 56 节点
- AveragePool 全局池化后 Reshape 成 (1,64)
- 5 个头（linear1~linear5，各 [26 或 27, 64]）**全部并联在同一个 64 维向量** /Reshape_output_0 上
- 头是单层 Gemm，无非线性隐藏层

### 五个待验证假设，实测否证 4 个
1. 全局池化丢位置信息 -> **否证**。抹掉第4槽时第4头输出改变 192/220，位置信息已隐式保留。
   判据2：只保留第4槽时第4头准确率降到 80.5%（完整图 97.7%），说明依赖上下文。
2. 二值化丢灰度 -> **反向否证**。灰度可分率 43.64% 低于二值 55.91%（5 个位置全部更差）。
   原因：灰度渐变是干扰而非信号，二值化是有效的归一化。这也解释了为何 11 种预处理变体全部无效。
   直接喂灰度给当前模型：0/220（分布外）。
3. 64 维太窄 -> **否证**。特征过度专属（跨集迁移仅 32~74%，集内 99~100%）=> 扩容量会加剧过拟合。
4. 输出头太浅 -> **部分成立**。2层MLP 比单层线性头高 3~8 个百分点（趋势支持，幅度小）。
5. 需要序列建模 CTC -> **否证**。长度判定零错误（4 字符码 blank 113/113 全对）。

### 关键诊断数据
- 位1/2/3/5 原生头 100%，位4 在留出集 98.00% —— 只有第4位弱。
- 过拟合诊断：调参集内部与留出集内部可分性都 99~100%（两集质量一致，非分布漂移）；
  但跨集迁移暴跌 26~67 个百分点。原图前景占比差 0.35pp、特征范数 27.90 对 27.86（分布无差异）。
- 结论：**主因是数据量不足（220 张支撑 26 类 x 5 头），不是架构**。

### 最终建议
- 真正突破：优先「增加定向数据（加权第4位 + 5 组混淆对 x/o, c/o, y/u, w/g, g/z）」+「增广/正则」，其次「加深头」。
- 不要做：扩宽 64 维、改灰度输入、CTC 序列解码、加回空间结构。
- 改架构天花板约 98.5%~99.0%（受 1~2px 笔画物理限制），且需重采集 + 重训 + 全链路回归。
- 零风险现成解仍是「低置信转人工/换图」（99.9% 阈值 -> 84.5% 全自动、100% 正确）。

### 本轮脚本
analysis/ 下新增：arch_probe.py, arch_probe2.py, check_eval.py, fix_diff.py, see_image.py,
arch_final.py, arch_verdict.py, gray_probe.py, plan_c_probe.py, plan_c_probe2.py, overfit_diag.py
产出：vm_dump/ARCH_BREAKTHROUGH_REPORT.md、vm_dump/mask_compare.png（原图 / 两种掩码对照）

### 经验教训
- **先验证预处理方向再分析**：>=156 与 <156 这个方向错误浪费了一轮分析。做掩码类分析前先打印图像像素直方图与可视化。
- 小样本（220 张 / 26 类）做 K 折交叉验证指标不可靠，应固定划分（调参集训 / 留出集测）。
- 在 bash 里用 python -c 内联写含反引号的内容会被命令替换吃掉，必须写独立脚本文件。
"""

with open(p, 'a', encoding='utf-8') as f:
    f.write(note)
print('written', os.path.getsize(p))
