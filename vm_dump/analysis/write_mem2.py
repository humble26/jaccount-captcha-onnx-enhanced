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

## 现成解落地 + 重构方案并行推进

### 线 1：现成解已落地（低置信自动换图重试）
油猴脚本 -> v4.5.0；扩展 -> app.js VERSION 1.0.5 / manifest 1.0.5。

关键改动：
- lowConfidence 从 0.60 提到 **0.999**。0.60 是**从未生效的死阈值** ——
  实测 5 个错误样本最低置信在 84.39%~99.58%，永远够不到 0.60。
- 新增 findRefreshButton()：9 个候选选择器兜底定位换图控件（抗改版，
  不用固定 id，要求 offsetWidth/Height>0 可见）。
- 新增 refreshCaptcha()：click 后等新图就绪，优先 img.decode()，1200ms 兜底超时。
- recognize() 重构：抽出 recognizeOnce()，主流程加入重试循环
  （最多 maxRefreshRetries=3 次），重试用尽则采用**置信最高**的那次 + 橙色标注。

修复的真实缺陷：best 原本与 res 比较，轮次置信忽高忽低（0.80->0.60->0.75）时
best 会被中间低值覆盖，丢掉真正最高的 0.80。改为与自身比较。

### 线 2：重构方案（脚本已可运行，但发现路径限制）
新增工具：
- analysis/targeted_sampling.py —— 定向采样配额表（已跑出结果）
- analysis/train_refactor.py —— extract / train / eval 三个子命令（已跑通）

定向采样缺口（现有 220 张）：
  8 个高危字符 c,g,o,u,w,x,y,z 要在第 4 位各达 60 例，还需补约 171 个字符位
  （约 38 张验证码即可覆盖，成本比想象低）
  2000 张目标配额：基础每字符 346 次，高危字符额外 +173（共 519）

**关键实测发现（推翻此前乐观估计）**：
在冻结的 64 维特征上重训头，**无论加宽加深都大幅劣于原生头**：
  位1 100%->63% / 位2 100%->68% / 位3 100%->58% / 位4 98%->70% / 位5 100%->30.43%
  平均 99.60% -> 57.89%（-41.71pp），所有位置 p=0.000
原因：原生头与 backbone 是端到端联合优化的；冻结特征后再换头，
      只是在已"过度专属"的特征空间里拟合 220 个样本，无法绕过该限制。
=> **重构必须端到端重训，不能只换头**。

### 交付物
- E:\\harness\\jAccount验证码识别-ResNet增强版.zip （56.1 KB，8 条目）
  含 README.txt / 安装说明.html / 脚本 / 验证码识别实测报告.html / 测试脚本/（新增）
- E:\\harness\\jAccount验证码识别-浏览器扩展版.zip （3.97 MB，10 条目）

### 校验结果（全部通过）
- analysis/test_retry.js：8/8 行为测试通过
- analysis/consistency_check.py：10/10 一致性通过（两版阈值/重试数/选择器全对齐，0.60 零残留）
- analysis/final_check.py：**26/26 交付前总校验通过**
- extension-src/verify_extension.js：220/220 与 Python 参考实现一致，97.7%
- node --check 两版语法均通过

### 本轮新增/修改文件
修改：jaccount-captcha-onnx-enhanced.user.js (72260->81543)、extension-src/app.js、
      extension-src/manifest.json
新增：analysis/test_retry.js、consistency_check.py、targeted_sampling.py、
      train_refactor.py、update_pkg_docs.py、final_check.py
产出：vm_dump/REFACTOR_PLAN.md、sampling_plan.json、backbone_feats.npz、refactor_eval.json

### 经验教训
- **改脚本前先核对原阈值是否真的生效**：0.60 这个值导致"低置信标注"功能形同虚设，
  单看代码完全发现不了，是靠实测数据（错误样本最低置信 84%~99.6%）才定位的。
- **"在冻结特征上换分类头"是常见误区**：端到端联合优化的头无法被离线重训复现，
  验证任何"换头"方案前必须先和原生头在同一划分上对比。
- 打包交付物前先跑总校验脚本（final_check.py），比逐个手工确认可靠。
"""

with open(p, 'a', encoding='utf-8') as f:
    f.write(note)
print('written', os.path.getsize(p))
