# jAccount 验证码自动识别（ResNet / ONNX 增强版）

上海交通大学 jAccount 登录页的验证码自动识别工具。本地推理，不联网、不上传任何数据。

**220 张人工标注实测准确率 97.7%**（原始 Tesseract 方案为 74.2%）。
当前版本：用户脚本 **4.5.2** / 浏览器扩展 **1.0.8**。

仓库同时保留了一条完整的**模型重构研究线**（第 4 位字符瓶颈攻坚）与一套**冻结的 520 张回归集**，
结论与「已验证无效的路径」都写在下面，避免后来者重复踩。

提供两种分发形式：

| 形式 | 需要什么 | 特点 |
|---|---|---|
| 用户脚本 | 暴力猴 / Tampermonkey | 安装最简单，模型按需下载并缓存 |
| 浏览器扩展 | Chrome / Edge（MV3） | 模型与推理引擎内置，**全程零网络请求** |

> 本项目仅供学习与个人便利使用。请遵守上海交通大学的相关使用规范，不要用于任何自动化批量登录、刷课、撞库等用途。

---

## 目录

- [这是什么 / 不是什么](#这是什么--不是什么)
- [快速开始](#快速开始)
- [准确率与实测方法](#准确率与实测方法)
- [实现要点](#实现要点)
- [项目结构](#项目结构)
- [复现实验](#复现实验)
- [重构研究：第 4 位瓶颈攻坚](#重构研究第-4-位瓶颈攻坚已收尾)
- [踩过的坑](#踩过的坑)
- [许可与致谢](#许可与致谢)

---

## 这是什么 / 不是什么

**是：** 一个把 jAccount 登录页上的验证码图片识别成字符串、并自动填入输入框的浏览器端工具。识别完全在本机完成。

**不是：** 不是一个"绕过登录"的工具。它不碰账号密码、不保存凭据、不模拟提交表单，只是替你把那 4~5 个字母敲进框里，登录按钮仍然要你自己点。

模型权重来自已有的开源项目（见[许可与致谢](#许可与致谢)），本项目的工作集中在**推理链路工程化**与**准确率验证**上 —— 原始项目里的 ONNX 推理代码实际上从未被调用过，实际生效的是一条 74.2% 的 Tesseract 路径。

---

## 快速开始

### 方式一：用户脚本

1. 浏览器安装 [暴力猴](https://violentmonkey.github.io/)（Violentmonkey）或 Tampermonkey
2. 新建脚本，把 `jaccount-captcha-onnx-enhanced.user.js` 的内容整个贴进去，保存
3. 打开 <https://jaccount.sjtu.edu.cn/jaccount/jalogin>

首次会下载推理引擎（约 0.57 MB）与模型（约 1.08 MB），之后走本地缓存。冷启动约 2.2 秒。

### 方式二：浏览器扩展

1. 构建扩展：

   ```bash
   python extension-src/build.py      # 产出 extension-build/jaccount-captcha-extension/
   ```

2. 打开 `edge://extensions`（Chrome 为 `chrome://extensions`），开启「开发人员模式」
3. 点「加载解压缩的扩展」，选择 `extension-build/jaccount-captcha-extension` 文件夹
4. 打开登录页即可

扩展版把模型和 wasm 都打进包里，通过 `chrome.runtime.getURL` 读成字节直接喂给 ONNX Runtime，**全程不发任何网络请求**，也就不存在 CDN 被墙的问题。

### 行为约定

| 场景 | 行为 |
|---|---|
| 验证码出现 | 自动识别并填入，焦点跳到用户名框 |
| 点「换一张」 | 重新识别新图，覆盖旧答案 |
| **你正在验证码框里打字** | **完全不动你的输入**（哪怕框里是空的） |
| 决策余量不足（模型对某一位没把握） | 自动点「换一张」重试（最多 3 次）；仍不足则加橙色描边提示核对 |
| 识别失败 | 在**验证码下方**显示具体失败原因 |

识别成功时页面上不会出现任何多余文字 —— 答案已经在输入框里了。诊断信息只在你需要行动时才出现，且一律留在文档流内，不会遮挡任何东西。

---

## 准确率与实测方法

| 引擎 / 配置 | 调参集 120 张 | 独立留出集 100 张 |
|---|---|---|
| Tesseract，原图直喂、每次新建 worker | 74.2% | — |
| Tesseract + 白名单 + PSM7 + worker 复用 | 74.2% | — |
| 原版 ONNX 后处理（前 26 类 + raw logit 启发式） | 95.8% | — |
| **本项目：ResNet ONNX，27 类 + blank 跳过** | **97.5%** | **98.0%** |

端到端复算（220 张，含预处理/推理/后处理全链路）：**215/220 = 97.7%**，与 Python 参考实现逐张比对 **220/220 完全一致**。差异只来自模型本身的误差，不是工程实现。

### 方法上做了三件事，避免自欺

1. **留出集独立采集**，不参与任何调参决策。本轮就有一个方案在调参集上"有效"、在留出集上归零 —— 没有留出集就会把它当成成果写进变更日志。
2. **真值人工逐张辨认**（`vm_dump/ground_truth_all.json`），不用任何模型的输出当真值，否则就是循环论证。
3. **报告用了哪个样本集**。20 张上看到的 "80% → 85%" 后来被 120 张推翻，实际是打平。小样本噪声是这类项目最常见的假成果来源。

准确率从 74.2% 到 97.5% 来自**真正启用 ONNX 路径**这一件事，McNemar 精确检验 p < 0.0001。其余改动（阈值修正、张量名动态获取、输出对齐）修的都是确定性缺陷，不是刷分 —— 这一点在变更日志里逐条标注了。

### 冻结的 520 张回归集

`vm_dump/analysis/_sampling_out/unified_gt_520.json` 把前后两批数据合并成一套可一键回归的评测集：

| 分段 | 张数 | 真值来源 | 原生模型整串准确率 |
|---|---|---|---|
| tune | 120 | 人工逐张辨认 | 97.50% |
| hold | 100 | 人工逐张辨认（独立采集，不参与调参） | 98.00% |
| new | 300 | 生产模型自标注 + 人工核对（修正 3 张） | 99.00% |
| **合计** | **520** | | **98.46%** |

⚠ **引用时请分开列。** `new` 段的真值来自该模型自己的标注，它上面的"准确率"是**自洽性**指标，
不是泛化准确率；只有 `tune` / `hold` 是人工标注，可作泛化参考。
`new` 段的正确用途是「改动有没有破坏原有行为」的回归检查。

```bash
python vm_dump/analysis/eval_unified.py            # 评估生产模型
python vm_dump/analysis/eval_unified.py xxx.onnx   # 评估任意模型
```

顺带一个反复被印证的观察：8 个误判样本**全部落在第 4 个字符位**，长度判定 520/520 全对。

---

## 实现要点

### 1. 受控作用域执行 UMD 库

这是本项目最有价值的一个坑，因为它**在干净沙箱里永远测不出来**。

ONNX Runtime Web 的 UMD 包装是四路分支：

```js
!function (e, t) {
    "object" == typeof exports && "object" == typeof module ? module.exports = t()
  : "function" == typeof define && define.amd            ? define([], t)
  : "object" == typeof exports                           ? exports.ort = t()
  :                                                        e.ort = t()   // ← 只有这里才挂到全局
}(self, ...)
```

**只要页面上存在 `define`（AMD 加载器）或 `exports`，`e.ort = t()` 这一行就永远不执行。** 表现是：`<script>` 下载成功、`onload` 正常触发、但 `window.ort` 永远是 `undefined`。

任何遗留的模块加载器、统计脚本、兼容 shim 都可能留下这些全局变量。而沙箱环境是干净的，永远走第 4 条分支成功 —— 于是 220 张端到端全过、用户却 100% 失败。

解决办法是不再把 `<script src>` 交给页面环境执行，改为**取回源码文本、在受控作用域里求值**，把模块系统标识符显式遮蔽成 `undefined`：

```js
const sandbox = { console };
sandbox.self = sandbox; sandbox.window = sandbox; sandbox.globalThis = sandbox;
const factory = new Function(
  'self', 'window', 'globalThis', 'define', 'exports', 'module', 'require',
  code + '\n;return (self && self["ort"]) || (window && window["ort"]) || null;'
);
const ort = factory(sandbox, sandbox, sandbox, undefined, undefined, undefined, undefined);
```

附带好处：也不再受页面 CSP 对 `<script src>` 的限制。取回对象后必须校验结构（`ort.env.wasm`、`ort.InferenceSession`），避免拿到被污染的半成品。

验证脚本：`vm_dump/analysis/umd_probe.js`、`umd_e2e.js` —— 在 `define` / `exports` / 干净三种环境下对照，旧做法在前两种下**均复现失败**，新做法三种全部成功。

### 2. 与官方 Python 实现严格对齐的预处理

官方 `ocr.py` 用 `Image.convert("L")` + `point([0]*156 + [1]*100, "1")`。两个容易被忽略的细节：

- 阈值是 `>= 156`，不是 `> 156`（差一个灰度级）
- **必须先 `Math.round` 再比阈值**。PIL 是整数运算 + 四舍五入，直接拿浮点比会在 220 张里产生 106 个像素的判定差异 —— 那样"我实测的准确率"和"你跑出来的准确率"就不是同一件事了

### 3. 4/5 位判定交给模型，不用启发式

第 5 个输出头是 **27 类**（26 字母 + 1 个 blank），前 4 个头是 26 类。原版对 5 个输出一律只读前 26 类，只能靠"第 5 位 raw logit < 10 或比值 < 0.6"猜是不是 4 位，会把 5 位码误截成 4 位。

本版按官方语义处理：argmax 覆盖全部类别，index ≥ 26 视为 blank 直接跳过。

### 4. 用「决策间隔」而不是置信度判断一张图能不能信

这个模型的输出非常"自信"：错误样本的最低字符概率在 **84%~99.6%**，正确样本最低可到 **94.6%** —— 两者完全重叠，靠概率阈值筛不出错例。

改用**决策间隔**（该位 top1 与 top2 的 logit 之差）后情况不同：它直接度量"离决策边界有多近"，且不受 logit 整体尺度影响。220 张实测（两个独立子集分别验证）：

| 判据 | 调参集 120 自动化率 / 误伤 | 留出集 100 | 220 张换图率 |
|---|---|---|---|
| min 概率 < 0.999 | 80.8% / 20 | 89.0% / 9 | 15.5% |
| **min 间隔 < 6** | **93.3% / 5** | **94.0% / 4** | **6.4%** |

两个子集都仍拦下各自全部错误。**另一面也要说清楚**：520 张整体是拦 **7/8**（漏掉 c269，间隔 6.78）—— 阈值提高到 7 能全拦，但调参集换图率会升到 20%（比旧判据还高，改善被吃光）。取 6 是明确的权衡：用"漏 1 个边缘错例"换"换图率减半"。

判据落地后做过一次独立复验：把 520 张的 logits 导出，喂给**从交付脚本里抽取的真实 `postprocess`**，与 numpy 参考逐张比对 —— 文字结果 0 差异、最小间隔最大差 4.77e-7（float32 舍入级），两版之间也逐张一致。

真正兜底的还有两件事：**长度校验**（不是 4/5 位就拒绝填入），以及"填进去的答案必须属于屏幕上那张图"的语义约束 —— 换图会让此前所有结果作废，不能跨图片比较置信度。

### 5. 诊断信息不能遮挡被诊断的对象

失败横幅曾在 4.4.3 用过 `position: fixed; top: 0; z-index: 2147483647`，结果**把验证码图片本身盖住了**。

现在所有注入元素一律插在验证码图片**下方的文档流内**，样式里不含任何 `position`。没有定位就不可能覆盖 —— 这是结构性保证，不是靠调位置调出来的。

---

## 项目结构

```
.
├── jaccount-captcha-onnx-enhanced.user.js   # 用户脚本（主交付物，单文件）
├── CHANGELOG.md                              # 逐版本变更记录（含每处改动的实测依据）
├── 验证码识别实测报告.html                    # 实测报告（浏览器打开）
├── 项目报告.html                              # 项目全貌：架构 / 数据 / 测试体系 / 17 版时间线
├── 复审报告-2026-09-22.md                     # 对研究线结论的独立复验与修正记录
├── extension-src/                            # 扩展源码
│   ├── app.js                                #   识别逻辑
│   ├── manifest.json                         #   MV3 清单
│   ├── build.py                              #   构建：拼 ORT + 打包资源
│   └── verify_extension.js                   #   产物校验 + 220 张端到端
└── vm_dump/
    ├── nn_model.onnx                         # ResNet-20 模型（1.08 MB）—— 生产模型
    ├── ground_truth_all.json                 # 220 张人工逐张辨认的真值
    ├── MODEL_CEILING_REPORT.md               # 模型优化空间评估（97.7% 已近判别极限）
    ├── ARCH_BREAKTHROUGH_REPORT.md           # 架构层面突破评估（5 个假设否证 4 个）
    ├── REFACTOR_PLAN.md                      # 重构方案与定向采样计划
    ├── _E2E_FINETUNE_PILOT_REPORT.md         # 端到端微调试点报告
    ├── _E2E_FINETUNE_ROUND2_ARCHIVE.md       # 端到端重训终局归档（含结论修订）
    ├── samples/ samples2/ holdout/            # 220 张人工标注样本（评测必需）
    ├── new300/                               # 300 张已核对样本（冻结回归集的一部分）
    └── analysis/                             # 实验、训练与测试脚本（见下）
```

> 预处理与 Tesseract 消融的对照图（`all_*/hold_*/tess_*/ablation`，共 580 张）**刻意不入库** ——
> 它们可由 `analysis/` 里的脚本重建，且评测不需要它们。

### 测试脚本（`vm_dump/analysis/`）

**一键全量检验（改完任何东西先跑这个）**

```bash
python vm_dump/analysis/_final_full_check.py
```

它把下面所有套件跑一遍，再附上语法检查与「工作区 vs 交付包 md5 核对」，最后给一句总判定。

**行为回归（都从真实源码切片，不手抄副本）**

| 脚本 | 验证内容 |
|---|---|
| `bug_hunt_retry.js` | 15 项：换图控件探测 / 换图等待语义 / 配置健全性 |
| `bug_hunt_flow.js` | 16 项：把**真实 `recognize()` 整段**跑起来（失败路径 / 重试竞态 / title） |
| `bug_hunt_v2.js` | 9 项：换图后结果归属、异步换图、`AbortError`、闸门误伤 |
| `race_retry_observer.js` | 7 项：换图自激环路收敛 + 闸门不误伤手动换图 |
| `test_retry.js` | 9 项：重试循环（含阈值边界 5.999 / 6.000 / 6.001） |
| `layout_check.js` / `repro_fail.js` / `overwrite_v2.js` / `dom_sim.js` | 注入不遮挡、完整链路沙箱、覆盖逻辑真值表、扩展版 DOM 沙箱 |
| `umd_probe.js` / `umd_e2e.js` | UMD 受控作用域方案的三环境对照 |

**交付一致性**

| 脚本 | 用途 |
|---|---|
| `consistency_check.py` | 25 项：两版版本号 / 参数 / 判据完全同构 + 防回归锁 |
| `final_check.py` | 30 项：交付包条目、版本、关键符号、资源完整性 |
| `_verify_margin_impl.js` | 把 520 张 logits 喂给**真实 `postprocess`**，与 numpy 参考逐张比对 |

**准确率评测**

| 脚本 | 用途 |
|---|---|
| `eval_unified.py` | **520 张统一回归集**一键评估（推荐） |
| `web_verify.js` | 油猴版真实 onnxruntime-web 路径 220 张端到端 |
| `holdout_eval.py` / `score_120.py` / `score_resnet.py` | 分集打分 |
| `tess_*.js` / `tune_accuracy.py` | Tesseract 基线与消融 |
| `confidence_and_budget.py` / `_audit_margin*.py` | 置信度与决策间隔的分布分析 |

**模型重构研究线**（NumPy 实现，不依赖 PyTorch）

| 脚本 | 用途 |
|---|---|
| `resnet20_np.py` | 纯 NumPy 的前向 / 反向（im2col 向量化），与 onnxruntime 逐层对齐 |
| `e2e_backward_check.py` | 数值梯度校验（657/657 通过） |
| `train_e2e.py` | 端到端微调：`train` / `eval` / `baseline` / `gen` |
| `_backport_onnx.py` / `_eval_backport_onnx.py` | 把微调权重回写进 ONNX 并用 onnxruntime 独立复验 |
| `_autolabel.py` / `_apply_labels.py` / `_qa_*.py` | 自动标注 + 人工核对流程 |
| `build_unified_gt.py` | 合并两批真值，生成 520 张统一回归集 |
| `_aug_mine.py` / `_verify_aug.py` | 定向难例增广与机制验证 |

**数据采集**：`collect_test.py` / `collect100.py` / `collect_holdout.py` / `_grab_captcha.py`。

---

## 复现实验

```bash
# 1) 依赖
python -m venv .venv && .venv/Scripts/pip install numpy onnxruntime pillow
node --version    # 测试脚本需要 Node 18+

# 2) 一把跑完（推荐）：10 个套件 + 语法检查 + 交付物 md5 核对
python vm_dump/analysis/_final_full_check.py

# 3) 520 张统一回归集评估
python vm_dump/analysis/eval_unified.py

# 4) 扩展版构建与端到端
python extension-src/build.py
node extension-src/verify_extension.js
```

### 路径与环境变量

所有脚本都用「环境变量优先、否则按文件位置推导」的方式定位仓库 —— clone 到任何目录都能直接跑，
不需要改任何硬编码路径。少数依赖外部工具的地方可用环境变量覆盖：

| 变量 | 用途 | 不设时的默认 |
|---|---|---|
| `PROD_WS` | 仓库根目录 | 按脚本位置向上推导 |
| `NODE_BIN` | node 可执行文件 | `node`（走 PATH） |
| `PY_BIN` | python 可执行文件 | 当前解释器 |
| `ORT_NODE_WORKSPACE` | 含 `onnxruntime-web` 的 node_modules | `<仓库根>/node_modules` |

`web_verify.js` 需要 `onnxruntime-web` 的 dist 目录（见下），所以通常要设 `ORT_NODE_WORKSPACE`。

复现提示：`vm_dump/ort.min.js` 与扩展构建产物里的 `ort-wasm-simd.wasm` 未入库（体积原因），需要时从 npm 取得：

```bash
npm pack onnxruntime-web@1.16.3
# ort.min.js           -> dist/ort.min.js           -> vm_dump/
# ort-wasm-simd.wasm   -> dist/ort-wasm-simd.wasm   -> extension-build/.../assets/
```

若要做逐行比对，还需要上游参考实现（本项目未收录，见[许可与致谢](#许可与致谢)），放到 `vm_dump/ref/`：

```
vm_dump/ref/PhotonQuantum_ocr.py          <- jaccount-captcha-solver/ocr.py
vm_dump/ref/PhotonQuantum_ocr_legacy.py   <- 同上（SVM 版）
vm_dump/ref/PhotonQuantum_utils.py        <- 同上
vm_dump/ref/danyang685_ocr.py             <- Tesseract 版脚本附带的 ocr.py
```

这些文件在仓库中**不存在**，是刻意排除的 —— 它们是他人作品、且分发时未附带许可声明，原样收录进本仓库不合适。真正需要比对再自行下载即可。

---

## 重构研究：第 4 位瓶颈攻坚（已收尾）

97.7% 的 5 个错例**全部落在第 4 个字符位**，因此针对性地做过一轮完整攻坚。

**结论：98% 是该模型 + 该数据的泛化天花板，训练侧无法在保留评估可信度的前提下突破。**

| 尝试 | 结果 |
|---|---|
| 冻结特征上换分类头（加宽 / 加深） | **−41.71 pp**（99.60% → 57.89%）。原生头与 backbone 是端到端联合优化的，离线在冻结特征上重训无法复现 |
| 端到端微调（lr=1e-4，激进） | 整串退化到 79% —— 第一步梯度就把权重推出良好区域 |
| 端到端微调（lr=3e-6，温和，300 步） | 训练集位4 98.57% → 99.76%，但**留出集全程 98.00%，未突破** |
| 定向增广难例（往 w→g / g→z 的边界推） | 75 个变体里 3 个命中 w→g、g→z 一个未命中；叠加训练仍不修正这两例 |
| 回写 ONNX 独立复验（用**最终权重**） | 520 张里 7 张预测改变（6 修正 / 1 退化），**全部落在训练集内**；留出集 0 变化 |

**为什么突破不了**：`h030`(w→g)、`h040`(g→z) 是该字体里极小概率的边缘字形 —— 在 520 张里各只出现 1 次，
训练集中不存在同类难例，随机扰动也复制不出这两个具体字形。靠采集碰运气命中同类字形，
需要约 1 万张量级，性价比极低。

**保留的资产**：300 张已核对样本、全套 NumPy 训练 / 回写 / 校验脚本、两份研究报告、统一 520 张回归集。

> ⚠ 研究线报告里有一处**"自我更正"本身是错的**：它把"微调确实修正了训练集难例"改成了"一处未修正"，
> 根因是回写验证用了改动最小的**早停权重**（step 25，改动量只有最终权重的 1/10）。
> 该处已在 `复审报告-2026-09-22.md` 与 `_E2E_FINETUNE_ROUND2_ARCHIVE.md` §4 修正。
> 保留这段历史是因为教训值钱：**用哪一版权重做验证，本身就是结论的一部分。**

---

## 踩过的坑

按教训价值排序，都是真实发生过、且都不是"读代码能看出来"的那类问题。

### 沙箱全绿而真实环境失败时，去列举环境差异，而不是重读逻辑

这个项目连续 5 轮修复无效，因为**所有测试都在某一个维度上是盲的**。当时逐项列举出的差异维度：

| 维度 | 沙箱默认 | 真实页面 |
|---|---|---|
| 存在哪些全局变量 | 无 | 站点加载什么就有什么 |
| 注入时 DOM 就绪度 | 按需构造 | `document-start`，`head` 可能是 `null` |
| 元素状态 | `naturalWidth` 硬编码 | 图片可能未解码 / 尺寸为 0 |
| 脚本来源 | 桩里直接 `onload` | 缓存、CSP、追踪拦截器 |
| 运行时身份 | Node | 浏览器（无 `__dirname`、无 `process`） |

第 1 行就是那个 5 轮才找到的 bug。**为每个维度单独构造测试**，是这一步唯一有效的做法。

### 会撒谎的测试桩比没有测试更糟

两处桩曾经给出**看似可信的错误结论**，各浪费一轮：

1. **样式桩不同步** —— 桩里 `style` 是普通对象，脚本设了 `display:none` 后测试读 `cssText` 仍为空，把"已折叠"误判成"仍占位"（假失败）。
2. **网络桩返回全零 buffer** —— 所有 URL 一律给 1 MB 的 `\0`，UMD 源码被 `new Function` 执行必然抛 `Invalid or unexpected token`，表现为"ONNX 路径全挂"（假失败）。

假失败浪费一轮，**假成功会把 bug 直接发出去**。现在网络桩按 URL 分支返回正确形状的数据；样式桩用 `Object.defineProperty` 让具名属性与 `cssText` 双向同步。

### 重构时确认每一行为什么存在，再删

`loadScript` 重写时漏掉了 `s.src = url;` —— `<script>` 没有地址、永不加载，表现为"所有源都超时"。这类缺陷在 diff 里极其显眼（少一行赋值），但当时没人重新读一遍完整 diff。

### 打包脚本必须从空目录开始

`copytree` 到子目录而不清顶层，导致上一版遗留的散落文件与新文件同时留在包里 —— 用户看到两套同名文件，很可能加载到**旧的那一套**，压缩包体积也白白翻倍（7.92 MB → 3.96 MB）。

### 批量改写代码时，「检查是否已处理」要检查定义而不是出现

把硬编码路径换成推导式时，注入辅助变量的判断写成了 `'_REPO' not in src`。但替换后的文本里**已经出现** `_REPO`（作为表达式的一部分），于是被误判成"已注入" —— 15 个文件带着未定义变量通过了 `ast.parse`（语法检查不做名字解析），运行时才 `NameError`。

**必须检查"定义"（`'_REPO = '`），而不是"出现"。** 顺带记住：语法检查通过 ≠ 能跑，批量改写后至少要实跑一个样本。

### 校验用的临时文件：要有正确扩展名，且别落在被遍历的目录里

同一轮里还踩了两个小的：

1. 把待校验的 JS 写成 `foo.js.check`，`node --check` 会走 ESM 加载并报 `get_format` 错误 —— 于是**所有文件都被判成"语法有问题"**而没有写回，看上去像"26 个文件全部失败"，实际一个都没改。
2. 临时文件写在被 `os.listdir` 遍历的目录里，下一轮跑时快照把上一轮残留的 `xxx.js.tmp.js` 当成待处理文件，直接 `FileNotFoundError` 崩掉。

现在：临时文件一律写到系统临时目录，并显式跳过任何含 `.tmp.` 的文件名。

### 判据换代时，改桥接点而不是改 20 多处测试桩

把"低置信"判据从 softmax 概率换成 logit 间隔后，`bug_hunt_flow.js` / `bug_hunt_v2.js` 里的桩全是按置信度写的，逐个改要动 20 多处，改错一处就会让用例悄悄失效。

实际上 `recognizeOnce` 是"桩 → 被测代码"的**唯一桥接点**，在那里加一层映射就够了：

```js
const marginFor = (conf) => (conf < 0.999 ? 3 : 12);   // 让旧用例语义保持不变
const recognizeOnce = async (...a) => {
    const r = await recognizeOnceImpl(...a);
    if (r && r.minMargin === undefined) r.minMargin = marginFor(r.minConfidence);
    return r;
};
```

改判据前先问一句：**这个变化能不能在"边界"处一次性吸收掉**，而不是让它渗透到每个用例里。

---

## 许可与致谢

本项目的模型权重与原始思路来自以下项目，**不是本项目原创**：

| 来源 | 内容 | 许可 |
|---|---|---|
| [PhotonQuantum/jaccount-captcha-solver](https://github.com/PhotonQuantum/jaccount-captcha-solver) | ResNet-20 模型结构、`ocr.py` 官方实现 | Apache-2.0 |
| [RyanStarFox/JAccountVerificationCode](https://github.com/RyanStarFox/JAccountVerificationCode) | `nn_model.onnx` 分发、ONNX 推理代码 | — |
| [danyang685](https://github.com/danyang685) 的 Tesseract 版用户脚本 | 原始用户脚本骨架 | MIT |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | ONNX Runtime Web v1.16.3 | MIT |

本仓库对上述内容的改造部分同样以 MIT 许可发布，详见 [LICENSE](LICENSE)。

---

## 免责声明

本项目为学习性质的技术实践，用于个人登录便利。请遵守上海交通大学的信息系统使用规定。作者不对任何滥用行为负责。

识别在本机完成，验证码图片与账号密码都不会离开你的设备。
