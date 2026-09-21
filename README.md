# jAccount 验证码自动识别（ResNet / ONNX 增强版）

上海交通大学 jAccount 登录页的验证码自动识别工具。本地推理，不联网、不上传任何数据。

**220 张真实验证码实测准确率 97.7%**（原始 Tesseract 方案为 74.2%）。

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
| 置信度偏低 | 输入框加橙色描边，提示你核对 |
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

### 4. 置信度不可用于筛选错误

这个模型的输出非常"自信"：错误样本的最低字符概率在 **84%~99.6%**，正确样本最低可到 **94.6%** —— 两者完全重叠，靠置信度阈值筛不出错例。真正兜底的是**长度校验**（不是 4/5 位就拒绝填入）。

### 5. 诊断信息不能遮挡被诊断的对象

失败横幅曾在 4.4.3 用过 `position: fixed; top: 0; z-index: 2147483647`，结果**把验证码图片本身盖住了**。

现在所有注入元素一律插在验证码图片**下方的文档流内**，样式里不含任何 `position`。没有定位就不可能覆盖 —— 这是结构性保证，不是靠调位置调出来的。

---

## 项目结构

```
.
├── jaccount-captcha-onnx-enhanced.user.js   # 用户脚本（主交付物，单文件）
├── 验证码识别实测报告.html                    # 实测报告（浏览器打开）
├── extension-src/                            # 扩展源码
│   ├── app.js                                #   识别逻辑
│   ├── manifest.json                         #   MV3 清单
│   ├── build.py                              #   构建：拼 ORT + 打包资源
│   └── verify_extension.js                   #   产物校验 + 220 张端到端
└── vm_dump/
    ├── nn_model.onnx                         # ResNet-20 模型（1.08 MB）
    ├── ground_truth_all.json                 # 人工逐张辨认的真值
    ├── ground_truth_holdout.json             # 独立留出集真值
    ├── final_120.json / holdout_eval.json    # 评测结果
    └── analysis/                             # 实验与测试脚本（见下）
```

### 测试脚本（`vm_dump/analysis/`）

按用途分组，都是有名字的自解释脚本：

**回归测试（改完代码必跑）**

| 脚本 | 验证内容 |
|---|---|
| `layout_check.js` | 注入元素不遮挡页面；ONNX 链路在 `exports` 污染环境下仍可用 |
| `repro_fail.js` | 完整链路 DOM 沙箱实跑（含 `document-start` 最恶劣时刻） |
| `overwrite_v2.js` | 自动填充覆盖逻辑 7 场景真值表（油猴版 + 扩展版） |
| `umd_probe.js` / `umd_e2e.js` | UMD 受控作用域方案：三环境对照 |
| `race_check.js` | 换图竞态 |
| `dom_sim.js` | 扩展版 DOM 沙箱 |

**准确率评测**

| 脚本 | 用途 |
|---|---|
| `web_verify.js` | 220 张端到端，与 Python 参考逐张比对 |
| `holdout_eval.py` | 独立留出集评测 |
| `score_120.py` / `score_resnet.py` | 分集打分 |
| `tess_*.js` / `tune_accuracy.py` | Tesseract 基线与消融 |
| `confidence_and_budget.py` | 置信度分布分析 |

**数据采集**

`collect_test.py` / `collect100.py` / `collect_holdout.py` —— 从登录页采集验证码样本。

---

## 复现实验

```bash
# 1) 依赖
python -m venv .venv && .venv/Scripts/pip install numpy onnxruntime pillow
node --version    # 测试脚本需要 Node 18+

# 2) 扩展版端到端（220 张，与 Python 参考比对）
python extension-src/build.py
node extension-src/verify_extension.js

# 3) 用户脚本回归
cd vm_dump/analysis
node layout_check.js
node repro_fail.js
node overwrite_v2.js
node umd_probe.js
```

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
