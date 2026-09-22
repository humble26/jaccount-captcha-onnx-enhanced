# 变更记录（Changelog）

本项目有两个并行的发行形态，逻辑保持同构（由 `consistency_check.py` 强制校验）：

- **油猴脚本** `jaccount-captcha-onnx-enhanced.user.js` —— 版本号 `4.x.x`
- **浏览器扩展（MV3）** `extension-src/` —— 版本号 `1.0.x`

所有条目均来自实跑验证，未验证的推测已单独标注。

---

## [4.5.2] / 扩展 [1.0.8] — 2026-09-22 · 换图判据改为「logit 决策间隔」

> 依据：220 张标注样本独立复测，并在两个独立子集上分别验证。
> **不是**靠调参刷出来的改动 —— 它在同样拦下全部错误的前提下，把换图率（用户体感：脚本自动换图的频率）砍掉一半以上。

### 变更

- **判据从「最小 softmax 置信 < 0.999」改为「最小 logit 决策间隔 < 6」**。
  「决策间隔」= 该位 top1 与 top2 的 logit 之差，比概率更直接地度量"离决策边界有多近"，
  且不受 logit 整体尺度影响（softmax 概率存在温度效应）。
- `postprocess()` 一次扫描同时取 top1 / top2，新增返回 `minMargin`（`margins` 数组一并返回）。
- `CFG` 新增 `lowMargin: 6`；`lowConfidence: 0.999` 保留，但**只用于显示与日志**，不再触发换图。
- `isLow` 改用 `minMargin`。Tesseract 路径返回 `minMargin: Infinity`，即不参与换图重试
  —— 它的置信度是引擎自评，与 ResNet 的 logit 间隔口径不可比。
- 提示文案随之更新：「置信度偏低（xx%）」→「决策余量不足（间隔 x.xx）」。

### 实测（220 张标注样本，两个子集分别验证）

| 判据 | 调参集 120 自动化率 / 误伤 | 留出集 100 自动化率 / 误伤 | 220 张换图率 | 300 张新样本换图率 |
|---|---|---|---|---|
| 旧：min 置信 < 0.999 | 80.8% / 20 | 89.0% / 9 | 15.5% | 14.0% |
| **新：min 间隔 < 6** | **93.3% / 5** | **94.0% / 4** | **6.4%** | **7.0%** |

两个子集都仍拦下各自的全部错误（调参集 3/3、留出集 2/2）。「误伤」= 把正确答案也换掉、白换一轮的张数。

### 注意

- 阈值 6 与实测最大错误间隔（5.799）只差 0.2，**余量薄**：它是在 5 个错误样本上标定的，
  应随新数据重新标定。两个独立子集都通过，是它当前可信度的主要来源。
- 加置信度兜底（`margin < 6 或 conf < 0.99`）实测与原方案**完全相同** —— 两个信号高度相关
  （rank 相关系数 0.979），兜底不增加任何拦截，故未采用，以免长期维护两个阈值。
- **520 张全量实测的召回是 7/8，不是"全拦"**（如实记录，避免日后被误引用）：
  阈值 6 漏掉 c269（margin 6.78，第 4 位 m→b；该样本真值由人工修正而来）。
  用**真实 postprocess** 实跑的阈值扫描：

  | 阈值 | tune(120) 换图率 / 拦错 | hold(100) | new(300) | ALL(520) |
  |---|---|---|---|---|
  | 5 | 3.3% / 2/3 | 3.0% / 1/2 | 5.7% / 2/3 | 4.6% / 5/8 |
  | **6（选用）** | **6.7% / 3/3** | **6.0% / 2/2** | 7.0% / 2/3 | **6.7% / 7/8** |
  | 7 | 20.0% / 3/3 | 10.0% / 2/2 | 12.7% / 3/3 | 13.8% / 8/8 |

  阈值 7 能全拦，但 tune 段换图率会飙到 20%（比旧判据的 15.5% 还高，改善被吃光）。
  取 6 是明确的权衡：**用"漏掉 1 个边缘错例"换"换图率减半"**。
  这也再次印证那条老结论 —— 模型"自信地错"的样本（c269 最低置信 0.9987）无法只靠阈值筛出。
- 判据落地后做过独立复验：把 520 张的 logits 导出，喂给**从交付脚本抽取的真实 `postprocess`**，
  与 numpy 参考逐张对比 —— text 0 差异、minMargin 最大差 4.77e-7（float32 舍入级），
  两版 `postprocess` 之间也逐张完全一致。即"选阈值时用的 numpy margin"与"线上跑出的 JS margin"
  是同一个量。

### 测试

- 新增两版一致性锁（`consistency_check.py` 22 → **25** 项）：必须存在 `lowMargin`、
  `postprocess` 必须返回 `minMargin`、`isLow` 必须用间隔且不得残留 conf 判据。
- `test_retry.js` 重写为按间隔书写（含 `minMargin = Infinity` 的 Tesseract 用例）。
- `bug_hunt_flow.js` / `bug_hunt_v2.js` 在桥接处加了「置信度 → 间隔」映射，
  使既有用例语义自动保持不变（`conf < 0.999` → `margin 3`），无需逐个改桩。
- 全量回归：15/15、16/16、9/9、7/7、8/8、25/25、220/220（97.7%），`node --check` 两版通过。

---

## [4.5.1] / 扩展 [1.0.7] — 2026-09-21 · 复审：修复 6 处语义错误

> 背景：上一版"15/15、16/16、7/7 全绿"，但修的是"能不能跑"（不再干等 1200 ms），
> 却引入了"跑得快但跑错"（在旧图上识别）。本轮不采信既有结论，重读代码 + 新写复现测试，
> **先让测试失败证明 bug 存在，再改代码**（两版修复前均为 2/9）。

### 修复

- **A（最严重）填入的答案不属于屏幕上的图**
  `recognize()` 保留历史最高置信 `best` 作为重试用尽的兜底。但换图后屏幕上已经不是那张图了——
  置信度的语义是"这一张能不能信"，不是跨图片比较的分数。
  日志铁证：`重试 3 结果 "abv3z" | 重试用尽，改用置信更高的那次结果："abv1z" (95%) | 填入 "abv1z"`
  ——屏幕是 v3 却填 v1，必然登录失败。
  **改为删除 `best` 回退，重试用尽即采用最后一次（对应屏幕当前那张图）。**

- **B 换图后新图识别失败仍填旧图答案**
  catch 与长度异常分支原本 `break` 后继续使用 `res`（旧图结果）。已换图 → 旧结果作废。
  改为 `res = null` + `showDiag(MSG_FAILED, '换图后未能重新识别，请手动输入')` + return。

- **C 异步换图站点：等待在旧图上就兑现**
  `img.decode()` 在"src 未变且旧图已解码"时会立即兑现。站点若由 ajax 延迟取新图，
  refreshCaptcha 会 0 ms 返回，随后对旧图重复识别 → 三次换图全白换。
  实测 `{"srcAtResolve":"A","ms":0}`。
  **改为以"src 真的变了"为解码前置条件；未变则 30 ms 轮询等待，总超时 1500 ms。**

- **D `decode()` 的 `AbortError` 被当成"图片已就绪"**
  原 `.catch(finish)` 把它当完成。实际上 AbortError 的含义是"本次解码被更新的 src 取消"
  ＝新图还在路上，此时放行等于在没加载完的图上开跑。实测 `ms=0, calls=1`。
  **改为 catch 内 `setTimeout(decodeNow, 25)` 重试，上限 20 次。**

- **E 换图闸门误伤用户手动换图**
  闸门是 2000 ms 时间窗，重试循环往往几百 ms 就结束，剩下的一秒多里用户点"换一张"会被静默忽略。
  **新增 `releaseObserver()`，在重试循环的 `finally` 中调用（仅当本轮真的换过图）。**
  配套新增 **M2 反向用例**：重试进行中闸门必须仍然有效——防止该修复把无限换图循环放出来。

- **F（逻辑审查发现，非测试发现）换图未生效时仍空转**
  换图后 src 始终未变时原本仍返回 `true` → 重试循环在同一张旧图上空转 3 × 1.5 s
  （浏览器不会为同一 URL 重新加载，再点也没用）。**改为返回 `false`，`recognize()` 直接放弃重试。**

### 测试

- 新增 `analysis/bug_hunt_v2.js`（J/K/L/M 四组，9 项），被测代码全部按大括号配平从真实源码抽取。
- **修正 6 条固化了错误预期的既有测试**（flow E4、F2/F3；retry B4/B6/B7；test_retry 用例 3/5）
  及 3 处从不真改 src 的测试桩。
- `consistency_check.py` 17 → **22 项**，新增两条防回归锁：
  禁 `let best = res` / `best.minConfidence`；要求 `refreshCaptcha` 同时含 `oldSrc` 与 `decodeNow`。
- `final_check.py` 去掉第三次出现的硬编码版本号（改为从源文件读取互比）。

### 校验

| 检查 | 结果 |
|---|---|
| `bug_hunt_retry.js` | 15/15 |
| `bug_hunt_flow.js` | 16/16 |
| `bug_hunt_v2.js`（新） | 9/9 |
| `race_retry_observer.js` | 7/7 |
| `test_retry.js` | 8/8 |
| `consistency_check.py` | 22/22 |
| `final_check.py` | 26/26 |
| `web_verify.js`（真实 ORT + 220 样本） | 220/220，与 Python 参考一致（logits 最大绝对差 1.001e-5） |
| `node --check` | 两版通过 |

---

## [4.5.0] / 扩展 [1.0.5→1.0.6] — 2026-09-21 · 现成解落地与严格审查

### 4.5.0 新功能：低置信自动换图重试

- `lowConfidence` 由 `0.60` 提升至 **`0.999`**。0.60 是**从未生效的死阈值**——
  实测 5 个错误样本最低置信在 84.39%~99.58% 之间，永远够不到 0.60。
  阈值选型依据（220 张实测）：99.9% → 自动化率 84.5%、自动部分准确率 **100%**、拦截 5/5 错误。
- 新增 `findRefreshButton()`：9 个候选选择器兜底定位换图控件（抗改版，要求可见且可点）。
- 新增 `refreshCaptcha()`：点击后等待新图就绪，优先 `img.decode()`，超时兜底。
- `recognize()` 重构：抽出 `recognizeOnce()`，主流程加入重试循环（`maxRefreshRetries: 3`）。

### 4.5.1（原 4.5.0 修订版，第十五轮）修掉的 4 个真实缺陷

1. **版本号不一致**：脚本头 `@version 4.5.0`，内部 `const VERSION` 还停在 `4.4.4` → 状态条上报旧版本。
2. **`refreshCaptcha()` 里 `img.decode()` 是死代码**：`oldSrc` 判断导致必然提前 return，每次换图固定白等 1200 ms。
3. **无限换图循环（最严重）**：环路 `recognize → refresh → src 变更 → MutationObserver → schedule → runFor → recognize`。
   既有两道护栏（`runToken` 每圈自新、80 ms 防抖每圈时间错开）都失效，实测换图次数
   `[33, 66, 99, 132, 165, 198]` 线性放大、永不收敛。
   → 新增换图闸门 `suppressObserver(2000)`，**必须在 `btn.click()` 之前调用**（站点同步改 src 时，放后面等于白设）。
   修复后收敛到 ≤3 次。
4. **`markInput` 无条件写 `title`**：把 `showDiag` 写入的诊断信息抹掉。改为仅低置信时写。

另修 5 个测试/工具自身缺陷（其中 2 个曾造成假通过：选择器正则非贪婪只比前 5 个、
`markInput` 是手抄的过时副本、`img.decode` 桩缺 setter 导致 decodeCalled 恒为 0）。

---

## [4.4.5] / 扩展 [1.0.4] — 2026-09-21 · 布局遮挡修复

- **失败横幅不再遮挡验证码图片**：4.4.3 的 `position: fixed; top: 0; z-index: 2147483647`
  永久占据页面顶部，把验证码图片本身盖住——排障信息挡住了待排查的对象。
  横幅与状态条一律改为插入验证码图片**下方的文档流内**（`insertBefore(el, img.nextSibling)`），
  样式中不含任何 position，默认 `display:none`。
- 新增 `hideBanner()`：识别成功时收起（此前一旦出现就永远挂着）。
- 成功路径不再常驻"识别成功"文字（答案已在输入框里，再挂一行是噪音）。
- `ensureStatus()` / `ensureBanner()` 找不到验证码图片时退回 `body`（document-start 阶段 DOM 未就绪）。
- 扩展版新增同款文档流内状态条（此前完全不注入 DOM，本就不遮挡，为排障一致性补齐）。
- 打包脚本 `package.py`：先清空顶层再重建（此前旧版本散落文件与新目录并存，体积翻倍 7.92 → 3.96 MB）；
  补生成扩展版专用安装说明。
- 修两处测试桩缺陷：cssText 与 display 不同步、网络桩对所有 URL 返回全零 ArrayBuffer。

---

## [4.4.4] / 扩展 [1.0.3] — 2026-09-21 · ★ 修复真正的根因：UMD 全局被页面污染

- 现象：`[加载 ORT] 脚本已加载但 window.ort 未出现`（由 4.4.3 的横幅首次捕获）。
- 根因：ORT 的 UMD 包装四路分支，
  `"function" == typeof define && define.amd` → `define([], t)`；
  `"object" == typeof exports` → `exports.ort = t()`；
  只有两者都不存在才走 `e.ort = t()`。
  **jAccount 登录页上存在 `define`/`exports` 类全局变量，于是 ort 永远挂不到 window 上。**
- 修复：`fetchText()` 下载 ORT 源码文本 → `evalUmdInControlledScope()` 用 `new Function`
  把 `define/exports/module/require` 显式声明为 undefined 形参遮蔽掉，强制走第四分支，再 `return self.ort`。
  附带好处：不再受页面 CSP 对 `<script src>` 的限制。Tesseract 同为 UMD，一并改造。
- `loadScript()` **保留**并标注"只用于非 UMD 脚本"——吸取 4.4.0 删除仍在用的代码的教训。
- 扩展版的 ORT 内联在 content.js、在隔离世界执行（通常无 define/exports），受影响概率低，
  但 `initOrt()` 也加了兜底检查，把原因说清楚而不是静默失败。

---

## [4.4.3] — 2026-09-21 · 让失败无法被忽略

- 页面顶部红底横幅：一次性摊开「脚本版本号 + 失败环节 + 完整错误」，带关闭按钮。
  横幅文案自带"脚本已加载"这一信息——可区分"脚本没跑"与"脚本跑了但失败"，
  这正是此前四轮最分不清的一件事。
- 预热失败立即弹横幅（预热失败＝之后每次识别必然失败）。
- "等了 10 秒没找到验证码"弹横幅，并区分两种成因：输入框在但图片选择器没匹配（页面改版）／两者都没有。
- 状态条永远带版本号，用户一眼能确认装的是哪一版。
- ⚠ 本版的 fixed 定位横幅导致遮挡问题，已在 4.4.5 修正。

### 本轮实测排除的嫌疑（避免重复劳动）

| 嫌疑 | 结果 |
|---|---|
| jAccount 页面 CSP 拦 `<script>` | 不成立，响应头只有 `upgrade-insecure-requests` |
| CDN 不可达 | 不成立，四个资源全部 200 + MIME 正确，1 s 内完成 |
| ORT 1.16.3 缺 glue 文件 | 不成立，jsdelivr 同样没有该文件，说明 ORT 不依赖它 |
| Node 跑 ort.min.js 报 `b.normalize is not a function` | Node 缺 fake-path 的环境问题，浏览器不出现，不能当证据 |

---

## [4.4.2] / 扩展 [1.0.2] — 2026-09-21 · 让失败可被看见

- placeholder 带具体原因（`识别失败… [ONNX: [推理] no available backend found ｜ Tess: Failed to fetch …]`），
  完整文本另存 `title`（悬停可见）。
- ONNX 路径拆五步标注：加载 ORT / 加载模型 / 预处理 / 推理 / 后处理；
  异常统一 `new Error('[' + step + '] ' + brief(e))` 再抛。
- 新增页面状态条（`CFG.showStatus`，验证码下方）：成功显示引擎与最低置信，失败显示具体原因。
- 补齐所有"静默 return"：找不到输入框 / 图片尺寸为 0 / 10 秒没发现验证码 / 预热失败。
- 补 `runFor` 里 `recognize()` 的 catch（原为无人认领的 rejected Promise，表现为"点了换图什么都没发生"）。
- `clearOwnPlaceholder` 改前缀匹配（否则带原因的新文案永远清不掉）。
- `sync_pack.py` 改为自动从脚本头读 `@version`。

---

## [4.4.1] — 2026-09-21 · 事故修复：漏了一行 `s.src = url;`

- 4.4.0 重写 `loadScript` 时漏掉赋 src → 浏览器根本不去加载 → onload 永不触发 →
  每个源走 15 s 超时 → 全部失败 → 回退 Tesseract（国内拉不到）→ 双双失败。
- 补回该行，位置在事件挂载**之后**（先赋 src，脚本命中强缓存时可能在同一个微任务内加载完成，
  事件派发时 onload 还是 null，会被永久错过）。
- `loadScript` 加超时（默认 15 s）：真实网络里 onload/onerror 可能一个都不来。
- `loadOrt` 不再只凭"脚本 onload 了"就认定 `window.ort` 存在，改为回读全局 + 换源重试 + 抛出真实原因。

---

## [4.4.0] / 扩展 [1.0.1] — 2026-09-21 · 8 个缺陷修复

- **A 覆盖逻辑副作用（最严重，两版都有）**：4.3.0 的修复导致用户 Ctrl+A 清空准备自己重打时，
  自动填的值立刻冲掉刚清空的输入位。改为**只要焦点在验证码框内就一律不动**（空值也是"我要自己输"的信号）。
  用 7 场景真值表验证（`overwrite_v2.js`）。
- **B wasm 回退链路断点（油猴版）**：`ortWasmPath` 写死第一个源，换源后 ORT 自取 10.4 MB wasm 时又撞回原源。
  改为从实际加载成功的脚本 URL 推导（`loadScript` 现在返回命中的 URL），与 ORT 同源。
- **C `loadScript` 可能永久挂起**：document-start 下 `appendChild` 抛 TypeError，
  异常跑在回调里不被外层 Promise 捕获 → Promise 永久 pending → ONNX 路径静默失效且无任何日志。
  修复：回调内自兜异常 + 25 ms 短轮询（最多 1 s）+ 拿不到宿主节点则换下一个源。
- **D 扩展版 CDN 兜底在 MV3 下不可用**：content script 的 fetch 受扩展 CSP 约束
  （默认 `connect-src 'self'`），跨域取 jsdelivr 必被拒，且会用一条 CSP 报错掩盖真正原因。
  修复：删掉假兜底，`loadAsset()` 只读扩展内资源，错误信息提示"点重新加载"。

---

## [4.3.0] — 2026-09-21 · 可运行性验证后

- **灰度取整**：PIL `convert("L")` 是整数四舍五入，而 JS 直接拿浮点比 `>= 156`，
  220 张 968,000 像素里有 **106 个像素判定不同**。改 `Math.round` 后为 **0 个**。
- 覆盖逻辑：改为只在"用户此刻正在该框内打字"时跳过填充（原判断会在用户改过验证码框后再点换图时丢弃新答案）。
- 预热加条件，避免在 jaccount 非登录页白下载 12 MB。
- API 逐个核对（对着 onnxruntime-web@1.16.3 的 bundle 反查）：
  `inputNames`/`outputNames` 是 `InferenceSession` 上的 getter；
  `wasmPaths`/`numThreads`/`simd` 均存在；`wasmPaths` 会被拼接文件名，必须以 `/` 结尾。

---

## [4.2.0] — 2026-09-21 · 冷启动提速

- `@run-at` 由 `document-end` 改为 **`document-start`**（下载与页面解析并行）。
- ORT 运行时与模型改为 `Promise.all` 并行下载（原串行约 0.6 s 变为重叠）。
- `loadScript` 兼容 document-start 下 `<head>` 尚不存在（退化挂 `documentElement`）。
- 预热调用移到脚本最外层。

---

## [4.1.0] — 2026-09-21 · 代码审查修 8 处

1. **换图可能漏识别（最严重）**：原用 `src + naturalWidth×naturalHeight` 做指纹去重，
   src 变更事件触发时尺寸还是上一张的值 → 指纹与 load 后相同 → 新验证码被跳过；同址重载完全绕不过。
   改为 load/decode 事件驱动 + 80 ms 防抖。
2. 可能读到旧帧：`img.complete` 在 src 刚改写时可能仍为 true → 改用 `await img.decode()`。
3. 吞掉了 ORT 的 `[E:onnxruntime` 错误日志 → 只静音 W 级。
4. ONNX 单次失败即永久降级到 Tesseract → 改为连续 2 次才放弃。
5. 验证码异步插入时直接放弃 → 改为轮询等待约 10 秒。
6. 无谓清空页面自己的 input placeholder → 去掉。
7. 用户正在别处打字时抢焦点 → 仅在用户未在输入框内时聚焦。
8. `session.outputNames` 缺失时 `.every()` 抛异常打断整条链路 → 退化为按 key 枚举。

---

## [4.0.0] — 2026-09-21 · 首个版本：接上被闲置的 ResNet 模型

用户原先安装的 Greasyfork 脚本「jAccount 验证码识别 - Tesseract 版 v3.1.0」里，
`recognizeWithONNX()` 是**死代码**——`recognize()` 只调 `recognizeWithTesseract()`，
99% 准确率的 ResNet 从未启用。本版修正：

- 调用真正的 ONNX 路径。
- 后处理改 27 类 + `blank@26` 跳过判定 4 位码（原版对 5 个头一律 `for (j=0; j<26; j++)`，永远选不出 blank，
  4 位码只能靠 logit 启发式硬猜，会把 `kwwkc` 误截成 `kwwk`）。
- 二值化阈值 `> 156` → **`>= 156`**（官方 LUT 是 `[0]*156 + [1]*100`，**前景为 1**）。
- 模型 URL 从 `raw.githubusercontent.com` 换成国内可达的多源 CDN。
- Tesseract 兜底加白名单 + PSM 7 + worker 复用（169 ms → 12 ms；准确率收益后来被证明不成立）。

---

## 未发布 / 待办

- 推送本轮新增与修改的测试脚本到 GitHub 仓库（当前仓库内为较早一批）。
- 模型重构线：定向采样约 38 张（8 个高危字符 `c,g,o,u,w,x,y,z` 在第 4 位各达 60 例）
  → 端到端重训 → 重导 ONNX → 更新模型 → 重跑全部校验。
  ⚠ 已实测否证「在冻结的 64 维特征上换分类头」：99.60% → 57.89%（−41.71 pp），必须端到端重训。
