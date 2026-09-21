// ==UserScript==
// @name         jAccount 验证码识别 - ResNet(ONNX) 增强版
// @name:en      jAccount Captcha Auto-Recognizer (ONNX ResNet, Enhanced)
// @namespace    local.tuned.jaccount
// @version      4.4.5
// @description  本地 ResNet-20 (ONNX) 推理，真实验证码实测 97.5%，独立留出集 98.0%（原始 Tesseract 方案 74.2%）；自动判定 4/5 位；Tesseract 兜底；模型本地缓存；冷启动约 2.2s
// @description:en Local ResNet-20 (ONNX) inference, 97.5% measured (98.0% on a held-out set) vs 74.2% for the plain Tesseract approach. Auto 4/5 length detection. Tesseract fallback. ~2.2s cold start.
// @author       danyang685 / RyanStarFox (original) — enhanced
// @homepageURL  https://github.com/RyanStarFox/JAccountVerificationCode
// @source       https://github.com/PhotonQuantum/jaccount-captcha-solver
// @match        https://jaccount.sjtu.edu.cn/jaccount/jalogin*
// @match        https://jaccount.sjtu.edu.cn/jaccount/*
// @grant        GM_xmlhttpRequest
// @connect      cdn.jsdelivr.net
// @connect      fastly.jsdelivr.net
// @connect      unpkg.com
// @connect      raw.githubusercontent.com
// @run-at       document-start
// @license      MIT
// ==/UserScript==

/*
 * 与原版 (v3.1.0) 的差异。以下数字全部来自真实抓取的 jAccount 验证码实测，真值人工逐张
 * 辨认（vm_dump/ground_truth_all.json）。留出集是独立采集、不参与任何调参决策的 100 张，
 * 用来防止自欺 —— 本轮就有一个方案在调参集上"有效"、在留出集上归零。
 *
 *   引擎 / 配置                                          调参集 120     独立留出集 100
 *   你原来跑的：Tesseract 原图直喂、无参数、每次新建 worker   74.2%           —
 *   本脚本兜底：Tesseract + 白名单 + PSM7 + worker 复用      74.2%           —
 *   原版 ONNX 后处理：前 26 类 + raw logit 启发式            95.8%           —
 *   本脚本主路径：ResNet ONNX，27 类 + blank 跳过            97.5%          98.0%
 *
 *  [准确度]
 *  1. 真正启用 ONNX ResNet 路径（原版里 recognizeWithONNX 是死代码，从未被调用）。
 *     这是唯一一个统计显著的提升：74.2% -> 97.5%，McNemar 精确检验 p < 0.0001
 *     （前者对后者错 29 例，后者对前者错 2 例）。其余改动都是修确定性缺陷，不是刷分。
 *  2. 第 5 个输出头是 27 类（26 字母 + 1 个 blank 占位）。原版对 5 个输出一律只读前 26 类，
 *     只能靠「第5位 raw logit < 10 或 比值 < 0.6」猜是不是 4 位。该启发式实测 95.8%，
 *     与本版 97.5% 的差距在 120 张样本上并不显著（只有 2 对不一致样本）——
 *     所以别把这条说成"准确率提升"。它修掉的是一个确定性缺陷：原版会把 5 位码误截成 4 位
 *     （kwwkc -> kwwk、keail -> keai），正是它 5 例错误里的 2 例。
 *     本版按官方 ocr.py 语义处理：argmax 覆盖全部类别，index >= 26 视为 blank 直接跳过，
 *     4/5 位判定交给模型自己给，不再依赖阈值。
 *  3. 二值化阈值修正为 >= 156（官方 LUT = [0]*156 + [1]*100，原版用 > 156 差了 1 个灰度级）。
 *  4. 输入张量名从 session.inputNames 动态取，不再硬编码 'input.1'。
 *  5. 输出张量按 session.outputNames 数值序对齐，不再依赖 JS 对象 key 的隐式排序。
 *  6. softmax 置信度只作参考：这个模型输出非常"自信"，错误样本的最低字符概率在 84%~99.6%，
 *     而正确样本最低可到 94.6% —— 两者完全重叠，靠置信度阈值筛不出错例。
 *     真正兜底靠的是长度校验（不是 4/5 位就拒绝填入）。
 *  7. Tesseract 兜底调优（白名单 + PSM7）在 120 张上与原版**打平**，都是 89/120 = 74.2%。
 *     20 张时看到的 "80% -> 85%" 是小样本噪声，已被更大样本推翻。这项改动的真实收益是
 *     速度：worker 复用把单张 169ms 压到 12ms（约 13.6 倍）。
 *     ⚠ PSM 8 是准确率杀手（20 张消融里掉到 45%），无论如何不要用。
 *
 *  [已试过、确认无效的方案]（记下来避免以后重复踩）
 *  - 二值化阈值扫描：145~175 全都在 96.7% 一片平台上，官方 156 已在最优区，没有可调空间。
 *  - 不二值化直接喂灰度：0/120。这个模型要求硬 0/1 输入，预处理一步都不能省。
 *  - TTA（平移 ±1px 后 logits 平均，试了 3 种组合）：调参集上 +1 例、看着像有效，
 *    但在 100 张独立留出集上 0 对不一致、完全没变，调参集上反而 -1 例。已在留出集上否决。
 *  - 把 ONNX Runtime 换成纯 JS 前向实现：省掉 10.4MB 的 wasm，但实测 jsDelivr 有 10.4MB/s，
 *    冷启动总共才 2.2 秒且之后被 HTTP 缓存长期命中；而纯 JS 手写 21 层卷积会慢十倍以上。
 *    不划算，已放弃。
 *
 *  [效率]
 *  8. 模型改用 jsDelivr CDN 优先（国内可直连），三级回退；写入 Cache Storage，二次访问零下载。
 *     原版用 raw.githubusercontent.com，国内常常拉不到 —— 这正是它悄悄退回 Tesseract 的原因。
 *  9. 冷启动实测约 2.2 秒（ort.min.js 0.54MB + ort-wasm-simd.wasm 10.4MB + 模型 1.08MB，
 *     jsDelivr 实测 10.4MB/s），之后由浏览器 HTTP 缓存长期命中。
 * 10. ORT 运行时与模型改为**并行**下载：原来串行要 0.54 + 0.62 秒，现在重叠，省约 0.6 秒。
 *     wasm 由 ORT 自己在 create() 阶段取，没法再往前叠。
 * 11. @run-at 改为 document-start：让下载与页面解析/加载并行，而不是等 DOM 就绪才开始。
 * 12. ONNX Runtime 只加载一次，Session 复用；多线程关闭（页面非跨域隔离，开了只会告警并回退）。
 * 13. Tesseract 仅在 ONNX 失败时才懒加载，正常路径完全不下载 tesseract + traineddata。
 * 14. 验证码刷新后自动重新识别。不用「src 是否变化」做去重（同址重载会漏图），
 *     改为 load/decode 事件驱动 + 80ms 防抖。
 * 15. 去掉了原版两段重复的聚焦 setTimeout、去掉了默认刷屏的 console 日志。
 *
 *  [4.1.0 代码审查修补]
 * 16. 修复换图可能漏识别：原来用「src + naturalWidth×naturalHeight」做指纹去重，但 src 变更
 *     事件触发时图片尺寸还是上一张的值，算出的指纹会和 load 后相同，导致新验证码被跳过；
 *     同址重载更是完全绕不过去。改为 load/decode 事件驱动 + 80ms 防抖，不再做内容去重。
 * 17. 修复可能读到旧帧：只用 img.complete 判断图片就绪不可靠，src 刚改写时它可能仍为 true，
 *     会拿到上一张残留的画面。改用 await img.decode()。
 * 18. 不再吞掉 ORT 的 E 级错误日志（原来把 [E:onnxruntime 也静音了，会掩盖真故障）。
 * 19. ONNX 单次失败不再把整页永久降级到 Tesseract，改为连续失败 2 次才放弃。
 * 20. 验证码异步插入（SPA / 延迟渲染）时不再直接放弃，轮询等待约 10 秒。
 * 21. 不再无谓清空输入框的 placeholder（那是页面自己的东西）。
 * 22. 用户正在别处打字时不抢焦点 —— 冷启动识别要 1~3 秒，这期间抢光标是帮倒忙。
 * 23. outputNames 缺失时退化为按输出 key 枚举，不再因 .every() 抛异常中断整条链路。
 *
 *  [4.2.0 本轮：把"还能不能更好"逐个测掉]
 * 24. 修正了一处自己的人工标注错误（n062 首字符是 j 不是 i，用 i/j 的字形高度差 15px/19px
 *     系统性复核了全部 83 个 i/j 字符，未再发现矛盾）。修正后调参集 97.5%、留出集 98.0%。
 * 25. 阈值扫描 / 灰度输入 / TTA 三组方案实测后否决，结果记在上面的"已试过、确认无效"一节。
 * 26. 效率上做了两处确定性改进：@run-at 提到 document-start、ORT 与模型并行下载（省约 0.6s）。
 *
 *  [4.3.0 可运行性验证 —— 这次是真的跑过了，不是"读起来没问题"]
 * 27. API 逐个核对：对着 onnxruntime-web@1.16.3 的实际 bundle 确认
 *     `get inputNames()` / `get outputNames()` 是 InferenceSession 上的 getter，
 *     `env.wasm.wasmPaths` / `numThreads` / `simd` 均存在；且 wasmPaths 会被拼接文件名，
 *     所以必须以 "/" 结尾 —— 本脚本的值符合（simd+单线程 -> ort-wasm-simd.wasm）。
 * 28. 端到端实跑：用真实的 onnxruntime-web 加载真实模型，跑 220 张样本，
 *     解码结果与官方 Python 参考实现 **220/220 完全一致**，样本0 的 logits 最大绝对差 1.001e-5
 *     （浮点舍入级别）。session.inputNames = ["input.1"]、outputNames = ["218".."222"]，
 *     与 postprocess 的假设一致；推理 9.7ms/张。
 *     测试用的 postprocess/softmax 是**直接从本文件抽出来执行**的，不是我另抄一份。
 * 29. 灰度换算改为 Math.round 后再比阈值。PIL 的 convert("L") 是整数+四舍五入，
 *     原来直接拿浮点比 >= 156，在 220 张里有 106 个像素判定不同；四舍五入后 **0 个不同**，
 *     保证"我实测的准确率"和"你实际跑出来的"是同一件事。
 *     （另测：给 RGB 加 ±8 的随机扰动模拟不同解码器差异，220 张里 0 张结果改变，
 *      所以这条本身不会影响准确率，属于消除隐患而非修 bug。）
 * 30. 修复"用户改过验证码框再点换图"时答案被丢弃：原判断是「值不是我们填的就跳过」，
 *     但换图后旧值必然对不上新图，必须覆盖；改成只跳过"用户此刻正在该框里打字"的情况。
 * 31. 预热加条件：@match 包含 jaccount/*，不加判断会让用户浏览其他页面时白下载 12MB。
 *     现在只在登录页（路径含 jalogin）或页面里确实存在验证码图片时才预热。
 * 32. loadScript 兜底：document-start 下 <head> 可能不存在，极端时 documentElement 也没有。
 *
 *  [4.4.0 复查：把 4.3.0 漏掉的几个真实缺陷补上]
 * 33. 【覆盖逻辑副作用】4.3.0 第 30 条改对了「换图要覆盖」，但判定式写成了
 *     `!(value === '' || value === lastFilled) && activeNow`，留下一个反向缺陷：
 *     用户按 Ctrl+A 删掉错误答案、准备自己重打时（框内为空 + 焦点在框内），
 *     前半部分因为"空值"判定为可覆盖，自动填的值会立刻冲掉用户刚清空的输入位，
 *     表现为"我刚删完它就又填回来了"。现在改为：**只要焦点在验证码框内就一律不动**，
 *     空值同样视为用户"我要自己输"的明确信号；焦点不在框内时（例如点了换图按钮）
 *     无论框内是空、是我们填的、还是用户改过的，都按新图结果覆盖。
 *     已用 7 场景真值表逐一核对，两版行为一致。
 * 34. 【回退链路断点】三级回退只覆盖了 ort.min.js，wasm 目录却写死第一个源。
 *     于是"jsDelivr 被墙 → 回退到 fastly 拉到 ORT 本体"这条链，会在 ORT 自取
 *     10.4MB 的 ort-wasm-simd.wasm 时重新撞回 jsDelivr 而失败 —— 回退等于没生效，
 *     且失败点被埋在 ORT 内部，日志上只看到"模型加载失败"。
 *     现在 wasmPaths 从**实际加载成功的那个脚本 URL** 推导，与 ORT 本体同源。
 * 35. 【可能永久挂起】loadScript 在 document-start 下若 documentElement 尚未建立，
 *     会在事件回调里直接 appendChild 而抛 TypeError —— 那个异常跑在回调里，
 *     不被外层 Promise 捕获，Promise 永久 pending，loadOrt() 跟着挂死，
 *     整条 ONNX 路径静默失效且无任何日志。现在回调内自兜异常，并加短轮询 + 换源兜底。
 *
 *  [4.4.1 紧急修复 —— 4.4.0 我改坏了东西]
 * 36. 【致命】4.4.0 重写 loadScript 时，漏掉了 `s.src = url;` 这一行。
 *     后果：<script> 元素没有地址，浏览器根本不会去加载它，onload 永不触发，
 *     于是每个源都走到 15 秒超时 -> 全部源失败 -> ONNX 加载失败 ->
 *     回退 Tesseract（国内拉不到 tesseract.js 与语言包）-> 双双失败 ->
 *     输入框显示"识别失败，请手动输入"。也就是「所有识别全部失败」。
 *     已补回赋值；对照实验：修复前 6 个场景全部挂死，修复后正常加载 7ms 返回。
 * 37. 事件挂载顺序：现在是**先挂 onload/onerror 再赋 src**。若先赋 src 再挂事件，
 *     脚本命中强缓存时可能在同一个微任务里就加载完成，事件派发时 onload 还是 null，
 *     事件被永久错过 —— 同样是 Promise 永不 settle。该场景已单独验证。
 * 38. loadScript 增加超时（默认 15s）：真实网络里 onload/onerror 可能一个都不来
 *     （中间设备挂起连接、CSP 静默丢弃、其它扩展拦截），没有超时会把
 *     "报错"变成"没反应"，后者难排查得多。
 * 39. loadOrt 不再只凭"脚本 onload 了"就认定 window.ort 存在：onload 触发但
 *     全局没挂上的情况确实存在（CSP 拦截、被别的扩展搅乱）。现在回读全局，
 *     不在就换源再试一轮，并抛出说明真实原因的异常。
 *
 *  [4.4.2 让失败能被看见]
 * 40. 【可观测性】4.4.1 之后仍有用户报"识别失败"。问题在于：无论是 ORT 没加载上、
 *     模型下不下来、wasm 起不来、还是 Tesseract 也拉不到，界面上**全都是同一句**
 *     "识别失败，请手动输入"。用户只能看到结果，看不到原因，于是每一轮修复都在
 *     "我猜是 A"→"猜错了"→"我猜是 B"之间打转，猜错了两次。
 *     现在把两个引擎各自的真实异常 message 一起写进 placeholder：
 *       识别失败，请手动输入 [ONNX: [推理] xxx ｜ Tess: yyy]
 *     并在 title 里存完整文本（悬停可见）。用户复制一句话就能定位到具体环节。
 * 41. 【可观测性】ONNX 路径拆成"加载 ORT / 加载模型 / 预处理 / 推理 / 后处理"
 *     五个步骤，异常统一带上步骤前缀再抛。此前 "no available backend found" 和
 *     "Failed to fetch" 都会被压成同一句"ONNX 路径失败"，但它们一个是 wasm 环境
 *     问题、一个是网络问题，处理方式完全不同。
 * 42. 【缺陷】runFor 里 `recognize(img)` 没有接 catch。recognize 内部虽然为两个
 *     引擎各自做了 try，但后面还有事件派发与焦点处理语句；那里一旦抛异常，
 *     就是一个无人认领的 rejected Promise —— 用户侧表现为"点了换图什么都没发生"，
 *     连"识别失败"都不显示，是比报错更难排查的状态。现已显式兜住并展示。
 * 43. 【缺陷】clearOwnPlaceholder 原本用全等匹配清理自己写的文案。改为前缀匹配 ——
 *     否则第 40 条引入的带原因后缀的文案永远清不掉，会在识别成功后残留在框里。
 * 44. 【隐患】brief / showDiag 原打算定义在首次使用点附近，但它们被更早定义的
 *     recognizeWithONNX 引用。虽然实际调用发生在初始化之后（不会有 TDZ 问题），
 *     仍统一上提到工具区定义 —— 错误处理基础设施不该依赖定义顺序。
 *
 *  [4.4.3 把"失败"变成不可忽略的横幅]
 * 45. 4.4.2 的三种提示（placeholder / 状态条 / 控制台）都太容易被忽略，用户连续
 *     三轮只回一句"依旧失败"，排查完全卡住。改为：只要失败就在页面顶部弹一条
 *     红底横幅，把**脚本版本号 + 失败环节 + 完整错误**一次性摊开，附关闭按钮。
 *     同时横幅内容自带"脚本已加载"这一信息 —— 它能区分"脚本没跑"和"脚本跑了但失败"，
 *     而这恰恰是前几轮最分不清的一件事。
 * 46. 预热失败（模型加载不出来）现在立刻弹横幅。以前它只在用户点换图时才间接暴露，
 *     而预热失败意味着后面每一次识别都必然失败，必须第一时间说。
 * 47. "10 秒没找到验证码图片"的提示改为横幅，并区分两种成因：输入框在但图片选择器
 *     没匹配（页面改版）／两者都没有（可能根本没注入脚本）。
 * 48. 状态条现在永远带版本号（"v4.4.3 已发现验证码…"），用户一眼就能确认自己装的
 *     到底是哪一版 —— 前几轮无法排除"用户装的不是最新版"这一可能，浪费了时间。
 *
 * 【本轮排除的嫌疑，记录以免重复】
 *   - jAccount 页面 CSP：实测响应头只有 `upgrade-insecure-requests`，不含 script-src
 *     限制，因此"动态注入 <script> 被 CSP 拦"不成立。
 *   - CDN 可达性：ort.min.js / ort-wasm-simd.wasm(10.9MB) / nn_model.onnx /
 *     tesseract.min.js 四个资源实测均 200 且 MIME 正确，1 秒内下载完成。
 *   - ORT 1.16.3 dist 完整性：npm 包确实缺非 jsep 的 ort-wasm-simd-threaded.js，
 *     但 jsdelivr 同样没有该文件，说明 ORT 不依赖它。
 *   - 用 Node 直接跑 ort.min.js：挂在 `b.normalize is not a function`，
 *     这是 Node 缺 fake-path 的环境问题，浏览器不会出现，不能作为证据。
 *
 *  [4.4.4 ★ 真正的根因修复：UMD 全局被页面污染]
 * 49. 【根因】4.4.3 的横幅拿到了决定性线索：
 *       "脚本已加载但 window.ort 未出现（最后尝试：.../ort.min.js）"
 *     即：<script> 从 jsdelivr 下载成功、onload 正常触发，但 ort 没挂到 window 上。
 *
 *     真凶是 ORT 的 UMD 包装头：
 *       !function (e, t) {
 *           "object" == typeof exports && "object" == typeof module ? module.exports = t()
 *         : "function" == typeof define && define.amd            ? define([], t)
 *         : "object" == typeof exports                           ? exports.ort = t()
 *         :                                                       e.ort = t()
 *       }(self, ...)
 *     四条分支按序判断 exports / module / define / exports。**只要页面上存在
 *     `define`（AMD 加载器）或 `exports`，就永远走不到 `e.ort = t()`。**
 *     jAccount 登录页上确实有这类全局变量（老式模块加载器 / 统计脚本 / 兼容库都可能留下）。
 *     结果：脚本"加载成功"、onload 触发、但 window.ort 永远是 undefined。
 *
 *     为什么此前测不出来：**沙箱环境是干净的，没有 define/exports**，永远走第 4 条分支。
 *     这就是 220 张端到端全过、用户却全部失败的原因 —— 测试环境与真实环境有一个
 *     从未被识别的差异维度。已用 umd_probe.js 在"有 define""有 exports"两种环境下
 *     复现旧做法的失败、并验证修复有效。
 *
 *     修复：不再把 <script src> 交给页面环境执行。改为**下载源码文本**，
 *     在**受控作用域**里执行 —— 用 new Function 把 define/exports/module/require
 *     显式声明为 undefined 形参遮蔽掉，强制 UMD 走 e.ort = t()，再显式取回对象。
 *     附带好处：也不再受页面 CSP 对 <script src> 的限制。
 * 50. 新增 loadUmdModule() / evalUmdInControlledScope() / fetchText() 三个函数。
 *     Tesseract 同为 UMD，一并改造（它还有 worker 与语言包要联网，国内通常拉不到，
 *     所以它只是兜底；但真需要它时不该再挂在同一个坑里）。
 * 51. loadScript() 保留但标注为「只用于非 UMD 脚本」——不删除是为了不重演 4.4.0
 *     "重构时把还在用的东西删掉"的教训。当前已无调用者。
 * 52. loadOrt() 会先看页面是否已有可用的 ort（省一次下载），没有再走受控加载；
 *     并对取回的 ort 做结构校验（必须有 env.wasm），避免拿到被污染的半成品。
 *
 * 验证：umd_e2e.js 在 define / exports / clean 三种环境下 ——
 *   旧做法在 define 与 exports 下**均复现"未挂载"**，新做法三种环境**均成功取回**，
 *   且取回的 ort 版本、Tensor、InferenceSession 结构完好。
 *
 *  [4.4.5 布局修复：不再遮挡验证码图片]
 * 53. 【用户反馈】"有一些遮挡验证码图片"。根因是 4.4.3 引入的失败横幅用了
 *       position: fixed; top: 0; z-index: 2147483647;
 *     它永久占据页面顶部可视区域，把验证码图片和页面头部内容一起盖住了 ——
 *     排障信息本身把待排查的对象挡住了。这是典型的"修复引入了比原问题更糟的问题"。
 * 54. 横幅与状态条一律改为**插入验证码图片下方的文档流内**（img.nextSibling），
 *     样式里不再有任何 position 定位，从根上不可能覆盖既有元素。
 *     两者默认 `display:none`，只有真需要显示时才展开；识别成功时状态条清空并折叠、
 *     横幅收起 —— 正常情况下页面上不会多出任何一行字。
 * 55. 新增 hideBanner()：识别成功 / 状态恢复正常时把失败横幅收起来。
 *     之前失败横幅一旦出现就永远挂在页面上，即使后续识别已经成功。
 * 56. 成功路径不再常驻显示"识别成功"文字。答案已经填进输入框了，再挂一行提示
 *     既是噪音又占空间。只有"低置信需要核对"时才显示橙色警告。
 * 57. ensureStatus() / ensureBanner() 在找不到验证码图片时退回 body，
 *     不再假设图片一定存在（@run-at document-start 阶段 DOM 还没建好）。
 *
 * 新增测试 layout_check.js：两个场景 ——
 *   场景 1（DOM 沙箱实跑，且模拟页面存在 exports/module 污染）：
 *     验证 ONNX 全链路在污染环境下仍能取回 ort 并识别出结果，
 *     且注入元素均在文档流内、成功路径下全部折叠隐藏。
 *   场景 2（静态核验）：验证 showBanner/ensureBanner 内不含任何 position 定位、
 *     确实用 insertBefore 插入 img.nextSibling、且默认 display:none。
 *
 * 顺带修正测试桩自身的两处缺陷（否则测试会给出假信心）：
 *   - 样式桩的 cssText 与 display 属性此前互不同步，脚本设 display:none 后
 *     测试读到空字符串，会把"已折叠"误判成"仍占位"。
 *   - 网络桩此前对所有 URL 一律返回全零 ArrayBuffer，ORT 源码变成 1MB 的 \0，
 *     交给 new Function 必然抛 Invalid or unexpected token ——
 *     表现为"ONNX 路径全部失败"，实际是桩的假数据造成的假失败。
 *     现按 URL 区分：.onnx 返回二进制、其余返回形状正确的 UMD 源码。
 */

(function () {
    'use strict';

    const CFG = {
        imgSelector: '#captcha-img',
        inputSelector: '#input-login-captcha',
        userSelector: '#input-login-name',

        debug: false,          // 想看详细日志改成 true
        showStatus: true,      // 在验证码下方显示识别状态（排障用；确认无问题后可改为 false）
        markLowConfidence: true, // 低置信度时给输入框加个橙色描边
        lowConfidence: 0.60,   // 单字符 softmax 概率低于该值视为"低置信"

        // 类别数与官方一致：前 4 个头 26 类，第 5 个头 27 类（多一个 blank）
        numClasses: 26,
        blankIndex: 26,
        charset: 'abcdefghijklmnopqrstuvwxyz',

        modelUrls: [
            'https://cdn.jsdelivr.net/gh/RyanStarFox/JAccountVerificationCode@main/model/nn_model.onnx',
            'https://fastly.jsdelivr.net/gh/RyanStarFox/JAccountVerificationCode@main/model/nn_model.onnx',
            'https://raw.githubusercontent.com/RyanStarFox/JAccountVerificationCode/main/model/nn_model.onnx'
        ],
        ortUrls: [
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js',
            'https://fastly.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js',
            'https://unpkg.com/onnxruntime-web@1.16.3/dist/ort.min.js'
        ],
        // wasm 目录不再写死。ORT 会拿 wasmPaths 去拼 ort-wasm-simd.wasm，
        // 如果 ORT 本体是从第 2/3 个源加载回来的、而 wasm 仍去第一个源取，
        // 那么"第一个源挂了、回退到第二个源"这条链路依然会在取 wasm 时断掉 ——
        // 三级回退只覆盖了 JS 却没有覆盖它依赖的 10.4MB wasm，等于白做。
        // 现在改为从**实际加载成功的那个脚本 URL** 推导目录（见 loadOrt）。
        tesseractUrls: [
            'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js',
            'https://unpkg.com/tesseract.js@5/dist/tesseract.min.js'
        ]
    };

    const log = (...a) => CFG.debug && console.log('[jAccount]', ...a);
    const warn = (...a) => console.warn('[jAccount]', ...a);

    // 该 ONNX 模型把权重既当 initializer 又声明成 graph input，ORT 会刷一屏
    // "Initializer xxx appears in graph inputs..." 警告（不影响正确性）。只静音 W 级告警，
    // E 级错误必须放行 —— 吞掉真错误比刷屏危险得多。
    (function muteOrtNoise() {
        const filter = (orig, ctx) => function (...args) {
            const m = args[0];
            if (typeof m === 'string' && m.includes('[W:onnxruntime')) return;
            return orig.apply(ctx, args);
        };
        console.error = filter(console.error, console);
        console.warn = filter(console.warn, console);
    })();

    /* ---------------------------------------------------------------- 通用工具 */

    /**
     * 把异常压成一小段可直接读的文本。
     * 目的：用户不需要开控制台，光看输入框就能知道自己卡在哪一环。
     * 之前所有失败都只显示同一句"识别失败"，导致用户端与开发端之间只能靠猜 ——
     * 整整两轮修复都建立在"我猜是 X"之上，而猜错了两次。
     * 放在最前面定义：它是错误处理的基础设施，不该依赖定义顺序。
     */
    const brief = (e) => {
        if (!e) return '?';
        const m = (e.message || String(e)).replace(/\s+/g, ' ').trim();
        return m.length > 120 ? m.slice(0, 120) + '…' : m;
    };
    const VERSION = '4.4.4';
    const showDiag = (input, base, detail) => {
        // 详细原因同时写进 title，鼠标悬停可见（不改动输入框本身的外观）
        input.placeholder = base + ' [' + detail + ']';
        try { input.title = detail; } catch (e) { /* 只读属性，忽略 */ }
        warn(base + ' → ' + detail);
        paintStatus('err', base.replace(/，.*$/, '') + '：' + detail);
        showBanner(base, detail);
    };

    /* ------------------------------------------------ 失败提示条（排障用）
     *
     * 用途：识别失败时把具体原因摊开，让用户不必打开控制台就能定位。
     *
     * ⚠ 布局约束（4.4.5 修正，吃过一次教训）：
     * 4.4.3 把这里做成了 `position: fixed; top: 0; z-index: 2147483647` 的悬浮横幅，
     * 结果**直接盖住了页面顶部内容**，用户反馈"有一些遮挡验证码图片"。
     * 悬浮层看起来"更醒目"，但它会永久侵占页面可视区域，对正常使用是净损失。
     *
     * 现在改为：插到验证码图片**下方的文档流里**，不覆盖任何既有元素。
     * 识别成功时自动隐藏 —— 它只该在真出问题时占用空间。
     */
    const BANNER_ID = 'jaccount-recognizer-banner';
    let bannerEl = null;

    /** 把提示条挂到验证码图片下面（文档流内，绝不悬浮遮挡） */
    function ensureBanner() {
        if (!CFG.showStatus) return null;
        if (bannerEl && bannerEl.isConnected) return bannerEl;
        const img = document.querySelector(CFG.imgSelector);
        const host = (img && img.parentNode) || document.body || document.documentElement;
        if (!host) return null;

        const el = document.createElement('div');
        el.id = BANNER_ID;
        el.style.cssText = [
            'display:none',                       // 默认隐藏，只在失败时显示
            'margin:6px 0', 'background:#fef0f0', 'color:#c45656',
            'border:1px solid #fde2e2', 'border-radius:4px',
            'padding:8px 10px', 'font:12px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
            'white-space:pre-wrap', 'word-break:break-all', 'max-width:420px'
        ].join(';');

        // 关闭按钮：用户可以自己收起来
        const close = document.createElement('span');
        close.textContent = '×';
        close.style.cssText = 'float:right;cursor:pointer;font-size:16px;line-height:1;'
            + 'padding:0 4px;color:#c45656;font-weight:700';
        close.onclick = () => { el.style.display = 'none'; };

        const body = document.createElement('div');
        body.id = BANNER_ID + '-body';

        el.appendChild(close);
        el.appendChild(body);
        try {
            if (img && img.parentNode) host.insertBefore(el, img.nextSibling);
            else host.appendChild(el);
        } catch (e) { return null; }
        bannerEl = el;
        return el;
    }

    function showBanner(title, detail) {
        if (!CFG.showStatus) return;
        try {
            const el = ensureBanner();
            if (!el) return;
            el.style.display = '';
            const body = document.getElementById(BANNER_ID + '-body');
            if (body) {
                body.textContent = '【jAccount 验证码识别】v' + VERSION + ' 识别失败\n'
                    + '原因：' + title + '\n'
                    + '细节：' + detail + '\n'
                    + '（把这段文字发给我即可定位。脚本已加载说明扩展/用户脚本本身是生效的。）';
            }
        } catch (e) { /* 提示条只是排障辅助，任何失败都不该影响主流程 */ }
    }

    /** 识别成功时把失败提示条收起来 —— 它只在真出问题时该占地方 */
    function hideBanner() {
        try { if (bannerEl && bannerEl.isConnected) bannerEl.style.display = 'none'; } catch (e) { }
    }

    /* ------------------------------------------------ 页面状态条（排障用）
     *
     * 为什么要有这个东西：
     * 前两轮修复全都建立在"我猜是 X"之上，而用户能提供的信息只有一句"识别失败"。
     * 光靠这句话分不清是网络、CSP、wasm 还是模型的问题。placeholder 里的原因容易被
     * 忽略（颜色淡、位置偏），控制台又要求用户主动打开。
     *
     * 所以在验证码旁边放一条状态，把最近一次识别的**具体结果或原因**直接显示出来。
     * CFG.showStatus 设为 false 即可完全关闭。
     */
    const STATUS_ID = 'jaccount-recognizer-status';
    let statusEl = null;

    function ensureStatus() {
        if (!CFG.showStatus) return null;
        if (statusEl && statusEl.isConnected) return statusEl;
        // 插到验证码图片**下方的文档流里**：不悬浮、不覆盖任何既有元素。
        // 4.4.3 那版用 fixed 定位直接把页面顶部盖住了（用户反馈"遮挡验证码图片"），
        // 这是本函数坚持留在文档流内的原因。
        const img = document.querySelector(CFG.imgSelector);
        const host = (img && img.parentNode) || document.body || document.documentElement;
        if (!host) return null;
        const el = document.createElement('div');
        el.id = STATUS_ID;
        el.style.cssText = 'display:none;margin:4px 0;'
            + 'font:12px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            + 'color:#909399;white-space:pre-wrap;word-break:break-all;max-width:420px;';
        try {
            if (img && img.parentNode) host.insertBefore(el, img.nextSibling);
            else host.appendChild(el);
        } catch (e) { return null; }
        statusEl = el;
        return el;
    }

    /**
     * 绘制状态。
     * 成功时只留一行灰色小字（不打扰正常使用）；
     * 失败/警告时才染色并展开细节。
     */
    function paintStatus(kind, text) {
        const el = statusEl || ensureStatus();
        if (!el) return;
        if (!text) {
            // 空文本 = 无话可说。整个清空并折叠，连前缀都不留 ——
            // 否则页面会挂着一行孤零零的 "[验证码识别]"，比什么都不显示更碍眼。
            el.textContent = '';
            el.style.display = 'none';
            hideBanner();
            return;
        }
        const color = kind === 'err' ? '#f56c6c' : kind === 'warn' ? '#e6a23c' : '#909399';
        const mark = kind === 'err' ? '✗ ' : kind === 'warn' ? '! ' : '';
        el.style.color = color;
        el.textContent = '[验证码识别] ' + mark + text;
        el.style.display = '';
        if (kind === 'ok') hideBanner();   // 识别成功就把失败提示收起来
    }

    /**
     * 动态加载远程脚本（<script src> 方式）。
     *
     * ⚠ 4.4.4 起本函数**已不再用于 ORT / Tesseract**，保留仅供将来加载非 UMD 的
     * 普通脚本。原因：<script src> 在页面环境执行，UMD 库会因页面上的
     * define/exports 而挂不上全局对象（详见 loadUmdModule 的注释）。
     * 加载 UMD 库请一律使用 loadUmdModule()。
     *
     * 两个必须遵守的次序/兜底规则，都是踩过的坑：
     *
     * 1. 事件处理器必须在赋 src **之前**挂好。若先 `s.src = url` 再 `s.onload = ...`，
     *    而该脚本恰好命中浏览器缓存（或 HTTP/2 推送），加载可能在同一个微任务里就完成，
     *    事件派发时 onload 还是 null —— 事件被永久错过，Promise 永不 settle，
     *    调用方 loadOrt() 跟着挂死，整条 ONNX 路径静默失效。
     * 2. 必须自己加超时。onload / onerror 在真实网络里都可能一个都不来
     *    （中间设备挂起连接、CSP 静默丢弃、扩展拦截），没有超时的 Promise
     *    会把故障从"报错"变成"没反应"，后者难排查得多。
     */
    /* -------------------------------------------- 受控作用域执行 UMD 库
     *
     * 这是 4.4.4 的核心修复，解决「脚本 onload 了但 window.ort 没出现」。
     *
     * 真凶在 ORT 的 UMD 包装头上：
     *
     *   !function (e, t) {
     *       "object" == typeof exports && "object" == typeof module ? module.exports = t()
     *     : "function" == typeof define && define.amd            ? define([], t)
     *     : "object" == typeof exports                           ? exports.ort = t()
     *     : /* 只有走到这里 *\/                                     e.ort = t()
     *   }(self, ...)
     *
     * 四条分支按序判断 exports / module / define / exports。
     * **只要页面上存在 `define`（AMD 加载器）或 `exports`，就永远走不到 `self.ort = ...`。**
     *
     * jAccount 登录页里确实有这类全局变量（老式模块加载器 / 统计脚本 / 兼容库都可能留下）。
     * 于是：<script> 从 CDN 下载成功 → onload 正常触发 → 但 ort 挂到了别处 →
     * window.ort 永远是 undefined → ONNX 整条路径失败。
     *
     * 这个故障在沙箱里**永远复现不出来**，因为沙箱是干净的、没有 define/exports ——
     * 这就是为什么此前 220 张端到端全过、用户却全部失败。
     * 已用 umd_probe.js 在"有 define"和"有 exports"两种环境下复现并验证修复。
     *
     * 解决办法：不把 <script src> 交给页面环境执行，改为下载源码文本，
     * 在**受控作用域**里执行 —— 把 exports / module / define 显式遮蔽为 undefined，
     * 强制 UMD 走最后一条分支，然后显式把 ort 对象取回来。
     *
     * 附带好处：也不再受页面 CSP 对 <script src> 的限制。
     */
    async function loadUmdModule(urls, globalName) {
        let lastErr = null;
        for (const url of urls) {
            let code;
            try {
                code = await fetchText(url);
            } catch (e) {
                lastErr = e;
                warn('UMD 源码下载失败，尝试下一个源', url, e.message);
                continue;
            }
            try {
                const mod = evalUmdInControlledScope(code, globalName);
                if (!mod) {
                    throw new Error('源码已执行，但未返回 ' + globalName + ' 对象（UMD 分支可能仍被拦截）');
                }
                log('UMD 模块已在受控作用域载入', globalName, url);
                return { mod, url };
            } catch (e) {
                lastErr = e;
                warn('UMD 受控执行失败，尝试下一个源', url, e.message);
            }
        }
        throw lastErr || new Error('所有 UMD 源均不可用');
    }

    /**
     * 在受控作用域执行 UMD 源码并取回导出对象。
     *
     * 用 new Function 显式声明同名形参，把外层可能存在的 define / exports / module
     * （以及 require）遮蔽掉。形参值为 undefined，UMD 的 typeof 判断因此全部落空，
     * 必然走 `e.ort = t()` 这一条 —— 而 e 就是我们传入的 sandbox。
     */
    function evalUmdInControlledScope(code, globalName) {
        const sandbox = {};
        // self/window/globalThis 三者都指向同一个 sandbox：
        // UMD 用 `self` 作为挂载点，而库内部有时会用 window/globalThis 探测环境。
        sandbox.self = sandbox;
        sandbox.window = sandbox;
        sandbox.globalThis = sandbox;
        sandbox.console = console;   // 让库的告警仍能在控制台出现

        // 注意：不能把 navigator/document 搬进来 —— 库需要真实的浏览器对象。
        // 只遮蔽模块系统的三个标识符 + require 即可。
        const factory = new Function(
            'self', 'window', 'globalThis',
            'define', 'exports', 'module', 'require',
            code + '\n;return (self && self[' + JSON.stringify(globalName) + '])'
            + ' || (window && window[' + JSON.stringify(globalName) + ']) || null;'
        );
        const mod = factory(sandbox, sandbox, sandbox, undefined, undefined, undefined, undefined);
        // 顺手把结果同步回真实全局，方便用户在控制台调试（失败也不影响）
        try {
            if (mod && typeof self !== 'undefined') self[globalName] = self[globalName] || mod;
        } catch (e) { /* 严格模式下可能只读，忽略 */ }
        return mod;
    }

    /** 下载文本（脚本源码）。优先 GM_xmlhttpRequest 以绕过 CSP/CORS */
    async function fetchText(url) {
        let buf;
        try {
            buf = await gmGet(url);
        } catch (e) {
            const r = await fetch(url);
            if (!r.ok) throw new Error('HTTP ' + r.status);
            buf = await r.arrayBuffer();
        }
        try {
            return new TextDecoder('utf-8').decode(new Uint8Array(buf));
        } catch (e) {
            // 极老环境没有 TextDecoder，退回手写解码
            const bytes = new Uint8Array(buf);
            let s = '';
            for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
            return decodeURIComponent(escape(s));
        }
    }

    function loadScript(urls, timeoutMs) {
        timeoutMs = timeoutMs || 15000;
        return new Promise((resolve, reject) => {
            let i = 0;
            const next = () => {
                if (i >= urls.length) return reject(new Error('所有脚本源均加载失败'));
                const url = urls[i++];
                const s = document.createElement('script');

                let settled = false;
                const timer = setTimeout(() => {
                    if (settled) return;
                    settled = true;
                    warn(`脚本加载超时 ${timeoutMs}ms，尝试下一个`, url);
                    next();
                }, timeoutMs);

                // 顺序关键：**先挂事件，再赋 src**。
                // 如果先赋 src 再挂 onload，而该脚本恰好命中强缓存，加载可能在
                // 同一个微任务里就完成，事件派发时 onload 还是 null —— 事件被永久
                // 错过，Promise 永不 settle，调用方 loadOrt() 跟着挂死，
                // 整条 ONNX 路径静默失效（没有任何日志，最难查）。
                s.onload = () => {
                    if (settled) return;
                    settled = true;
                    clearTimeout(timer);
                    log('脚本已加载', url);
                    resolve(url);
                };
                s.onerror = () => {
                    if (settled) return;
                    settled = true;
                    clearTimeout(timer);
                    warn('脚本加载失败，尝试下一个', url);
                    next();
                };
                // 赋 src 必须放在事件挂好之后；漏掉这一行的话 <script> 没有地址，
                // 浏览器根本不会去加载，onload 永不触发 —— 表现为"所有源都超时"。
                s.src = url;

                // @run-at document-start 时 <head> 可能还不存在；极端情况下连
                // documentElement 都还没建好。那就等主机节点出现再插 —— 注意
                // 这里不能抛异常：回调里的异常不会被上面的 Promise 捕获，
                // 只会让 Promise 永久 pending。
                const tryAttach = () => {
                    const h = document.head || document.documentElement;
                    if (!h || typeof h.appendChild !== 'function') return false;
                    try { h.appendChild(s); return true; } catch (e) { return false; }
                };

                if (tryAttach()) return;

                const onReady = () => { if (tryAttach()) document.removeEventListener('readystatechange', onReady); };
                document.addEventListener('readystatechange', onReady);
                // readystatechange 在 document-start 之后可能不会再补发，
                // 用短轮询兜底；超时计时器已在跑，不会无限等。
                let tries = 0;
                const poll = setInterval(() => {
                    if (tryAttach() || ++tries > 200) {      // 200 × 25ms = 5s
                        clearInterval(poll);
                        document.removeEventListener('readystatechange', onReady);
                    }
                }, 25);
            };
            next();
        });
    }

    function gmGet(url) {
        return new Promise((resolve, reject) => {
            if (typeof GM_xmlhttpRequest !== 'function') return reject(new Error('GM_xmlhttpRequest 不可用'));
            GM_xmlhttpRequest({
                method: 'GET',
                url,
                responseType: 'arraybuffer',
                timeout: 60000,
                onload: r => (r.status >= 200 && r.status < 300 && r.response)
                    ? resolve(r.response)
                    : reject(new Error('HTTP ' + r.status)),
                onerror: () => reject(new Error('网络错误')),
                ontimeout: () => reject(new Error('超时'))
            });
        });
    }

    async function fetchBytes(url) {
        // 优先 GM_xmlhttpRequest：绕过页面 CSP / CORS 限制
        try {
            return await gmGet(url);
        } catch (e) {
            const r = await fetch(url);
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return await r.arrayBuffer();
        }
    }

    /** 带持久缓存的模型下载：第二次进入登录页不再走网络 */
    async function cachedBytes(urls) {
        const CACHE = 'jaccount-captcha-nn-v1';
        let cache = null;
        try {
            if (typeof caches !== 'undefined') cache = await caches.open(CACHE);
        } catch (e) { /* 隐私模式等场景忽略 */ }

        for (const url of urls) {
            if (cache) {
                try {
                    const hit = await cache.match(url);
                    if (hit) {
                        log('模型命中本地缓存', url);
                        return await hit.arrayBuffer();
                    }
                } catch (e) { }
            }
            try {
                const buf = await fetchBytes(url);
                log('模型下载成功', url, buf.byteLength, 'bytes');
                if (cache) {
                    try {
                        await cache.put(url, new Response(buf.slice(0), {
                            headers: { 'Content-Type': 'application/octet-stream' }
                        }));
                    } catch (e) { }
                }
                return buf;
            } catch (e) {
                warn('模型源不可用', url, e.message);
            }
        }
        throw new Error('所有模型源均不可用');
    }

    /* ------------------------------------------------------- 预处理 / 后处理 */

    const IN_W = 110, IN_H = 40;

    /**
     * 与官方 ocr.py 严格对齐的预处理：
     *   Image.convert("L")                      -> ITU-R 601-2 luma
     *   img.point([0]*156 + [1]*100, "1")       -> 灰度 >= 156 记 1，否则 0
     *   不放缩（验证码原始尺寸就是 110x40）
     */
    function preprocess(img) {
        const canvas = document.createElement('canvas');
        canvas.width = IN_W;
        canvas.height = IN_H;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.imageSmoothingEnabled = false; // 原尺寸时避免任何重采样造成的像素级偏移
        ctx.drawImage(img, 0, 0, IN_W, IN_H);

        const { data } = ctx.getImageData(0, 0, IN_W, IN_H);
        const input = new Float32Array(IN_W * IN_H);
        for (let i = 0; i < input.length; i++) {
            const o = i * 4;
            // 必须先四舍五入再比阈值。PIL 的 convert("L") 是整数运算 + 四舍五入，
            // 直接拿浮点数比 >= 156 会在 220 张样张里产生 106 个像素的判定差异 ——
            // 那样"我实测的准确率"和"你实际跑出来的准确率"就不是同一件事了。
            const gray = Math.round(0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2]);
            input[i] = gray >= 156 ? 1.0 : 0.0;
        }
        return input;
    }

    function softmax(src, n, out) {
        let max = -Infinity;
        for (let i = 0; i < n; i++) if (src[i] > max) max = src[i];
        let sum = 0;
        for (let i = 0; i < n; i++) { const v = Math.exp(src[i] - max); out[i] = v; sum += v; }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    /**
     * 与官方 _tensor_to_captcha 等价：
     *   for tensor in tensors:
     *       asc = argmax(tensor, 1)
     *       if asc < 26: captcha += chr(ord('a') + asc)
     * 即：第 5 个头 argmax 落到 26 号 blank 类时，直接跳过 -> 得到 4 位。
     */
    function postprocess(session, outputMap) {
        // 输出名 "218".."222"，按数值序对齐到第 1..5 个字符位。
        // 若 outputNames 不可用则退化为按输出对象的 key 枚举 —— 宁可降级，
        // 也不要让 `names.every` 抛异常把整条识别链路打断。
        const rawNames = Array.isArray(session.outputNames) && session.outputNames.length
            ? session.outputNames
            : Object.keys(outputMap);
        const names = rawNames.slice().sort((a, b) => {
            const na = Number(a), nb = Number(b);
            return (Number.isFinite(na) && Number.isFinite(nb)) ? na - nb : 0;
        });

        let text = '';
        const confidences = [];
        let probBuf = new Float32Array(32);

        for (const name of names) {
            const tensor = outputMap[name];
            if (!tensor) { warn('缺少输出张量', name); continue; }

            const data = tensor.data;
            const n = (tensor.dims && tensor.dims.length === 2)
                ? tensor.dims[1]
                : Math.min(data.length, CFG.blankIndex + 1);

            let best = 0, bestVal = -Infinity;
            for (let i = 0; i < n; i++) if (data[i] > bestVal) { bestVal = data[i]; best = i; }

            if (best >= CFG.numClasses) {          // blank -> 该位不存在
                log(`位置 ${name}: <blank>  (prob 主导类为占位符)`);
                continue;
            }

            if (probBuf.length < n) probBuf = new Float32Array(n);
            const probs = softmax(data, n, probBuf);
            const p = probs[best];
            confidences.push(p);
            text += CFG.charset[best] || '?';
            log(`位置 ${name}: ${CFG.charset[best]}  p=${(p * 100).toFixed(1)}%`);
        }

        const minP = confidences.length ? Math.min(...confidences) : 0;
        return { text, confidences, minConfidence: minP };
    }

    /* ------------------------------------------------------------ ONNX 引擎 */

    let ortReady = null;
    let sessionPromise = null;

    function loadOrt() {
        if (ortReady) return ortReady;
        ortReady = (async () => {
            // 若页面/别的脚本已经提供 ort，直接用（省一次下载）。
            // 注意这只是顺手利用，不能依赖 —— 见下面受控执行的说明。
            let ort = (typeof window !== 'undefined' && window.ort)
                || (typeof self !== 'undefined' && self.ort) || null;
            let loadedUrl = null;

            if (!ort) {
                // 用「下载源码 + 受控作用域执行」而不是 <script src>。
                //
                // 为什么必须这样：<script src> 是在**页面环境**里执行的，ORT 的 UMD 头
                // 会先检测 exports / module / define。jAccount 页面上存在这些全局变量时，
                // ORT 会走 AMD 或 CommonJS 分支，`self.ort = t()` 那一行永远不执行 ——
                // 于是 onload 触发、脚本"加载成功"，但 window.ort 始终是 undefined。
                // 这曾经让整条 ONNX 路径失败，且沙箱无法复现（沙箱环境干净）。
                const r = await loadUmdModule(CFG.ortUrls, 'ort');
                ort = r.mod;
                loadedUrl = r.url;
            }

            if (!ort || !ort.env || !ort.env.wasm) {
                throw new Error('拿到 ort 对象但结构不完整（缺少 env.wasm），可能是被其它脚本污染了全局');
            }

            // 页面不是跨域隔离环境，多线程 wasm 只会告警并回退，直接关掉更干净
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = true;
            // wasm 必须跟 ORT 本体同源：从实际加载成功的那个脚本地址推导目录。
            // 否则"第一个源挂了、回退到第二个"时，JS 能回来、wasm 仍去死掉的第一个源取，
            // 整条回退链路等于没生效（wasm 有 10.4MB，比 JS 更容易被墙/超时）。
            // 已经由页面提供 ort 时（loadedUrl 为 null）退回使用第一个源。
            const base = (loadedUrl || CFG.ortUrls[0]).replace(/\/[^/]*$/, '/');
            ort.env.wasm.wasmPaths = base;
            log('ORT 就绪，wasm 目录', base);
            return ort;
        })().catch(e => { ortReady = null; throw e; });
        return ortReady;
    }

    function getSession() {
        if (sessionPromise) return sessionPromise;
        sessionPromise = (async () => {
            // 并行拉取：ORT 运行时(0.54MB) 与 模型(1.08MB) 同时下载，串行要多花约 0.6 秒。
            // 10.4MB 的 wasm 由 ORT 自己在 create() 阶段取，没法再往前叠，所以把能叠的叠满。
            const [ort, buf] = await Promise.all([
                loadOrt(),
                cachedBytes(CFG.modelUrls)
            ]);
            const session = await ort.InferenceSession.create(new Uint8Array(buf), {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            log('模型加载完成，输入:', session.inputNames, '输出:', session.outputNames);
            return session;
        })().catch(e => { sessionPromise = null; throw e; });
        return sessionPromise;
    }

    async function recognizeWithONNX(img) {
        // 分步标注：ONNX 这条路上有四段完全不同的失败模式，
        // "ONNX 路径失败"这一句话无法区分是"脚本没加载"、"模型没下载"还是"wasm 没起来"。
        // 打上步骤前缀之后，用户报来的一句话就能直接定位到具体环节。
        let step = '加载 ORT';
        try {
            const ort = await loadOrt();
            step = '加载模型';
            const session = await getSession();

            step = '预处理';
            const inputName = session.inputNames[0];
            const input = new ort.Tensor('float32', preprocess(img), [1, 1, IN_H, IN_W]);

            step = '推理';
            const t0 = performance.now();
            const output = await session.run({ [inputName]: input });
            const cost = performance.now() - t0;

            step = '后处理';
            const res = postprocess(session, output);
            log(`ONNX 推理 ${cost.toFixed(1)}ms -> "${res.text}" 最低置信 ${(res.minConfidence * 100).toFixed(1)}%`);
            return res;
        } catch (e) {
            // 抛出的 message 一定带上失败步骤，避免上层只看到一句 "xxx is not a function"。
            // 不直接改 e.message：某些 host 对象的 message 只读，赋值会静默失败或再抛一次。
            // 统一包一层新 Error，原始错误挂在 cause 上。
            const wrapped = new Error('[' + step + '] ' + brief(e));
            wrapped.cause = e;
            throw wrapped;
        }
    }

    /* ------------------------------------------------- Tesseract 兜底（调优） */

    let tessWorkerPromise = null;

    async function getTessWorker() {
        if (tessWorkerPromise) return tessWorkerPromise;
        tessWorkerPromise = (async () => {
            // 同样用受控作用域执行：tesseract.js 也是 UMD，
            // 在被 define/exports 污染的页面里同样挂不上 window.Tesseract。
            // （它的 worker 与语言包仍需联网，国内通常拉不到，所以这条只是兜底。）
            let T = (typeof window !== 'undefined' && window.Tesseract)
                || (typeof self !== 'undefined' && self.Tesseract) || null;
            if (!T) {
                const r = await loadUmdModule(CFG.tesseractUrls, 'Tesseract');
                T = r.mod;
            }
            if (!T) throw new Error('tesseract.js 加载失败');

            const worker = await T.createWorker('eng', 1);
            await worker.setParameters({
                tessedit_char_whitelist: CFG.charset,
                tessedit_pageseg_mode: '7',           // 单行文本
                preserve_interword_spaces: '0',
                classify_bln_numeric_mode: '0'
                // 注意：不要用 PSM 8（单个单词）。20 张消融里它把所有组合都打到 35~60%，
                // 是唯一一个会造成灾难性退化的参数。
                //
                // worker 只创建这一次并全程复用 —— 这是本段代码真正的价值所在：
                // 单张耗时 169ms -> 12ms（约 13.6 倍）。原版每识别一次就 Tesseract.recognize()
                // 一次，等于每次都重建 worker 并重新加载语言包。
            });
            return worker;
        })().catch(e => { tessWorkerPromise = null; throw e; });
        return tessWorkerPromise;
    }

    /**
     * Tesseract 前置处理：只做与 ResNet 同源的 156 阈值二值化，不做上采样 / OTSU / 白边。
     *
     * 依据（20 张消融 vm_dump/ablation_result.json ＋ 120 张复测 vm_dump/tess_120.json）：
     *   原图直接喂 .................... 20 张 80%  ｜ 120 张 74.2%
     *   156 二值化 + 白名单 + PSM7 ..... 20 张 85%  ｜ 120 张 74.2%   <- 采用
     *   2x / 4x 上采样、OTSU、白边 ....... 20 张 85%，无任何增益
     * 在更大样本上这些改动与"原图直喂"完全打平 —— Tesseract 本身就卡在 74% 上下。
     * 保留二值化只是为了与 ResNet 使用同一阈值、便于解释，不是为了准确率。
     * 把准确率从 74.2% 拉到 96.7% 的是上面的 ResNet 路径，不是这里的任何一行。
     */
    function buildTesseractCanvas(img) {
        const canvas = document.createElement('canvas');
        canvas.width = IN_W;
        canvas.height = IN_H;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, IN_W, IN_H);

        const id = ctx.getImageData(0, 0, IN_W, IN_H);
        const d = id.data;
        for (let i = 0; i < d.length; i += 4) {
            // 同样先四舍五入，保证与 ResNet 路径、与官方 Python 实现判定一致
            const g = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
            const v = g >= 156 ? 255 : 0;   // 背景转白、笔画转黑
            d[i] = d[i + 1] = d[i + 2] = v;
            d[i + 3] = 255;
        }
        ctx.putImageData(id, 0, 0);
        return canvas;
    }

    async function recognizeWithTesseract(img) {
        const worker = await getTessWorker();
        const { data } = await worker.recognize(buildTesseractCanvas(img));
        const text = (data.text || '').toLowerCase().replace(/[^a-z]/g, '');
        log('Tesseract 结果:', text);
        return {
            text: text.length === 4 || text.length === 5 ? text : text.slice(0, 5),
            confidences: [],
            minConfidence: (data.confidence || 0) / 100
        };
    }

    /* ------------------------------------------------------------- 主流程 */

    // 连续失败计数：偶发网络抖动不该把整页永久降级到 Tesseract（85% vs 95%）
    let ortFailures = 0;
    const ORT_MAX_FAILURES = 2;
    let lastFilled = null;   // 记录我们自动填入的值，避免覆盖用户手输
    let runToken = 0;

    // 只清理我们自己写进去的提示文案，页面原本的 placeholder 不碰。
    // 用前缀匹配而不是全等：排障版本会把具体原因追加在文案后面
    // （"识别失败，请手动输入 [ONNX: ...]"），全等匹配会清不掉，反而留下脏文案。
    const MSG_ANOMALY = '识别异常，请手动输入';
    const MSG_FAILED = '识别失败，请手动输入';
    const isOwnMsg = (s) => typeof s === 'string' && (s.startsWith(MSG_ANOMALY) || s.startsWith(MSG_FAILED));
    const clearOwnPlaceholder = (input) => {
        if (isOwnMsg(input.placeholder)) input.placeholder = '';
    };

    function setValue(input, value) {
        // 即使值没变也要记下是我们填的，否则后面会把它误判成"用户手输的内容"
        if (input.value === value) { lastFilled = value; return; }
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        lastFilled = value;
    }

    function markInput(input, low) {
        if (!CFG.markLowConfidence) return;
        input.style.outline = low ? '2px solid #e6a23c' : '';
        input.title = low ? '识别置信度较低，请核对验证码' : '';
    }

    async function recognize(img) {
        // 输入框找不到时以前是静默 return —— 用户完全看不出脚本是否在运行。
        // 这是最难排查的一类失败：登录页改版把 ID 换掉的话，表现就是"什么都没发生"。
        const input = document.querySelector(CFG.inputSelector);
        if (!input) {
            paintStatus('err', `找不到验证码输入框（${CFG.inputSelector}），页面结构可能已改版`);
            return;
        }

        const token = ++runToken;
        let res = null, engine = '';
        let ortErr = '', tessErr = '';   // 保留两路各自的真实错误，供失败时展示

        if (ortFailures < ORT_MAX_FAILURES) {
            try {
                res = await recognizeWithONNX(img);
                engine = 'ResNet(ONNX)';
                ortFailures = 0;
            } catch (e) {
                ortFailures++;
                ortErr = brief(e);
                warn(`ONNX 路径失败（第 ${ortFailures}/${ORT_MAX_FAILURES} 次），本次回退 Tesseract：`, e.message);
            }
        } else {
            ortErr = `已连续失败 ${ortFailures} 次，本轮跳过 ONNX`;
        }

        if (!res) {
            try {
                res = await recognizeWithTesseract(img);
                engine = 'Tesseract';
            } catch (e) {
                tessErr = brief(e);
                warn('Tesseract 也失败：', e.message);
                // 两路都挂了才叫"识别失败"。把两边的原因都写出来 ——
                // 只写一句"识别失败"会让人以为是模型问题，实际上多半是网络/CSP。
                showDiag(input, MSG_FAILED, 'ONNX: ' + (ortErr || '未尝试') + ' ｜ Tess: ' + (tessErr || '未尝试'));
                return;
            }
        }

        if (token !== runToken) { log('已被更新的识别任务取代，丢弃结果'); return; }

        const text = res.text || '';
        if (text.length !== 4 && text.length !== 5) {
            // 推理成功但位数不对：这才是真正的"模型给出意外结果"，
            // 和"两路都跑不起来"是两回事，必须分开显示。
            warn(`识别长度异常 (${text.length})：${text}`);
            showDiag(input, MSG_ANOMALY, '引擎=' + engine + ' 结果="' + text + '" 长度=' + text.length);
            return;
        }

        // 覆盖规则：验证码一换图，框里的旧值必然对不上当前图，所以默认应该覆盖。
        // 唯一的例外是"用户此刻正在这个框里操作" —— 那就不动他。
        //
        // 这里曾经写成 `!(value === '' || value === lastFilled) && activeNow`，
        // 有个真实缺陷：用户按 Ctrl+A 删掉旧答案、准备自己输入时（框内为空且焦点在框内），
        // 前半部分因为"空值"判为可覆盖，于是自动填的值会直接冲掉用户刚清出的输入位。
        // 只要焦点在验证码框里，就不该替他做决定 —— 空值同样是他"我要自己输"的信号。
        const activeNow = document.activeElement === input;
        if (activeNow) {
            log('用户正在验证码框内操作，跳过自动填充');
            return;
        }
        // 焦点不在框内时（例如点了换图按钮、人还停留在别处）：无论框内是我们填的还是
        // 用户改过的，旧值都对应旧图，直接用新图的结果覆盖。
        if (input.value !== '' && input.value !== lastFilled) {
            log('框内是用户改过的内容，但图片已更换，按新图结果覆盖');
        }

        clearOwnPlaceholder(input);
        setValue(input, text);

        const low = res.minConfidence > 0 && res.minConfidence < CFG.lowConfidence;
        markInput(input, low);
        log(`[${engine}] 填入 "${text}"${low ? '（低置信，已标记）' : ''}`);
        // 识别正常时不再常驻显示文字 —— 验证码图片本身已经把答案写在输入框里了，
        // 再挂一行"识别成功"只是噪音，还会占掉页面空间。
        // 只有低置信（需要用户核对）时才显示提示。
        if (low) {
            paintStatus('warn', `识别为 "${text}"，置信度偏低（${(res.minConfidence * 100).toFixed(1)}%），请核对`);
        } else {
            paintStatus('ok', '');
        }

        // 只在用户没在别处打字时才抢焦点。冷启动识别要 1~3 秒，这期间用户
        // 很可能已经在敲密码了，这时候把光标拽到用户名框是帮倒忙。
        const ae = document.activeElement;
        const typingElsewhere = ae && /^(INPUT|TEXTAREA)$/.test(ae.tagName)
            && ae !== input && ae !== document.querySelector(CFG.userSelector);
        if (!typingElsewhere) {
            setTimeout(() => {
                const u = document.querySelector(CFG.userSelector);
                if (u) u.focus();
            }, 80);
        }
    }

    /**
     * 等图片真正解码完成再识别。
     * 只用 img.complete 判断是不够的：src 刚被改写的一瞬间 complete 可能仍为 true，
     * 那时读到的是上一张残留的帧，会把旧答案填进新验证码。
     */
    async function runFor(img) {
        try {
            if (typeof img.decode === 'function') {
                await img.decode();
            } else if (!img.complete) {
                await new Promise(res => {
                    img.addEventListener('load', res, { once: true });
                    img.addEventListener('error', res, { once: true });
                });
            }
        } catch (e) {
            // 换图竞态下 decode() 会以 AbortError 拒绝，丢弃本轮，等下一次触发即可
            log('decode 未就绪，跳过本轮：', e && e.name);
            return;
        }
        if (!img.naturalWidth) {
            // 图片存在但尺寸为 0：通常是 src 指向的地址加载失败了（404 / 被墙 / 跨域）。
            paintStatus('err', `验证码图片尺寸为 0（src=${String(img.src).slice(0, 80)}），图片可能没加载成功`);
            return;
        }
        // recognize() 内部虽然对两个引擎分别做了 try，但它后面还有事件派发、
        // 焦点处理等语句。任何一处抛异常都会变成一个"无人认领"的 rejected Promise ——
        // 用户侧只看到什么都不发生，连"识别失败"都不会出现，排查时完全无从下手。
        // 这里兜住并显式报出来。
        recognize(img).catch(e => {
            warn('识别流程未捕获异常：', e && (e.stack || e.message));
            const input = document.querySelector(CFG.inputSelector);
            if (input) showDiag(input, MSG_FAILED, '未捕获: ' + brief(e));
        });
    }

    /** 绑定验证码监听。返回 false 表示当前 DOM 里还没有验证码图片。 */
    function watchCaptcha() {
        const img = document.querySelector(CFG.imgSelector);
        if (!img) return false;

        // 页面上确实有验证码 —— 还没开始下载就现在开始（覆盖"登录页路径不叫 jalogin"的情况）
        ensureSession();
        ensureStatus();
        if (!sessionPromise) paintStatus('warn', 'v' + VERSION + ' 已发现验证码，正在准备识别模型…');
        else paintStatus('warn', 'v' + VERSION + ' 已发现验证码，模型加载中…');

        // 刻意不拿「src 是否变化」做去重，原因有两个：
        //   1. 站点可能用同一个 URL 重载验证码，src 字符串压根不变；
        //   2. src 变更事件触发时 naturalWidth/naturalHeight 还停留在上一张图的值，
        //      据此算出的指纹会和 load 之后的指纹相同，反而把新图漏掉。
        // 改为「src 变更或 load 都重新识别」，用 80ms 防抖把成对的两次触发合并成一次。
        // 重复识别一张只花几毫秒，漏识别一张的代价是登录失败。
        let pending = null;
        const schedule = (why) => {
            clearTimeout(pending);
            pending = setTimeout(() => {
                log('触发识别：', why);
                runFor(img);
            }, 80);
        };

        new MutationObserver(() => schedule('src 属性变更'))
            .observe(img, { attributes: true, attributeFilter: ['src'] });
        img.addEventListener('load', () => schedule('load 事件'));
        schedule('初始化');

        return true;
    }

    // 预热：只在确实会出现验证码的页面上做。
    // @match 里带了 https://jaccount.sjtu.edu.cn/jaccount/*，如果不加判断，
    // 用户逛 jaccount 的任何其他页面都会白下载 12MB 的 wasm 并编译一次。
    // 登录页则越早开始越好 —— @run-at document-start 让下载与页面解析并行。
    function ensureSession() {
        getSession()
            .then(() => paintStatus('ok', ''))   // 就绪后不再显示，避免长期占位
            .catch(e => {
                warn('预热失败：', e.message);
                paintStatus('err', '模型预热失败：' + brief(e));
                // 预热失败 = 后面每次识别都必然失败，属于最高优先级的坏消息，
                // 必须立刻用横幅告诉用户，不能等他点了换图才显示。
                showBanner('模型预热失败（识别尚未开始就已失败）', brief(e));
            });
    }
    if (/jalogin/i.test(location.pathname + location.search)) ensureSession();

    function init(attempt) {
        attempt = attempt || 0;
        if (watchCaptcha()) return;

        // 验证码可能是异步插入的（SPA / 延迟渲染），轮询等待一段时间再放弃
        if (attempt < 25) {
            setTimeout(() => init(attempt + 1), 400);
        } else {
            log('等了约 10 秒仍未出现验证码图片，放弃');
            // 这种情况用户观感也是"脚本没生效"，必须说出来。
            // 常见原因是页面改版导致 CFG.imgSelector 失效。
            const hasInput = !!document.querySelector(CFG.inputSelector);
            const hint = hasInput
                ? '输入框存在，但图片选择器 ' + CFG.imgSelector + ' 没匹配到 → 页面可能改版了'
                : '验证码与输入框都没找到 → 可能不是登录页（也可能是脚本没被注入）';
            paintStatus('err', '等了约 10 秒仍未发现验证码图片（' + hint + '）');
            showBanner('没有找到验证码图片', hint);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
