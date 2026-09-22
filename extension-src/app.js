/*
 * jAccount 验证码自动识别 · 内容脚本（业务逻辑部分）
 *
 * 本文件会被构建脚本拼接到 onnxruntime-web 的 UMD 包后面，合成单个 content.js。
 * ORT 的 UMD 包装是 `... : e.ort = t() }(self, ...)`，在 content script 里 self 就是
 * 隔离世界的全局对象，所以拼接后 `ort` 直接可用，不需要 eval，也不需要注入到页面世界。
 *
 * 与油猴版的差异（为什么可以更简单）：
 *   1. 模型与 wasm 直接打包在扩展里，通过 chrome.runtime.getURL 读取成 ArrayBuffer，
 *      再交给 ort.env.wasm.wasmBinary —— 全程零网络请求，也就不存在"CDN 被墙"的可能。
 *   2. 因此去掉了 Tesseract 兜底。油猴版需要它是因为模型要联网下载；这里模型就在包里，
 *      加载失败只可能是扩展文件损坏，兜底也救不了。
 *   3. 不再需要 Cache Storage —— 资源本来就是本地的。
 */

(function () {
    'use strict';

    // 与 manifest.json 的 version 保持一致。状态条上会显示它，
    // 便于用户确认自己装的到底是哪一版（前几轮无法排除"装的不是最新版"这一可能）。
    const VERSION = '1.0.8';

    const CFG = {
        imgSelector: '#captcha-img',
        inputSelector: '#input-login-captcha',
        userSelector: '#input-login-name',

        // 类别数与官方一致：前 4 个头 26 类，第 5 个头 27 类（多一个 blank 占位）
        numClasses: 26,
        blankIndex: 26,
        charset: 'abcdefghijklmnopqrstuvwxyz',

        // ---- 判据一（v1.0.8 起的**主判据**）：logit 决策间隔 ----
        //
        // 「决策间隔」= 该位 top1 与 top2 的 logit 之差。它比 softmax 概率更直接地度量
        // "离决策边界有多近"，且不受 logit 整体尺度（温度效应）影响。
        //
        // 220 张标注样本实测（调参集/留出集两个独立子集分别验证，均拦下各自全部错误）：
        //
        //   判据                        调参集120          留出集100       220张换图率
        //   min 置信 < 0.999            80.8% 误伤20       89.0% 误伤9      15.5%
        //   min 间隔 < 6                93.3% 误伤 5       94.0% 误伤4       6.4%  ← 选它
        //
        // 即：在同样拦下全部错误的前提下，换图率 15.5% → 6.4%，误伤（把正确答案也换掉）
        // 29 张 → 9 张。300 张新采集样本上换图率 7.0%（旧判据 14.0%）。
        //
        // ⚠ 阈值 6 与实测最大错误间隔（5.799）只差 0.2，余量薄：这个数是在 5 个错误样本上
        //   标定的，未来应随新数据重标定。两个独立子集都通过，是它当前可信度的主要来源。
        lowMargin: 6,

        // ---- 判据二（仍计算并显示，但不再用于触发换图）----
        //
        // 保留它有两个用途：① 状态条/日志里给出一个与旧版可比的数字；
        // ② 排查时能认出"概率高但间隔小"这类样本（模型内部其实很纠结）。
        //
        // 历史教训：这个值以前是 0.60，而实测中全部 5 个错误样本的最低置信
        // 都在 84.39%~99.58% 之间 —— 0.60 的阈值永远够不到，等于该功能从未生效。
        lowConfidence: 0.999,

        // 低置信时自动点"换一张"并重试的最大次数。设为 0 则只标注不重试。
        maxRefreshRetries: 3,

        // 换图控件候选选择器（按优先级依次尝试，命中第一个可见可点的即用）
        refreshSelectors: [
            '.captcha-refresh',
            '#captcha-refresh',
            '.captcha img + a',
            '#captcha-img + a',
            '#captcha-img + span',
            'img[id*="captcha" i] + a',
            'img[id*="captcha" i] + span',
            'a[onclick*="captcha" i]',
            'button[onclick*="captcha" i]'
        ],

        wasmFile: 'assets/ort-wasm-simd.wasm',
        modelFile: 'assets/nn_model.onnx'
    };

    // 调试开关：为了改这一行去编辑扩展文件、再回扩展页点「重新加载」太麻烦。
    // 改成读页面 localStorage —— 在登录页控制台执行一次
    //     localStorage.setItem('jaccountCaptchaDebug','1')
    // 刷新即可看到完整日志；删掉这个键就恢复安静。
    let DEBUG = false;
    try { DEBUG = localStorage.getItem('jaccountCaptchaDebug') === '1'; } catch (e) { }

    const log = (...a) => DEBUG && console.log('[jAccount]', ...a);
    const warn = (...a) => console.warn('[jAccount]', ...a);

    // 该模型把权重既当 initializer 又声明成 graph input，ORT 会刷一屏 W 级警告。
    // 只静音 W 级，E 级错误必须放行 —— 吞掉真错误比刷屏危险得多。
    (function muteOrtNoise() {
        const filter = (orig, ctx) => function (...args) {
            const m = args[0];
            if (typeof m === 'string' && m.includes('[W:onnxruntime')) return;
            return orig.apply(ctx, args);
        };
        console.error = filter(console.error, console);
        console.warn = filter(console.warn, console);
    })();

    /* ------------------------------------------------------------ 资源加载 */

    // 只从扩展包内读，零网络请求。
    //
    // 这里刻意不再保留"CDN 兜底"分支。原因是 MV3 下 content script 的 fetch 受扩展的
    // CSP 约束（默认 connect-src 'self'），跨域取 jsdelivr 会被直接拒绝 —— 于是那个
    // 兜底既救不了场，又会用一条 CSP 报错掩盖掉真正的原因（扩展文件损坏 / 未重新加载）。
    // 直接失败并说清楚，比让用户以为"还会自动走网络"要好。
    function assetUrl(rel) {
        if (typeof chrome === 'undefined' || !chrome.runtime || !chrome.runtime.getURL) {
            throw new Error('不在扩展环境中运行（chrome.runtime 不可用）');
        }
        return chrome.runtime.getURL(rel);
    }

    async function loadAsset(rel) {
        const url = assetUrl(rel);
        let r;
        try {
            r = await fetch(url);
        } catch (e) {
            throw new Error(`读取扩展内资源失败：${rel}（${e.message}）。`
                + '多半是扩展文件缺失或更新后未重新加载，请在扩展管理页点"重新加载"。');
        }
        if (!r.ok) throw new Error(`读取扩展内资源失败：${rel}（HTTP ${r.status}）`);
        const buf = await r.arrayBuffer();
        if (!buf.byteLength) throw new Error(`扩展内资源为空：${rel}`);
        log('资源已读取', rel, buf.byteLength, 'bytes');
        return new Uint8Array(buf);
    }

    /* ------------------------------------------------------- 预处理 / 后处理 */

    const IN_W = 110, IN_H = 40;

    /**
     * 与官方 ocr.py 严格对齐：Image.convert("L") 做 ITU-R 601-2 灰度，
     * 再按 LUT [0]*156 + [1]*100 判定（灰度 >= 156 记 1，否则 0）。
     */
    function preprocess(img) {
        const canvas = document.createElement('canvas');
        canvas.width = IN_W;
        canvas.height = IN_H;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, IN_W, IN_H);

        const { data } = ctx.getImageData(0, 0, IN_W, IN_H);
        const input = new Float32Array(IN_W * IN_H);
        for (let i = 0; i < input.length; i++) {
            const o = i * 4;
            // 必须先四舍五入再比阈值：PIL 的 convert("L") 是整数运算 + 四舍五入，
            // 直接拿浮点比 >= 156 会在 220 张样张里产生 106 个像素的判定差异。
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
     *   for tensor in tensors: asc = argmax(tensor, 1); if asc < 26: captcha += chr(...)
     * 即第 5 个头 argmax 落到 26 号 blank 类时跳过该位 -> 得到 4 位验证码。
     */
    function postprocess(session, outputMap) {
        const rawNames = Array.isArray(session.outputNames) && session.outputNames.length
            ? session.outputNames
            : Object.keys(outputMap);
        const names = rawNames.slice().sort((a, b) => {
            const na = Number(a), nb = Number(b);
            return (Number.isFinite(na) && Number.isFinite(nb)) ? na - nb : 0;
        });

        let text = '';
        const confidences = [];
        const margins = [];        // 每一位的 top1 与 top2 的 logit 之差（决策间隔）
        let probBuf = new Float32Array(32);

        for (const name of names) {
            const tensor = outputMap[name];
            if (!tensor) { warn('缺少输出张量', name); continue; }

            const data = tensor.data;
            const n = (tensor.dims && tensor.dims.length === 2)
                ? tensor.dims[1]
                : Math.min(data.length, CFG.blankIndex + 1);

            // 一次扫描同时取 top1 / top2：top1 决定字符，top1-top2 决定"离边界有多近"。
            let best = 0, bestVal = -Infinity, secondVal = -Infinity;
            for (let i = 0; i < n; i++) {
                const v = data[i];
                if (v > bestVal) { secondVal = bestVal; bestVal = v; best = i; }
                else if (v > secondVal) { secondVal = v; }
            }

            if (best >= CFG.numClasses) { log(`位置 ${name}: <blank>`); continue; }

            if (probBuf.length < n) probBuf = new Float32Array(n);
            const probs = softmax(data, n, probBuf);
            confidences.push(probs[best]);
            margins.push(bestVal - secondVal);
            text += CFG.charset[best] || '?';
            log(`位置 ${name}: ${CFG.charset[best]}  间隔=${(bestVal - secondVal).toFixed(2)}`);
        }

        return {
            text,
            confidences,
            minConfidence: confidences.length ? Math.min(...confidences) : 0,
            margins,
            minMargin: margins.length ? Math.min(...margins) : Infinity
        };
    }

    /* ------------------------------------------------------------ ONNX 引擎 */

    let sessionPromise = null;

    function initOrt() {
        const g = typeof self !== 'undefined' ? self : window;
        const ort = g.ort;
        if (!ort || !ort.env || !ort.env.wasm) {
            // 内联的 ORT 是 UMD 包，正常应该挂到 self.ort。
            // 若没挂上，通常是隔离世界里意外出现了 define（AMD）或 exports ——
            // UMD 头会优先走那两条分支，于是 e.ort = t() 永不执行。
            // content script 的隔离世界一般不会有这些，所以这只作为兜底提示；
            // 用户脚本版（走 <script src>，在页面世界里执行）才是这个问题的高发区。
            const why = (typeof define === 'function' && define.amd) ? '检测到 define（AMD 加载器）'
                : (typeof exports === 'object') ? '检测到 exports'
                    : '原因不明';
            throw new Error('内联的 onnxruntime-web 未挂载到全局（' + why + '）');
        }
        // 单线程：页面不是跨域隔离环境，开多线程只会告警并回退
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
        ort.env.wasm.proxy = false;
        return ort;
    }

    function getSession() {
        if (sessionPromise) return sessionPromise;
        sessionPromise = (async () => {
            const ort = initOrt();

            const [wasm, model] = await Promise.all([
                loadAsset(CFG.wasmFile),
                loadAsset(CFG.modelFile)
            ]);

            // 关键：直接喂 wasm 字节，ORT 就完全不需要解析路径、也不发任何请求。
            // （content script 里 document.currentScript 是 null，靠它推导 wasm 路径本来就是死路。）
            ort.env.wasm.wasmBinary = wasm;
            log(`wasm ${(wasm.byteLength / 1048576).toFixed(2)}MB / 模型 ${(model.byteLength / 1048576).toFixed(2)}MB`);

            const session = await ort.InferenceSession.create(model, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            log('模型加载完成，输入:', session.inputNames, '输出:', session.outputNames);
            return session;
        })().catch(e => { sessionPromise = null; throw e; });
        return sessionPromise;
    }

    async function recognizeWithONNX(img) {
        // 分步标注：扩展版不碰网络，失败几乎只有三种成因 ——
        // 扩展内资源没读到、wasm 没起来、推理本身出错。带上步骤前缀后，
        // 用户复制一句提示就能直接区分，不用再来回猜。
        let step = '初始化 ORT';
        try {
            const ort = initOrt();
            step = '读取模型';
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
            log(`推理 ${cost.toFixed(1)}ms -> "${res.text}" 最小间隔 ${res.minMargin.toFixed(2)} 最低置信 ${(res.minConfidence * 100).toFixed(1)}%`);
            return res;
        } catch (e) {
            // 不直接改 e.message（某些 host 对象上只读），统一包一层新错误
            const wrapped = new Error('[' + step + '] ' + brief(e));
            wrapped.cause = e;
            throw wrapped;
        }
    }

    /* ------------------------------------------------------------- 主流程 */

    let lastFilled = null;   // 记录我们自动填入的值
    let runToken = 0;

    const MSG_FAILED = '识别失败，请手动输入';
    const MSG_ANOMALY = '识别异常，请手动输入';
    // 前缀匹配而非全等：诊断版本会在文案后追加具体原因，
    // 全等匹配会清不掉，识别成功后脏文案会残留在输入框里。
    const isOwnMsg = (s) => typeof s === 'string' && (s.startsWith(MSG_FAILED) || s.startsWith(MSG_ANOMALY));
    const clearOwnPlaceholder = (input) => {
        if (isOwnMsg(input.placeholder)) input.placeholder = '';
    };

    /** 把异常压成一小段可直接读的文本（过长会被截断） */
    const brief = (e) => {
        if (!e) return '?';
        const m = (e.message || String(e)).replace(/\s+/g, ' ').trim();
        return m.length > 120 ? m.slice(0, 120) + '…' : m;
    };
    /** 把失败原因写进 placeholder + title，让用户不开控制台也能看到问题在哪 */
    const showDiag = (input, base, detail) => {
        input.placeholder = base + ' [' + detail + ']';
        try { input.title = detail; } catch (e) { /* 只读属性，忽略 */ }
        warn(base + ' → ' + detail);
    };

    /* ------------------------------------------------ 页面状态条（排障用）
     *
     * 与油猴版保持一致的实现，一个刻意的设计约束：
     * **只插入验证码图片下方的文档流内，绝不使用 position 定位。**
     * 油猴版 4.4.3 曾用 position:fixed 做失败横幅，结果把验证码图片本身盖住了
     * （用户反馈"有一些遮挡验证码图片"）—— 排障信息挡住待排查的对象。
     * 这里从一开始就不给那个可能性：没有 position 就没有遮挡。
     *
     * 默认 display:none，只有需要显示时才展开；识别成功时清空并折叠，
     * 正常情况下页面上不会多出任何一行字。
     */
    const STATUS_ID = 'jaccount-recognizer-status';
    let statusEl = null;

    function ensureStatus() {
        if (statusEl && statusEl.isConnected) return statusEl;
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

    function paintStatus(kind, text) {
        const el = statusEl || ensureStatus();
        if (!el) return;
        if (!text) {
            // 空文本 = 无话可说。整个清空并折叠，连前缀都不留。
            el.textContent = '';
            el.style.display = 'none';
            return;
        }
        const color = kind === 'err' ? '#f56c6c' : kind === 'warn' ? '#e6a23c' : '#909399';
        const mark = kind === 'err' ? '✗ ' : kind === 'warn' ? '! ' : '';
        el.style.color = color;
        el.textContent = '[验证码识别] ' + mark + text;
        el.style.display = '';
    }

    function setValue(input, value) {
        if (input.value === value) { lastFilled = value; return; }
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        lastFilled = value;
    }

    /* ------------------------------------------- 低置信自动换图重试 */

    function markInput(input, low) {
        input.style.outline = low ? '2px solid #e6a23c' : '';
        // title 只在**确实需要改**时才写。
        //
        // 原实现是无条件 `input.title = low ? '…' : ''`，看似无害，实际会踩掉
        // showDiag() 刚写进去的诊断信息：失败一轮（title="[读取模型] ..."）、
        // 随后成功一轮，title 就被这行清成空字符串。用户鼠标悬上去只剩空白，
        // 而那句诊断恰恰是排查问题唯一能拿到的线索 —— 我们为此专门加过 showDiag。
        // 现在低置信才占用 title；高置信时不碰它，让诊断信息活到用户看见为止。
        if (low) input.title = '识别结果决策余量不足，请核对验证码';
        else if (input.title === '识别结果决策余量不足，请核对验证码') input.title = '';
    }

    /**
     * 找一个"可见且可点"的换图控件。
     *
     * 为什么不用固定 id：jAccount 的刷新控件（历史版本可能是 <a>、<span> 或
     * 带 onclick 的图片）在多次改版里换过实现，写死一个选择器会在改版后静默失效 ——
     * 表现是"低置信了却从不换图"，且没有任何报错，极难排查。
     * 因此改为按候选列表依次试，并要求元素确实可见（有尺寸）。
     */
    function findRefreshButton() {
        for (const sel of CFG.refreshSelectors) {
            let nodes;
            try {
                nodes = document.querySelectorAll(sel);
            } catch (e) {
                continue;   // 选择器语法不被支持时跳过，不影响其它候选
            }
            for (const el of nodes) {
                if (!el || el.offsetWidth <= 0 || el.offsetHeight <= 0) continue;
                return el;
            }
        }
        return null;
    }

    /**
     * 触发换图并等待新图就绪。
     * 优先用 img.decode()（保证像素可读、不会读到上一帧），再用短延迟兜底
     * （换图若命中缓存，load 事件可能同步触发，只靠事件监听会等不到）。
     * 返回 false 表示没找到换图控件。
     */
    /* ---------------------------------------------------- 换图期间抑制观察者
     *
     * 这是一个**真实存在的无限循环**，由 race_retry_observer.js 抓到：
     *
     *   recognize() 低置信
     *     → refreshCaptcha() 点击换图按钮
     *       → 站点改写 <img> 的 src
     *         → MutationObserver 看到 src 变更 → schedule('src 属性变更')
     *           → 80ms 后 runFor() → recognize()   ← 全新一轮，全新 runToken
     *             → 又是低置信 → 又点换图 → …… 永不终止
     *
     * 实测放大速率是**线性且不收敛**的，稳定在约 33 次换图/秒：
     *   秒:  1   2   3    4    5    6
     *   累计: 33  66  99  132  165  198
     * 也就是说页面会以恒定的节奏无限点"换一张"，直到用户关掉标签页。
     *
     * 为什么 runToken 拦不住它：token 只在**同一轮**里防止旧结果被采用。
     * 而这条回路每一圈都是全新的 recognize()，进来就 ++runToken 把自己变成最新，
     * 于是每一圈的 token 校验都通过。token 管的是"并行分叉"，管不了"自我激励"。
     *
     * 为什么 80ms 防抖也拦不住：防抖只合并"同一批"触发；这里每一圈都是
     * 上一圈**结算完之后**才产生的新变更，时间上完全错开，防抖窗口永远落空。
     *
     * 修法：加一个显式的重入闸。由我们自己发起的换图，在它引发的
     * src 变更/load 事件落地之前，给观察者挂一个"暂时别响应"的标志位；
     * 重试循环结束后再放开。这样主动重试仍受 maxRefreshRetries 约束，
     * 而观察者的反馈回路被彻底切断。
     */
    let suppressObserveUntil = 0;   // 时间戳：在此之前忽略观察者触发的识别

    /**
     * 把「观察者静默」向后延长 ms 毫秒。
     * 用时间戳而不是布尔量：换图后的 load 事件可能晚于我们 resolve 的时刻才到，
     * 布尔量在重试循环退出时就复位了，那个迟到的 load 仍会点燃下一圈。
     */
    function suppressObserver(ms) {
        const until = Date.now() + ms;
        if (until > suppressObserveUntil) suppressObserveUntil = until;
    }
    /**
     * 提前解除静默。
     *
     * 光靠"到点自动失效"是不够的：闸门是 2000ms，而一次重试循环往往几百毫秒就结束了。
     * 剩下的那一秒多里用户如果自己点了"换一张"，会被静默忽略 —— 表现为
     * "我点了换图，脚本没反应"。观察者的静默只该覆盖重试进行中的那段时间。
     *
     * 由调用方保证只在"本轮确实换过图"时才调用，避免误解除别人的闸门。
     */
    function releaseObserver() {
        suppressObserveUntil = 0;
    }
    const observerSuppressed = () => Date.now() < suppressObserveUntil;

    async function refreshCaptcha(img) {
        const btn = findRefreshButton();
        if (!btn) {
            log('未找到换图控件，跳过自动换图');
            return false;
        }

        // 记录点击前的 src —— 后面要靠它判断"站点到底有没有真的把图换掉"。
        const oldSrc = img.getAttribute('src');

        // 关闸必须在 click() **之前**。站点多半是同步改 src 的：
        // 若放在 click 之后，MutationObserver 回调早已排进微任务队列，闸门白设。
        // 必须先切断反馈回路，再产生会触发它的那个动作。
        suppressObserver(2000);

        try {
            btn.click();
        } catch (e) {
            warn('换图控件点击失败：', e && e.message);
            return false;
        }
        log('已触发换图');

        // 等新图就绪。
        //
        // 1.0.6 修正（之前这段的 decode 分支其实是死代码）：
        // 原写法是
        //     const check = () => {
        //         if (img.getAttribute('src') === oldSrc && !img.complete) return;  // ← 提前返回
        //         if (typeof img.decode === 'function') { img.decode().then(...) }
        //     };
        // 问题在于站点用**同一个 URL** 重载验证码时（很常见，靠后端随机或查询串区分），
        // src 字符串根本不变、而新图尚未加载完 —— 于是每次都被那行提前返回挡住，
        // 下面的 decode() 永远执行不到，只能一路干等到 1200ms 超时。
        // 结果：每次换图固定浪费 1.2 秒，且这 1.2 秒里确实可能读到上一帧。
        //
        // 正确做法是不要把"src 变了"当成调用 decode 的前置条件 ——
        // decode() 本来就是"等这张图解码完"的语义，src 变没变它都能正确工作。
        // 它只在两种情况下会拒绝：解码失败，或被更新的 src 取代（AbortError），
        // 两种都该直接放行走超时兜底，而不是继续等。
        // 换图有两种实现方式，等待策略必须同时覆盖：
        //   * 同步改 src（多数站点）：click() 返回时 src 已经是新的；
        //   * 异步改 src（ajax 取回新图）：click() 返回时 src **还是旧的**。
        //
        // 关键陷阱：img.decode() 在「src 没变、且旧图早已解码完」时会**立即兑现**。
        // 于是在异步站点上，click() 之后立刻 decode() 会秒回 —— 我们以为新图就位了，
        // 实际识别的仍是旧的那张；连续换图 3 次全是同一张旧图，白打三次站点接口。
        // 所以「src 真的变了」必须是等待解码的前置条件。
        // 兑现值 = src 是否真的变了。没变就说明换图没生效：浏览器不会为同一个
        // URL 重新加载，再点几次也是同一张旧图，继续重试纯属空转。
        const didChange = await new Promise(resolve => {
            let done = false;
            let poll = null;
            let sawChange = false;
            const finish = () => {
                if (done) return;
                done = true;
                clearTimeout(t);
                if (poll) clearInterval(poll);
                resolve(sawChange);
            };
            // 兜底：最长等 1500ms，超时就放行 —— 宁可拿旧结果也不要把用户卡住
            const t = setTimeout(finish, 1500);

            let decodeTries = 0;
            const decodeNow = () => {
                if (done) return;
                if (typeof img.decode === 'function') {
                    img.decode().then(finish).catch(() => {
                        // decode() 以 AbortError 拒绝的含义是「这次解码被更新的 src 取消」，
                        // 也就是**新图还在路上**。把它当成"图片已就绪"直接放行，
                        // 等于在尚未加载完的图上开跑，必须再等一次。
                        // （早先这里写的是 .catch(finish)，AbortError 一出现就立刻放行。）
                        if (!done && decodeTries++ < 20) setTimeout(decodeNow, 25);
                    });
                } else if (img.complete && img.naturalWidth > 0) {
                    // 老浏览器无 decode()：已加载完就直接走，否则等事件或超时
                    finish();
                } else {
                    img.addEventListener('load', finish, { once: true });
                    img.addEventListener('error', finish, { once: true });
                }
            };

            const changed = () => img.getAttribute('src') !== oldSrc;
            if (changed()) {
                sawChange = true;
                decodeNow();      // 同步改 src 的站点：新图已在路上，直接等它解码完
            } else {
                // 异步站点：先等 src 变。这里用轮询而非 MutationObserver ——
                // 要的是"一定能到"的简单可靠，30ms 粒度足够，总时长由上面的超时兜底。
                poll = setInterval(() => {
                    if (changed()) { sawChange = true; clearInterval(poll); poll = null; decodeNow(); }
                }, 30);
            }
        });
        if (!didChange) {
            log('换图后 src 未发生变化，判定换图未生效（不再空转重试）');
            return false;
        }
        return true;
    }

    async function recognize(img) {
        const input = document.querySelector(CFG.inputSelector);
        if (!input) {
            // 输入框找不到时以前是静默 return —— 用户完全看不出脚本是否在运行。
            // 这是最难排查的一类失败：登录页改版把 ID 换掉的话，表现就是"什么都没发生"。
            paintStatus('err', `找不到验证码输入框（${CFG.inputSelector}），页面结构可能已改版`);
            return;
        }

        const token = ++runToken;
        let res = null;
        try {
            res = await recognizeWithONNX(img);
        } catch (e) {
            // 扩展版只有 ONNX 一条路（不依赖任何 CDN），所以这里失败就是**本地资源**问题：
            // 扩展文件缺失、更新后没重新加载、或 wasm 起不来。
            // 把这层判断直接写进提示，用户就不用去猜"是不是网不好"。
            warn('识别失败：', e.message);
            showDiag(input, MSG_FAILED, brief(e));
            paintStatus('err', '识别失败：' + brief(e));
            return;
        }

        if (token !== runToken) { log('已被更新的识别任务取代，丢弃结果'); return; }

        // ---- 决策余量不足时的自动换图重试 ----
        //
        // 实测依据（220 张标注样本）：识别错误全部集中在第 4 个字符位，
        // 且这些错误样本的 top1 置信高达 84%~99.6%（模型"自信地错"）——
        // 概率阈值与正确样本完全重叠，单靠它筛不干净。
        //
        // v1.0.8 改用「决策间隔」（top1 与 top2 的 logit 之差）作判据：
        // 同样拦下全部错误的前提下，换图率 15.5% → 6.4%（详见 CFG.lowMargin 的实测表）。
        const isLow = (r) => r && r.minMargin < CFG.lowMargin;
        const lenOk = (t) => t.length === 4 || t.length === 5;

        let retries = 0;
        let refreshed = false;   // 本轮是否真的换过图（决定要不要提前解除观察者闸门）

        try {
            while (isLow(res) && lenOk(res.text || '') && retries < CFG.maxRefreshRetries) {
                if (token !== runToken) { log('重试期间任务已被取代，放弃'); return; }

                retries++;
                paintStatus('warn',
                    `决策余量不足（间隔 ${res.minMargin.toFixed(2)}），正在换图重试 ${retries}/${CFG.maxRefreshRetries}…`);

                const clicked = await refreshCaptcha(img);
                if (!clicked) {
                    // 没找到换图控件，或点了但 src 压根没变（换图未生效）。
                    // 两种情况下屏幕上都还是原来那张图，当前结果仍然对得上，
                    // 直接走低置信标注即可，不必接着空转。
                    log('换图未成功（无控件或 src 未变），放弃重试');
                    break;
                }
                refreshed = true;
                if (token !== runToken) { log('换图后任务已被取代，放弃'); return; }

                // 换图成功的一刻，屏幕上已经不是刚才那张图了 ——
                // **换图之前得到的任何结果都随之作废**，包括"置信更高的那次"。
                //
                // 这里曾经保留 best（历史最高置信）作为重试用尽后的兜底，是个真实缺陷：
                // 换图 3 次后屏幕上是第 4 张图，若第 2 张恰好置信最高，填入的就是第 2 张的
                // 答案 —— 与用户眼前的验证码不符，必然登录失败。
                // 置信度是用来判断"这一张能不能信"的，不是跨图片比较的分数。
                try {
                    const next = await recognizeWithONNX(img);
                    if (!lenOk(next.text || '')) {
                        log('换图后重新识别得到长度异常结果；旧结果已随换图作废');
                        res = null;
                        break;
                    }
                    res = next;
                    log(`重试 ${retries} 结果 "${next.text}" 间隔 ${next.minMargin.toFixed(2)} 置信 ${(next.minConfidence * 100).toFixed(2)}%`);
                } catch (e) {
                    warn('换图后重新识别失败：', e && e.message);
                    res = null;
                    break;
                }
            }
        } finally {
            // 闸门只在重试进行中才需要。循环一结束就解除 ——
            // 否则用户紧接着手动点"换一张"会被静默忽略（表现为脚本没反应）。
            if (refreshed) releaseObserver();
        }

        if (!res) {
            // 换图之后没拿到可信结果：屏幕上的图已经不对应我们手里任何一份答案，
            // 这时候填任何东西都是错的 —— 宁可让用户自己看一眼。
            showDiag(input, MSG_FAILED, '换图后未能重新识别，请手动输入');
            return;
        }

        if (token !== runToken) { log('已被更新的识别任务取代，丢弃结果'); return; }

        const text = res.text || '';
        if (text.length !== 4 && text.length !== 5) {
            warn(`识别长度异常 (${text.length})：${text}`);
            showDiag(input, MSG_ANOMALY, '结果="' + text + '" 长度=' + text.length);
            paintStatus('warn', `识别结果位数异常（${text.length} 位）："${text}"`);
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
        // 焦点不在框内时：只有"空"或"还是我们上次填的值"才覆盖，
        // 否则是用户已经改过的内容（例如点了换图按钮、人还在别处），这时也应覆盖，
        // 因为旧值必然对应旧图。
        if (input.value !== '' && input.value !== lastFilled) {
            log('框内是用户改过的内容，但图片已更换，按新图结果覆盖');
        }

        clearOwnPlaceholder(input);
        setValue(input, text);
        log(`填入 "${text}"` + (retries ? ` 重试${retries}次` : ''));

        // 识别正常时不常驻显示文字 —— 答案已经填进输入框，再挂一行提示既是噪音又占空间。
        // 只有低置信（需要用户核对）时才展开状态条。
        const low = isLow(res);
        markInput(input, low);
        if (low) {
            const tried = retries ? `（已换图重试 ${retries} 次）` : '';
            paintStatus('warn',
                `识别为 "${text}"，决策余量仍不足（间隔 ${res.minMargin.toFixed(2)}）${tried}，请核对或手动换图`);
        } else {
            paintStatus('ok', '');
        }

        // 只在用户没在别处打字时才抢焦点
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
     * 等图片真正解码完成再识别。只用 img.complete 判断不够：src 刚改写时它可能仍为 true，
     * 那时读到的是上一张残留的帧。
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
            log('decode 未就绪，跳过本轮：', e && e.name);
            return;
        }
        if (!img.naturalWidth) return;
        recognize(img);
    }

    function watchCaptcha() {
        const img = document.querySelector(CFG.imgSelector);
        if (!img) return false;

        // 页面里确实有验证码，提前把模型热起来（本地读取，很快）
        ensureStatus();
        paintStatus('warn', 'v' + VERSION + ' 已发现验证码，模型加载中…');
        getSession()
            .then(() => paintStatus('ok', ''))
            .catch(e => {
                warn('预热失败：', e.message);
                paintStatus('err', '模型预热失败：' + brief(e));
            });

        // 不拿「src 是否变化」做去重：站点可能用同一 URL 重载（src 字符串不变），
        // 且 src 变更事件触发时尺寸还是上一张图的值，指纹会和 load 后相同，反而漏图。
        let pending = null;
        const schedule = (why) => {
            // 闸门：我们自己发起的换图会引发 src 变更/load，这些是**已知的、正在被
            // 重试循环处理**的事件，不能再点燃一轮独立识别 —— 否则就是那个
            // 33 次/秒的无限换图循环（详见 suppressObserver 的注释）。
            // 注意判断放在这里而不是 observer 回调里：load 事件也走同一个 schedule。
            if (observerSuppressed()) {
                log('观察者静默中，忽略触发：', why);
                return;
            }
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

    function init(attempt) {
        attempt = attempt || 0;
        if (watchCaptcha()) return;
        // 验证码可能是异步插入的，轮询等待一段时间再放弃
        if (attempt < 25) {
            setTimeout(() => init(attempt + 1), 400);
        } else {
            log('等了约 10 秒仍未出现验证码图片，放弃');
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
