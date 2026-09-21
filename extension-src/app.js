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
    const VERSION = '1.0.4';

    const CFG = {
        imgSelector: '#captcha-img',
        inputSelector: '#input-login-captcha',
        userSelector: '#input-login-name',

        // 类别数与官方一致：前 4 个头 26 类，第 5 个头 27 类（多一个 blank 占位）
        numClasses: 26,
        blankIndex: 26,
        charset: 'abcdefghijklmnopqrstuvwxyz',

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

            if (best >= CFG.numClasses) { log(`位置 ${name}: <blank>`); continue; }

            if (probBuf.length < n) probBuf = new Float32Array(n);
            const probs = softmax(data, n, probBuf);
            confidences.push(probs[best]);
            text += CFG.charset[best] || '?';
        }

        return {
            text,
            minConfidence: confidences.length ? Math.min(...confidences) : 0
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
            log(`推理 ${cost.toFixed(1)}ms -> "${res.text}" 最低置信 ${(res.minConfidence * 100).toFixed(1)}%`);
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
        log(`填入 "${text}"`);

        // 识别正常时不常驻显示文字 —— 答案已经填进输入框，再挂一行提示既是噪音又占空间。
        // 只有低置信（需要用户核对）时才展开状态条。
        const low = res.minConfidence > 0 && res.minConfidence < 0.60;
        if (low) {
            paintStatus('warn', `识别为 "${text}"，置信度偏低（${(res.minConfidence * 100).toFixed(1)}%），请核对`);
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
