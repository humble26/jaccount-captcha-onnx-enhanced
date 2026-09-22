// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 缺陷狩猎 · 第三轮：针对「低置信换图重试」这条新链路的**边界语义**做复现。
 *
 * 前两轮修的是「能不能跑」，这一轮问的是「跑出来的结果对不对」：
 *   J 组  换图之后，先前那次识别的结果还算不算数？（填入值必须对应屏幕上的图）
 *   K 组  站点异步换图时，等待逻辑会不会在旧图上就兑现？
 *   L 组  decode() 以 AbortError 拒绝时，是不是被当成了「图片已就绪」？
 *   M 组  换图闸门会不会误伤用户自己的换图操作？
 *
 * 与 bug_hunt_flow.js 一样：被测代码全部从**真实源码**按大括号配平抽取，
 * 不手抄。实现漂移会让测试因符号缺失而报错，而不是继续测一份过时拷贝。
 *
 * 用法: node bug_hunt_v2.js <monkey|ext>
 */
const fs = require('fs');

const WS = _REPO;
const TARGET = process.argv[2] || 'monkey';
const FILE = TARGET === 'monkey'
    ? WS + '\\jaccount-captcha-onnx-enhanced.user.js'
    : WS + '\\extension-src\\app.js';
const raw = fs.readFileSync(FILE, 'utf8');

/* ------------------------------------------------------------ 源码抽取 */

function extractBalanced(src, sig) {
    const i = src.indexOf(sig);
    if (i < 0) throw new Error('找不到定义: ' + sig);
    let depth = 0, started = false;
    for (let k = i; k < src.length; k++) {
        const c = src[k];
        if (c === '{') { depth++; started = true; }
        else if (c === '}') {
            depth--;
            if (started && depth === 0) return src.slice(i, k + 1);
        }
    }
    throw new Error('大括号未配平: ' + sig);
}

function extractArrow(src, sig) {
    const i = src.indexOf(sig);
    if (i < 0) throw new Error('找不到定义: ' + sig);
    let depth = 0, started = false;
    for (let k = src.indexOf('{', i); k < src.length; k++) {
        if (src[k] === '{') { depth++; started = true; }
        else if (src[k] === '}') { depth--; if (started && depth === 0) return src.slice(i, k + 1) + ';'; }
    }
    throw new Error('大括号不配平: ' + sig);
}

function extractBlock(src, sig, endMarker) {
    const i = src.indexOf(sig);
    if (i < 0) throw new Error('找不到定义: ' + sig);
    const j = src.indexOf(endMarker, i);
    if (j < 0) throw new Error('找不到终点: ' + endMarker);
    return src.slice(i, j + endMarker.length);
}

const CFG_SRC = extractBlock(raw, 'const CFG = {', '\n    };');
const GATE_SRC = (function () {
    const i = raw.indexOf('let suppressObserveUntil = 0;');
    if (i < 0) throw new Error('源码里找不到换图闸门（suppressObserveUntil）');
    const j = raw.indexOf('async function refreshCaptcha(img) {', i);
    if (j < 0) throw new Error('找不到 refreshCaptcha 起点，无法切出闸门段');
    return raw.slice(i, j);
})();
const FIND_SRC = extractBlock(raw, 'function findRefreshButton() {', '\n    }');
const REFRESH_SRC = extractBlock(raw, 'async function refreshCaptcha(img) {', '\n    }');
const REC_SRC = extractBalanced(raw, 'async function recognize(img) {');
const SCHED_SRC = extractArrow(raw, 'const schedule = (why) => {');
const MARK_SRC = extractBalanced(raw, 'function markInput(input, low) {');

/* ------------------------------------------------------------ 桩环境 */

class El {
    constructor(tag) {
        this.tagName = (tag || 'div').toUpperCase();
        this._ls = {}; this.attrs = {}; this.style = {};
        this.textContent = ''; this.isConnected = true;
        this.offsetWidth = 30; this.offsetHeight = 30;
        this.value = ''; this.placeholder = ''; this.title = '';
        this.decode = null;             // 由用例按需装成函数
        this.complete = true;
        this.naturalWidth = 110;
    }
    addEventListener(t, fn) { (this._ls[t] = this._ls[t] || []).push(fn); }
    removeEventListener() { }
    dispatchEvent(e) { (this._ls[e.type] || []).forEach(f => f.call(this, e)); return true; }
    setAttribute(k, v) { this.attrs[k] = v; if (k === 'src') this.src = v; }
    getAttribute(k) { return this.attrs[k]; }
    focus() { doc.activeElement = this; }
    click() { this.clicked = (this.clicked || 0) + 1; }
}

const doc = {
    readyState: 'complete', activeElement: null, _els: {}, _qsa: {},
    createElement: (t) => new El(t),
    querySelector(sel) { return this._els[sel] || null; },
    querySelectorAll(sel) { return this._qsa[sel] || []; },
    addEventListener() { },
    documentElement: new El('html'),
    body: new El('body'),
    getElementById() { return null; }
};

const img = new El('img');
const input = new El('input');
const user = new El('input');
const refreshBtn = new El('a');
doc._els['#captcha-img'] = img;
doc._els['#input-login-captcha'] = input;
doc._els['#input-login-name'] = user;
doc._qsa['.captcha-refresh'] = [refreshBtn];

const logBuf = [], warnBuf = [], statusLog = [], runForCalls = [];
const log = (...a) => logBuf.push(a.map(String).join(' '));
const warn = (...a) => warnBuf.push(a.map(String).join(' '));
const paintStatus = (kind, text) => statusLog.push({ kind, text });
const brief = (e) => String((e && e.message) || e || '?').slice(0, 120);
const showDiag = (inp, base, detail) => { inp.placeholder = base + ' [' + detail + ']'; };
const MSG_ANOMALY = '识别异常，请手动输入';
const MSG_FAILED = '识别失败，请手动输入';
const isOwnMsg = (s) => typeof s === 'string' && (s.startsWith(MSG_ANOMALY) || s.startsWith(MSG_FAILED));
const clearOwnPlaceholder = (inp) => { if (isOwnMsg(inp.placeholder)) inp.placeholder = ''; };

const RT = { v: 0 };     // runToken
const LF = { v: null };  // lastFilled
const setValue = (inp, value) => {
    if (inp.value === value) { LF.v = value; return; }
    inp.value = value;
    inp.dispatchEvent(new Event('input', { bubbles: true }));
    LF.v = value;
};
const markInput = new Function('CFG', MARK_SRC + '; return markInput;')({ markLowConfidence: true });

let recognizeOnceImpl = async () => { throw new Error('未设置 recognizeOnceImpl'); };

// v4.5.2 起生产判据是「决策间隔」minMargin < CFG.lowMargin(6)，不再是 softmax 置信度。
// 各用例的桩仍按置信度书写（0.80 = 低、0.99995 = 高），这里按同一分界线换算成间隔，
// 既有用例语义不变：
//   conf < 0.999（原本触发换图）-> margin = 3（同样触发）
//   否则                       -> margin = 12（同样不触发）
const marginFor = (conf) => (conf < 0.999 ? 3 : 12);
const recognizeOnce = async (...a) => {
    const r = await recognizeOnceImpl(...a);
    if (r && r.minMargin === undefined) r.minMargin = marginFor(r.minConfidence);
    return r;
};

/* ------------------------------------------------------------ 组装 */

const factory = new Function(
    'document', 'log', 'warn', 'brief', 'paintStatus', 'showDiag',
    'clearOwnPlaceholder', 'setValue', 'markInput', 'MSG_FAILED', 'MSG_ANOMALY',
    'RT', 'LF', 'recognizeOnce', 'runFor', 'statusLog', 'img',
    `
    ${CFG_SRC}
    ${GATE_SRC}
    ${FIND_SRC}
    ${REFRESH_SRC}
    let pending = null;
    ${SCHED_SRC}
    const hideBanner = () => {};
    const showBanner = () => {};
    ${REC_SRC
        .replace(/\brunToken\b/g, 'RT.v')
        .replace(/\blastFilled\b/g, 'LF.v')
        .replace(/\brecognizeWithONNX\b/g, 'recognizeOnce')}
    return { recognize: recognize, refreshCaptcha: refreshCaptcha, schedule: schedule,
             suppressObserver: suppressObserver, observerSuppressed: observerSuppressed };
    `
);
const api = factory(
    doc, log, warn, brief, paintStatus, showDiag,
    clearOwnPlaceholder, setValue, markInput, MSG_FAILED, MSG_ANOMALY,
    RT, LF, recognizeOnce,
    (im) => { runForCalls.push(im.getAttribute('src')); },
    statusLog, img
);

/* ------------------------------------------------------------ 用例框架 */

let pass = 0, fail = 0;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function check(name, fn, expect) {
    logBuf.length = 0; warnBuf.length = 0; statusLog.length = 0; runForCalls.length = 0;
    input.value = ''; input.placeholder = ''; input.style = {}; input.title = '';
    user.value = ''; doc.activeElement = null;
    LF.v = null; RT.v = 0;
    refreshBtn.clicked = 0;
    doc._qsa['.captcha-refresh'] = [refreshBtn];
    img.attrs = {}; img.src = undefined; img.complete = true; img.naturalWidth = 110;
    img.decode = null; img._ls = {};

    let got, err = null;
    try { got = await fn(); } catch (e) { err = e; }
    let ok = false, why = '';
    if (err) { why = '抛出异常: ' + (err.message || err); }
    else {
        try { ok = expect(got); } catch (e) { why = '断言异常: ' + e.message; }
    }
    if (ok) { pass++; console.log('  ✓ ' + name); }
    else {
        fail++;
        console.log('  ✗ ' + name + (why ? '  —— ' + why : ''));
        console.log('      实得: ' + JSON.stringify(got === undefined ? null : got));
        if (logBuf.length) console.log('      log: ' + logBuf.slice(-6).join(' | '));
    }
}

function section(t) { console.log('\n[' + t + ']'); }

(async function main() {
    console.log('=== 缺陷狩猎 v2 · ' + TARGET + ' ===\n');

    /* ============ J 组：换图之后，旧结果还算不算数 ============ */
    section('J 换图后的结果归属（填入值必须对应屏幕上的那张图）');

    // 每次换图都把 src 推进一版；recognizeOnce 返回"当前 src"对应的答案。
    // 这样就能追踪：最终填入的答案，到底属于屏幕上那张图，还是早已被换掉的旧图。
    //
    // ⚠ 构造数据时踩过一次坑：最初把 text 写成 'code_' + src，长度 7，
    // 被 recognize 里的 lenOk()（只接受 4/5 位）挡掉，重试循环压根没进，
    // 5 个用例全部"假失败"。答案必须是合法的 4/5 位。
    let imgVer = 0;
    let seq = [];
    let idx = 0;

    // 'v0' -> 'abv0z'（5 位，合法），既能过 lenOk 又能反查是哪张图
    const codeOf = (src) => 'ab' + src + 'z';

    function setSrc(v) { img.setAttribute('src', v); }

    function installClick() {
        refreshBtn.click = function () {
            this.clicked = (this.clicked || 0) + 1;
            imgVer++;
            setSrc('v' + imgVer);          // 同步换图
        };
    }

    function setupRetry() {
        imgVer = 0; idx = 0;
        setSrc('v0');
        installClick();
        recognizeOnceImpl = async () => ({
            text: codeOf(img.getAttribute('src')),
            minConfidence: seq[idx++],
            engine: 'ONNX'
        });
    }

    // J1 核心：置信忽高忽低时，重试用尽后到底填哪张图的答案
    seq = [0.80, 0.95, 0.88, 0.70];
    await check('J1 重试用尽后，必须填入**屏幕上当前图**的结果（不是历史最高置信那张）',
        async () => {
            setupRetry();
            await api.recognize(img);
            return { filled: input.value, screen: img.getAttribute('src') };
        },
        (g) => g.filled === codeOf(g.screen)
    );

    // J2 三次都低置信且单调下降：best 就是第一次，最容易暴露问题
    seq = [0.90, 0.85, 0.80, 0.75];
    await check('J2 置信单调下降时，同样必须填当前图',
        async () => {
            setupRetry();
            await api.recognize(img);
            return { filled: input.value, screen: img.getAttribute('src') };
        },
        (g) => g.filled === codeOf(g.screen)
    );

    // J3 中途命中高置信 -> 提前退出，此时当前图就是高置信那张，两种实现一致
    seq = [0.80, 0.99995];
    await check('J3 中途命中高置信：提前退出，填入当前图（对照组）',
        async () => {
            setupRetry();
            await api.recognize(img);
            return { filled: input.value, screen: img.getAttribute('src') };
        },
        (g) => g.filled === codeOf(g.screen) && imgVer === 1
    );

    // J4 换图后新图识别抛错：不能拿旧图答案凑数
    await check('J4 换图后新图识别抛错：不得把旧图答案填进去',
        async () => {
            imgVer = 0; idx = 0;
            setSrc('v0');
            installClick();
            let n = 0;
            recognizeOnceImpl = async () => {
                n++;
                if (n === 1) return { text: codeOf(img.getAttribute('src')), minConfidence: 0.80, engine: 'ONNX' };
                throw new Error('ENGINE_DOWN');
            };
            await api.recognize(img);
            return { filled: input.value, screen: img.getAttribute('src'), old: codeOf('v0') };
        },
        (g) => g.filled !== g.old     // 屏幕已是新图，旧图答案一律不得填入
    );

    // J5 换图后新图长度异常：同样不能回退到旧图答案
    await check('J5 换图后新图长度异常：不得回退到旧图答案',
        async () => {
            imgVer = 0; idx = 0;
            setSrc('v0');
            installClick();
            let n = 0;
            recognizeOnceImpl = async () => {
                n++;
                if (n === 1) return { text: codeOf(img.getAttribute('src')), minConfidence: 0.80, engine: 'ONNX' };
                return { text: 'abc', minConfidence: 0.99, engine: 'ONNX' };  // 3 位，长度异常
            };
            await api.recognize(img);
            return { filled: input.value, screen: img.getAttribute('src'), old: codeOf('v0') };
        },
        (g) => g.filled !== g.old
    );

    /* ============ K 组：站点异步换图 ============ */
    section('K 站点异步换图时，等待是否在旧图上就兑现');

    await check('K1 点击后 300ms 才改 src：必须等 src 真的变了才算换图完成',
        async () => {
            imgVer = 0;
            setSrc('A');
            img.complete = true; img.naturalWidth = 110;
            img.decode = () => Promise.resolve();     // 旧图已就绪 -> 会立即兑现
            // 模拟"点击后由 ajax 取回新图，300ms 后才改 src"的站点
            refreshBtn.click = function () {
                this.clicked = (this.clicked || 0) + 1;
                setTimeout(() => setSrc('B'), 300);
            };
            const t0 = Date.now();
            const okv = await api.refreshCaptcha(img);
            const dt = Date.now() - t0;
            return { okv, srcAtResolve: img.getAttribute('src'), ms: dt };
        },
        (g) => g.okv === true && g.srcAtResolve === 'B'
    );

    /* ============ L 组：decode 以 AbortError 拒绝 ============ */
    section('L decode() 被 AbortError 拒绝时的处理');

    await check('L1 AbortError 意味着"有新图在路上"，不能当成图片已就绪立刻放行',
        async () => {
            img.setAttribute('src', 'A');
            img.complete = false; img.naturalWidth = 0;
            let calls = 0;
            img.decode = () => {
                calls++;
                if (calls === 1) {
                    const e = new Error('The source image cannot be decoded.');
                    e.name = 'AbortError';
                    return Promise.reject(e);
                }
                return new Promise(r => setTimeout(r, 120));
            };
            const t0 = Date.now();
            await api.refreshCaptcha(img);
            return { ms: Date.now() - t0, calls };
        },
        (g) => g.ms >= 100      // 至少该等到第二次 decode 兑现
    );

    /* ============ M 组：闸门是否误伤用户自己的换图 ============ */
    section('M 换图闸门不得误伤用户手动换图');

    // 装一个"每次识别耗时 delay 毫秒"的引擎桩。
    // M2 要靠它保证采样时重试循环确实还在跑 —— 否则 recognize 可能瞬间就结束了。
    function installSeq(s, delay) {
        idx = 0; seq = s;
        recognizeOnceImpl = async () => {
            if (delay) await sleep(delay);
            const k = Math.min(idx++, seq.length - 1);
            return { text: codeOf(img.getAttribute('src')), minConfidence: seq[k], engine: 'ONNX' };
        };
    }

    // M1 必须走**完整路径**（recognize 触发换图 -> 循环结束 -> 用户手动换图）。
    // 直接调 refreshCaptcha 会绕过重试循环退出时的闸门解除，测不到真实场景。
    await check('M1 脚本刚换完图（重试已结束）时用户手动换图，必须照常识别',
        async () => {
            imgVer = 0;
            setSrc('v0');
            installClick();
            img.complete = true; img.naturalWidth = 110;
            img.decode = () => Promise.resolve();
            installSeq([0.80, 0.99995], 0);   // 换图 1 次后命中高置信 -> 提前退出循环
            await api.recognize(img);
            const filledAfterAuto = input.value;
            runForCalls.length = 0;
            img.setAttribute('src', 'B');           // 用户紧接着手动换图
            api.schedule('用户手动换图');
            await sleep(150);                       // 越过 80ms 防抖
            return { runs: runForCalls.length, filledAfterAuto };
        },
        (g) => g.runs === 1 && g.filledAfterAuto !== ''
    );

    // M2 反向：重试**进行中**闸门必须仍然有效，不能因为 M1 的修复而复发循环。
    // 这一条是防止"提前解除闸门"这个修复把无限换图循环又放出来。
    await check('M2 重试进行中，换图引发的观察者事件必须仍被静默（防循环复发）',
        async () => {
            imgVer = 0;
            setSrc('v0');
            installClick();
            img.complete = true; img.naturalWidth = 110;
            img.decode = () => Promise.resolve();
            installSeq([0.80, 0.80, 0.80, 0.80], 50);   // 持续低置信 -> 跑满 3 次换图
            runForCalls.length = 0;
            const p = api.recognize(img);
            // 在重试循环进行中塞进一个"由换图引发"的观察者事件
            await sleep(80);
            api.schedule('src 属性变更（由换图引发）');
            await p;
            return { runs: runForCalls.length, refreshed: imgVer };
        },
        (g) => g.refreshed >= 1 && g.runs === 0
    );

    console.log('\n============================================================================');
    console.log('通过 ' + pass + ' / 失败 ' + fail);
    console.log('============================================================================');
    process.exit(fail ? 1 : 0);
})();
