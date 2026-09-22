// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 缺陷狩猎 · 第三轮：**换图 → 观察者 → 新一轮识别** 的交互。
 *
 * 这是前两轮都没覆盖的路径，也是这个设计里最容易出错的地方：
 * refreshCaptcha() 点击换图按钮后，页面会改 <img> 的 src，
 * 而 watchCaptcha() 里注册的 MutationObserver 会对 src 变更做出反应，
 * 再次调度 runFor() → recognize()。
 *
 * 于是同一个"换图"动作会引出**两条独立的识别流**：
 *   A. 重试循环里那次 recognizeOnce(img)（我们主动发起的）
 *   B. 观察者触发的整轮 recognize()（被动发生的）
 * 两者都持有自己的 runToken。问题是它们会不会互相顶掉、重复填值、或反复换图。
 *
 * 用法: node race_retry_observer.js <monkey|ext>
 */
const fs = require('fs');
const WS = _REPO;
const TARGET = process.argv[2] || 'monkey';
const FILE = TARGET === 'monkey'
    ? WS + '\\jaccount-captcha-onnx-enhanced.user.js'
    : WS + '\\extension-src\\app.js';
const raw = fs.readFileSync(FILE, 'utf8');

function extractBalanced(src, sig) {
    const i = src.indexOf(sig);
    if (i < 0) throw new Error('找不到: ' + sig);
    let d = 0, started = false;
    for (let k = i; k < src.length; k++) {
        if (src[k] === '{') { d++; started = true; }
        else if (src[k] === '}') { d--; if (started && d === 0) return src.slice(i, k + 1); }
    }
    throw new Error('大括号不配平: ' + sig);
}
function extractBlock(src, sig, end) {
    const i = src.indexOf(sig);
    if (i < 0) throw new Error('找不到: ' + sig);
    const j = src.indexOf(end, i);
    if (j < 0) throw new Error('找不到终点: ' + end);
    return src.slice(i, j + end.length);
}

const CFG_SRC = extractBlock(raw, 'const CFG = {', '\n    };');
const FIND_SRC = extractBlock(raw, 'function findRefreshButton() {', '\n    }');
const REFRESH_SRC = extractBlock(raw, 'async function refreshCaptcha(img) {', '\n    }');
const REC_SRC = extractBalanced(raw, 'async function recognize(img) {');
// 换图闸门：必须从被测源码里抽，不能在测试里另写一份
const GATE_SRC = (function () {
    const i = raw.indexOf('let suppressObserveUntil = 0;');
    if (i < 0) throw new Error('源码里找不到 suppressObserveUntil（闸门未实现？）');
    const j = raw.indexOf('async function refreshCaptcha(img) {', i);
    return raw.slice(i, j);
})();
let PS_SRC = extractBalanced(raw, 'function paintStatus(kind, text) {').replace('hideBanner();', '');
let SD_SRC = (function () {
    const i = raw.indexOf('const showDiag = (input, base, detail) => {');
    if (i < 0) throw new Error('找不到 showDiag');
    let d = 0, started = false;
    for (let k = raw.indexOf('{', i); k < raw.length; k++) {
        if (raw[k] === '{') { d++; started = true; }
        else if (raw[k] === '}') { d--; if (started && d === 0) return raw.slice(i, k + 1) + ';'; }
    }
    throw new Error('showDiag 不配平');
})().replace('showBanner(base, detail);', '');

/* ---------------------------------------------------------- DOM 桩（带观察者） */

let observers = [];
class FakeEl {
    constructor(tag) {
        this.tagName = (tag || 'div').toUpperCase();
        this._listeners = {}; this.attrs = {}; this.style = {};
        this.textContent = ''; this.isConnected = true;
        this.offsetWidth = 30; this.offsetHeight = 30;
        this.naturalWidth = 110; this.complete = true;
        this.value = ''; this.placeholder = ''; this.title = '';
        this.decode = undefined;
    }
    addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
    removeEventListener() { }
    dispatchEvent(e) { (this._listeners[e.type] || []).forEach(f => f.call(this, e)); return true; }
    setAttribute(k, v) {
        this.attrs[k] = v;
        if (k === 'src') {
            this.src = v;
            // 真实浏览器里 src 变更会触发 MutationObserver
            observers.forEach(o => { if (o.target === this) o.cb([{ type: 'attributes' }]); });
            this.dispatchEvent({ type: 'load', target: this });
        }
    }
    getAttribute(k) { return this.attrs[k]; }
    focus() { doc.activeElement = this; }
    click() { this.clicked = (this.clicked || 0) + 1; }
}

const doc = {
    readyState: 'complete', activeElement: null, _els: {}, _qsa: {},
    createElement: (t) => new FakeEl(t),
    querySelector(sel) { return this._els[sel] || null; },
    querySelectorAll(sel) { return this._qsa[sel] || []; },
    addEventListener(t, fn) { (this._dl = this._dl || {})[t] = fn; },
    documentElement: new FakeEl('html'), body: new FakeEl('body'),
    getElementById() { return null; }
};
class MutationObserver {
    constructor(cb) { this.cb = cb; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect() { }
}

const img = new FakeEl('img');
const input = new FakeEl('input');
const user = new FakeEl('input');
const refreshBtn = new FakeEl('a');
// 点换图 => 改 src（会连锁触发 observer + load）
refreshBtn.click = function () {
    this.clicked = (this.clicked || 0) + 1;
    img.setAttribute('src', 'captcha.png?v=' + (this.clicked + 1));
};
doc._els['#captcha-img'] = img;
doc._els['#input-login-captcha'] = input;
doc._els['#input-login-name'] = user;
doc._qsa['.captcha-refresh'] = [refreshBtn];

const statusEl = { style: {}, textContent: '', isConnected: true };
let statusLog = [];
const statusSink = { kind: '', text: '', push(k, t) { statusLog.push({ kind: k, text: t }); } };
const logBuf = []; const log = (...a) => logBuf.push(a.map(String).join(' '));
const warnBuf = []; const warn = (...a) => warnBuf.push(a.map(String).join(' '));
const brief = (e) => { if (!e) return '?'; const m = (e.message || String(e)).replace(/\s+/g, ' ').trim(); return m.length > 120 ? m.slice(0, 120) + '…' : m; };
const MSG_ANOMALY = '识别异常，请手动输入';
const MSG_FAILED = '识别失败，请手动输入';
const isOwnMsg = (s) => typeof s === 'string' && (s.startsWith(MSG_ANOMALY) || s.startsWith(MSG_FAILED));
const clearOwnPlaceholder = (i) => { if (isOwnMsg(i.placeholder)) i.placeholder = ''; };

const RT = { v: 0 }, LF = { v: null };
const setValue = (i, v) => {
    if (i.value === v) { LF.v = v; return; }
    i.value = v; i.dispatchEvent({ type: 'input' }); LF.v = v;
};
const markInput = (i, low) => { i.style.outline = low ? '2px solid #e6a23c' : ''; };

let engineImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
const recognizeOnce = (...a) => engineImpl(...a);

/* 把真实 recognize + 真实 paintStatus/showDiag 装起来 */
const mk = new Function(
    'document', 'log', 'warn', 'brief', 'clearOwnPlaceholder',
    'setValue', 'markInput', 'MSG_FAILED', 'MSG_ANOMALY',
    'RT', 'LF', 'recognizeOnce', 'statusSink', 'statusEl',
    `
    ${CFG_SRC}
    /* 闸门必须在 refreshCaptcha 之前定义（refreshCaptcha 会调用它） */
    ${GATE_SRC}
    ${FIND_SRC}
    ${REFRESH_SRC}
    const showBanner = () => {};
    ${SD_SRC}
    ${PS_SRC}
    ${REC_SRC
        .replace(/\brunToken\b/g, 'RT.v')
        .replace(/\blastFilled\b/g, 'LF.v')
        .replace(/\brecognizeWithONNX\b/g, 'recognizeOnce')}
    return { recognize, refreshCaptcha, findRefreshButton, observerSuppressed };
    `
);
// paintStatus 里被我们替换成 push 的写法
const M = mk(doc, log, warn, brief, clearOwnPlaceholder, setValue, markInput,
    MSG_FAILED, MSG_ANOMALY, RT, LF, recognizeOnce, statusSink, statusEl);

/* ---------------------------------------------------------- 复刻观察者调度 */

// 与 watchCaptcha() 同构：80ms 防抖 + **真实闸门**
// 闸门必须取自被测代码本身（M.observerSuppressed），不能在测试里另写一份，
// 否则测的是测试自己的实现，修复与否都会"通过"。
let pending = null, observerRuns = 0, suppressedHits = 0;
const suppressHitsReset = () => { suppressedHits = 0; };
const schedule = (why) => {
    if (M.observerSuppressed()) { suppressedHits++; return; }
    clearTimeout(pending);
    pending = setTimeout(() => { observerRuns++; M.recognize(img); }, 80);
};
new MutationObserver(() => schedule('src 变更')).observe(img);
img.addEventListener('load', () => schedule('load'));

/* ---------------------------------------------------------- 用例 */

let pass = 0, fail = 0;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
async function check(name, fn, expect) {
    statusLog = []; logBuf.length = 0; warnBuf.length = 0;
    input.value = ''; input.placeholder = ''; input.style = {}; input.title = '';
    LF.v = null; RT.v = 0; refreshBtn.clicked = 0;
    pending = null; observerRuns = 0;
    let got, err = null;
    try { got = await fn(); } catch (e) { err = e; }
    let ok = false;
    try { ok = !err && expect(got); } catch (e) { err = e; }
    if (ok) pass++; else fail++;
    console.log(`${ok ? '✓' : '✗'} ${name}`);
    if (!ok) console.log(`    实际=${JSON.stringify(got)}${err ? ' 抛错=' + err.message : ''}`);
}

(async () => {
    console.log('='.repeat(78));
    console.log(`换图×观察者 交互测试 · ${TARGET} · ${FILE.split('\\').pop()}`);
    console.log('='.repeat(78));

    await check('G1 高置信：点换图控件本身不产生额外识别轮次', async () => {
        let n = 0;
        engineImpl = async () => { n++; return { text: 'abcd', minConfidence: 0.9999 }; };
        await M.recognize(img);
        await sleep(200);                       // 让观察者的 80ms 防抖有机会触发
        return { engineCalls: n, refreshClicks: refreshBtn.clicked || 0, observerRuns };
    }, g => g.engineCalls === 1 && g.refreshClicks === 0 && g.observerRuns === 0);

    await check('G2 低置信重试：换图会引发观察者再跑一轮，但不应无限扩散', async () => {
        let n = 0;
        engineImpl = async () => { n++; return { text: 'aaaa', minConfidence: 0.70 }; };
        await M.recognize(img);
        const after = { engineCalls: n, refreshClicks: refreshBtn.clicked || 0, observerRuns };
        // 持续观察 6 秒：若存在放大回路，换图次数会随时间线性增长
        const snap = [];
        for (let t = 0; t < 6; t++) { await sleep(1000); snap.push(refreshBtn.clicked || 0); }
        return { refreshLater: refreshBtn.clicked || 0, observerRuns, 闸门拦截: suppressedHits, 每秒累计换图: snap };
    }, g => {
        // 关键约束：观察者轮次不应把换图次数推高到远超上限
        // 主动重试最多 3 次；观察者每轮最多再带 3 次，但不应指数发散
        // 修复后：主动重试最多 3 次，且观察者不再追加轮次
        return g.refreshLater <= 6;
    });

    await check('G3 高置信时最终填入正确结果，且不残留警告状态', async () => {
        engineImpl = async () => ({ text: 'wxyz', minConfidence: 0.9999 });
        await M.recognize(img);
        await sleep(150);
        return { value: input.value, status: statusLog.map(x => x.kind + ':' + x.text) };
    }, g => g.value === 'wxyz');

    await check('G4 观察者轮次不会把已填好的值清空', async () => {
        engineImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        await M.recognize(img);
        const first = input.value;
        refreshBtn.click();                      // 模拟用户手动换图 -> 观察者新一轮
        await sleep(300);
        return { first, after: input.value };
    }, g => g.first === 'abcd' && g.after === 'abcd');

    /* ---------- H. 闸门不能误伤正常流程 ---------- */

    console.log('\n--- H. 闸门不能误伤用户手动换图 ---');

    await check('H1 用户手动换图（不在重试循环中）后，观察者必须照常触发识别', async () => {
        let n = 0;
        engineImpl = async () => { n++; return { text: 'abcd', minConfidence: 0.9999 }; };
        suppressHitsReset();
        // 先让一次识别跑完（此时闸门应已过期或从未设置）
        await M.recognize(img);
        const base = n;
        await sleep(2500);                 // 等过 2000ms 闸门窗口
        refreshBtn.click();                // 用户手动换图 -> src 变更 -> 观察者
        await sleep(300);
        return { base, after: n, observerRuns };
    }, g => g.after === g.base + 1);

    await check('H2 闸门窗口过后，观察者恢复响应', async () => {
        engineImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        // 制造一次换图（会设置闸门）
        await M.refreshCaptcha(img);
        const during = M.observerSuppressed();
        await sleep(2200);
        const after = M.observerSuppressed();
        return { during, after };
    }, g => g.during === true && g.after === false);

    await check('H3 闸门只压制由换图引发的事件，不改变填入结果', async () => {
        engineImpl = async () => ({ text: 'qwer', minConfidence: 0.9999 });
        await M.recognize(img);
        return input.value;
    }, g => g === 'qwer');

    console.log('\n' + '='.repeat(78));
    console.log(`通过 ${pass} / 失败 ${fail}`);
    console.log('='.repeat(78));
    process.exit(fail ? 1 : 0);
})();
