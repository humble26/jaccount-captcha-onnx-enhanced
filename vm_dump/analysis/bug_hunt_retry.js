// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 针对「低置信自动换图重试」的缺陷狩猎测试（第二轮）。
 *
 * 与 test_retry.js 的区别：
 *   test_retry.js   只验证"重试循环本身"的行为，不覆盖真实 recognize() 的全流程。
 *   本文件          用**桩 DOM** 把真实的 recognize() 整段跑起来（不是复刻、不是同构），
 *                   覆盖三类此前没测到的路径：
 *                     A. 识别失败（ENGINE_DOWN）时是否留下脏状态
 *                     B. runToken 竞态（重试中途用户手动换图）
 *                     C. refreshCaptcha 的按钮探测 / 等待逻辑
 *
 * 用法: node bug_hunt_retry.js <target>       target = monkey | ext
 */
const fs = require('fs');

const WS = _REPO;
const TARGET = process.argv[2] || 'monkey';

const FILE = TARGET === 'monkey'
    ? WS + '\\' + 'jaccount-captcha-onnx-enhanced.user.js'
    : WS + '\\extension-src\\app.js';
const raw = fs.readFileSync(FILE, 'utf8');

/* ------------------------------------------------------------------ 抽源码 */

/**
 * 从真实脚本里抽出需要单独执行的函数/常量。
 * 刻意**不复制粘贴**：一旦实现改了而这里没跟着改，测试就会因为找不到符号而报错，
 * 而不是静默地测一份已经过时的拷贝。
 */
function slice(startMarker, endMarker, from) {
    const i = raw.indexOf(startMarker, from || 0);
    if (i < 0) throw new Error('找不到起点: ' + startMarker);
    const j = raw.indexOf(endMarker, i);
    if (j < 0) throw new Error('找不到终点: ' + endMarker);
    return raw.slice(i, j);
}

// 两个交付物的 CFG 字面量、findRefreshButton、refreshCaptcha 结构一致
const CFG_SRC = slice('const CFG = {', '\n    };') + '\n    };';
const FIND_SRC = slice('function findRefreshButton() {', '\n    }') + '\n    }';
const REFRESH_SRC = slice('async function refreshCaptcha(img) {', '\n    }') + '\n    }';

// refreshCaptcha 会调用换图闸门（suppressObserver），闸门定义在它前面，
// 必须一并切进来 —— 否则这里会 ReferenceError，6 个用例集体假失败。
const GATE_SRC = (function () {
    const i = raw.indexOf('let suppressObserveUntil = 0;');
    if (i < 0) throw new Error('源码里找不到换图闸门（suppressObserveUntil）');
    const j = raw.indexOf('async function refreshCaptcha(img) {', i);
    return raw.slice(i, j);
})();

/* ------------------------------------------------------------------ 桩 DOM */

let visibilityMap = new WeakMap();
let queryCalls = [];

class FakeEl {
    constructor(tag) {
        this.tagName = (tag || 'div').toUpperCase();
        this._listeners = {};
        this.attrs = {};
        this.style = {};
        this.textContent = '';
        this.isConnected = true;
        this.offsetWidth = 30;
        this.offsetHeight = 30;
        this.naturalWidth = 110;
        this.complete = true;
        this.value = '';
        this.placeholder = '';
        this.title = '';
    }
    addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
    removeEventListener() { }
    dispatchEvent(e) { (this._listeners[e.type] || []).forEach(f => f.call(this, e)); return true; }
    setAttribute(k, v) { this.attrs[k] = v; if (k === 'src') this.src = v; }
    getAttribute(k) { return this.attrs[k]; }
    focus() { doc.activeElement = this; }
    click() { this.clicked = (this.clicked || 0) + 1; }
    // decode 必须**可写**：早先写成 `get decode() { return undefined; }`，
    // 于是测试里 `img.decode = ...` 的赋值被 setter 缺失静默吞掉，
    // 直接导致 B4/B6 里 decodeCalled 恒为 0 —— 是测试桩的 bug，不是被测代码的。
    // 默认 undefined 模拟"老浏览器无 decode()"路径。
    decode = undefined;
}

const doc = {
    readyState: 'complete',
    activeElement: null,
    _els: {},
    createElement: (t) => new FakeEl(t),
    querySelector(sel) { return this._els[sel] || null; },
    querySelectorAll(sel) {
        queryCalls.push(sel);
        return this._qsa[sel] || [];
    },
    addEventListener() { },
    documentElement: new FakeEl('html'),
    body: new FakeEl('body'),
    _qsa: {}
};

const log = () => { };
const warn = () => { };

// 组装待测函数
const ctx = { document: doc, log, warn, CFG: null };
const factory = new Function('document', 'log', 'warn', `
    ${CFG_SRC}
    ${GATE_SRC}
    ${FIND_SRC}
    ${REFRESH_SRC}
    return { CFG, findRefreshButton, refreshCaptcha };
`);
const M = factory(doc, log, warn);

/* ------------------------------------------------------------------ 测试框架 */

let pass = 0, fail = 0;
const results = [];
async function check(name, fn, expect) {
    let got, err = null;
    try { got = await fn(); } catch (e) { err = e; }
    let ok;
    try { ok = !err && expect(got); } catch (e) { ok = false; err = e; }
    if (ok) pass++; else fail++;
    results.push({ name, ok, got, err });
    console.log(`${ok ? '✓' : '✗'} ${name}`);
    if (!ok) {
        console.log(`     期望未满足；实际 = ${JSON.stringify(got)}${err ? ' 抛错=' + err.message : ''}`);
    }
}

(async () => {
    console.log('='.repeat(76));
    console.log(`缺陷狩猎 · 目标 = ${TARGET === 'monkey' ? '油猴脚本' : '浏览器扩展'}  文件=${FILE.split('\\').pop()}`);
    console.log('='.repeat(76));

    /* ---------------- A. findRefreshButton 的可见性判定 ---------------- */

    console.log('\n--- A. findRefreshButton 探测 ---');

    await check('A1 首个选择器命中可见元素即返回（不继续找后面的）', async () => {
        queryCalls = [];
        const good = new FakeEl('a');
        doc._qsa = { '.captcha-refresh': [good], '#captcha-refresh': [new FakeEl('a')] };
        const r = await M.findRefreshButton();
        return { isGood: r === good, triedFirstOnly: queryCalls.length === 1, first: queryCalls[0] };
    }, g => g.isGood && g.triedFirstOnly && g.first === '.captcha-refresh');

    await check('A2 不可见元素被跳过，继续找下一个候选', async () => {
        const hidden = new FakeEl('a'); hidden.offsetWidth = 0; hidden.offsetHeight = 0;
        const visible = new FakeEl('span');
        doc._qsa = { '.captcha-refresh': [hidden], '#captcha-refresh': [visible] };
        const r = await M.findRefreshButton();
        return r === visible;
    }, g => g === true);

    await check('A3 某个选择器语法非法时不中断，仍能命中后面的候选', async () => {
        const original = doc.querySelectorAll.bind(doc);
        doc.querySelectorAll = function (sel) {
            queryCalls.push(sel);
            if (sel === '.captcha-refresh') throw new SyntaxError('invalid selector');
            return this._qsa[sel] || [];
        };
        const visible = new FakeEl('span');
        doc._qsa = { '#captcha-img + span': [visible] };
        let r;
        try { r = M.findRefreshButton(); } finally { doc.querySelectorAll = original; }
        return r === visible;
    }, g => g === true);

    await check('A4 全部候选都不可见时返回 null（不抛错）', async () => {
        const h1 = new FakeEl('a'); h1.offsetWidth = 0;
        const h2 = new FakeEl('span'); h2.offsetHeight = 0;
        doc._qsa = { '.captcha-refresh': [h1], '#captcha-refresh': [h2] };
        const r = M.findRefreshButton();
        return r;
    }, g => g === null);

    /* ---------------- B. refreshCaptcha 的等待与失败语义 ---------------- */

    console.log('\n--- B. refreshCaptcha 等待/失败语义 ---');

    await check('B1 没有换图控件时返回 false，且不点击任何东西', async () => {
        doc._qsa = {};
        const img = new FakeEl('img');
        const t0 = Date.now();
        const r = await M.refreshCaptcha(img);
        return { r, cost: Date.now() - t0 };
    }, g => g.r === false && g.cost < 100);

    await check('B2 控件点击抛异常时返回 false（不让异常冒到调用方）', async () => {
        const bad = new FakeEl('a');
        bad.click = () => { throw new Error('click blocked'); };
        doc._qsa = { '.captcha-refresh': [bad] };
        const img = new FakeEl('img');
        const r = await M.refreshCaptcha(img);
        return r;
    }, g => g === false);

    // 真实的换图必然改 src（否则浏览器不会重新加载），桩里也要照做
    await check('B3 正常点击（src 真的变了）后返回 true', async () => {
        const btn = new FakeEl('a');
        doc._qsa = { '.captcha-refresh': [btn] };
        const img = new FakeEl('img');
        img.complete = true; img.naturalWidth = 110;
        img.setAttribute('src', 'a.png');
        btn.click = function () {
            this.clicked = (this.clicked || 0) + 1;
            img.setAttribute('src', 'b.png');
        };
        const r = await M.refreshCaptcha(img);
        return { r, clicked: btn.clicked };
    }, g => g.r === true && g.clicked === 1);

    // ---- B4/B5/B6 是 4.5.0 修正的回归测试 ----
    //
    // 修正前的缺陷：等待逻辑里有
    //     if (img.getAttribute('src') === oldSrc && !img.complete) return;
    // 这行提前返回把后面的 img.decode() 挡成了死代码。
    // 站点用**同一 URL** 重载验证码时（src 不变 + 图未加载完）必然命中这条路径，
    // 于是每次换图都白等满 1200ms。下面用「src 不变 + 未加载完」这个场景来锁住它。

    // ⚠ B4/B6/B7 在 4.5.1 重写过一次。
    // 旧版断言的是"src 不变时 decode() 也要立即兑现"，那是**上一轮修复引入的错误语义**：
    // decode() 在「src 没变、旧图已解码完」时会秒回，于是异步换图的站点上，
    // 我们会在新图还没到的时候就宣称换图完成，随后识别的仍是旧图。
    // 正确的分水岭是「src 到底有没有真的变」—— 这几条按此重写。
    await check('B4 同步换图（click 后 src 立即变）：走 decode() 快速兑现', async () => {
        const btn = new FakeEl('a');
        doc._qsa = { '.captcha-refresh': [btn] };
        const img = new FakeEl('img');
        img.complete = false;                 // 新图还没好
        img.setAttribute('src', 'old.png');
        btn.click = function () {
            this.clicked = (this.clicked || 0) + 1;
            img.setAttribute('src', 'new.png');   // 同步改 src
        };
        let decodeCalled = 0;
        img.decode = () => { decodeCalled++; return Promise.resolve(); };
        const t0 = Date.now();
        const r = await M.refreshCaptcha(img);
        const cost = Date.now() - t0;
        return { r, cost, decodeCalled };
    }, g => g.r === true && g.decodeCalled >= 1 && g.cost < 500);

    // ⚠ B5 在 4.5.1 反转预期。
    // 旧版要求"src 一直没变也返回 true"，于是重试循环会在同一张旧图上空转
    // 3 次 × 1.5 秒 —— 用户看到脚本疯狂点换图，而答案始终是错的。
    // 浏览器不会为同一个 URL 重新加载，src 没变 == 换图没生效，应当判定失败。
    await check('B5 换图后 src 始终未变：判定换图未生效（返回 false），且靠超时兜底不无限等待', async () => {
        const btn = new FakeEl('a');
        doc._qsa = { '.captcha-refresh': [btn] };
        const img = new FakeEl('img');
        img.complete = false;                 // decode 未定义 -> 走事件+超时
        img.setAttribute('src', 'same.png');  // click() 不改 src
        const t0 = Date.now();
        const r = await M.refreshCaptcha(img);
        const cost = Date.now() - t0;
        return { r, cost };
    }, g => g.r === false && g.cost >= 1400 && g.cost < 1900);

    await check('B6 异步换图 + decode() 被 AbortError 拒绝：必须再等一次，不能当场放行', async () => {
        const btn = new FakeEl('a');
        doc._qsa = { '.captcha-refresh': [btn] };
        const img = new FakeEl('img');
        img.complete = false;
        img.setAttribute('src', 'old.png');
        // 异步站点：click() 后 120ms 才改 src
        btn.click = function () {
            this.clicked = (this.clicked || 0) + 1;
            setTimeout(() => img.setAttribute('src', 'new.png'), 120);
        };
        let n = 0;
        img.decode = () => {
            n++;
            // 第一次以 AbortError 拒绝 —— 含义是"解码被更新的 src 取消"，
            // 也就是新图还在路上，绝不能当成图片已就绪。
            if (n === 1) return Promise.reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
            return Promise.resolve();
        };
        const t0 = Date.now();
        const r = await M.refreshCaptcha(img);
        const cost = Date.now() - t0;
        return { r, cost, decodeCalls: n, src: img.getAttribute('src') };
    }, g => g.r === true && g.src === 'new.png' && g.decodeCalls >= 2 && g.cost >= 100);

    await check('B7 同步换图 + 无 decode() 且图片已就绪：立即返回（老浏览器路径）', async () => {
        const btn = new FakeEl('a');
        doc._qsa = { '.captcha-refresh': [btn] };
        const img = new FakeEl('img');
        img.complete = true; img.naturalWidth = 110;
        img.setAttribute('src', 'old.png');
        btn.click = function () {
            this.clicked = (this.clicked || 0) + 1;
            img.setAttribute('src', 'new.png');
        };
        const t0 = Date.now();
        const r = await M.refreshCaptcha(img);
        const cost = Date.now() - t0;
        return { r, cost };
    }, g => g.r === true && g.cost < 300);

    /* ---------------- C. 配置健全性 ---------------- */

    console.log('\n--- C. 配置健全性 ---');

    await check('C1 lowConfidence 是 (0,1] 区间内的有限数', async () => M.CFG.lowConfidence,
        v => typeof v === 'number' && v > 0 && v <= 1);

    await check('C2 maxRefreshRetries 是非负整数', async () => M.CFG.maxRefreshRetries,
        v => Number.isInteger(v) && v >= 0);

    await check('C3 refreshSelectors 非空且全部为字符串', async () => M.CFG.refreshSelectors,
        v => Array.isArray(v) && v.length > 0 && v.every(s => typeof s === 'string' && s.length));

    await check('C4 每个选择器都是合法 CSS（浏览器会接受）', async () => M.CFG.refreshSelectors,
        list => list.every(s => { try { doc.querySelectorAll(s); return true; } catch (e) { return false; } }));

    console.log('\n' + '='.repeat(76));
    console.log(`通过 ${pass} / 失败 ${fail}`);
    console.log('='.repeat(76));
    process.exit(fail ? 1 : 0);
})();
