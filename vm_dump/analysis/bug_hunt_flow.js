// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 缺陷狩猎 · 第二轮：把**真实的 recognize() 整体**跑起来（桩 DOM + 桩引擎），
 * 覆盖 test_retry.js 测不到的路径 —— 只在"失败""竞态""多轮交错"时才出现的问题。
 *
 * 设计要点：
 *   * 不复制粘贴实现。recognize() 由源码按大括号配平切出，CFG / findRefreshButton /
 *     refreshCaptcha 同理。实现一旦漂移，测试会因为符号缺失而报错，而不是继续
 *     测一份已经过时的拷贝。
 *   * 可变状态（runToken / lastFilled）用共享对象 S 双向桥接。不能传 getter：
 *     recognize 里有 `++runToken` 这种赋值，getter 返回的只是拷贝，写不回去。
 *
 * 用法: node bug_hunt_flow.js <monkey|ext>
 */
const fs = require('fs');

const WS = _REPO;
const TARGET = process.argv[2] || 'monkey';
const FILE = TARGET === 'monkey'
    ? WS + '\\jaccount-captcha-onnx-enhanced.user.js'
    : WS + '\\extension-src\\app.js';
const raw = fs.readFileSync(FILE, 'utf8');

/* ------------------------------------------------------------ 源码抽取 */

/**
 * 按大括号配平抽一个完整函数体。
 * 最初用 'u.focus();' 之类的文本锚点定尾，切出来的片段语法不完整，
 * 结果 13 个用例全部以 "recognize is not defined" 假失败 —— 又得回头查测试本身。
 * 现在切完立刻用 new Function 试解析，不完整就当场抛错。
 */
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

function extractBlock(src, sig, endMarker) {
    const i = src.indexOf(sig);
    if (i < 0) throw new Error('找不到定义: ' + sig);
    const j = src.indexOf(endMarker, i);
    if (j < 0) throw new Error('找不到终点: ' + endMarker);
    return src.slice(i, j + endMarker.length);
}

const CFG_SRC = extractBlock(raw, 'const CFG = {', '\n    };');
// 真实的 paintStatus：D2/F4 要验证"失败必须可见"，用桩替代等于没测。
// 它依赖 statusEl / ensureStatus —— 沙箱里给一个极简实现即可。
let PS_SRC = extractBalanced(raw, 'function paintStatus(kind, text) {');
// showDiag：油猴版里是 const 箭头函数，扩展版是 const 箭头函数
let SD_SRC = (function () {
    const i = raw.indexOf('const showDiag = (input, base, detail) => {');
    if (i < 0) throw new Error('找不到 showDiag');
    let depth = 0, started = false;
    for (let k = raw.indexOf('{', i); k < raw.length; k++) {
        if (raw[k] === '{') { depth++; started = true; }
        else if (raw[k] === '}') { depth--; if (started && depth === 0) return raw.slice(i, k + 1) + ';'; }
    }
    throw new Error('showDiag 大括号不配平');
})();
SD_SRC = SD_SRC.replace('showBanner(base, detail);', '/* showBanner 已桩掉 */');
PS_SRC = PS_SRC.replace('hideBanner();', '/* hideBanner 已桩掉（本沙箱不测横幅） */');
// 让真实 paintStatus 顺手把内容喂给 statusSink，测试才断言得到
PS_SRC = PS_SRC
    .replace('el.textContent = \'\';', "statusSink.push('', ''), el.textContent = '';")
    .replace("el.textContent = '[验证码识别] ' + mark + text;",
             "statusSink.push(kind, text), el.textContent = '[验证码识别] ' + mark + text;");
const FIND_SRC = extractBlock(raw, 'function findRefreshButton() {', '\n    }');
// 换图闸门：refreshCaptcha 会调用 suppressObserver/observerSuppressed，
// 它们定义在 refreshCaptcha 之前，必须一并切进来，否则调用时 ReferenceError。
const GATE_SRC = (function () {
    const i = raw.indexOf('let suppressObserveUntil = 0;');
    if (i < 0) throw new Error('源码里找不到换图闸门（suppressObserveUntil）');
    const j = raw.indexOf('async function refreshCaptcha(img) {', i);
    return raw.slice(i, j);
})();
const REFRESH_SRC = extractBlock(raw, 'async function refreshCaptcha(img) {', '\n    }');
const REC_SRC = extractBalanced(raw, 'async function recognize(img) {');

/* ------------------------------------------------------------ 桩环境 */

class FakeEl {
    constructor(tag) {
        this.tagName = (tag || 'div').toUpperCase();
        this._listeners = {}; this.attrs = {}; this.style = {};
        this.textContent = ''; this.isConnected = true;
        this.offsetWidth = 30; this.offsetHeight = 30;
        this.naturalWidth = 110; this.complete = true;
        this.value = ''; this.placeholder = ''; this.title = '';
    }
    addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
    removeEventListener() { }
    dispatchEvent(e) { (this._listeners[e.type] || []).forEach(f => f.call(this, e)); return true; }
    setAttribute(k, v) { this.attrs[k] = v; if (k === 'src') this.src = v; }
    getAttribute(k) { return this.attrs[k]; }
    focus() { doc.activeElement = this; }
    click() { this.clicked = (this.clicked || 0) + 1; }
    get decode() { return undefined; }
}

const doc = {
    readyState: 'complete', activeElement: null, _els: {}, _qsa: {},
    createElement: (t) => new FakeEl(t),
    querySelector(sel) { return this._els[sel] || null; },
    querySelectorAll(sel) { return this._qsa[sel] || []; },
    addEventListener() { },
    documentElement: new FakeEl('html'),
    body: new FakeEl('body'),
    getElementById() { return null; }
};

const img = new FakeEl('img');
const input = new FakeEl('input');
const user = new FakeEl('input');
const refreshBtn = new FakeEl('a');
doc._els['#captcha-img'] = img;
doc._els['#input-login-captcha'] = input;
doc._els['#input-login-name'] = user;
doc._qsa['.captcha-refresh'] = [refreshBtn];

// 换图必须真的改 src。4.5.1 起 refreshCaptcha 以"src 是否变化"判定换图是否生效；
// 桩里若只累加 clicked 而不改 src，模拟的其实是"点了但没换成功"，
// 所有重试用例都会在第一次 refreshCaptcha 就退出，根本测不到重试逻辑。
let shotVer = 0;
refreshBtn.click = function () {
    this.clicked = (this.clicked || 0) + 1;
    shotVer++;
    img.setAttribute('src', 'shot' + shotVer + '.png');
};

let statusLog = [];
/* statusEl 给真实 paintStatus 用（它只做 style/textContent 操作） */
const statusEl = { style: {}, textContent: '', isConnected: true };
/* statusSink 是真实 paintStatus 与测试断言之间的桥：
   paintStatus 非空文本时记录一条，空文本（=折叠）时也记一条，方便断言"是否被清空" */
const statusSink = {
    kind: '', text: '',
    push(kind, text) { statusLog.push({ kind, text }); }
};
const logBuf = [];
const log = (...a) => logBuf.push(a.map(String).join(' '));
const warnBuf = [];
const warn = (...a) => warnBuf.push(a.map(String).join(' '));

const brief = (e) => {
    if (!e) return '?';
    const m = (e.message || String(e)).replace(/\s+/g, ' ').trim();
    return m.length > 120 ? m.slice(0, 120) + '…' : m;
};
const showDiag = (inp, base, detail) => {
    inp.placeholder = base + ' [' + detail + ']';
    warnBuf.push(base + ' → ' + detail);
};
const MSG_ANOMALY = '识别异常，请手动输入';
const MSG_FAILED = '识别失败，请手动输入';
const isOwnMsg = (s) => typeof s === 'string' && (s.startsWith(MSG_ANOMALY) || s.startsWith(MSG_FAILED));
const clearOwnPlaceholder = (inp) => { if (isOwnMsg(inp.placeholder)) inp.placeholder = ''; };

/* 共享可变状态：recognize() 要能 ++runToken、要能读写 lastFilled。
   用「持有者对象」而不是裸变量 —— 抽取出来的源码里会把标识符重写成 RT.v / LF.v。 */
const RT = { v: 0 };            // runToken
const LF = { v: null };         // lastFilled
const setValue = (inp, value) => {
    if (inp.value === value) { LF.v = value; return; }
    inp.value = value;
    inp.dispatchEvent(new Event('input', { bubbles: true }));
    LF.v = value;
};
// markInput 从被测源码里抽，不手抄 —— 手抄的那份会随源码修复而过时，
// I1/I2 于是测的是"我记忆中的实现"而不是真家伙（第一版就是这样漏掉修复的）。
const MARK_SRC = (function () {
    const i = raw.indexOf('function markInput(input, low) {');
    if (i < 0) throw new Error('找不到 markInput');
    let d = 0, started = false;
    for (let k = raw.indexOf('{', i); k < raw.length; k++) {
        if (raw[k] === '{') { d++; started = true; }
        else if (raw[k] === '}') { d--; if (started && d === 0) return raw.slice(i, k + 1); }
    }
    throw new Error('markInput 不配平');
})();
// CFG.markLowConfidence 在油猴版里存在（可关闭描边），扩展版没有 -> 用可选链兜底
const markInput = new Function('CFG', MARK_SRC + '; return markInput;')({ markLowConfidence: true });

/* recognizeOnce 每轮用例都换实现，必须每次调用时重新查找 —— 不能绑定当时的函数 */
let recognizeOnceImpl = async () => { throw new Error('未设置 recognizeOnceImpl'); };

// v4.5.2 起生产判据换成了「决策间隔」minMargin < CFG.lowMargin(6)，不再看 softmax 置信度。
// 各用例的桩仍是按置信度写的（0.70 = 低、0.9999 = 高），这里按同一分界线换算成间隔，
// 于是既有用例的语义完全不变，也不必逐个改桩：
//   conf < 0.999（原本会触发换图）-> margin = 3（同样触发）
//   否则                        -> margin = 12（同样不触发）
const marginFor = (conf) => (conf < 0.999 ? 3 : 12);
const recognizeOnce = async (...a) => {
    const r = await recognizeOnceImpl(...a);
    if (r && r.minMargin === undefined) r.minMargin = marginFor(r.minConfidence);
    return r;
};

/* ------------------------------------------------------------ 组装 */

const factory = new Function(
    'document', 'log', 'warn', 'brief',
    'clearOwnPlaceholder', 'setValue', 'markInput', 'MSG_FAILED', 'MSG_ANOMALY',
    'RT', 'LF', 'recognizeOnce', 'statusSink', 'statusEl',
    `
    ${CFG_SRC}
    ${GATE_SRC}
    ${FIND_SRC}
    ${REFRESH_SRC}
    /* showDiag 也用真实实现：它内部会调 paintStatus，用桩会让"失败是否可见"漏测 */
    const showBanner = () => {};          // 横幅不在本沙箱范围内
    ${SD_SRC}
    /* paintStatus 用真实实现（从源码抽出），把结果写进 statusSink。
       之前用桩替代，导致 D2/F4 断言 status 恒为空 —— 测了个寂寞。 */
    ${PS_SRC}
    /* recognize() 以裸标识符读写 runToken / lastFilled —— 原脚本里它们是模块级 let。
       沙箱里这两个状态归 RT / LF 两个持有者管：RT.v 是数字，LF.v 是上一次填入值。
       直接给 new Function 传值行不通（有 ++runToken 这类赋值，传进去的只是拷贝），
       所以把标识符整体重写成 RT.v / LF.v —— 属性访问天然是双向的。 */
    ${REC_SRC
        .replace(/\brunToken\b/g, 'RT.v')
        .replace(/\blastFilled\b/g, 'LF.v')
        // 扩展版直接调用 recognizeWithONNX，油猴版调用 recognizeOnce；
        // 统一映射到沙箱里的 recognizeOnce，两版才能跑同一套用例。
        .replace(/\brecognizeWithONNX\b/g, 'recognizeOnce')}
    return { recognize: recognize };
    `
);
const recognize = factory(
    doc, log, warn, brief,
    clearOwnPlaceholder, setValue, markInput, MSG_FAILED, MSG_ANOMALY,
    RT, LF, recognizeOnce, statusSink, statusEl
).recognize;

/* ------------------------------------------------------------ 用例框架 */

let pass = 0, fail = 0;
async function check(name, fn, expect) {
    statusLog = []; logBuf.length = 0; warnBuf.length = 0;
    input.value = ''; input.placeholder = ''; input.style = {}; input.title = '';
    user.value = ''; doc.activeElement = null;
    LF.v = null; RT.v = 0;
    refreshBtn.clicked = 0;
    shotVer = 0;
    img.setAttribute('src', 'shot0.png');
    img.complete = true; img.naturalWidth = 110; img.decode = null;
    doc._qsa['.captcha-refresh'] = [refreshBtn];

    let got, err = null;
    try { got = await fn(); } catch (e) { err = e; }
    let ok = false;
    try { ok = !err && expect(got); } catch (e) { err = e; }
    if (ok) pass++; else fail++;
    console.log(`${ok ? '✓' : '✗'} ${name}`);
    if (!ok) {
        console.log(`    实际=${JSON.stringify(got)}${err ? ' 抛错=' + err.message : ''}`);
        if (warnBuf.length) console.log('    warn: ' + warnBuf.join(' | ').slice(0, 220));
    }
}

const snap = () => ({
    value: input.value,
    placeholder: input.placeholder,
    outline: input.style.outline || '',
    status: statusLog.map(x => (x.kind || 'hide') + ':' + (x.text || '')),
    clicks: refreshBtn.clicked || 0,
    lastFilled: LF.v,
    warns: warnBuf.slice()
});

(async () => {
    console.log('='.repeat(78));
    console.log(`流程缺陷狩猎 · ${TARGET === 'monkey' ? '油猴' : '扩展'} · ${FILE.split('\\').pop()}`);
    console.log(`切片长度: CFG=${CFG_SRC.length} find=${FIND_SRC.length} refresh=${REFRESH_SRC.length} recognize=${REC_SRC.length}`);
    console.log('='.repeat(78));

    /* ---------- D. 识别失败路径是否留下脏状态 ---------- */

    console.log('\n--- D. 识别失败路径 ---');

    await check('D1 引擎彻底挂掉：失败提示到位，且不清空用户已填内容', async () => {
        recognizeOnceImpl = async () => {
            const e = new Error('ENGINE_DOWN');
            e.ortErr = 'wasm 挂了'; e.tessErr = 'CDN 被墙';
            throw e;
        };
        input.value = 'keepme';
        await recognize(img);
        return snap();
    }, s => s.value === 'keepme' && /识别失败/.test(s.placeholder));

    await check('D2 引擎失败后状态条必须给出错误（不能静默）', async () => {
        recognizeOnceImpl = async () => { const e = new Error('ENGINE_DOWN'); e.ortErr = 'x'; e.tessErr = 'y'; throw e; };
        await recognize(img);
        return snap();
    }, s => s.status.some(t => t.startsWith('err')));

    await check('D3 长度异常：给出异常提示且不填值', async () => {
        recognizeOnceImpl = async () => ({ text: 'abc', minConfidence: 0.99 });
        await recognize(img);
        return snap();
    }, s => s.value === '' && /识别异常/.test(s.placeholder));

    await check('D4 长度异常不触发换图（长度问题 ≠ 置信问题）', async () => {
        recognizeOnceImpl = async () => ({ text: 'abc', minConfidence: 0.99 });
        await recognize(img);
        return snap();
    }, s => s.clicks === 0);

    /* ---------- E. 重试循环 ---------- */

    console.log('\n--- E. 重试循环与竞态 ---');

    await check('E1 重试期间被更新的任务取代：放弃本轮且不填值', async () => {
        let n = 0;
        recognizeOnceImpl = async () => {
            n++;
            if (n === 1) return { text: 'abcd', minConfidence: 0.70 };
            RT.v++;                 // 模拟用户此刻手动换了图
            await new Promise(r => setImmediate(r));
            return { text: 'zzzz', minConfidence: 0.9999 };
        };
        await recognize(img);
        return snap();
    }, s => s.value === '');

    await check('E2 高置信一次命中：不点换图、正常填值、状态条清空', async () => {
        recognizeOnceImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        await recognize(img);
        return snap();
    }, s => s.value === 'abcd' && s.clicks === 0
        && s.status.length > 0 && s.status[s.status.length - 1] === 'hide:');

    await check('E3 低置信 → 换图命中高置信：填高置信那次', async () => {
        let n = 0;
        recognizeOnceImpl = async () => (++n === 1
            ? { text: 'aaaa', minConfidence: 0.70 }
            : { text: 'bbbb', minConfidence: 0.9999 });
        await recognize(img);
        return snap();
    }, s => s.value === 'bbbb' && s.clicks === 1 && s.status[s.status.length - 1] === 'hide:');

    // ⚠ E4 在 4.5.1 反转过预期。
    // 旧版断言"填入历史最高置信那次（0.95 -> 'bbbb'）"，这是**错误的**：
    // 换图 3 次之后屏幕上是第 4 张图，'bbbb' 属于早已被换掉的第 2 张。
    // 把它的答案填进输入框，与用户眼前看到的验证码不符 —— 必然登录失败。
    // 置信度是"这一张能不能信"的判断，不是跨图片比较的分数。
    await check('E4 重试用尽仍低置信：填**最后一次**（屏幕当前那张），不是历史最高', async () => {
        // 文本必须是**合法的 4 位码**，否则会先撞上长度校验，测不到选择逻辑
        const seq = [0.90, 0.95, 0.80, 0.85];
        const txt = ['aaaa', 'bbbb', 'cccc', 'dddd'];
        let n = 0;
        recognizeOnceImpl = async () => ({ text: txt[n], minConfidence: seq[n++] });
        await recognize(img);
        return snap();
    }, s => s.clicks === 3 && s.value === 'dddd' && /决策余量仍不足/.test(s.status[s.status.length - 1]));

    await check('E5 焦点在验证码框内：识别成功也不抢填', async () => {
        recognizeOnceImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        input.focus();
        await recognize(img);
        return snap();
    }, s => s.value === '' && doc.activeElement === input);

    await check('E6 低置信加橙色描边，高置信清除', async () => {
        recognizeOnceImpl = async () => ({ text: 'aaaa', minConfidence: 0.70 });
        await recognize(img);
        const low = input.style.outline;
        recognizeOnceImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        await recognize(img);
        return { low, high: input.style.outline };
    }, g => /e6a23c/.test(g.low) && !/e6a23c/.test(g.high || ''));

    /* ---------- I. title 不被 success 路径抹掉 ---------- */

    console.log('\n--- I. 诊断信息（title）保留 ---');

    await check('I1 失败写入的 title 在随后一次成功识别后仍然保留', async () => {
        recognizeOnceImpl = async () => { const e = new Error('ENGINE_DOWN'); e.ortErr = 'wasm 起不来'; e.tessErr = 'x'; throw e; };
        await recognize(img);
        const afterFail = input.title;
        recognizeOnceImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        await recognize(img);
        return { afterFail, afterSuccess: input.title };
    }, g => {
        // 断言"失败时写了 title"且"成功后 title 没被抹掉"，不写死具体文案：
        //   油猴版 detail 是 'ONNX: … ｜ Tess: …'，扩展版是 brief(e) = 'ENGINE_DOWN'
        // 纠结文案会把一个正确的版本差异误判为缺陷。
        return !!g.afterFail && g.afterSuccess === g.afterFail;
    });

    await check('I2 低置信标注 title 后，高置信一轮应清掉该标注', async () => {
        recognizeOnceImpl = async () => ({ text: 'aaaa', minConfidence: 0.70 });
        await recognize(img);
        const low = input.title;
        recognizeOnceImpl = async () => ({ text: 'abcd', minConfidence: 0.9999 });
        await recognize(img);
        return { low, high: input.title };
    }, g => /决策余量不足/.test(g.low) && g.high === '');

    /* ---------- F. 换图控件缺失 / 重试异常 ---------- */

    console.log('\n--- F. 换图控件缺失与重试异常 ---');

    await check('F1 找不到换图控件：不空转，用首次结果继续', async () => {
        doc._qsa['.captcha-refresh'] = [];
        recognizeOnceImpl = async () => ({ text: 'abcd', minConfidence: 0.70 });
        await recognize(img);
        return snap();
    }, s => s.clicks === 0 && s.value === 'abcd');

    // ⚠ F2/F3 在 4.5.1 反转过预期：旧版要求"保留首次结果"。
    // 但既然已经换过图，屏幕上就不是首次那张图了 —— 保留它的答案等于填错的码。
    // 正确行为是作废并给出可见提示，由用户自己看一眼。
    await check('F2 换图后新图识别抛错：旧结果随之作废，且必须可见提示', async () => {
        let n = 0;
        recognizeOnceImpl = async () => {
            if (++n === 1) return { text: 'abcd', minConfidence: 0.70 };
            throw new Error('推理中途崩了');
        };
        await recognize(img);
        return snap();
    }, s => s.value === '' && s.clicks === 1 && /识别失败/.test(s.placeholder));

    await check('F3 换图后新图长度异常：旧结果随之作废，且必须可见提示', async () => {
        let n = 0;
        recognizeOnceImpl = async () => (++n === 1
            ? { text: 'abcd', minConfidence: 0.70 }
            : { text: 'ab', minConfidence: 0.99 });
        await recognize(img);
        return snap();
    }, s => s.value === '' && s.clicks === 1 && /识别失败/.test(s.placeholder));

    await check('F4 首次识别就抛非引擎类错误：必须给出可见失败提示，不能静默', async () => {
        recognizeOnceImpl = async () => { throw new Error('意想不到的崩溃'); };
        await recognize(img);
        return snap();
    }, s => {
        // 两版的语义差异是**刻意的**，不是缺陷：
        //   油猴版有 ONNX + Tesseract 两路，抛错只是"这一路挂了" -> 报"识别异常"
        //   扩展版只有 ONNX 一路（资源全在本地，失败=包损坏）-> 报"识别失败"
        // 共同底线是「必须有可见提示」，这里就只断言这一点，
        // 否则会把一个正确的设计差异误报成 bug。
        const expected = (TARGET === 'monkey') ? /识别异常/ : /识别失败/;
        return expected.test(s.placeholder) && s.status.some(t => t.startsWith('err'));
    });

    console.log('\n' + '='.repeat(78));
    console.log(`通过 ${pass} / 失败 ${fail}`);
    console.log('='.repeat(78));
    process.exit(fail ? 1 : 0);
})();
