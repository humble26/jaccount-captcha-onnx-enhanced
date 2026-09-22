// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
// 核心验证：把导出的 logits 喂给「从交付脚本抽取的真实 postprocess」，
// 与 numpy 参考逐张对比。两版（油猴 / 扩展）都要跑，并互相比对。
//
// 为什么这样做：我选阈值时用的是 numpy 版 margin，落地的是 JS 版。
// 两者必须逐张一致，否则「换图率 6.4%」这类结论在真实脚本上不成立。
const fs = require('fs');
const path = require('path');

const W = _REPO;
const US = path.join(W, 'jaccount-captcha-onnx-enhanced.user.js');
const EX = path.join(W, 'extension-src', 'app.js');
const DIR = path.join(W, 'vm_dump', 'audit_sheets');
const META = JSON.parse(fs.readFileSync(path.join(DIR, 'logits_520.json'), 'utf8'));
const BIN = fs.readFileSync(path.join(DIR, 'logits_520.bin'));

// ---------- 括号配平抽取（与既有测试同一套方法）----------
function extractBalanced(src, header) {
    const i = src.indexOf(header);
    if (i < 0) throw new Error('未找到: ' + header);
    let j = src.indexOf('{', i), depth = 1, k = j + 1;
    while (k < src.length && depth > 0) {
        const ch = src[k];
        if (ch === '{') depth++;
        else if (ch === '}') depth--;
        k++;
    }
    return src.slice(i, k);
}

const OUT_NAMES = ['218', '219', '220', '221', '222'];
const DIMS = [26, 26, 26, 26, 27];

function buildPostprocess(file, label) {
    const src = fs.readFileSync(file, 'utf8');
    const softmaxSrc = extractBalanced(src, 'function softmax(');
    const postSrc = extractBalanced(src, 'function postprocess(');
    const CFG = { numClasses: 26, blankIndex: 26, charset: 'abcdefghijklmnopqrstuvwxyz' };
    const logs = [];
    const log = (...a) => logs.push(a.join(' '));
    const warn = (...a) => logs.push('WARN ' + a.join(' '));
    const fn = new Function('CFG', 'log', 'warn',
        softmaxSrc + '\n' + postSrc + '\nreturn postprocess;');
    const pp = fn(CFG, log, warn);
    console.log(`  [${label}] softmax 切片 ${softmaxSrc.length} 字符, postprocess 切片 ${postSrc.length} 字符`);
    return pp;
}

function mkOutputMap(off) {
    const out = {};
    let p = off;
    for (let i = 0; i < 5; i++) {
        const n = DIMS[i];
        const data = new Float32Array(n);
        for (let k = 0; k < n; k++) { data[k] = BIN.readFloatLE(p); p += 4; }
        out[OUT_NAMES[i]] = { data, dims: [1, n] };
    }
    return out;
}

const session = { outputNames: OUT_NAMES };

// ---------- 跑两版 ----------
console.log('抽取真实 postprocess：');
const PP = {
    '油猴4.5.2': buildPostprocess(US, '油猴'),
    '扩展1.0.8': buildPostprocess(EX, '扩展'),
};
console.log();

const results = {};
for (const [label, pp] of Object.entries(PP)) {
    const rows = [];
    for (const m of META) {
        const r = pp(session, mkOutputMap(m.off));
        rows.push({
            key: m.key, split: m.split, truth: m.truth,
            text: r.text,
            minMargin: r.minMargin,
            minConfidence: r.minConfidence,
            marginsLen: r.margins ? r.margins.length : -1,
            confLen: r.confidences ? r.confidences.length : -1,
            ref: m,
        });
    }
    results[label] = rows;
    console.log(`${label}: 跑完 ${rows.length} 张`);
}

// ---------- 对比 1：JS vs numpy 参考 ----------
console.log();
console.log('='.repeat(78));
console.log('对比 1：真实 JS postprocess  vs  numpy 参考');
console.log('='.repeat(78));
for (const [label, rows] of Object.entries(results)) {
    let dText = 0, dMargin = 0, dConf = 0, dLen = 0, maxMd = 0, maxCd = 0;
    const examples = [];
    for (const r of rows) {
        const md = Math.abs(r.minMargin - r.ref.ref_minMargin);
        const cd = Math.abs(r.minConfidence - r.ref.ref_minConf);
        if (r.text !== r.ref.ref_text) { dText++; if (examples.length < 5) examples.push(['text', r.key, r.text, r.ref.ref_text]); }
        if (md > 1e-5) { dMargin++; if (examples.length < 8) examples.push(['margin', r.key, r.minMargin, r.ref.ref_minMargin]); }
        if (cd > 1e-5) { dConf++; if (examples.length < 10) examples.push(['conf', r.key, r.minConfidence, r.ref.ref_minConf]); }
        if (r.marginsLen !== strlen(r.text)) dLen++;
        maxMd = Math.max(maxMd, md); maxCd = Math.max(maxCd, cd);
    }
    console.log(`  ${label}`);
    console.log(`    text 不一致      : ${dText} 张`);
    console.log(`    minMargin 不一致 : ${dMargin} 张  (最大绝对差 ${maxMd.toExponential(2)})`);
    console.log(`    minConf 不一致   : ${dConf} 张  (最大绝对差 ${maxCd.toExponential(2)})`);
    console.log(`    margins 长度 != 字符数: ${dLen} 张`);
    for (const e of examples) console.log('       ', JSON.stringify(e));
}

function strlen(t) { return t.length; }

// ---------- 对比 2：两版互相一致 ----------
console.log();
console.log('='.repeat(78));
console.log('对比 2：油猴版 postprocess  vs  扩展版 postprocess（应逐张完全一致）');
console.log('='.repeat(78));
{
    const a = results['油猴4.5.2'], b = results['扩展1.0.8'];
    let dt = 0, dm = 0, dc = 0, dl = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i].text !== b[i].text) dt++;
        if (Math.abs(a[i].minMargin - b[i].minMargin) > 1e-9) dm++;
        if (Math.abs(a[i].minConfidence - b[i].minConfidence) > 1e-9) dc++;
        if (a[i].marginsLen !== b[i].marginsLen) dl++;
    }
    console.log(`  text 不一致 ${dt} | minMargin 不一致 ${dm} | minConf 不一致 ${dc} | margins 长度不一致 ${dl}`);
}

// ---------- 对比 3：用 JS 的 margin 重算换图率与拦截 ----------
console.log();
console.log('='.repeat(78));
console.log('对比 3：用真实 JS 的 minMargin 重算判据效果（阈值 6）');
console.log('='.repeat(78));
{
    const rows = results['油猴4.5.2'];
    for (const th of [5, 6, 6.5, 7, 8]) {
        const line = [];
        for (const split of ['tune', 'hold', 'new', 'ALL']) {
            const sub = split === 'ALL' ? rows : rows.filter(r => r.split === split);
            const low = sub.filter(r => r.minMargin < th);
            const errs = sub.filter(r => r.text !== r.truth);
            const blocked = low.filter(r => r.text !== r.truth).length;
            line.push(`${split}: 换图${(100 * low.length / sub.length).toFixed(1)}% 拦${blocked}/${errs.length}`);
        }
        console.log(`  阈值 ${String(th).padEnd(4)} ` + line.join('  |  '));
    }
    console.log();
    console.log('  错误样本的 margin（tune/hold 为真实误判，new 为人工修正过的那几张）：');
    for (const r of rows) {
        if (r.text !== r.truth) {
            console.log(`    [${r.split.padEnd(4)}] ${r.key.padEnd(8)} margin=${r.minMargin.toFixed(2)}  conf=${r.minConfidence.toFixed(5)}  ${r.text} -> ${r.truth}`);
        }
    }
}

// ---------- 对比 4：边界行为 ----------
console.log();
console.log('='.repeat(78));
console.log('对比 4：阈值边界行为（isLow 的判定必须是严格的 < ）');
console.log('='.repeat(78));
{
    const CFG_LOW = 6;
    const cases = [5.9999, 6.0, 6.0001];
    for (const m of cases) {
        console.log(`  minMargin = ${m}  ->  isLow = ${m < CFG_LOW}  (期望 ${m < CFG_LOW})`);
    }
    // 用真实数据验证：是否存在 margin 恰好等于 6 的样本
    const rows = results['油猴4.5.2'];
    const exactly = rows.filter(r => Math.abs(r.minMargin - 6) < 1e-9);
    console.log(`  真实样本里 margin 恰好 = 6 的: ${exactly.length} 张`);
    const near = rows.filter(r => Math.abs(r.minMargin - 6) < 0.05);
    console.log(`  margin 落在 5.95~6.05 的: ${near.length} 张`,
        near.slice(0, 6).map(r => `${r.key}=${r.minMargin.toFixed(4)}`).join(' '));
}
