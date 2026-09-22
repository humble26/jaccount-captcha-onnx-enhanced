// 验证「决策余量不足 → 自动换图重试」逻辑：抽出核心循环做行为测试
// 不依赖真实 DOM/ONNX，用桩模拟 recognizeOnce / refreshCaptcha 的返回序列。
const fs = require('fs');

const SRC = process.argv[2];
const src = fs.readFileSync(SRC, 'utf8');

// ---- 从脚本里抽出关键常量，确保测试的阈值与实现一致 ----
function pick(re, name) {
    const m = src.match(re);
    if (!m) throw new Error('没找到 ' + name);
    return m;
}
const lowM = pick(/lowMargin:\s*([0-9.]+)/, 'lowMargin');
const maxM = pick(/maxRefreshRetries:\s*(\d+)/, 'maxRefreshRetries');
const LOW = Number(lowM[1]);
const MAXR = Number(maxM[1]);
console.log(`从脚本读出: lowMargin=${LOW}  maxRefreshRetries=${MAXR}`);
if (LOW !== 6) throw new Error('lowMargin 不是预期的 6');
if (MAXR !== 3) throw new Error('maxRefreshRetries 不是预期的 3');

// ---- 复刻 recognize 里的重试循环（与实现同构）----
//
// ⚠ 这是**复刻**，不是真实源码。一旦源码改了而这里没跟着改，它照样会通过 ——
// 之前正是因此把"填入历史最高置信那次"这个错误行为锁成了预期。
// 真实行为以 bug_hunt_flow.js / bug_hunt_v2.js 为准（那两个是从交付脚本里
// 抽真实 recognize() 来跑的）。这里保留的价值是：
//   * 从脚本里读出 lowMargin / maxRefreshRetries 做配置校验；
//   * 对"重试次数上限""无控件不空转"这类结构性约束做快速回归。
//
// 4.5.1 同步了「换图后旧结果作废」的语义修正（此前这里还留着 best 回退）。
// 4.5.2 判据由 minConfidence(<0.999) 改为 minMargin(<6)：本文件同步为按间隔书写。
async function runRecognize(seq, canRefresh = true) {
    let i = 0;
    const recognizeOnce = async () => {
        if (i >= seq.length) throw new Error('序列耗尽');
        return { ...seq[i++] };
    };
    let refreshCount = 0;
    const refreshCaptcha = async () => {
        if (!canRefresh) return false;
        refreshCount++;
        return true;
    };
    const isLow = (r) => r && r.minMargin < LOW;
    const lenOk = (t) => t.length === 4 || t.length === 5;

    let res = await recognizeOnce();
    let retries = 0;
    while (isLow(res) && lenOk(res.text || '') && retries < MAXR) {
        retries++;
        const clicked = await refreshCaptcha();
        if (!clicked) break;          // 没换成图 -> res 仍对应屏幕，可用
        const next = await recognizeOnce();
        // 换图之后屏幕已不是刚才那张，旧结果一律作废，不再保留"历史最高置信"
        if (!lenOk(next.text || '')) { res = null; break; }
        res = next;
    }
    return { res, retries, refreshCount, low: isLow(res) };
}

(async () => {
    let pass = 0, fail = 0;
    const check = async (name, fn, expect) => {
        try {
            const got = await fn();
            const ok = expect(got);
            console.log(`${ok ? '✓' : '✗'} ${name}  ->  ${JSON.stringify(got)}`);
            ok ? pass++ : fail++;
        } catch (e) {
            console.log(`✗ ${name}  抛错: ${e.message}`);
            fail++;
        }
    };
    // 间隔约定：3 = 余量不足（触发换图）；12 = 余量充足（不触发）

    // 用例 1：首次就余量充足 -> 不应触发换图
    await check('余量充足一次命中，不换图',
        () => runRecognize([{ text: 'abcd', minMargin: 12 }]),
        g => g.refreshCount === 0 && g.retries === 0 && g.res.text === 'abcd');

    // 用例 2：前两次余量不足，第三次充足 -> 应换图 2 次并采用第三次结果
    await check('余量不足 -> 换图 2 次后命中',
        () => runRecognize([
            { text: 'abcd', minMargin: 3 },
            { text: 'abce', minMargin: 3 },
            { text: 'abcf', minMargin: 12 },
        ]),
        g => g.refreshCount === 2 && g.res.text === 'abcf' && !g.low);

    // 用例 3：一直余量不足 -> 换图达到上限后停止。
    // ⚠ 4.5.1 反转预期：此前断言 'abce'（置信最高那次），那是**已被换掉的第 2 张图**；
    // 换图 3 次后屏幕上是第 4 张，必须填它的结果 'abcg'。
    await check('持续余量不足，换图达上限后采用**最后一次**（屏幕当前那张）',
        () => runRecognize([
            { text: 'abcd', minMargin: 3 },
            { text: 'abce', minMargin: 4 },
            { text: 'abcf', minMargin: 5 },
            { text: 'abcg', minMargin: 3 },
        ]),
        g => g.refreshCount === 3 && g.retries === 3 && g.res.text === 'abcg');

    // 用例 4：找不到换图控件 -> 不空转，立即停止
    await check('无换图控件时立即停止',
        () => runRecognize([{ text: 'abcd', minMargin: 3 }], false),
        g => g.refreshCount === 0 && g.res.text === 'abcd');

    // 用例 5：换图后重试得到长度异常 -> 旧结果随之作废（不是"保留上次结果"）
    // ⚠ 4.5.1 反转预期：既然已经换了图，'abcd' 对应的那张已经不在屏幕上，
    // 保留它就等于填一个必然错误的码。正确行为是作废，由用户自己看一眼。
    await check('换图后重试得到长度异常：旧结果作废，不回退',
        () => runRecognize([
            { text: 'abcd', minMargin: 3 },
            { text: 'abc', minMargin: 3 },
        ]),
        g => g.retries === 1 && g.res === null);

    // 用例 6：余量不足但长度为 5 也允许重试
    await check('5 位码余量不足同样触发重试',
        () => runRecognize([
            { text: 'abcde', minMargin: 3 },
            { text: 'abcdf', minMargin: 12 },
        ]),
        g => g.refreshCount === 1 && g.res.text === 'abcdf');

    // 用例 7：Tesseract 路径 minMargin = Infinity（口径不可比）-> 不应触发重试
    await check('minMargin=Infinity（Tesseract 口径）不触发重试',
        () => runRecognize([{ text: 'abcd', minMargin: Infinity, minConfidence: 0.6 }]),
        g => g.refreshCount === 0);

    // 用例 8：绝不无限循环 —— 给一个超长余量不足序列，确认只换 MAXR 次
    const longSeq = Array.from({ length: 50 }, () => ({ text: 'abcd', minMargin: 2 }));
    await check('超长余量不足序列不会无限循环',
        () => runRecognize(longSeq),
        g => g.refreshCount === MAXR && g.retries === MAXR);

    // 用例 9：阈值边界必须严格小于（margin 恰好 = 6 不触发，5.999 触发）
    // ⚠ 单测里最容易被忽略的一类缺陷：把 `<` 写成 `<=`。
    // 实测 520 张里没有任何样本落在 5.95~6.05，所以线上不会立刻暴露，
    // 但语义边界必须由测试钉死。
    await check('阈值边界：5.999 触发 / 6.0 不触发 / 6.001 不触发',
        async () => {
            const a = await runRecognize(Array.from({ length: 4 }, () => ({ text: 'abcd', minMargin: 5.999 })));
            const b = await runRecognize([{ text: 'abcd', minMargin: 6.0 }]);
            const c = await runRecognize([{ text: 'abcd', minMargin: 6.001 }]);
            return {
                low59: a.low, refresh59: a.refreshCount,
                low60: b.low, refresh60: b.refreshCount,
                low61: c.low, refresh61: c.refreshCount,
            };
        },
        g => g.low59 === true && g.refresh59 === MAXR
            && g.low60 === false && g.refresh60 === 0
            && g.low61 === false && g.refresh61 === 0);

    console.log(`\n通过 ${pass} / 失败 ${fail}`);
    process.exit(fail ? 1 : 0);
})();
