/**
 * 稳健版：把两版的 recognize() 原文抽出，配桩执行，观察最终是否写入了 input.value。
 * 这样测的是真实函数行为，不依赖字符串截取的边界猜测。
 */
const fs = require('fs');
const path = require('path');
const WS = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40';

// 抽出 recognize 函数体（从 "async function recognize(img) {" 到下一个顶层 "}"）
function extractRecognize(src) {
  const start = src.indexOf('async function recognize(img) {');
  if (start < 0) throw new Error('未找到 recognize');
  let depth = 0, i = src.indexOf('{', start), end = -1;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
  }
  return src.slice(start, end);
}

const monkeySrc = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');
const extSrc = fs.readFileSync(path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js'), 'utf8');

const recM = extractRecognize(monkeySrc);
const recE = extractRecognize(extSrc);

function makeHarness(recSrc, label) {
  const state = { filled: null, lastFilled: 'abcd' };
  const input = { value: '', placeholder: '', style: {}, title: '' };
  const userEl = { tagName: 'INPUT', focus() { } };

  const doc = {
    activeElement: null,
    querySelector(sel) {
      if (sel === '#input-login-captcha') return input;
      if (sel === '#input-login-name') return userEl;
      return null;
    }
  };
  const CFG = {
    inputSelector: '#input-login-captcha',
    userSelector: '#input-login-name',
    numClasses: 26, blankIndex: 26, charset: 'abcdefghijklmnopqrstuvwxyz'
  };
  // 断言用：只有 setValue 真正改了值才算"覆盖"
  const origValue = { v: '' };
  let wrote = false;
  Object.defineProperty(input, 'value', {
    get() { return origValue.v; },
    set(v) { origValue.v = v; wrote = true; }
  });

  const ctx = {
    input, doc, CFG, state,
    recognizeWithONNX: async () => ({ text: 'wxyz', minConfidence: 0.99 }),
    recognizeWithTesseract: async () => ({ text: 'wxyz', minConfidence: 0.9 }),
    ortFailures: 0,
    setValue(input, value) {
      if (input.value === value) { state.lastFilled = value; return; }
      input.value = value;
      state.lastFilled = value;
    },
    clearOwnPlaceholder() { },
    markInput() { },
    MSG_ANALYSIS: '', MSG_FAILED: '',
    // 4.4.2 新增依赖：诊断与状态输出。桩成 no-op —— 本测试只关心"是否覆盖"，
    // 不关心提示文案。缺了它们会 ReferenceError，测试会以"跑到一半崩"的形式失败，
    // 那种失败很容易被误读成用例不通过。
    paintStatus() { },
    brief: (e) => (e && e.message) || '?',
    isOwnMsg: () => false,
    showDiag() { },
    log() { }, warn() { },
  };

  // 用 with 把 stub 注入作用域
  const fn = new Function('ctx', 'holder', `
    const { input, doc, CFG, state, recognizeWithONNX, recognizeWithTesseract, clearOwnPlaceholder, markInput, log, warn } = ctx;
    const paintStatus = ctx.paintStatus, brief = ctx.brief, isOwnMsg = ctx.isOwnMsg, showDiag = ctx.showDiag;
    let setValue = ctx.setValue;
    const document = doc;
    let lastFilled = state.lastFilled;
    let runToken = 0;
    let ortFailures = 0;
    const ORT_MAX_FAILURES = 2;
    const MSG_ANALYSIS = '识别异常，请手动输入';
    const MSG_FAILED = '识别失败，请手动输入';
    ${recSrc}
    return async function (img, activeIsInput, initialValue, lastFill) {
      state.lastFilled = lastFill;
      input.value = initialValue;
      doc.activeElement = activeIsInput ? input : { tagName: 'BUTTON' };
      holder.wrote = false;
      // 包一层：只要 recognize 决定调用 setValue，就算"发生了覆盖判定"
      setValue = function (inp, value) {
        state.lastFilled = value;
        holder.wrote = true;
        if (inp.value !== value) inp.value = value;
      };
      await recognize(img);
      return holder.wrote;
    };
  `);

  // 需要让 wrote 可见
  const holder = { wrote: false };
  Object.defineProperty(input, 'value', {
    get() { return origValue.v; },
    set(v) { origValue.v = v; holder.wrote = true; }
  });
  // setValue 里 "值相同则提前 return"，那也是一种"决定填入"（只是无需改 DOM）。
  // 所以判定标准改成：recognize 是否走到了 setValue 这一步。
  ctx.setValue = function (inp, value) {
    state.lastFilled = value;
    if (inp.value === value) { holder.wrote = true; return; }  // 决定填入了，只是值没变
    inp.value = value;
  };
  const runner = fn(ctx, holder);
  return { runner, input, holder, doc };
}

(async () => {
  const scenarios = [
    { name: '用户清空准备重新输入',            focus: true,  value: '',     last: 'abcd', want: false },
    { name: '用户已手输 2 个字符',             focus: true,  value: 'xy',   last: 'abcd', want: false },
    { name: '用户已手输完整答案',              focus: true,  value: 'wxyz', last: 'abcd', want: false },
    { name: '框内是我们填的、焦点仍在框内',     focus: true,  value: 'abcd', last: 'abcd', want: false },
    { name: '空值、焦点在别处(自动流程)',       focus: false, value: '',     last: 'abcd', want: true },
    { name: '是我们填的、焦点在别处',           focus: false, value: 'abcd', last: 'abcd', want: true },
    { name: '用户改过值、焦点已离开',           focus: false, value: 'xy',   last: 'abcd', want: true },
  ];

  for (const [label, src] of [['油猴版', recM], ['扩展版', recE]]) {
    console.log(`\n===== ${label} =====`);
    console.log('    用户行为                              焦点在内  框内值     是否覆盖  期望');
    console.log('    ' + '-'.repeat(72));
    let bad = 0;
    for (const sc of scenarios) {
      const h = makeHarness(src, label);
      const got = await h.runner({}, sc.focus, sc.value, sc.last);
      const okk = got === sc.want;
      if (!okk) bad++;
      console.log(`    ${okk ? '✓' : '✗'} ${sc.name.padEnd(34)} ${String(sc.focus).padEnd(9)} ${('"' + sc.value + '"').padEnd(10)} ${String(got).padEnd(9)} ${sc.want}`);
    }
    console.log(`    ${bad ? '✗ ' + bad + ' 项不符' : '✓ 全部符合预期'}`);
  }
})();
