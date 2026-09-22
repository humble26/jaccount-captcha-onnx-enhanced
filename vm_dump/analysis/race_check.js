// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 检查 runFor/img.decode 的竞态处理。
 * 真实浏览器里：img.src 改变后立刻调用 decode()，会以 AbortError 拒绝（当解码被新的 src 取代）。
 * 关键问题：decode() 拒绝后我们直接 return（丢弃本轮），那么这一轮还会不会有人重试？
 */
const fs = require('fs');
const path = require('path');
const WS = _REPO;
const src = fs.readFileSync(path.join(WS, 'extension-src', 'app.js'), 'utf8');

// 抽 runFor
const s0 = src.indexOf('async function runFor(img) {');
const s1 = src.indexOf('\n    }', src.indexOf('recognize(img);', s0)) + 6;
const runForSrc = src.slice(s0, s1);
console.log('--- 抽出的 runFor ---');
console.log(runForSrc);
console.log('--- end ---\n');

let calls = [];
const mkImg = (opts) => ({
  complete: opts.complete !== false,
  naturalWidth: opts.naturalWidth === undefined ? 110 : opts.naturalWidth,
  decode: opts.decodeThrows
    ? () => Promise.reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))
    : (opts.decode === undefined ? undefined : opts.decode),
  addEventListener(t, fn) { calls.push('addEventListener:' + t); }
});

const recognizeStub = (img) => { calls.push('recognize'); };

const runFor = new Function('log', 'recognize', runForSrc + '; return runFor;')(() => { }, recognizeStub);

(async () => {
  let bad = 0;
  const t = async (name, img, wantRecognize) => {
    calls = [];
    await runFor(img);
    await new Promise(r => setTimeout(r, 10));
    const did = calls.includes('recognize');
    const mark = did === wantRecognize ? '✓' : '✗';
    if (did !== wantRecognize) bad++;
    console.log(`  ${mark} ${name.padEnd(46)} recognize=${did} (期望 ${wantRecognize})`);
  };

  console.log('场景:');
  await t('decode 存在且成功 -> 识别', mkImg({ decode: () => Promise.resolve() }), true);
  await t('decode 抛 AbortError -> 本轮丢弃', mkImg({ decodeThrows: true }), false);
  await t('无 decode, complete=true -> 识别', mkImg({ decode: undefined, complete: true }), true);
  await t('无 decode, complete=false -> 等 load 后识别', mkImg({ decode: undefined, complete: false }), true);
  await t('naturalWidth=0 (图加载失败) -> 不识别', mkImg({ decode: () => Promise.resolve(), naturalWidth: 0 }), false);

  console.log(`\n${bad ? bad + ' 项异常' : '全部符合预期'}`);
  console.log('\n注意 AbortError 场景：runFor 直接 return，**没有任何重试**。');
  console.log('但 watchCaptcha 的 MutationObserver 会在下一次 src 变更/load 时再次触发，');
  console.log('所以 AbortError 只在"图已换成最终版本、但事件已派发完"这一瞬间才可能漏图。');
  process.exit(0);
})();
