// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 单独测 loadScript：直接从**当前脚本**抽取并执行，验证"先挂事件后赋 src"是否修好，
 * 以及超时是否生效。用最小桩，不用整个脚本。
 */
const fs = require('fs');
const path = require('path');
const WS = _REPO;
const src = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');

// 抽取：从 "function loadScript(urls, timeoutMs) {" 到其配对右括号
function sliceBalanced(s, head) {
  const i = s.indexOf(head);
  if (i < 0) throw new Error('头部未找到: ' + head);
  let d = 0, started = false;
  for (let k = i; k < s.length; k++) {
    const c = s[k];
    if (c === '"' || c === "'" || c === '`') {   // 跳过字符串
      const q = c; k++;
      while (k < s.length) {
        if (s[k] === '\\') { k += 2; continue; }
        if (s[k] === q) break;
        k++;
      }
      continue;
    }
    if (s[k] === '/' && s[k + 1] === '/') { const j = s.indexOf('\n', k); k = j < 0 ? s.length : j; continue; }
    if (s[k] === '/' && s[k + 1] === '*') { const j = s.indexOf('*/', k + 2); k = j < 0 ? s.length : j + 1; continue; }
    if (c === '{') { d++; started = true; }
    else if (c === '}') { d--; if (started && d === 0) return s.slice(i, k + 1); }
  }
  throw new Error('花括号不平衡');
}

const body = sliceBalanced(src, 'function loadScript(urls, timeoutMs) {');
console.log('抽出的 loadScript:', body.length, '字符\n');

const logs = [];
const warn = (...a) => logs.push('WARN ' + a.join(' '));
const log = (...a) => logs.push('LOG ' + a.join(' '));

function makeEnv(behavior, hasHost) {
  const inserted = [];
  const doc = {
    head: hasHost ? { appendChild(n) { inserted.push(n); } } : null,
    documentElement: hasHost ? { appendChild(n) { inserted.push(n); } } : null,
    addEventListener() { }, removeEventListener() { },
    createElement() {
      const el = { _src: '', onload: null, onerror: null };
      Object.defineProperty(el, 'src', {
        get() { return el._src; },
        set(v) {
          el._src = v;
          setTimeout(() => {
            if (behavior === 'ok') { el.onload && el.onload(); }
            else if (behavior === 'fail') { el.onerror && el.onerror(); }
            else if (behavior === 'ok-cached-sync') {
              // 极端：src 赋值微任务内就完成（模拟强缓存）
              Promise.resolve().then(() => el.onload && el.onload());
            }
            // 'silent' 什么都不触发
          }, 1);
        }
      });
      return el;
    }
  };
  return { doc, inserted };
}

async function run(name, behavior, hasHost, timeoutMs) {
  logs.length = 0;
  const env = makeEnv(behavior, hasHost);
  const fn = new Function('document', 'log', 'warn', 'setTimeout', 'clearTimeout',
    'setInterval', 'clearInterval',
    body + '\nreturn loadScript;')(env.doc, log, warn, setTimeout, clearTimeout, setInterval, clearInterval);

  const t0 = Date.now();
  let state = 'pending', val = null;
  const p = fn(['https://a/ort.js', 'https://b/ort.js'], timeoutMs).then(
    v => { state = 'resolved'; val = v; },
    e => { state = 'rejected'; val = e.message; });
  await Promise.race([p, new Promise(r => setTimeout(r, (timeoutMs || 15000) + 500))]);
  const dt = Date.now() - t0;

  const mark = (state === 'resolved') ? '✓' : (state === 'pending' ? '✗ 挂死' : '✗ 被拒');
  console.log(`  ${mark} ${name}`);
  console.log(`      状态=${state} 值=${val} 耗时=${dt}ms 插入script=${env.inserted.length}`);
  logs.forEach(l => console.log('      ' + l));
  return state;
}

(async () => {
  console.log('=== loadScript 修复后行为 ===\n');
  await run('A 正常加载', 'ok', true);
  await run('B 强缓存（src 赋值后微任务即完成）', 'ok-cached-sync', true);
  await run('C 所有源 onerror', 'fail', true);
  await run('D 静默无事件（应超时后换源->最终 reject）', 'silent', true, 300);
  await run('E host 不存在（应轮询兜底，1s 内失败）', 'ok', false, 2000);
  await run('F 全部源超时 -> reject', 'silent', true, 200);
  process.exit(0);
})();
