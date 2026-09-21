/**
 * 精确复现 loadScript 新逻辑的致命缺陷。
 *
 * 看走正常路径（host 存在）：
 *     const host = document.head || document.documentElement;
 *     if (host) { host.appendChild(s); return; }
 *
 * 这里 return 的是 next() 这个箭头函数内部的 return —— 正确，不再往下走。
 * 但注意：**如果 host 存在，我们 appendChild 之后，脚本要靠 s.onload 来 resolve。**
 *
 * 而真实故障场景是：**浏览器把脚本缓存了 / 或者 CSP 拦截 / 或者 onerror 与 onload 都不触发**
 * 此时 Promise 永不 settle。但这个在 4.3.0 也一样。
 *
 * 真正的新增风险：轮询分支的 setInterval 在 attach() 成功后 clearInterval，
 * **但 tries 计数与 clearInterval 的判定顺序有 bug：**
 *     if (attach() || ++tries > 40) { clearInterval(timer); if (tries > 40) next(); }
 * attach() 返回 true 时短路，++tries 不执行 —— 这没问题。
 * 但 **attach() 每次调用都会 appendChild**，如果第一次 attach() 返回 true，
 * 就 return true 短路，不会重复 append。OK。
 *
 * 那问题在哪？—— 让我实际模拟"onload 被重复触发"和"脚本执行顺序"。
 */
const fs = require('fs');
const path = require('path');
const WS = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40';

const cur = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');
const mk = (s) => {
  const i = s.indexOf('function loadScript(urls) {');
  let d = 0, st = s.indexOf('{', i), e = -1;
  for (let k = st; k < s.length; k++) {
    if (s[k] === '{') d++;
    else if (s[k] === '}') { d--; if (!d) { e = k + 1; break; } }
  }
  return s.slice(i, e);
};

const src = mk(cur);

// ---- 场景模拟 ----
function makeEnv(opts) {
  const appended = [];
  const timers = [];
  const doc = {
    head: opts.head ? { appendChild: (n) => { appended.push(n); } } : null,
    documentElement: opts.html ? { appendChild: (n) => { appended.push(n); } } : null,
    addEventListener() { },
    removeEventListener() { },
    createElement: () => {
      const el = { src: '', onload: null, onerror: null };
      // 模拟 src 赋值后浏览器开始加载
      Object.defineProperty(el, 'src', {
        get() { return el._src; },
        set(v) {
          el._src = v;
          setTimeout(() => {
            if (opts.behavior === 'ok') { setTimeout(() => el.onload && el.onload(), 1); }
            else if (opts.behavior === 'all-fail') { setTimeout(() => el.onerror && el.onerror(), 1); }
            else if (opts.behavior === 'silent') { /* 什么都不触发 */ }
          }, 1);
        }
      });
      return el;
    }
  };
  return { doc, appended, timers };
}

const log = () => { };
const warn = (...a) => console.log('    warn:', a.join(' '));

async function run(name, opts) {
  const env = makeEnv(opts);
  const g = { document: env.doc, setTimeout, clearTimeout, setInterval, clearInterval };
  const fn = new Function('document', 'log', 'warn', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval',
    src + '; return loadScript;')(env.doc, log, warn, setTimeout, clearTimeout, setInterval, clearInterval);

  let state = 'pending', result = null;
  const p = fn(['https://a/ort.min.js', 'https://b/ort.min.js', 'https://c/ort.min.js']);
  p.then(v => { state = 'resolved'; result = v; }, e => { state = 'rejected'; result = e.message; });

  await new Promise(r => setTimeout(r, 1500));
  console.log(`  ${name}`);
  console.log(`    最终状态: ${state}${result ? ' -> ' + result : ''}`);
  console.log(`    实际插入 <script>: ${env.appended.length}`);
  return state;
}

(async () => {
  console.log('=== loadScript 各场景 ===\n');
  await run('A. head 存在，脚本正常加载', { head: true, html: true, behavior: 'ok' });
  await run('B. head 存在，所有源都 onerror', { head: true, html: true, behavior: 'all-fail' });
  await run('C. head 存在，但 onload/onerror 都不触发', { head: true, html: true, behavior: 'silent' });
  await run('D. head/html 都不存在（document-start 极早期）', { head: false, html: false, behavior: 'ok' });
  process.exit(0);
})();
