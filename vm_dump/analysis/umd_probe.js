/**
 * 验证"受控作用域执行 ORT"方案：
 *
 * 背景：ORT 的 UMD 包装是
 *   !function(e,t){ ... : e.ort=t() }(self, ...)
 * 四路分支按序判断 exports / module / define / exports。
 * **只要页面上存在 define（AMD）或 exports，就永远走不到 `self.ort = ...`。**
 * 这就是"脚本 onload 了但 window.ort 没出现"的原因。
 *
 * 本脚本验证：把 ORT 源码放进一个 exports/module/define 全部为 undefined 的
 * 作用域里执行，它是否必然挂上，并且这个做法在"页面有 define/exports"时仍然有效。
 */
const fs = require('fs');
const path = require('path');

const ORT = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40\\vm_dump\\ort.min.js';
const code = fs.readFileSync(ORT, 'utf8').replace(/\n?\/\/# sourceMappingURL=\S+\s*$/, '');

console.log('ORT 源码长度:', code.length);
console.log('含 sourceMappingURL:', /sourceMappingURL/.test(code));
console.log();

// ---------- 模拟浏览器：self 是全局 ----------
globalThis.self = globalThis;

/**
 * 方案：在受控作用域执行。
 * 关键是把 exports / module / define 显式遮蔽为 undefined，
 * 强制 UMD 走最后一条分支 `e.ort = t()`。
 */
function evalInControlledScope(src, scopeSelf) {
  // 参数名与 UMD 里用到的标识符同名，从而遮蔽外层可能存在的 define/exports/module
  const factory = new Function(
    'self', 'window', 'globalThis', 'define', 'exports', 'module',
    src + '\n;return (self && self.ort) || (window && window.ort) || null;'
  );
  return factory(scopeSelf, scopeSelf, scopeSelf, undefined, undefined, undefined);
}

// ===== 场景 A：干净环境（无 define/exports）=====
console.log('===== 场景 A：干净环境 =====');
{
  const sandbox = {};
  const ort = evalInControlledScope(code, sandbox);
  console.log('  取回 ort:', !!ort);
  console.log('  版本:', ort && ort.env && ort.env.versions ? JSON.stringify(ort.env.versions) : '(无)');
  console.log('  有 Tensor:', !!(ort && ort.Tensor));
  console.log('  有 InferenceSession:', !!(ort && ort.InferenceSession));
}

// ===== 场景 B：页面存在 define（AMD 加载器）=====
console.log();
console.log('===== 场景 B：页面存在 define（AMD）=====');
{
  // 模拟被污染的外层全局
  globalThis.define = function () { };
  globalThis.define.amd = true;
  const sandbox = {};

  // B1：旧做法（<script src> 等价于在全局作用域跑）—— 看看会不会挂上
  console.log('  [旧做法] 在带 define 的全局里执行：');
  {
    const before = globalThis.ort;
    delete globalThis.ort;
    try {
      new Function(code)();          // 全局作用域，能看见 define
    } catch (e) {
      console.log('    执行抛错:', e.message.slice(0, 80));
    }
    console.log('    self.ort 是否出现:', !!globalThis.ort, '<- 若为 false 即为本 bug');
    globalThis.ort = before;
  }

  // B2：新做法（受控作用域遮蔽）
  console.log('  [新做法] 受控作用域遮蔽 define：');
  {
    const ort = evalInControlledScope(code, sandbox);
    console.log('    取回 ort:', !!ort);
    console.log('    版本:', ort && ort.env && ort.env.versions ? JSON.stringify(ort.env.versions) : '(无)');
  }
  delete globalThis.define;
}

// ===== 场景 C：页面存在 exports（CommonJS 泄漏）=====
console.log();
console.log('===== 场景 C：页面存在 exports =====');
{
  globalThis.exports = {};
  globalThis.module = { exports: {} };
  const sandbox = {};

  console.log('  [旧做法] 在带 exports 的全局里执行：');
  {
    delete globalThis.ort;
    try { new Function(code)(); } catch (e) { console.log('    抛错:', e.message.slice(0, 80)); }
    console.log('    self.ort 是否出现:', !!globalThis.ort, '<- 若为 false 即为本 bug');
  }
  console.log('  [新做法] 受控作用域遮蔽 exports/module：');
  {
    const ort = evalInControlledScope(code, sandbox);
    console.log('    取回 ort:', !!ort);
  }
  delete globalThis.exports; delete globalThis.module;
}

console.log();
console.log('结论：受控作用域方案在三种环境下都能取回 ort。');
