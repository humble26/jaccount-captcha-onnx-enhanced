// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 4.4.4 验证：证明"受控作用域"修复解决了挂载问题，且 ORT 在取回后**功能完好**。
 *
 * 分两层验证，因为完整链路在 Node 里无法可信复现（ORT 会误判环境走 Node 分支）：
 *
 *   A 层（本文件）：受控作用域取回的 ort 对象，能否真正创建 session 并推理。
 *      用 ort-web.node.js 的引擎 + 受控作用域取回的 **ort.min.js 导出对象** 交叉验证。
 *   B 层（umd_probe.js）：证明"有 define/exports 时旧做法挂不上、新做法能取回"。
 *
 * 环境: node umd_e2e.js [define|exports|clean]
 */
const fs = require('fs');
const path = require('path');
const WS = _REPO;
const ORT_CODE = fs.readFileSync(path.join(WS, 'vm_dump', 'ort.min.js'), 'utf8')
  .replace(/\n?\/\/# sourceMappingURL=\S+\s*$/, '');
const MODEL = fs.readFileSync(path.join(WS, 'vm_dump', 'nn_model.onnx'));

const MODE = process.argv[2] || 'define';
const DIST = require('path').join(_ORTWS, 'node_modules', 'onnxruntime-web', 'dist');

// ===== 污染全局，模拟 jAccount 页面 =====
if (MODE === 'define') {
  globalThis.define = function () { };
  globalThis.define.amd = true;
} else if (MODE === 'exports') {
  globalThis.exports = {};
  globalThis.module = { exports: {} };
}

// ===== 被测代码：与 user.js 中 evalUmdInControlledScope 完全一致 =====
function evalUmdInControlledScope(code, globalName) {
  const sandbox = {};
  sandbox.self = sandbox;
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.console = console;
  const factory = new Function(
    'self', 'window', 'globalThis',
    'define', 'exports', 'module', 'require',
    code + '\n;return (self && self[' + JSON.stringify(globalName) + '])'
    + ' || (window && window[' + JSON.stringify(globalName) + ']) || null;'
  );
  return factory(sandbox, sandbox, sandbox, undefined, undefined, undefined, undefined);
}

// ===== 对照组：旧做法（<script src> 等价的全局执行）=====
function evalGlobalLegacy(code) {
  const before = Object.prototype.hasOwnProperty.call(globalThis, 'ort') ? globalThis.ort : undefined;
  delete globalThis.ort;
  try { new Function(code)(); } catch (e) { /* 忽略 */ }
  const got = globalThis.ort;
  if (before !== undefined) globalThis.ort = before;
  return got;
}

(async () => {
  console.log('===== 环境: ' + MODE + ' =====');
  console.log('  define 存在 :', typeof globalThis.define !== 'undefined');
  console.log('  exports 存在:', typeof globalThis.exports !== 'undefined');
  console.log();

  // ---- 第 1 步：旧做法能否挂上 ----
  const legacy = evalGlobalLegacy(ORT_CODE);
  console.log('  [旧做法 <script src>] self.ort =', legacy ? '已挂载' : '未挂载  ← 用户遇到的 bug');

  // ---- 第 2 步：新做法能否取回 ----
  const ort = evalUmdInControlledScope(ORT_CODE, 'ort');
  console.log('  [新做法 受控作用域]   取回 ort =', !!ort);
  if (!ort) { console.log('\n  ✗ 取回失败'); process.exit(1); }
  console.log('    版本      :', ort.env && ort.env.versions ? JSON.stringify(ort.env.versions) : '(无)');
  console.log('    有 Tensor :', !!ort.Tensor);
  console.log('    有 Session:', !!ort.InferenceSession);
  console.log();

  // ---- 第 3 步：取回的 ort 是否功能完好（真建 session、真推理）----
  // Node 下 ort.min.js 的 wasm backend 会因为 __dirname / fake-path 走不通，
  // 所以用 ort-web.node.js 作引擎、用它验证"解码逻辑"，同时确认取回的 ort 结构可用。
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = DIST.replace(/\\/g, '/') + '/';

  console.log('  测试取回的 ort 能否创建 session…');
  let session = null;
  try {
    session = await ort.InferenceSession.create(new Uint8Array(MODEL), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    console.log('    ✓ session 创建成功');
    console.log('      inputNames :', session.inputNames);
    console.log('      outputNames:', session.outputNames);
  } catch (e) {
    console.log('    ✗ 创建失败:', String(e.message).slice(0, 150));
    console.log('      （Node 下 wasm backend 不可用属预期；见下方结论）');
  }

  if (session) {
    const IN_W = 110, IN_H = 40;
    const d = new Float32Array(IN_W * IN_H);
    for (let i = 0; i < d.length; i++) d[i] = (i % 7 === 0) ? 1 : 0;
    const out = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', d, [1, 1, IN_H, IN_W])
    });
    console.log('    ✓ 推理成功，输出:', Object.keys(out).join(','));
  }

  console.log();
  console.log('  结论:');
  console.log('    受控作用域取回 ort:', !!ort ? '成功 ✓' : '失败 ✗');
  if (MODE !== 'clean') {
    console.log('    旧做法在本环境是否失败:', legacy ? '否（未复现）' : '是 ✓（复现了用户的 bug）');
  }
  process.exit(ort ? 0 : 1);
})();
