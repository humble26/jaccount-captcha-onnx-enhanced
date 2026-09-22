// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 验证扩展产物的数据通路：完全不碰路径解析，直接喂 wasm 字节 + 模型字节，
 * 跑 220 张样本，与官方 Python 参考实现比对。
 *
 * 这正是扩展在浏览器里走的那条路：
 *   fetch(扩展内 wasm) -> ort.env.wasm.wasmBinary
 *   fetch(扩展内模型)  -> InferenceSession.create(Uint8Array)
 * 因为是直接喂字节，浏览器版/Node 版 bundle 唯一不同的"资源获取"分支压根不会被执行，
 * 所以用 Node 构建来验证这条通路是等价的。
 */
const fs = require('fs');
const path = require('path');

const WS = _ORTWS;
const ORT_NODE = path.join(WS, 'node_modules', 'onnxruntime-web', 'dist', 'ort-web.node.js');
const EXT = require('path').join(_REPO, 'extension-build', 'jaccount-captcha-extension');
const W = _VD;
const WEB = path.join(W, 'webcheck');

const fail = [];
const ok = (cond, msg) => { console.log(`  ${cond ? '✓' : '✗'} ${msg}`); if (!cond) fail.push(msg); };

(async () => {
  // ---------- 1. 产物静态检查 ----------
  console.log('=== 1. 产物静态检查 ===');
  const manifest = JSON.parse(fs.readFileSync(path.join(EXT, 'manifest.json'), 'utf8'));
  ok(manifest.manifest_version === 3, 'manifest_version = 3');
  ok(!!manifest.name && !!manifest.version, `name/version 存在（${manifest.name} v${manifest.version}）`);
  ok(manifest.content_scripts[0].js.includes('content.js'), 'content_scripts 指向 content.js');
  ok(manifest.content_scripts[0].run_at === 'document_start', 'run_at = document_start');
  const war = manifest.web_accessible_resources[0].resources;
  ok(war.includes('assets/ort-wasm-simd.wasm') && war.includes('assets/nn_model.onnx'),
     'wasm 与模型都已声明为 web_accessible_resources');
  ok(!manifest.permissions, '未申请任何多余权限');

  for (const f of ['content.js', 'assets/ort-wasm-simd.wasm', 'assets/nn_model.onnx',
                   'icons/icon16.png', 'icons/icon48.png', 'icons/icon128.png']) {
    ok(fs.existsSync(path.join(EXT, f)), `存在 ${f}`);
  }
  const png = fs.readFileSync(path.join(EXT, 'icons/icon128.png'));
  ok(png.slice(0, 8).equals(Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])), '图标是合法 PNG');

  const content = fs.readFileSync(path.join(EXT, 'content.js'), 'utf8');
  ok(content.includes('ONNX Runtime Web v1.16.3'), 'content.js 内联了 ORT');
  ok(!content.includes('sourceMappingURL'), '已剥离 sourceMappingURL（不留必然会 404 的请求）');

  // ---------- 2. 从产物里抽出真实的 postprocess ----------
  console.log('\n=== 2. 从产物 content.js 中抽取 postprocess ===');
  const start = content.indexOf('function softmax(');
  const end = content.indexOf('/* ------------------------------------------------------------ ONNX 引擎');
  ok(start > 0 && end > start, `定位到 softmax/postprocess（${end - start} 字符）`);
  const CFG = { numClasses: 26, blankIndex: 26, charset: 'abcdefghijklmnopqrstuvwxyz' };
  const { postprocess } = new Function('CFG', 'log', 'warn',
    content.slice(start, end) + '; return { softmax, postprocess };')(CFG, () => {}, () => {});
  ok(typeof postprocess === 'function', 'postprocess 可执行');

  // ---------- 3. 走扩展的真实数据通路跑 220 张 ----------
  console.log('\n=== 3. 扩展数据通路（wasmBinary + 模型字节，零路径解析）===');
  globalThis.self = globalThis;
  const ort = require(ORT_NODE);

  const wasmBuf = fs.readFileSync(path.join(EXT, 'assets', 'ort-wasm-simd.wasm'));
  const modelBuf = fs.readFileSync(path.join(EXT, 'assets', 'nn_model.onnx'));
  console.log(`  wasm ${(wasmBuf.length / 1048576).toFixed(2)}MB  模型 ${(modelBuf.length / 1048576).toFixed(2)}MB`);

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmBinary = new Uint8Array(wasmBuf);   // ← 与扩展完全一致
  // 故意不设 wasmPaths：如果代码仍试图走路径解析，这里就会暴露出来
  ok(ort.env.wasm.wasmPaths === undefined, 'wasmPaths 未设置（证明没有依赖任何外部路径）');

  const t0 = Date.now();
  const session = await ort.InferenceSession.create(new Uint8Array(modelBuf), {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all'
  });
  console.log(`  InferenceSession.create 成功，耗时 ${Date.now() - t0}ms`);
  ok(JSON.stringify(session.inputNames) === '["input.1"]', `inputNames = ${JSON.stringify(session.inputNames)}`);
  ok(JSON.stringify(session.outputNames) === '["218","219","220","221","222"]',
     `outputNames = ${JSON.stringify(session.outputNames)}`);

  const ids = JSON.parse(fs.readFileSync(path.join(WEB, 'ids.json'), 'utf8'));
  const expected = JSON.parse(fs.readFileSync(path.join(WEB, 'expected.json'), 'utf8'));
  const gt = JSON.parse(fs.readFileSync(path.join(W, 'ground_truth_all.json'), 'utf8'));
  const bin = fs.readFileSync(path.join(WEB, 'inputs.bin'));
  const N = 110 * 40;

  let same = 0, hit = 0;
  const inName = session.inputNames[0];
  const tRun = Date.now();
  for (let k = 0; k < ids.length; k++) {
    const f32 = new Float32Array(bin.buffer, bin.byteOffset + k * N * 4, N);
    const out = await session.run({ [inName]: new ort.Tensor('float32', f32, [1, 1, 40, 110]) });
    const res = postprocess(session, out);
    if (res.text === expected[ids[k]]) same++;
    if (res.text === gt[ids[k]]) hit++;
  }
  const ms = Date.now() - tRun;
  console.log(`  220 张总耗时 ${ms}ms，平均 ${(ms / ids.length).toFixed(1)}ms/张`);
  ok(same === ids.length, `与 Python 参考实现完全一致：${same}/${ids.length}`);
  console.log(`  （与人工真值比对准确率：${hit}/${ids.length} = ${(hit / ids.length * 100).toFixed(1)}%）`);

  await session.release();
  console.log(`\n结论: ${fail.length ? '有 ' + fail.length + ' 项未通过 ✗' : '全部通过 ✓'}`);
  process.exit(fail.length ? 1 : 0);
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
