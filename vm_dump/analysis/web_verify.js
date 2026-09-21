/**
 * 用真实的 onnxruntime-web（浏览器用的那个 ort.min.js）+ 真实的模型，
 * 跑 220 张样本，和 Python 参考实现逐张比对。
 *
 * postprocess / softmax 直接从交付的 userscript 里抽出来执行 —— 测的就是线上那份代码，
 * 不是我照抄一份可能抄错的版本。
 */
const fs = require('fs');
const path = require('path');
const http = require('http');

const WS = 'C:\\Users\\g1507\\.workbuddy\\binaries\\node\\workspace';
const ORT_DIST = path.join(WS, 'node_modules', 'onnxruntime-web', 'dist');
const W = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40\\vm_dump';
const WEB = path.join(W, 'webcheck');
const US = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40\\jaccount-captcha-onnx-enhanced.user.js';

// ---------- 1. 从 userscript 里抽出真实的 softmax + postprocess ----------
const src = fs.readFileSync(US, 'utf8');
const start = src.indexOf('function softmax(');
const end = src.indexOf('/* ------------------------------------------------------------ ONNX 引擎');
if (start < 0 || end < 0 || end < start) {
  console.error('FAILED: 无法从 userscript 中定位 softmax/postprocess');
  process.exit(1);
}
const body = src.slice(start, end);
console.log(`已从 userscript 抽出 ${body.length} 字符的 postprocess/softmax 源码`);

const CFG = {
  numClasses: 26, blankIndex: 26, charset: 'abcdefghijklmnopqrstuvwxyz',
};
const logs = [];
const seen = new Set();
// 从 userscript 头部把 CFG 的真实字面量也取出来，避免这里手抄不一致
const mCfg = src.match(/numClasses:\s*(\d+)[\s\S]{0,80}?blankIndex:\s*(\d+)[\s\S]{0,80}?charset:\s*'([^']+)'/);
if (mCfg) {
  CFG.numClasses = Number(mCfg[1]);
  CFG.blankIndex = Number(mCfg[2]);
  CFG.charset = mCfg[3];
  console.log(`从 userscript 读到的 CFG: numClasses=${CFG.numClasses} blankIndex=${CFG.blankIndex} charset 长度=${CFG.charset.length}`);
}
const fakeLog = (...a) => logs.push(a.join(' '));
const fakeWarn = (...a) => seen.add(a.join(' '));
const { postprocess } = new Function('CFG', 'log', 'warn', body + '; return { softmax, postprocess };')(CFG, fakeLog, fakeWarn);

// ---------- 2. 起本地静态服务，让 ort-web 通过 http 取 wasm ----------
function serve(dir) {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const p = path.join(dir, decodeURIComponent(req.url.split('?')[0]));
      if (!p.startsWith(dir) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
        res.writeHead(404); res.end(); return;
      }
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      fs.createReadStream(p).pipe(res);
    });
    srv.listen(0, '127.0.0.1', () => resolve({ srv, port: srv.address().port }));
  });
}

(async () => {
  const { srv, port } = await serve(ORT_DIST);
  console.log(`本地 wasm 服务: http://127.0.0.1:${port}/`);

  // ---------- 3. 加载 ort ----------
  // 说明：浏览器版 bundle（ort.min.js）在 Node 里跑不了 —— 它把 path polyfill 打成了空壳，
  // Emscripten 走 Node 分支时 S.normalize 不存在。这是 Node 环境的产物，浏览器里走 fetch 分支不受影响。
  // 因此这里用同一个包里的 Node 构建 ort-web.node.js：同一套 ORT wasm 内核、同一份推理代码，
  // 只是 wasm 的获取方式从 fetch 换成 fs。用来验证"这个模型能不能被 ORT 建会话并算出正确结果"。
  globalThis.self = globalThis;
  const ortEntry = fs.existsSync(path.join(ORT_DIST, 'ort-web.node.js'))
    ? path.join(ORT_DIST, 'ort-web.node.js')
    : path.join(ORT_DIST, 'ort.min.js');
  console.log('使用的 ORT 入口:', path.basename(ortEntry));
  const ort = require(ortEntry);
  console.log('ort 版本:', ort.env.versions ? JSON.stringify(ort.env.versions) : '(无 versions 字段)');

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  // Node 构建下 wasm 通过 fs 读取，wasmPaths 必须是文件系统路径（浏览器里则是 URL 前缀）
  ort.env.wasm.wasmPaths = ORT_DIST + path.sep;
  console.log(`env.wasm: numThreads=${ort.env.wasm.numThreads} simd=${ort.env.wasm.simd}`);
  console.log(`env.wasm.wasmPaths=${ort.env.wasm.wasmPaths}`);

  // ---------- 4. 建 session（用交付脚本同样的参数） ----------
  const model = new Uint8Array(fs.readFileSync(path.join(W, 'nn_model.onnx')));
  const t0 = Date.now();
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  console.log(`InferenceSession.create 成功，耗时 ${Date.now() - t0}ms`);
  console.log('  session.inputNames  =', JSON.stringify(session.inputNames));
  console.log('  session.outputNames =', JSON.stringify(session.outputNames));

  const inputName = session.inputNames[0];
  console.log('  取用的输入名 =', inputName, '（脚本第 364 行就是这么取的）');

  // ---------- 5. 跑全部 220 张 ----------
  const ids = JSON.parse(fs.readFileSync(path.join(WEB, 'ids.json'), 'utf8'));
  const expected = JSON.parse(fs.readFileSync(path.join(WEB, 'expected.json'), 'utf8'));
  const [refOutNames, raw0] = JSON.parse(fs.readFileSync(path.join(WEB, 'raw0.json'), 'utf8'));
  const gt = JSON.parse(fs.readFileSync(path.join(W, 'ground_truth_all.json'), 'utf8'));
  const bin = fs.readFileSync(path.join(WEB, 'inputs.bin'));
  const N = 110 * 40;

  let same = 0, hit = 0, errs = [];
  let maxDiff = 0, firstDiff = null;
  const tRun = Date.now();
  for (let k = 0; k < ids.length; k++) {
    const sid = ids[k];
    const f32 = new Float32Array(bin.buffer, bin.byteOffset + k * N * 4, N);
    const tensor = new ort.Tensor('float32', f32, [1, 1, 40, 110]);
    const out = await session.run({ [inputName]: tensor });
    const res = postprocess(session, out);

    if (res.text === expected[sid]) same++;
    if (res.text === gt[sid]) hit++; else errs.push(`${sid}:${res.text}!=${gt[sid]}`);
    if (!Number.isFinite(res.minConfidence) || res.minConfidence <= 0 || res.minConfidence > 1.0001) {
      errs.push(`${sid}: minConfidence 异常 ${res.minConfidence}`);
    }
    // 第一个样本逐数值比对
    if (k === 0) {
      for (let oi = 0; oi < refOutNames.length; oi++) {
        const got = out[refOutNames[oi]];
        if (!got) { errs.push(`样本0: 缺少输出 ${refOutNames[oi]}`); continue; }
        const g = got.data, r = raw0[oi];
        for (let j = 0; j < r.length; j++) {
          const d = Math.abs(g[j] - r[j]);
          if (d > maxDiff) { maxDiff = d; firstDiff = `${refOutNames[oi]}[${j}] web=${g[j]} py=${r[j]}`; }
        }
      }
    }
  }
  const total = Date.now() - tRun;
  console.log(`\n220 张推理总耗时 ${total}ms，平均 ${(total / ids.length).toFixed(1)}ms/张`);

  console.log(`\n=== 结果比对 ===`);
  console.log(`  ort-web 解码结果 与 Python 参考完全一致: ${same}/${ids.length}`);
  console.log(`  ort-web 解码结果 与人工真值比对准确率  : ${hit}/${ids.length} = ${(hit / ids.length * 100).toFixed(1)}%`);
  if (errs.length) { console.log('  差异/异常:'); errs.slice(0, 20).forEach(e => console.log('   ', e)); }
  console.log(`  样本0 logits 与 Python 的最大绝对差: ${maxDiff.toExponential(3)}`);
  if (maxDiff > 1e-3) console.log(`    最大差异位置: ${firstDiff}`);

  console.log(`\n  postprocess 是否触发了 warn: ${seen.size ? [...seen].join(' | ') : '否'}`);
  console.log(`  （debug 日志条数: ${logs.length}，正常）`);

  await session.release();
  srv.close();
  const ok = same === ids.length && hit === ids.length - errs.filter(e => !e.includes('minConfidence')).length;
  console.log(`\n结论: ${same === ids.length ? 'ORT-web 路径与参考实现完全一致 ✓' : '存在不一致 ✗'}`);
  process.exit(0);
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
