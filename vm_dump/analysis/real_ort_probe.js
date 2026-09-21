/**
 * 真实 ORT 探针：用**用户脚本实际加载的那个文件** dist/ort.min.js，
 * 以与用户脚本 loadOrt() 完全相同的方式配置 env，然后跑真实模型推理。
 *
 * 目的：沙箱里的假 ort 永远 "成功"，掩盖了真实 ORT 在浏览器里可能的失败。
 * 这里用真 ORT 跑一遍，看它到底会不会抛、抛什么。
 */
const fs = require('fs');
const path = require('path');

const ORT_DIST = process.argv[2] || path.join(
  'C:\\Users\\g1507\\.workbuddy\\binaries\\node\\workspace', 'node_modules', 'onnxruntime-web', 'dist');
const MODEL = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40\\vm_dump\\nn_model.onnx';

// ort.min.js 是浏览器 UMD 包，挂在 self 上；Node 里没有 self，先补一个
globalThis.self = globalThis;

(async () => {
  const dist = ORT_DIST.replace(/\\/g, '/') + '/';
  console.log('dist =', dist);

  // 关键：脚本用 <script src=...ort.min.js> 加载，这里用同样的文件
  const ort = require(path.join(ORT_DIST, 'ort.min.js'));
  console.log('ORT 版本:', ort.env.versions ? JSON.stringify(ort.env.versions) : '(unknown)');

  // 与用户脚本 loadOrt() 完全一致的配置
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = dist;
  console.log('wasmPaths =', ort.env.wasm.wasmPaths);
  console.log('numThreads =', ort.env.wasm.numThreads, ' simd =', ort.env.wasm.simd);

  const buf = fs.readFileSync(MODEL);
  console.log('模型字节:', buf.length);

  let session;
  try {
    session = await ort.InferenceSession.create(new Uint8Array(buf), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    console.log('✓ session 创建成功');
    console.log('  inputNames =', session.inputNames);
    console.log('  outputNames =', session.outputNames);
  } catch (e) {
    console.log('✗ session 创建失败:', e && e.message);
    console.log('   stack:', e && e.stack && e.stack.split('\n').slice(0, 6).join('\n    '));
    process.exit(1);
  }

  // 构造输入：全 0（与任一验证码等价的结构）
  const IN_W = 110, IN_H = 40;
  const data = new Float32Array(IN_W * IN_H);
  for (let i = 0; i < data.length; i++) data[i] = (i % 7 === 0) ? 1 : 0;
  const inputName = session.inputNames[0];
  const input = new ort.Tensor('float32', data, [1, 1, IN_H, IN_W]);

  try {
    const out = await session.run({ [inputName]: input });
    console.log('✓ 推理成功，输出 keys =', Object.keys(out));
    for (const k of Object.keys(out)) {
      const t = out[k];
      console.log(`   ${k}: dims=${JSON.stringify(t.dims)} len=${t.data.length}`);
    }
  } catch (e) {
    console.log('✗ 推理失败:', e && e.message);
    process.exit(2);
  }

  // 检查 ORT 实际去找了哪些 wasm 文件（列出 dist 里有没有）
  console.log('\n--- dist 中 wasm/glue 文件盘点 ---');
  const files = fs.readdirSync(ORT_DIST);
  const need = ['ort-wasm.wasm', 'ort-wasm-simd.wasm', 'ort-wasm-threaded.wasm',
    'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.js', 'ort-wasm-simd.js',
    'ort-wasm.min.js', 'ort-wasm-simd.min.js'];
  for (const n of need) {
    console.log(`  ${files.includes(n) ? '✓' : '✗ 缺失'} ${n}`);
  }
  process.exit(0);
})();
