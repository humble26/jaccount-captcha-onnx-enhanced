// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 实测 Tesseract 在真实验证码上的表现，对比两种调用方式：
 *   A 组 = 你当前脚本的方式：Tesseract.recognize(dataURL, 'eng')，原尺寸 JPEG，无任何参数
 *   B 组 = 新脚本兜底方式：复用 worker + 字母白名单 + PSM=8 + 4x上采样/OTSU/白边
 * 使用 tesseract.js@5，与脚本里加载的版本一致。
 */
const fs = require('fs');
const path = require('path');
const Tesseract = require('tesseract.js');

const W = _VD;
const A = path.join(W, 'tess_A_original');
const B = path.join(W, 'tess_B_prepared');
const OUT = path.join(W, 'tesseract_result.json');

const TRUTH = {
  c00: 'yvcew', c01: 'fcvtf', c02: 'ymoad', c03: 'qgjnf', c04: 'txju',
  c05: 'fomr', c06: 'bbmbh', c07: 'wxmh', c08: 'kwwkc', c09: 'afex',
  c10: 'fsyeu', c11: 'jgvc', c12: 'riixo', c13: 'oanu', c14: 'ugal',
  c15: 'odeb', c16: 'mutww', c17: 'ohlf', c18: 'rbrmw', c19: 'whto',
};

const idx = f => path.basename(f).replace(/\.(png|jpg)$/, '');
const cleanOld = s => (s || '').replace(/[^a-zA-Z0-9]/g, '');
const cleanNew = s => (s || '').toLowerCase().replace(/[^a-z]/g, '');

const score = (pred, truth) => {
  if (!pred) return { ok: false, why: 'empty' };
  if (pred === truth) return { ok: true, why: '' };
  if (pred.length !== truth.length) return { ok: false, why: `长度 ${pred.length}≠${truth.length}` };
  let diff = 0;
  for (let i = 0; i < truth.length; i++) if (pred[i] !== truth[i]) diff++;
  return { ok: false, why: `${diff} 个字符错` };
};

(async () => {
  const report = { A: [], B: [], meta: {} };

  console.log('================ A 组：你当前脚本的调用方式 ================');
  const tA0 = Date.now();
  for (const f of fs.readdirSync(A).filter(x => x.endsWith('.jpg')).sort()) {
    const t0 = Date.now();
    const { data } = await Tesseract.recognize(path.join(A, f), 'eng');
    const ms = Date.now() - t0;
    const raw = data.text || '';
    const pred = cleanOld(raw);
    const truth = TRUTH[idx(f)];
    const s = score(pred, truth);
    report.A.push({ id: idx(f), raw, pred, truth, ms, ...s });
    console.log(`  ${idx(f)}  真值=${truth.padEnd(6)} 识别=${pred.padEnd(8)} ${s.ok ? '✓' : '✗ ' + s.why}  (${ms}ms)`);
  }
  report.meta.A_total_ms = Date.now() - tA0;

  console.log('\n================ B 组：新脚本兜底方式 ================');
  const tB0 = Date.now();
  const worker = await Tesseract.createWorker('eng', 1);
  await worker.setParameters({
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz',
    tessedit_pageseg_mode: '8',
    preserve_interword_spaces: '0',
    classify_bln_numeric_mode: '0',
  });
  const warm = Date.now() - tB0;
  console.log(`  （worker 初始化含语言包加载: ${warm}ms，只发生一次）`);
  for (const f of fs.readdirSync(B).filter(x => x.endsWith('.png')).sort()) {
    const t0 = Date.now();
    const { data } = await worker.recognize(path.join(B, f));
    const ms = Date.now() - t0;
    const pred = cleanNew(data.text);
    const truth = TRUTH[idx(f)];
    const s = score(pred, truth);
    report.B.push({ id: idx(f), pred, truth, ms, conf: data.confidence, ...s });
    console.log(`  ${idx(f)}  真值=${truth.padEnd(6)} 识别=${pred.padEnd(8)} ${s.ok ? '✓' : '✗ ' + s.why}  (${ms}ms, conf=${data.confidence})`);
  }
  await worker.terminate();
  report.meta.B_total_ms = Date.now() - tB0;
  report.meta.B_warmup_ms = warm;

  const acc = k => report[k].filter(r => r.ok).length;
  report.meta.A_acc = acc('A'); report.meta.B_acc = acc('B'); report.meta.n = report.A.length;
  console.log('\n================ 汇总 ================');
  console.log(`A 组（当前脚本）: ${acc('A')}/${report.A.length}  总耗时 ${report.meta.A_total_ms}ms`);
  console.log(`B 组（新脚本）  : ${acc('B')}/${report.B.length}  总耗时 ${report.meta.B_total_ms}ms （含一次性预热 ${warm}ms）`);
  const warmAvg = report.B.reduce((a, r) => a + r.ms, 0) / (report.B.length || 1);
  console.log(`B 组热态单张平均: ${warmAvg.toFixed(0)}ms`);
  console.log(`A 组单张平均    : ${(report.meta.A_total_ms / report.A.length).toFixed(0)}ms`);

  fs.writeFileSync(OUT, JSON.stringify(report, null, 1), 'utf8');
  console.log('\n结果已写入 ' + OUT);
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
