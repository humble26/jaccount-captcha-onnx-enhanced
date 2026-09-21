/** 用脚本最终采用的配置跑逐样本结果：156二值化 + 白名单 + PSM7，worker 复用 */
const fs = require('fs');
const path = require('path');
const Tesseract = require('tesseract.js');

const W = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40\\vm_dump';
const DIR = path.join(W, 'ablation', 'bin156');
const TRUTH = JSON.parse(fs.readFileSync(path.join(W, 'ground_truth.json'), 'utf8'));
const clean = s => (s || '').toLowerCase().replace(/[^a-z]/g, '');

(async () => {
  const t0 = Date.now();
  const worker = await Tesseract.createWorker('eng', 1);
  await worker.setParameters({
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz',
    tessedit_pageseg_mode: '7',
    preserve_interword_spaces: '0',
    classify_bln_numeric_mode: '0',
  });
  const warm = Date.now() - t0;

  const rows = [];
  let hit = 0;
  for (const f of fs.readdirSync(DIR).filter(x => x.endsWith('.png')).sort()) {
    const id = f.replace('.png', '');
    const t = Date.now();
    const { data } = await worker.recognize(path.join(DIR, f));
    const ms = Date.now() - t;
    const pred = clean(data.text);
    const ok = pred === TRUTH[id];
    if (ok) hit++;
    rows.push({ id, truth: TRUTH[id], pred, ms, conf: data.confidence, ok });
    console.log(`  ${id}  真值=${TRUTH[id].padEnd(6)} 识别=${(pred || '∅').padEnd(7)} ${ok ? '✓' : '✗'}  (${ms}ms)`);
  }
  await worker.terminate();
  console.log(`\n最终配置: ${hit}/${rows.length} = ${(hit / rows.length * 100).toFixed(0)}%  预热 ${warm}ms`);
  fs.writeFileSync(path.join(W, 'tess_final_result.json'),
    JSON.stringify({ warmupMs: warm, hit, n: rows.length, rows }, null, 1), 'utf8');
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
